"""Fast oracle-label validation for perception, action learning, and transfer."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from sdpo_result_feedback.actions import ACTION_LABELS
from sdpo_result_feedback.env_adapter import InvalidScene, PandaPushOutcomePilot
from sdpo_result_feedback.evaluation import json_write
from sdpo_result_feedback.policy import PolicyConfig, QwenActionPolicy
from sdpo_result_feedback.scenes import create_directional_long_scene_splits
from sdpo_result_feedback.trainer import observation_transition


ROOT = Path(__file__).resolve().parents[1]
ACTION_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}
DIRECTION_ACTION = {'+x': 'A', '-x': 'B', '+y': 'C', '-y': 'D'}
INPUT_CONDITIONS = ('rgb', 'target_direction', 'oracle_action')
TRAIN_CONDITIONS = ('rgb', 'target_direction')


def target_direction(scene) -> str:
    delta = np.asarray(scene.goal_xy) - np.asarray(scene.object_xy)
    if abs(delta[0]) >= abs(delta[1]):
        return '+x' if delta[0] > 0 else '-x'
    return '+y' if delta[1] > 0 else '-y'


def context_for(condition: str, row: dict) -> str | None:
    if condition == 'rgb':
        return None
    if condition == 'target_direction':
        return (f"Verified target direction from cube to target: "
                f"{row['target_direction']}. Use it as diagnostic input.")
    if condition == 'oracle_action':
        return (f"Verified simulator oracle action for this exact state: "
                f"{row['oracle_action']}. Follow this action exactly; this is an "
                "action-hint upper-bound check, not a learned-action measurement.")
    raise ValueError(f'Unknown condition: {condition}')


def collect_oracle_rows(scenes, seed: int) -> tuple[list[dict], list[np.ndarray]]:
    env = PandaPushOutcomePilot(max_decisions=1, seed=seed)
    rows = []
    images = []
    try:
        for scene in scenes:
            try:
                observation = env.reset(f'oracle_{scene.scene_id}', scene)
            except InvalidScene as error:
                raise RuntimeError(f'Invalid frozen scene {scene.scene_id}: {error}') from error
            snapshot = env.snapshot()
            try:
                env.restore(snapshot)
                initial = observation_transition(env, observation)
                candidates = {}
                for label in ACTION_LABELS:
                    env.restore(snapshot)
                    _, transition = env.step(label)
                    candidates[label] = {
                        'success': bool(transition.result.success),
                        'progress_m': float(transition.result.progress_m),
                        'contact': bool(transition.result.contact),
                        'safety_failure': transition.safety_failure,
                        'distance_after_m': float(transition.distance_after_m),
                    }
                oracle = max(
                    ACTION_LABELS,
                    key=lambda label: (
                        candidates[label]['success'],
                        candidates[label]['safety_failure'] is None,
                        candidates[label]['progress_m'],
                        candidates[label]['contact'],
                    ),
                )
                rows.append({
                    'scene_id': scene.scene_id,
                    'split': scene.split,
                    'object_xy': list(scene.object_xy),
                    'goal_xy': list(scene.goal_xy),
                    'ee_xy': list(scene.ee_xy),
                    'target_direction': target_direction(scene),
                    'direction_action': DIRECTION_ACTION[target_direction(scene)],
                    'oracle_action': oracle,
                    'candidates': candidates,
                    'transition': initial.as_json_record(),
                })
                images.append(initial.image)
            finally:
                env.remove_snapshot(snapshot)
    finally:
        env.close()
    return rows, images


def load_policy(seed: int, model_id: str | None = None) -> QwenActionPolicy:
    config = PolicyConfig(seed=seed)
    if model_id is not None:
        config = PolicyConfig(seed=seed, model_id=model_id)
    return QwenActionPolicy(config)


def transitions_from_rows(rows):
    from types import SimpleNamespace

    transitions = []
    for row in rows:
        record = row['transition']
        transitions.append(SimpleNamespace(
            episode_id=record['episode_id'],
            decision_index=record['decision_index'],
            image=None,
            instruction=record['instruction'],
            proprio=np.asarray(record['proprio'], dtype=np.float32),
            action_label='E',
            result=None,
            scene_id=row['scene_id'],
        ))
    return transitions


def predict(policy, rows, condition: str, batch_size: int = 8) -> np.ndarray:
    transitions = transitions_from_rows(rows)
    for transition, row in zip(transitions, rows):
        transition.image = row['_image']
    probs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        batch = transitions[start:start + batch_size]
        contexts = [context_for(condition, row) for row in batch_rows]
        with torch.no_grad():
            probs.append(policy.log_probs(
                batch, grad=False, additional_context=contexts,
            ).exp().cpu().numpy())
    return np.concatenate(probs)


def score(rows, probs) -> dict:
    predicted = [ACTION_LABELS[int(index)] for index in probs.argmax(axis=1)]
    oracle = [row['oracle_action'] for row in rows]
    direction = [row['direction_action'] for row in rows]
    chosen = [row['candidates'][label] for row, label in zip(rows, predicted)]
    return {
        'scenes': len(rows),
        'oracle_action_accuracy': float(np.mean(np.asarray(predicted) == np.asarray(oracle))),
        'target_direction_action_accuracy': float(np.mean(np.asarray(predicted) == np.asarray(direction))),
        'chosen_one_step_success_rate': float(np.mean([item['success'] for item in chosen])),
        'chosen_mean_progress_m': float(np.mean([item['progress_m'] for item in chosen])),
        'greedy_action_counts': dict(Counter(predicted)),
        'mean_action_probabilities': probs.mean(axis=0).tolist(),
        'predicted_actions': predicted,
    }


def train_one_condition(name: str, train_rows: list[dict], dev_rows: list[dict], output: Path,
                        epochs: int, batch_size: int, seed: int, label_key: str,
                        model_id: str | None = None) -> dict:
    policy = load_policy(seed, model_id)
    history = []
    train_transitions = transitions_from_rows(train_rows)
    for transition, row in zip(train_transitions, train_rows):
        transition.image = row['_image']
    target = torch.tensor([ACTION_INDEX[row[label_key]] for row in train_rows], device=policy.device)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        train_probs = predict(policy, train_rows, name)
        dev_probs = predict(policy, dev_rows, name)
        history.append({
            'epoch': epoch,
            'train': score(train_rows, train_probs),
            'held_out': score(dev_rows, dev_probs),
        })
        if epoch == epochs:
            break
        order = rng.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch = [train_transitions[index] for index in indices]
            contexts = [context_for(name, train_rows[index]) for index in indices]
            logits = policy.log_probs(batch, grad=True, additional_context=contexts)
            loss = F.nll_loss(logits, target[torch.as_tensor(indices, device=policy.device)])
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), 1.)
            policy.optimizer.step()
            losses.append({'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        history[-1]['optimizer'] = {
            'batches': len(losses),
            'mean_loss': float(np.mean([item['loss'] for item in losses])),
            'mean_gradient_norm': float(np.mean([item['gradient_norm'] for item in losses])),
        }
    checkpoint = output / name / 'checkpoint'
    policy.save(checkpoint)
    result = {'condition': name, 'history': history,
              'elapsed_seconds': time.perf_counter() - started,
              'checkpoint': str(checkpoint)}
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-scenes', type=int, default=64)
    parser.add_argument('--held-out-scenes', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--label', choices=('oracle_action', 'direction_action'),
                        default='oracle_action')
    parser.add_argument('--model-id', type=str, default=None)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite existing run: {args.output}')
    args.output.mkdir(parents=True)
    if args.train_scenes < 8 or args.held_out_scenes < 8:
        raise SystemExit('Need at least 8 train and held-out scenes')

    splits = create_directional_long_scene_splits(ROOT / 'artifacts/directional_long_scenes_v1.json')
    train_scenes = splits['train'][:args.train_scenes]
    held_out_scenes = splits['dev'][:args.held_out_scenes]
    started = time.perf_counter()
    train_rows, train_images = collect_oracle_rows(train_scenes, args.seed)
    held_out_rows, held_out_images = collect_oracle_rows(held_out_scenes, args.seed + 1)
    all_rows = train_rows + held_out_rows
    all_images = train_images + held_out_images
    for row, image in zip(all_rows, all_images):
        row['_image'] = image
    serializable_rows = []
    for row in all_rows:
        serializable_rows.append({key: value for key, value in row.items() if key != '_image'})
    json_write(args.output / 'oracle_dataset.json', {
        'train': serializable_rows[:len(train_rows)],
        'held_out': serializable_rows[len(train_rows):],
        'oracle_definition': 'success first, then safe one-step progress, then contact',
        'candidate_actions': list(ACTION_LABELS),
    })
    np.savez_compressed(args.output / 'pre_action_images.npz', images=np.stack(all_images))

    frozen_policy = load_policy(args.seed, args.model_id)
    frozen_probe = {}
    for condition in INPUT_CONDITIONS:
        frozen_probe[condition] = {
            'train': score(train_rows, predict(frozen_policy, train_rows, condition)),
            'held_out': score(held_out_rows, predict(frozen_policy, held_out_rows, condition)),
        }
    del frozen_policy
    gc.collect()
    torch.cuda.empty_cache()

    results = {'frozen_probe': frozen_probe, 'supervised': {}, 'config': {
        'train_scenes': args.train_scenes, 'held_out_scenes': args.held_out_scenes,
        'epochs': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
        'input_conditions': list(INPUT_CONDITIONS),
        'supervised_conditions': list(TRAIN_CONDITIONS),
        'model': 'Qwen2.5-VL-3B-Instruct',
        'supervision': 'simulator exhaustive one-step oracle action',
        'oracle_action_input_role': 'diagnostic hint-following upper bound only; excluded from learned-action claims',
        'label': args.label,
        'model_id': args.model_id or str(PolicyConfig().model_id),
    }}
    json_write(args.output / 'results.json', results)
    for condition in TRAIN_CONDITIONS:
        result = train_one_condition(condition, train_rows, held_out_rows, args.output,
                                     args.epochs, args.batch_size, args.seed, args.label,
                                     args.model_id)
        results['supervised'][condition] = result
        json_write(args.output / 'results.json', results)
        print(json.dumps({
            'condition': condition,
            'initial_held_out_accuracy': frozen_probe[condition]['held_out']['oracle_action_accuracy'],
            'final_held_out_accuracy': result['history'][-1]['held_out']['oracle_action_accuracy'],
            'final_held_out_success': result['history'][-1]['held_out']['chosen_one_step_success_rate'],
        }), flush=True)
    json_write(args.output / 'completed.json', {
        'status': 'complete', 'elapsed_seconds': time.perf_counter() - started,
        'train_rows': len(train_rows), 'held_out_rows': len(held_out_rows),
    })
    print('FAST_ORACLE_VALIDATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
