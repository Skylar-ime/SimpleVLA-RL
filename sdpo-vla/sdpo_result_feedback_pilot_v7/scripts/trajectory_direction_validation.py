"""Train a direction policy on expert intermediate states."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import json

import numpy as np
import torch
import torch.nn.functional as F

from sdpo_result_feedback.actions import ACTION_LABELS
from sdpo_result_feedback.env_adapter import InvalidScene, PandaPushOutcomePilot
from sdpo_result_feedback.policy import PolicyConfig, QwenActionPolicy
from sdpo_result_feedback.scenes import create_directional_long_scene_splits


ROOT = Path(__file__).resolve().parents[1]
DIRECTION_ACTION = {'+x': 'A', '-x': 'B', '+y': 'C', '-y': 'D'}
ACTION_INDEX = {label: index for index, label in enumerate(ACTION_LABELS)}


def target_direction(scene) -> str:
    delta = np.asarray(scene.goal_xy) - np.asarray(scene.object_xy)
    if abs(delta[0]) >= abs(delta[1]):
        return '+x' if delta[0] > 0 else '-x'
    return '+y' if delta[1] > 0 else '-y'


def context_for(direction: str) -> str:
    return (f'Verified target direction from cube to target: {direction}. '
            'Use it as diagnostic input.')


def make_transition(env, observation, scene_id: str, index: int):
    return SimpleNamespace(
        episode_id=f'trajectory_{scene_id}',
        decision_index=index,
        image=observation['image'],
        instruction=observation['instruction'],
        proprio=np.asarray(observation['proprio'], dtype=np.float32),
        action_label='E',
        result=None,
        scene_id=scene_id,
    )


def collect_examples(scenes, seed: int, max_decisions: int):
    env = PandaPushOutcomePilot(max_decisions=max_decisions, seed=seed)
    examples = []
    episode_stats = []
    try:
        for scene in scenes:
            direction = target_direction(scene)
            label = DIRECTION_ACTION[direction]
            try:
                observation = env.reset(f'trajectory_{scene.scene_id}', scene)
            except InvalidScene as error:
                episode_stats.append({'scene_id': scene.scene_id, 'status': 'invalid_reset',
                                      'reason': str(error)})
                continue
            length = 0
            while length < max_decisions:
                examples.append({'transition': make_transition(env, observation,
                                                                 scene.scene_id, length),
                                 'direction': direction, 'label': label})
                observation, transition = env.step(label)
                length += 1
                if transition.terminated or transition.truncated:
                    episode_stats.append({
                        'scene_id': scene.scene_id,
                        'status': 'success' if transition.result.success else 'timeout',
                        'success': bool(transition.result.success),
                        'length': length,
                    })
                    break
    finally:
        env.close()
    return examples, episode_stats


def load_policy(seed: int, model_id: str | None):
    config = PolicyConfig(seed=seed, model_id=model_id) if model_id else PolicyConfig(seed=seed)
    return QwenActionPolicy(config)


def predict(policy, examples, batch_size: int):
    predictions = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        transitions = [item['transition'] for item in batch]
        contexts = [context_for(item['direction']) for item in batch]
        with torch.no_grad():
            log_probs = policy.log_probs(transitions, grad=False,
                                         additional_context=contexts)
        predictions.extend(ACTION_LABELS[int(index)]
                           for index in log_probs.argmax(-1).cpu())
    return predictions


def score(examples, predictions):
    labels = [item['label'] for item in examples]
    return {
        'examples': len(examples),
        'direction_action_accuracy': float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        'predicted_action_counts': dict(Counter(predictions)),
        'label_counts': dict(Counter(labels)),
    }


def train(args):
    splits = create_directional_long_scene_splits(
        ROOT / 'artifacts/directional_long_scenes_v1.json')
    train_examples, train_episodes = collect_examples(
        splits['train'][:args.train_scenes], args.seed, args.max_decisions)
    dev_examples, dev_episodes = collect_examples(
        splits['dev'][:args.held_out_scenes], args.seed + 1, args.max_decisions)
    policy = load_policy(args.seed, args.model_id)
    target = torch.tensor([ACTION_INDEX[item['label']] for item in train_examples],
                          device=policy.device)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs + 1):
        train_pred = predict(policy, train_examples, args.batch_size)
        dev_pred = predict(policy, dev_examples, args.batch_size)
        record = {
            'epoch': epoch,
            'train': score(train_examples, train_pred),
            'held_out': score(dev_examples, dev_pred),
        }
        history.append(record)
        if epoch == args.epochs:
            break
        order = rng.permutation(len(train_examples))
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = [train_examples[int(index)] for index in indices]
            transitions = [item['transition'] for item in batch]
            contexts = [context_for(item['direction']) for item in batch]
            logits = policy.log_probs(transitions, grad=True,
                                      additional_context=contexts)
            selected = torch.as_tensor(indices, device=policy.device)
            loss = F.nll_loss(logits, target[selected])
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), 1.)
            policy.optimizer.step()
            losses.append({'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        record['optimizer'] = {
            'batches': len(losses),
            'mean_loss': float(np.mean([item['loss'] for item in losses])),
            'mean_gradient_norm': float(np.mean([item['gradient_norm'] for item in losses])),
        }
    checkpoint = args.output / 'target_direction' / 'checkpoint'
    policy.save(checkpoint)
    result = {
        'config': {
            'train_scenes': args.train_scenes,
            'held_out_scenes': args.held_out_scenes,
            'max_decisions': args.max_decisions,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'model_id': args.model_id or str(PolicyConfig().model_id),
            'supervision': 'direction_action on expert intermediate states',
        },
        'train_examples': len(train_examples),
        'held_out_examples': len(dev_examples),
        'train_episodes': train_episodes,
        'held_out_episodes': dev_episodes,
        'history': history,
        'checkpoint': str(checkpoint),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    del policy
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-scenes', type=int, default=64)
    parser.add_argument('--held-out-scenes', type=int, default=64)
    parser.add_argument('--max-decisions', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--model-id', type=str, default=None)
    train(parser.parse_args())


if __name__ == '__main__':
    main()
