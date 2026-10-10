"""Compare frozen and oracle-supervised policies on held-out closed-loop scenes."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import gc
import json
import time

import numpy as np
import torch

from sdpo_result_feedback.env_adapter import InvalidScene, PandaPushOutcomePilot
from sdpo_result_feedback.evaluation import json_write
from sdpo_result_feedback.policy import PolicyConfig, QwenActionPolicy
from sdpo_result_feedback.scenes import create_directional_long_scene_splits
from sdpo_result_feedback.trainer import observation_transition


ROOT = Path(__file__).resolve().parents[1]


def context_for(condition, scene):
    if condition == 'rgb':
        return None
    delta = np.asarray(scene.goal_xy) - np.asarray(scene.object_xy)
    if abs(delta[0]) >= abs(delta[1]):
        direction = '+x' if delta[0] > 0 else '-x'
    else:
        direction = '+y' if delta[1] > 0 else '-y'
    return f'Verified target direction from cube to target: {direction}. Use it as diagnostic input.'


def evaluate(policy, scenes, condition, batch_size=4, max_decisions=12):
    started = time.perf_counter()
    envs = [PandaPushOutcomePilot(max_decisions=max_decisions) for _ in range(batch_size)]
    episodes = []
    interactions = 0
    try:
        for offset in range(0, len(scenes), batch_size):
            batch = scenes[offset:offset + batch_size]
            observations = {}
            active = []
            initial_distances = {}
            action_history = {}
            for local_index, scene in enumerate(batch):
                try:
                    observation = envs[local_index].reset(f'heldout_{scene.scene_id}', scene)
                except InvalidScene as error:
                    episodes.append({'scene_id': scene.scene_id, 'success': False,
                                     'status': 'invalid_reset', 'reason': str(error)})
                    continue
                observations[local_index] = observation
                initial_distances[local_index] = envs[local_index]._distance()
                action_history[local_index] = []
                active.append(local_index)
            while active:
                transitions = [observation_transition(envs[index], observations[index])
                               for index in active]
                contexts = [context_for(condition, envs[index].scene) for index in active]
                with torch.no_grad():
                    log_probs = policy.log_probs(transitions, grad=False,
                                                 additional_context=contexts)
                    labels = ['ABCDE'[int(index)] for index in log_probs.argmax(-1).cpu()]
                next_active = []
                for index, label in zip(active, labels):
                    action_history[index].append(label)
                    observation, transition = envs[index].step(label)
                    interactions += 1
                    if transition.terminated or transition.truncated:
                        episodes.append({
                            'scene_id': transition.scene_id,
                            'success': bool(transition.result.success),
                            'status': 'safety_failure' if transition.safety_failure else
                                      'success' if transition.result.success else 'timeout',
                            'length': transition.decision_index + 1,
                            'initial_distance_m': initial_distances[index],
                            'final_distance_m': transition.distance_after_m,
                            'progress_m': initial_distances[index] - transition.distance_after_m,
                            'actions': action_history[index],
                        })
                    else:
                        observations[index] = observation
                        next_active.append(index)
                active = next_active
    finally:
        for env in envs:
            env.close()
    valid = [row for row in episodes if row['status'] != 'invalid_reset']
    successes = sum(row['success'] for row in valid)
    return {
        'condition': condition,
        'scenes_requested': len(scenes),
        'scenes_valid': len(valid),
        'successes': successes,
        'success_rate': successes / len(valid) if valid else None,
        'mean_progress_m': float(np.mean([row['progress_m'] for row in valid])) if valid else None,
        'safety_failures': sum(row['status'] == 'safety_failure' for row in episodes),
        'interactions': interactions,
        'elapsed_seconds': time.perf_counter() - started,
        'action_counts': dict(Counter(action for row in valid for action in row.get('actions', []))),
        'episodes': episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-decisions', type=int, default=12)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--conditions', nargs='+', default=[
        'frozen_rgb', 'trained_rgb', 'trained_target_direction'])
    parser.add_argument('--model-id', type=str, default=None)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    output = (args.output or (args.run.resolve() / 'closed_loop_evaluation.json')).resolve()
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing evaluation: {output}')
    splits = create_directional_long_scene_splits(ROOT / 'artifacts/directional_long_scenes_v1.json')
    scenes = splits['dev'][:args.scenes]
    report = {'protocol': {'split': 'dev held-out', 'scenes': len(scenes),
                           'max_decisions': args.max_decisions,
                           'same_scenes_for_all_conditions': True}, 'conditions': {}}
    for condition in args.conditions:
        policy_config = PolicyConfig(seed=args.seed)
        if args.model_id is not None:
            policy_config = PolicyConfig(seed=args.seed, model_id=args.model_id)
        policy = QwenActionPolicy(policy_config)
        if condition.startswith('trained_'):
            adapter = args.run.resolve() / condition.removeprefix('trained_') / 'checkpoint'
            policy.model.load_adapter(str(adapter), adapter_name='diagnostic', is_trainable=False)
            policy.model.set_adapter('diagnostic')
        model_condition = 'target_direction' if 'target_direction' in condition else 'rgb'
        result = evaluate(policy, scenes, model_condition, args.batch_size, args.max_decisions)
        report['conditions'][condition] = result
        json_write(output, report)
        print(json.dumps({key: result[key] for key in (
            'condition', 'successes', 'scenes_valid', 'success_rate', 'mean_progress_m',
            'safety_failures', 'interactions', 'elapsed_seconds', 'action_counts')}), flush=True)
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = 'complete'
    json_write(output, report)
    print('CLOSED_LOOP_EVALUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
