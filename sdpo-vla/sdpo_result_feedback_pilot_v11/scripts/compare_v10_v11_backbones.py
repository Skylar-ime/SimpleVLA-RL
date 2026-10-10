from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


V10_CONDITIONS = ('sdpo_result', 'sdpo_binary', 'sdpo_causal',
                  'sdpo_directional', 'sdpo_wrong_directional')
V11_CONDITIONS = V10_CONDITIONS + ('sdpo_directional_residual',
                                  'sdpo_wrong_directional_residual')
MATCHED_ARGUMENTS = ('seed', 'transitions', 'updates', 'microbatch_size',
                     'parallel_envs', 'dev_scenes', 'scene_profile',
                     'one_step_rollouts', 'grouped_candidates',
                     'expect_feedback_only')


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(root: Path, conditions: tuple[str, ...], model_size: str) -> dict:
    audit = read(root / 'feedback_only_audit.json')
    if audit['status'] != 'passed':
        raise ValueError(f'Feedback-only audit failed: {root}')
    if len(conditions) == len(V11_CONDITIONS):
        residual_audit = read(root / 'v11_audit.json')
        if residual_audit['status'] != 'passed':
            raise ValueError(f'V11 residual audit failed: {root}')
    scenes_sha256 = digest(root.parent / 'directional_long_scenes_v1.json')
    runs = {}
    for condition in conditions:
        directory = root / condition
        completed = read(directory / 'completed.json')
        summary = read(directory / 'summary.json')
        config = read(directory / 'run_config.json')
        policy = read(directory / 'policy_audit.json')
        evaluation = read(directory / 'dev_final.json')
        arguments = config['arguments']
        if completed['status'] != 'complete' or completed['mode'] != 'train':
            raise ValueError(f'Incomplete training run: {directory}')
        if f'-{model_size}-' not in policy['model_id']:
            raise ValueError(f'Unexpected model in {directory}: {policy["model_id"]}')
        if len(summary['updates']) != arguments['updates']:
            raise ValueError(f'Incomplete update series: {directory}')
        if evaluation['summary'] != summary['evaluation']:
            raise ValueError(f'Evaluation mismatch: {directory}')
        episodes = {row['scene_id']: row for row in evaluation['episodes']}
        if (len(episodes) != arguments['dev_scenes']
                or len(evaluation['episodes']) != len(episodes)
                or any(row['status'] == 'invalid_reset' for row in episodes.values())):
            raise ValueError(f'Incomplete or duplicated evaluation scenes: {directory}')
        expected_residual = condition.endswith('_residual')
        if any(update.get('subtract_noop_baseline', False) != expected_residual
               for update in summary['updates']):
            raise ValueError(f'Unexpected residual setting: {directory}')
        runs[condition] = {
            'arguments': {key: arguments[key] for key in MATCHED_ARGUMENTS},
            'model': {'id': policy['model_id'], 'revision': policy['revision'],
                      'local_path': policy['local_path']},
            'protocol_sha256': config['protocol_sha256'],
            'first_rollout_sha256': digest(directory / 'update_001' / 'transitions.jsonl'),
            'evaluation': summary['evaluation'],
            'episodes': episodes,
            'updates': len(summary['updates']),
        }
    arguments = [run['arguments'] for run in runs.values()]
    if any(value != arguments[0] for value in arguments[1:]):
        raise ValueError(f'Condition settings differ inside {root}')
    if len({run['first_rollout_sha256'] for run in runs.values()}) != 1:
        raise ValueError(f'First rollout differs across conditions in {root}')
    if len({run['protocol_sha256'] for run in runs.values()}) != 1:
        raise ValueError(f'Protocol differs across conditions in {root}')
    return {'root': str(root), 'scenes_sha256': scenes_sha256,
            'arguments': arguments[0], 'conditions': runs}


def paired(first: dict, second: dict, condition: str) -> dict:
    left = first['conditions'][condition]
    right = second['conditions'][condition]
    left_episodes, right_episodes = left['episodes'], right['episodes']
    if left_episodes.keys() != right_episodes.keys():
        raise ValueError(f'Paired evaluation scenes differ for {condition}')
    scenes = sorted(left_episodes)
    gained = [scene for scene in scenes if not left_episodes[scene]['success']
              and right_episodes[scene]['success']]
    lost = [scene for scene in scenes if left_episodes[scene]['success']
            and not right_episodes[scene]['success']]
    progress_differences = [right_episodes[scene]['progress_m']
                            - left_episodes[scene]['progress_m'] for scene in scenes]
    return {
        'first_successes': left['evaluation']['successes'],
        'second_successes': right['evaluation']['successes'],
        'success_difference': len(gained) - len(lost),
        'gained_scene_ids': gained,
        'lost_scene_ids': lost,
        'mean_progress_difference_m': sum(progress_differences) / len(progress_differences),
    }


def compare(paths: dict[str, Path]) -> dict:
    runs = {name: load_run(root, V10_CONDITIONS if name.startswith('v10')
                          else V11_CONDITIONS, '3B' if name.endswith('3b') else '7B')
            for name, root in paths.items()}
    if len({run['scenes_sha256'] for run in runs.values()}) != 1:
        raise ValueError('Scene manifests differ')
    if len({tuple(run['arguments'].items()) for run in runs.values()}) != 1:
        raise ValueError('Experiment dimensions differ')
    for version in ('v10', 'v11'):
        for size in ('3b', '7b'):
            run = runs[f'{version}_{size}']
            ids = {row['model']['id'] for row in run['conditions'].values()}
            revisions = {row['model']['revision'] for row in run['conditions'].values()}
            if len(ids) != 1 or len(revisions) != 1:
                raise ValueError(f'Model differs within {version}_{size}')
    for size in ('3b', '7b'):
        first = runs[f'v10_{size}']['conditions'][V10_CONDITIONS[0]]
        second = runs[f'v11_{size}']['conditions'][V10_CONDITIONS[0]]
        if first['first_rollout_sha256'] != second['first_rollout_sha256']:
            raise ValueError(f'V10/V11 first rollout differs for {size}')
    result = {'scene_sha256': next(iter(runs.values()))['scenes_sha256'],
              'matched_arguments': next(iter(runs.values()))['arguments'],
              'runs': {}, 'paired': {}}
    for name, run in runs.items():
        result['runs'][name] = {
            'root': run['root'],
            'conditions': {condition: {key: value for key, value in data.items()
                                      if key != 'episodes'}
                           for condition, data in run['conditions'].items()},
        }
    for condition in V10_CONDITIONS:
        for size in ('3b', '7b'):
            result['paired'][f'v10_vs_v11_{size}_{condition}'] = paired(
                runs[f'v10_{size}'], runs[f'v11_{size}'], condition)
        for version in ('v10', 'v11'):
            result['paired'][f'3b_vs_7b_{version}_{condition}'] = paired(
                runs[f'{version}_3b'], runs[f'{version}_7b'], condition)
    for size in ('3b', '7b'):
        run = runs[f'v11_{size}']
        for raw, residual in (('sdpo_directional', 'sdpo_directional_residual'),
                              ('sdpo_wrong_directional', 'sdpo_wrong_directional_residual')):
            result['paired'][f'raw_vs_residual_{size}_{raw}'] = paired(
                {'conditions': {raw: run['conditions'][raw]}},
                {'conditions': {raw: run['conditions'][residual]}}, raw)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('v10_3b', 'v11_3b', 'v10_7b', 'v11_7b'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    paths = {name: getattr(arguments, name).resolve()
             for name in ('v10_3b', 'v11_3b', 'v10_7b', 'v11_7b')}
    report = compare(paths)
    if arguments.output.exists():
        raise SystemExit(f'Refusing to overwrite existing report: {arguments.output}')
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Comparison written: {arguments.output}')
