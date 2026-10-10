from pathlib import Path
import argparse
import hashlib
import json


V10_CONDITIONS = (
    'sdpo_result', 'sdpo_binary', 'sdpo_causal',
    'sdpo_directional', 'sdpo_wrong_directional',
)
V11_CONDITIONS = V10_CONDITIONS + (
    'sdpo_directional_residual', 'sdpo_wrong_directional_residual',
)
MATCHED_ARGUMENTS = (
    'seed', 'transitions', 'updates', 'microbatch_size', 'parallel_envs',
    'dev_scenes', 'scene_profile', 'one_step_rollouts', 'grouped_candidates',
    'expect_feedback_only',
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(root, version):
    root = Path(root)
    conditions = V10_CONDITIONS if version == 'v10' else V11_CONDITIONS
    audit = read(root / 'feedback_only_audit.json')
    if audit['status'] != 'passed':
        raise ValueError(f'feedback-only audit failed: {root}')
    if version == 'v11' and read(root / 'v11_audit.json')['status'] != 'passed':
        raise ValueError(f'V11 residual audit failed: {root}')
    runs = {}
    for condition in conditions:
        directory = root / condition
        completed = read(directory / 'completed.json')
        summary = read(directory / 'summary.json')
        config = read(directory / 'run_config.json')
        policy = read(directory / 'policy_audit.json')
        evaluation = read(directory / 'dev_final.json')
        arguments = config['arguments']
        if completed.get('status') != 'complete' or completed.get('mode') != 'train':
            raise ValueError(f'incomplete training run: {directory}')
        if len(summary['updates']) != int(arguments['updates']):
            raise ValueError(f'incomplete updates: {directory}')
        if evaluation['summary'] != summary['evaluation']:
            raise ValueError(f'evaluation mismatch: {directory}')
        episodes = {row['scene_id']: row for row in evaluation['episodes']}
        if (len(episodes) != int(arguments['dev_scenes'])
                or len(evaluation['episodes']) != len(episodes)
                or any(row['status'] == 'invalid_reset' for row in episodes.values())):
            raise ValueError(f'incomplete evaluation scenes: {directory}')
        expected_residual = condition.endswith('_residual')
        if any(bool(update.get('subtract_noop_baseline', False)) != expected_residual
               for update in summary['updates']):
            raise ValueError(f'unexpected residual setting: {directory}')
        runs[condition] = {
            'arguments': {key: arguments[key] for key in MATCHED_ARGUMENTS},
            'model': {'id': policy['model_id'], 'revision': policy['revision'],
                      'architecture': policy['architecture']},
            'protocol_sha256': config['protocol_sha256'],
            'first_rollout_sha256': digest(directory / 'update_001' / 'transitions.jsonl'),
            'evaluation': summary['evaluation'],
            'updates': len(summary['updates']),
        }
    values = [run['arguments'] for run in runs.values()]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f'condition settings differ: {root}')
    if len({run['first_rollout_sha256'] for run in runs.values()}) != 1:
        raise ValueError(f'first rollout differs across conditions: {root}')
    if len({run['protocol_sha256'] for run in runs.values()}) != 1:
        raise ValueError(f'protocol differs across conditions: {root}')
    return {
        'root': str(root),
        'scene_sha256': digest(root.parent / 'directional_long_scenes_v1.json'),
        'arguments': values[0],
        'conditions': runs,
    }


def paired(first, second, condition, second_condition=None):
    second_condition = second_condition or condition
    left = first['conditions'][condition]
    right = second['conditions'][second_condition]
    left_episodes = read(Path(first['root']) / condition / 'dev_final.json')['episodes']
    right_episodes = read(Path(second['root']) / second_condition / 'dev_final.json')['episodes']
    left_episodes = {row['scene_id']: row for row in left_episodes}
    right_episodes = {row['scene_id']: row for row in right_episodes}
    if left_episodes.keys() != right_episodes.keys():
        raise ValueError(f'paired scene mismatch: {condition}')
    gained = [scene for scene in sorted(left_episodes)
              if not left_episodes[scene]['success'] and right_episodes[scene]['success']]
    lost = [scene for scene in sorted(left_episodes)
            if left_episodes[scene]['success'] and not right_episodes[scene]['success']]
    differences = [right_episodes[scene]['progress_m'] - left_episodes[scene]['progress_m']
                   for scene in sorted(left_episodes)]
    return {
        'first_successes': left['evaluation']['successes'],
        'second_successes': right['evaluation']['successes'],
        'success_difference': len(gained) - len(lost),
        'gained_scene_ids': gained,
        'lost_scene_ids': lost,
        'mean_progress_difference_m': sum(differences) / len(differences),
    }


def compare(paths):
    runs = {}
    for version in ('v10', 'v11'):
        for backbone in ('qwen3', 'molmo2'):
            runs[f'{version}_{backbone}'] = load_run(paths[version][backbone], version)
    if len({run['scene_sha256'] for run in runs.values()}) != 1:
        raise ValueError('scene manifests differ')
    if len({tuple(run['arguments'].items()) for run in runs.values()}) != 1:
        raise ValueError('experiment dimensions differ')
    result = {
        'scene_sha256': next(iter(runs.values()))['scene_sha256'],
        'matched_arguments': next(iter(runs.values()))['arguments'],
        'runs': runs,
        'paired': {},
    }
    for version, conditions in (('v10', V10_CONDITIONS), ('v11', V11_CONDITIONS)):
        for condition in conditions:
            result['paired'][f'{version}_qwen3_vs_molmo2_{condition}'] = paired(
                runs[f'{version}_qwen3'], runs[f'{version}_molmo2'], condition)
    for backbone in ('qwen3', 'molmo2'):
        for condition in V10_CONDITIONS:
            result['paired'][f'v10_vs_v11_{backbone}_{condition}'] = paired(
                runs[f'v10_{backbone}'], runs[f'v11_{backbone}'], condition)
        for raw, residual in (
            ('sdpo_directional', 'sdpo_directional_residual'),
            ('sdpo_wrong_directional', 'sdpo_wrong_directional_residual'),
        ):
            result['paired'][f'raw_vs_residual_{backbone}_{raw}'] = paired(
                runs[f'v11_{backbone}'], runs[f'v11_{backbone}'], raw, residual)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for version in ('v10', 'v11'):
        for backbone in ('qwen3', 'molmo2'):
            parser.add_argument(f'--{version}-{backbone}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {
        version: {backbone: getattr(args, f'{version}_{backbone}')
                  for backbone in ('qwen3', 'molmo2')}
        for version in ('v10', 'v11')
    }
    report = compare(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Comparison written: {args.output}')
