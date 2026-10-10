from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np

from audit_v12 import CONDITIONS, audit_run, digest


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    values = [float(value) for value in values]
    count = len(values)
    mean = float(np.mean(values)) if values else None
    deviation = float(np.std(values, ddof=1)) if count > 1 else None
    half_width = 2.776445105573 if count == 5 else None
    if deviation is None or half_width is None:
        interval = None
    else:
        margin = half_width * deviation / math.sqrt(count)
        interval = [mean - margin, mean + margin]
    return {'n': count, 'mean': mean, 'std': deviation, 'ci95': interval,
            'values': values}


def load_backbone(root: Path, backbone: str, seeds, protocol: str):
    runs = {}
    for seed in seeds:
        for condition in CONDITIONS:
            run = root / f'seed_{seed}' / condition
            audit_path = run / 'v12_audit.json'
            audit = audit_run(run)
            if audit_path.exists() and read(audit_path) != audit:
                raise ValueError(f'stale audit: {run}')
            config = read(run / 'run_config.json')
            if config['arguments']['backbone'] != backbone:
                raise ValueError(f'wrong backbone: {run}')
            if int(config['arguments']['seed']) != seed:
                raise ValueError(f'seed mismatch: {run}')
            if audit['protocol'] != protocol:
                raise ValueError(f'wrong protocol: {run}')
            arguments = config['arguments']
            dimensions = (int(arguments['transitions']), int(arguments['updates']),
                          int(arguments['grouped_candidates']), int(arguments['dev_scenes']))
            if dimensions != (128, 12, 8, 64):
                raise ValueError(f'wrong v12 dimensions: {run}')
            runs[(seed, condition)] = {
                'audit': audit,
                'config': config,
                'summary': read(run / 'summary.json'),
                'evaluation': read(run / 'dev_final.json'),
            }
    for seed in seeds:
        configs = [runs[(seed, condition)]['config']['arguments'] for condition in CONDITIONS]
        reference = {key: value for key, value in configs[0].items()
                     if key not in {'feedback_mode', 'output'}}
        for config in configs[1:]:
            candidate = {key: value for key, value in config.items()
                         if key not in {'feedback_mode', 'output'}}
            if candidate != reference:
                raise ValueError(f'condition settings differ for {backbone}, seed {seed}')
    for condition in CONDITIONS:
        protocol_hashes = {runs[(seed, condition)]['audit']['protocol_sha256'] for seed in seeds}
        scene_hashes = {runs[(seed, condition)]['audit']['scene_sha256'] for seed in seeds}
        if len(protocol_hashes) != 1 or len(scene_hashes) != 1:
            raise ValueError(f'manifest mismatch for {backbone}, {condition}')
    source_sets = {json.dumps(run['config']['sources'], sort_keys=True)
                   for run in runs.values()}
    if len(source_sets) != 1:
        raise ValueError(f'source changes across {backbone} runs')
    return runs


def metric(run, name):
    return float(run['evaluation']['summary'][name])


def teacher_alignment(run):
    root = Path(run['audit']['run'])
    correct = 0
    informative = 0
    expected_advantage = []
    for update in range(1, run['audit']['updates'] + 1):
        update_dir = root / f'update_{update:03d}'
        rows = [json.loads(line) for line in
                (update_dir / 'transitions.jsonl').read_text().splitlines()]
        probabilities = np.exp(np.asarray(read(update_dir / 'update.json')['teacher_log_probs']))
        progress = np.asarray([row['measurements']['counterfactual_progress_m']
                               for row in rows], dtype=float)
        if probabilities.shape != progress.shape:
            raise ValueError(f'teacher/counterfactual shape mismatch: {update_dir}')
        relative = progress - progress[:, 4:5]
        mask = np.ptp(relative, axis=1) > 0.002
        informative += int(mask.sum())
        correct += int((probabilities.argmax(axis=1)[mask]
                        == relative.argmax(axis=1)[mask]).sum())
        expected_advantage.extend(((probabilities * relative).sum(axis=1)
                                   - relative.mean(axis=1))[mask].tolist())
    return {
        'informative_transitions': informative,
        'true_best_action_accuracy': correct / informative if informative else None,
        'expected_true_advantage_vs_uniform_m': (
            float(np.mean(expected_advantage)) if expected_advantage else None),
    }


def first_update_teacher_sensitivity(runs, seeds):
    report = {}
    for condition in CONDITIONS:
        if condition == 'sdpo_advantage':
            continue
        variations = []
        for seed in seeds:
            baseline = Path(runs[(seed, 'sdpo_advantage')]['audit']['run']) / 'update_001'
            comparison = Path(runs[(seed, condition)]['audit']['run']) / 'update_001'
            baseline_rows = [json.loads(line) for line in
                             (baseline / 'transitions.jsonl').read_text().splitlines()]
            comparison_rows = [json.loads(line) for line in
                               (comparison / 'transitions.jsonl').read_text().splitlines()]
            for left, right in zip(baseline_rows, comparison_rows):
                for key in ('scene_id', 'rollout_group_id', 'rollout_candidate_index',
                            'action_label', 'proprio'):
                    if left[key] != right[key]:
                        raise ValueError(f'unpaired first-update state: {comparison}')
            if len(baseline_rows) != len(comparison_rows):
                raise ValueError(f'first-update row count mismatch: {comparison}')
            with np.load(baseline / 'pre_action_images.npz') as left_images, \
                    np.load(comparison / 'pre_action_images.npz') as right_images:
                if not np.array_equal(left_images['images'], right_images['images']):
                    raise ValueError(f'unpaired first-update RGB: {comparison}')
            left_probs = np.exp(np.asarray(read(baseline / 'update.json')['teacher_log_probs']))
            right_probs = np.exp(np.asarray(read(comparison / 'update.json')['teacher_log_probs']))
            variations.append(float((0.5 * np.abs(left_probs - right_probs).sum(axis=1)).mean()))
        report[condition] = stats(variations)
    return report


def paired_scene_delta(left, right):
    left_rows = {row['scene_id']: row for row in left['evaluation']['episodes']}
    right_rows = {row['scene_id']: row for row in right['evaluation']['episodes']}
    if left_rows.keys() != right_rows.keys():
        raise ValueError('paired dev scene mismatch')
    success_delta = sum(int(right_rows[key]['success']) - int(left_rows[key]['success'])
                        for key in left_rows)
    progress_delta = [right_rows[key]['progress_m'] - left_rows[key]['progress_m']
                      for key in sorted(left_rows)]
    return {'success_delta': success_delta, 'progress_delta_m': stats(progress_delta)}


def paired_condition_delta(runs, seeds, baseline, condition):
    success_deltas = []
    progress_deltas = []
    safety_deltas = []
    scene_success_deltas = []
    scene_progress_deltas = []
    for seed in seeds:
        left, right = runs[(seed, baseline)], runs[(seed, condition)]
        success_deltas.append(metric(right, 'successes') - metric(left, 'successes'))
        progress_deltas.append(metric(right, 'mean_progress_m') - metric(left, 'mean_progress_m'))
        safety_deltas.append(metric(right, 'safety_failures') - metric(left, 'safety_failures'))
        scene = paired_scene_delta(left, right)
        scene_success_deltas.append(scene['success_delta'])
        scene_progress_deltas.extend(scene['progress_delta_m']['values'])
    return {
        'success_count_delta_by_seed': stats(success_deltas),
        'mean_progress_delta_m_by_seed': stats(progress_deltas),
        'safety_failure_delta_by_seed': stats(safety_deltas),
        'paired_scene_success_delta_by_seed': stats(scene_success_deltas),
        'paired_scene_progress_delta_m': stats(scene_progress_deltas),
    }


def summarize(runs, seeds):
    result = {}
    for condition in CONDITIONS:
        condition_runs = [runs[(seed, condition)] for seed in seeds]
        alignments = [teacher_alignment(run) for run in condition_runs]
        result[condition] = {
            'successes': stats([metric(run, 'successes') for run in condition_runs]),
            'success_rate': stats([metric(run, 'success_rate') for run in condition_runs]),
            'mean_progress_m': stats([metric(run, 'mean_progress_m') for run in condition_runs]),
            'safety_failures': stats([metric(run, 'safety_failures') for run in condition_runs]),
            'action_counts': {
                action: int(sum(run['evaluation']['summary']['action_counts'].get(action, 0)
                               for run in condition_runs))
                for action in 'ABCDE'
            },
            'teacher_alignment': {
                'informative_transitions': int(sum(
                    alignment['informative_transitions'] for alignment in alignments)),
                'true_best_action_accuracy_by_seed': stats([
                    alignment['true_best_action_accuracy'] for alignment in alignments]),
                'expected_true_advantage_vs_uniform_m_by_seed': stats([
                    alignment['expected_true_advantage_vs_uniform_m']
                    for alignment in alignments]),
            },
        }
    baseline = 'sdpo_result'
    result['paired_vs_result'] = {}
    for condition in CONDITIONS:
        if condition == baseline:
            continue
        result['paired_vs_result'][condition] = paired_condition_delta(
            runs, seeds, baseline, condition)
    result['advantage_vs_controls'] = {
        control: paired_condition_delta(runs, seeds, control, 'sdpo_advantage')
        for control in ('sdpo_advantage_shuffled', 'sdpo_advantage_reversed')
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen3', type=Path, required=True)
    parser.add_argument('--molmo2', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[12, 13, 14, 15, 16])
    parser.add_argument('--protocol', choices=['protocol_v12.json', 'protocol_v12b.json'],
                        default='protocol_v12.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected_seeds = [12, 13, 14, 15, 16]
    if args.seeds != expected_seeds:
        raise ValueError(f'v12 requires exact protocol seeds: {expected_seeds}')
    qwen = load_backbone(args.qwen3, 'qwen3', args.seeds, args.protocol)
    molmo = load_backbone(args.molmo2, 'molmo2', args.seeds, args.protocol)
    if qwen[(args.seeds[0], CONDITIONS[0])]['config']['sources'] != \
            molmo[(args.seeds[0], CONDITIONS[0])]['config']['sources']:
        raise ValueError('source changes across backbones')
    for condition in CONDITIONS:
        for seed in args.seeds:
            q_config = qwen[(seed, condition)]['config']['arguments']
            m_config = molmo[(seed, condition)]['config']['arguments']
            for key in ('transitions', 'updates', 'parallel_envs', 'dev_scenes',
                        'scene_profile', 'one_step_rollouts', 'grouped_candidates',
                        'expect_feedback_only'):
                if q_config[key] != m_config[key]:
                    raise ValueError(f'backbone settings differ: {key}, {seed}, {condition}')
    report = {
        'protocol': args.protocol,
        'seeds': args.seeds,
        'conditions': CONDITIONS,
        'qwen3': summarize(qwen, args.seeds),
        'molmo2': summarize(molmo, args.seeds),
    }
    report['qwen3']['first_update_teacher_tv_vs_advantage'] = first_update_teacher_sensitivity(
        qwen, args.seeds)
    report['molmo2']['first_update_teacher_tv_vs_advantage'] = first_update_teacher_sensitivity(
        molmo, args.seeds)
    report['backbone_paired_by_condition'] = {}
    for condition in CONDITIONS:
        report['backbone_paired_by_condition'][condition] = {
            'success_count_delta_molmo_minus_qwen_by_seed': stats([
                metric(molmo[(seed, condition)], 'successes')
                - metric(qwen[(seed, condition)], 'successes') for seed in args.seeds]),
            'mean_progress_delta_molmo_minus_qwen_by_seed': stats([
                metric(molmo[(seed, condition)], 'mean_progress_m')
                - metric(qwen[(seed, condition)], 'mean_progress_m') for seed in args.seeds]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Comparison written: {args.output}')


if __name__ == '__main__':
    main()
