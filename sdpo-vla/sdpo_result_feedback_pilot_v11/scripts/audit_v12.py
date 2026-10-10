from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


CONDITIONS = (
    'sdpo_result', 'sdpo_advantage',
    'sdpo_advantage_shuffled', 'sdpo_advantage_reversed',
)
ACTION_LABELS = ('A', 'B', 'C', 'D', 'E')
REPO_ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_grouped_rollout(update_dir, expected_transitions, candidates):
    rows = [json.loads(line) for line in (update_dir / 'transitions.jsonl').read_text().splitlines()]
    if len(rows) != expected_transitions:
        raise ValueError(f'{update_dir}: expected {expected_transitions} transitions, got {len(rows)}')
    images = np.load(update_dir / 'pre_action_images.npz')['images']
    if images.shape[0] != len(rows):
        raise ValueError(f'{update_dir}: image count mismatch')
    groups = {}
    for index, row in enumerate(rows):
        group = row.get('rollout_group_id')
        if not group:
            raise ValueError(f'{update_dir}: missing rollout group')
        row['_image'] = images[index]
        groups.setdefault(group, []).append(row)
        table = row.get('measurements', {}).get('counterfactual_progress_m')
        if table is None or len(table) != len(ACTION_LABELS):
            raise ValueError(f'{update_dir}: malformed counterfactual progress table')
        if not np.isfinite(np.asarray(table, dtype=float)).all():
            raise ValueError(f'{update_dir}: nonfinite counterfactual progress')
        action = row['action_label']
        if abs(float(row['result']['progress_m']) - float(table[ACTION_LABELS.index(action)])) > 1e-12:
            raise ValueError(f'{update_dir}: executed action differs from counterfactual replay')
        if abs(float(row['measurements']['progress_m']) - float(row['result']['progress_m'])) > 1e-12:
            raise ValueError(f'{update_dir}: result progress mismatch')
        expected_advantage = float(table[ACTION_LABELS.index(action)]) - float(table[4])
        actual_advantage = float(row['measurements'].get('action_advantage_m'))
        if abs(expected_advantage - actual_advantage) > 1e-12:
            raise ValueError(f'{update_dir}: action-relative advantage mismatch')
    if any(len(rows_for_group) != candidates for rows_for_group in groups.values()):
        raise ValueError(f'{update_dir}: candidate count mismatch')
    for rows_for_group in groups.values():
        first = rows_for_group[0]
        for row in rows_for_group[1:]:
            if row['scene_id'] != first['scene_id'] or row['decision_index'] != first['decision_index']:
                raise ValueError(f'{update_dir}: group state metadata mismatch')
            if row['proprio'] != first['proprio'] or not np.array_equal(row['_image'], first['_image']):
                raise ValueError(f'{update_dir}: same-state replay mismatch')
            if row['measurements']['counterfactual_progress_m'] != first['measurements']['counterfactual_progress_m']:
                raise ValueError(f'{update_dir}: counterfactual table mismatch')
    return rows


def audit_run(root: Path):
    config = read(root / 'run_config.json')
    arguments = config['arguments']
    reproducibility = config.get('reproducibility')
    if not reproducibility:
        raise ValueError(f'{root}: missing reproducibility snapshot')
    manifest_path = root / reproducibility['source_manifest']
    if not manifest_path.is_file():
        raise ValueError(f'{root}: missing source manifest')
    manifest = read(manifest_path)
    if hashlib.sha256(json.dumps(manifest['files'], sort_keys=True).encode()).hexdigest() != manifest['sha256']:
        raise ValueError(f'{root}: source manifest hash mismatch')
    for row in manifest['files']:
        path = root / 'reproducibility/source' / row['path']
        if not path.is_file() or digest(path) != row['sha256']:
            raise ValueError(f'{root}: source snapshot file mismatch: {row["path"]}')
    if arguments['feedback_mode'] not in CONDITIONS:
        raise ValueError(f'{root}: not a v12 condition')
    if arguments['scene_profile'] != 'directional_long' or not arguments['one_step_rollouts']:
        raise ValueError(f'{root}: wrong scene protocol')
    if (arguments['grouped_candidates'] < 2
            or arguments['transitions'] < arguments['grouped_candidates']
            or arguments['transitions'] % arguments['grouped_candidates']):
        raise ValueError(f'{root}: wrong grouped rollout dimensions')
    protocol_name = Path(arguments['protocol_file']).name
    if protocol_name not in {'protocol_v12.json', 'protocol_v12b.json'}:
        raise ValueError(f'{root}: unsupported v12 protocol')
    if config['protocol_sha256'] != digest(REPO_ROOT / protocol_name):
        raise ValueError(f'{root}: protocol hash mismatch')
    completed = read(root / 'completed.json')
    if completed.get('status') != 'complete' or completed.get('mode') != 'train':
        raise ValueError(f'{root}: incomplete run')
    summary = read(root / 'summary.json')
    evaluation = read(root / 'dev_final.json')
    if summary['evaluation'] != evaluation['summary']:
        raise ValueError(f'{root}: evaluation summary mismatch')
    if len(summary['updates']) != arguments['updates']:
        raise ValueError(f'{root}: update count mismatch')
    update_rows = []
    for update in range(1, arguments['updates'] + 1):
        update_rows.extend(_check_grouped_rollout(
            root / f'update_{update:03d}', int(arguments['transitions']),
            int(arguments['grouped_candidates'])))
        if protocol_name == 'protocol_v12b.json':
            update_dir = root / f'update_{update:03d}'
            context = read(update_dir / 'teacher_context.json')
            if context['solution_context_count'] or any(
                    action is not None for action in context['solution_actions']):
                raise ValueError(f'{update_dir}: self-solution context violates feedback-only protocol')
            if any(not use for use in context['include_outcome_feedback']):
                raise ValueError(f'{update_dir}: outcome feedback suppressed for a solution')
    if arguments['feedback_mode'] != 'sdpo_result':
        for update in range(1, arguments['updates'] + 1):
            update_dir = root / f'update_{update:03d}'
            context = read(update_dir / 'teacher_context.json')
            if context['condition'] != arguments['feedback_mode']:
                raise ValueError(f'{update_dir}: teacher context condition mismatch')
            rows = [json.loads(line) for line in
                    (update_dir / 'transitions.jsonl').read_text().splitlines()]
            if len(context['records']) != len(rows):
                raise ValueError(f'{update_dir}: feedback record count mismatch')
            if not all(record['counterfactual_advantage_m'] is not None
                       for record in context['records']):
                raise ValueError(f'{update_dir}: missing advantage feedback')
            for index, (record, row) in enumerate(zip(context['records'], rows)):
                table = row['measurements']['counterfactual_progress_m']
                action_index = ACTION_LABELS.index(row['action_label'])
                if arguments['feedback_mode'] == 'sdpo_advantage_shuffled':
                    donor = context['donor_indices'][index]
                    if donor is None or not 0 <= donor < len(rows):
                        raise ValueError(f'{update_dir}: invalid shuffled donor')
                    donor_row = rows[donor]
                    if donor_row['rollout_group_id'] == row['rollout_group_id']:
                        raise ValueError(f'{update_dir}: shuffled donor shares original state')
                    table = donor_row['measurements']['counterfactual_progress_m']
                sign = -1 if arguments['feedback_mode'] == 'sdpo_advantage_reversed' else 1
                expected_vector = [sign * (float(value) - float(table[4])) for value in table]
                actual_vector = record['counterfactual_advantage_m']
                if len(actual_vector) != len(ACTION_LABELS) or not np.allclose(
                        actual_vector, expected_vector, rtol=0, atol=1e-12):
                    raise ValueError(f'{update_dir}: feedback advantage vector mismatch')
                if not np.allclose(record['counterfactual_progress_m'], table,
                                   rtol=0, atol=1e-12):
                    raise ValueError(f'{update_dir}: feedback progress source mismatch')
                expected = expected_vector[action_index]
                if abs(float(record['action_advantage_m']) - expected) > 1e-12:
                    raise ValueError(f'{update_dir}: feedback advantage mismatch')
                if protocol_name == 'protocol_v12b.json':
                    if any(record[field] is not None for field in (
                            'contact', 'object_motion', 'goal_progress', 'success',
                            'progress_m', 'distance_before_m', 'distance_after_m')):
                        raise ValueError(f'{update_dir}: outcome leaked into v12b feedback')
    return {
        'status': 'passed',
        'run': str(root.resolve()),
        'backbone': arguments['backbone'],
        'condition': arguments['feedback_mode'],
        'seed': arguments['seed'],
        'updates': len(summary['updates']),
        'transitions': len(update_rows),
        'evaluation': evaluation['summary'],
        'protocol': protocol_name,
        'protocol_sha256': config['protocol_sha256'],
        'scene_sha256': digest(REPO_ROOT / 'artifacts' / 'directional_long_scenes_v1.json'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = audit_run(args.run)
    (args.run / 'v12_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"v12 audit passed: {args.run}")


if __name__ == '__main__':
    main()
