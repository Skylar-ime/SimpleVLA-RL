from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


CONDITIONS = ('sdpo_result', 'sdpo_binary', 'sdpo_shuffled', 'sdpo_neutral', 'sdpo_causal')


def read_json(path):
    return json.loads(path.read_text())


def audit(root):
    errors = []
    first_hashes = {}
    counts = {}
    for condition in CONDITIONS:
        run = root / condition
        if not (run / 'completed.json').exists() or not (run / 'summary.json').exists():
            errors.append(f'{condition}: incomplete')
            continue
        config = read_json(run / 'run_config.json')
        updates = int(config['arguments']['updates'])
        expected = int(config['arguments']['transitions'])
        if config['arguments']['scene_profile'] != 'directional_local':
            errors.append(f'{condition}: wrong scene profile')
        if config['arguments']['grouped_candidates'] != 8:
            errors.append(f'{condition}: wrong candidate count')
        first_hashes[condition] = hashlib.sha256(
            (run / 'update_001' / 'transitions.jsonl').read_bytes()).hexdigest()
        solution_count = 0
        feedback_count = 0
        shuffled_count = 0
        for update_index in range(1, updates + 1):
            batch = run / f'update_{update_index:03d}'
            rows = [json.loads(line) for line in (batch / 'transitions.jsonl').read_text().splitlines()]
            context = read_json(batch / 'teacher_context.json')
            update = read_json(batch / 'update.json')
            if len(rows) != expected or update['executed'] != expected or update['optimizer_steps'] != 1:
                errors.append(f'{condition}/{update_index}: budget')
            probabilities = np.exp(np.asarray(update['teacher_log_probs'], dtype=np.float64))
            if probabilities.shape != (expected, 5) or not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
                errors.append(f'{condition}/{update_index}: teacher distribution')
            groups = defaultdict(list)
            for index, row in enumerate(rows):
                groups[row['rollout_group_id']].append(index)
            if len(groups) != expected // 8:
                errors.append(f'{condition}/{update_index}: group count')
            for members in groups.values():
                first = rows[members[0]]
                for index in members[1:]:
                    row = rows[index]
                    if (row['scene_id'] != first['scene_id']
                            or row['decision_index'] != first['decision_index']
                            or row['instruction'] != first['instruction']):
                        errors.append(f'{condition}/{update_index}: state metadata mismatch')
                success = [index for index in members
                           if rows[index]['safety_failure'] is None and rows[index]['result']['success']]
                for index in members:
                    donors = [other for other in success if other != index]
                    expected_donor = donors[0] if donors else None
                    if context['solution_donor_indices'][index] != expected_donor:
                        errors.append(f'{condition}/{update_index}/{index}: solution donor')
                    expected_solution = None if expected_donor is None else rows[expected_donor]['action_label']
                    if context['solution_actions'][index] != expected_solution:
                        errors.append(f'{condition}/{update_index}/{index}: solution action')
            if context['include_outcome_feedback'] != [action is None for action in context['solution_actions']]:
                errors.append(f'{condition}/{update_index}: feedback mask')
            solution_count += context['solution_context_count']
            feedback_count += sum(context['include_outcome_feedback'])
            if condition == 'sdpo_shuffled':
                for index, donor in enumerate(context['donor_indices']):
                    if donor is None:
                        continue
                    shuffled_count += 1
                    if (rows[donor]['rollout_group_id'] == rows[index]['rollout_group_id']
                            or rows[donor]['action_label'] != rows[index]['action_label']
                            or rows[donor]['result']['success'] != rows[index]['result']['success']):
                        errors.append(f'{condition}/{update_index}/{index}: invalid shuffled donor')
        audit = read_json(run / 'policy_audit.json')
        if audit.get('distillation') != 'js' or audit.get('teacher_update_rate') != 0.05:
            errors.append(f'{condition}: SDPO teacher configuration')
        counts[condition] = {'solution_contexts': solution_count,
                             'feedback_used_rows': feedback_count,
                             'shuffled_donors': shuffled_count}
    if len(first_hashes) == len(CONDITIONS) and len(set(first_hashes.values())) != 1:
        errors.append('first rollout batches differ across conditions')
    result = {'status': 'passed' if not errors else 'failed',
              'errors': errors, 'first_rollout_hashes': first_hashes, 'coverage': counts}
    (root / 'artifact_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    audit(parser.parse_args().directory.resolve())
