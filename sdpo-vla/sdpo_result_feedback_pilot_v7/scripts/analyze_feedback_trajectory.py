from pathlib import Path
import argparse
import json

import numpy as np


CONDITIONS = ('sdpo_result', 'sdpo_binary', 'sdpo_shuffled', 'sdpo_neutral', 'sdpo_causal')


def load_update(root, condition, update_index):
    path = root / condition / f'update_{update_index:03d}'
    update = json.loads((path / 'update.json').read_text())
    context = json.loads((path / 'teacher_context.json').read_text())
    return np.exp(np.asarray(update['teacher_log_probs'], dtype=np.float64)), update, context


def analyze(root):
    result = {'updates': {}, 'conditions': {}}
    complete = []
    for condition in CONDITIONS:
        summary = json.loads((root / condition / 'summary.json').read_text())
        complete.append(condition)
        result['conditions'][condition] = {
            'evaluation': summary['evaluation'],
            'updates': len(summary['updates']),
        }
    update_count = min(result['conditions'][condition]['updates'] for condition in complete)
    for update_index in range(1, update_count + 1):
        distributions = {}
        updates = {}
        contexts = {}
        for condition in CONDITIONS:
            distributions[condition], updates[condition], contexts[condition] = load_update(
                root, condition, update_index
            )
        base = distributions['sdpo_result']
        base_context = contexts['sdpo_result']
        feedback_rows = ~np.asarray([action is not None for action in base_context['solution_actions']])
        comparisons = {}
        for condition in CONDITIONS[1:]:
            other = distributions[condition]
            tv = np.abs(base - other).sum(axis=1) / 2.0
            comparisons[condition] = {
                'teacher_tv_feedback_rows': float(tv[feedback_rows].mean()),
                'teacher_argmax_changed_feedback_rows': int(
                    (base[feedback_rows].argmax(1) != other[feedback_rows].argmax(1)).sum()
                ),
                'result_teacher_oracle_accuracy': updates['sdpo_result']['teacher_oracle_accuracy'],
                'condition_teacher_oracle_accuracy': updates[condition]['teacher_oracle_accuracy'],
                'result_teacher_entropy': updates['sdpo_result']['mean_teacher_entropy'],
                'condition_teacher_entropy': updates[condition]['mean_teacher_entropy'],
                'feedback_record_unique_count': updates[condition].get('feedback_record_unique_count'),
            }
        result['updates'][str(update_index)] = comparisons
    (root / 'feedback_trajectory_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory.resolve())
