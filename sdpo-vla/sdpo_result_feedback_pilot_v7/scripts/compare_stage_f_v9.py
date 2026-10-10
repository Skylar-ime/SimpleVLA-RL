from pathlib import Path
import argparse
import json

import numpy as np


CONDITIONS = ('sdpo_result', 'sdpo_binary', 'sdpo_causal', 'sdpo_directional')


def load_update(root, condition, update_index):
    path = root / condition / f'update_{update_index:03d}'
    update = json.loads((path / 'update.json').read_text())
    context = json.loads((path / 'teacher_context.json').read_text())
    return np.exp(np.asarray(update['teacher_log_probs'], dtype=np.float64)), update, context


def compare(root):
    summaries = {}
    for condition in CONDITIONS:
        summary = json.loads((root / condition / 'summary.json').read_text())
        summaries[condition] = summary
    update_count = min(len(summary['updates']) for summary in summaries.values())
    trajectory = {}
    for update_index in range(1, update_count + 1):
        distributions = {}
        updates = {}
        contexts = {}
        for condition in CONDITIONS:
            distributions[condition], updates[condition], contexts[condition] = load_update(
                root, condition, update_index)
        base = distributions['sdpo_result']
        feedback_rows = ~np.asarray([action is not None
                                     for action in contexts['sdpo_result']['solution_actions']])
        comparisons = {}
        for condition in CONDITIONS[1:]:
            other = distributions[condition]
            tv = np.abs(base - other).sum(axis=1) / 2.0
            changed = base.argmax(1) != other.argmax(1)
            comparisons[condition] = {
                'teacher_tv_feedback_rows': float(tv[feedback_rows].mean()),
                'teacher_argmax_changed_feedback_rows': int(changed[feedback_rows].sum()),
                'teacher_oracle_accuracy': updates[condition]['teacher_oracle_accuracy'],
                'feedback_record_unique_count': updates[condition]['feedback_record_unique_count'],
                'teacher_argmax_counts': updates[condition]['teacher_argmax_counts'],
            }
        trajectory[str(update_index)] = comparisons
    report = {
        'conditions': {
            condition: {'evaluation': summaries[condition]['evaluation'],
                        'updates': len(summaries[condition]['updates'])}
            for condition in CONDITIONS
        },
        'updates': trajectory,
    }
    (root / 'stage_f_v9_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    compare(parser.parse_args().directory.resolve())
