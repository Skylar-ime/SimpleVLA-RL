from pathlib import Path
import argparse
import json


CONDITIONS = (
    'sdpo_result', 'sdpo_binary', 'sdpo_causal',
    'sdpo_directional', 'sdpo_wrong_directional',
)


def compare(root: Path) -> dict:
    result = {'conditions': {}, 'updates': {}}
    summaries = {}
    for condition in CONDITIONS:
        run = root / condition
        completed = run / 'completed.json'
        summary = run / 'summary.json'
        if not completed.exists() or not summary.exists():
            raise SystemExit(f'incomplete condition: {condition}')
        summaries[condition] = json.loads(summary.read_text())
        result['conditions'][condition] = {
            'evaluation': summaries[condition]['evaluation'],
            'updates': len(summaries[condition]['updates']),
        }
    update_count = min(len(summary['updates']) for summary in summaries.values())
    for update_index in range(update_count):
        key = str(update_index + 1)
        result['updates'][key] = {}
        for condition in CONDITIONS:
            update = summaries[condition]['updates'][update_index]
            result['updates'][key][condition] = {
                'valid': update['valid'],
                'loss': update['loss'],
                'teacher_oracle_accuracy': update['teacher_oracle_accuracy'],
                'teacher_argmax_counts': update['teacher_argmax_counts'],
                'teacher_entropy': update['mean_teacher_entropy'],
                'teacher_student_tv': update['mean_teacher_student_total_variation'],
            }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = compare(args.directory.resolve())
    output = args.directory.resolve() / 'stage_i_v10_comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
