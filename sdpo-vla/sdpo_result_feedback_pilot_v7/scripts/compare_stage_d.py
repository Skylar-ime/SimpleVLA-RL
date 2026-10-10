from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    names = ['sdpo_result', 'sdpo_binary', 'sdpo_shuffled', 'sdpo_neutral', 'sdpo_causal']
    rows = {}
    for name in names:
        run = args.directory / name
        completed = run / 'completed.json'
        summary = run / 'summary.json'
        if not completed.exists() or not summary.exists():
            raise SystemExit(f'incomplete condition: {name}')
        data = json.loads(summary.read_text())
        rows[name] = {'evaluation': data['evaluation'], 'updates': data['updates']}
    result = rows['sdpo_result']['evaluation']
    report = {'conditions': rows, 'comparisons': {}}
    for name in names[1:]:
        evaluation = rows[name]['evaluation']
        report['comparisons'][f'sdpo_result_minus_{name}'] = {
            'success_rate_delta': result['success_rate'] - evaluation['success_rate'],
            'mean_progress_delta_m': result['mean_progress_m'] - evaluation['mean_progress_m'],
            'action_counts': result['action_counts'],
        }
    (args.directory / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
