from pathlib import Path
import argparse
import json


CONDITIONS = (
    'chunk_suffix_result', 'chunk_suffix_directional',
    'chunk_suffix_directional_residual', 'chunk_suffix_wrong_directional',
    'chunk_suffix_wrong_directional_residual',
)
MATCHED_ARGUMENTS = ('transitions', 'updates', 'chunk_length',
                     'grouped_candidates', 'seed', 'microbatch_size',
                     'dev_scenes', 'expect_feedback_only')


def read(path):
    return json.loads(Path(path).read_text())


def load(root):
    root = Path(root)
    runs = {}
    for condition in CONDITIONS:
        run = root / condition
        audit = read(run / 'chunk_v13_audit.json')
        if audit['status'] != 'passed':
            raise ValueError(f'audit failed: {run}')
        config = read(run / 'run_config.json')
        if config.get('protocol') != 'protocol_chunk_v13.json':
            raise ValueError(f'wrong protocol: {run}')
        summary = read(run / 'summary.json')
        arguments = config['arguments']
        if len(summary['updates']) != int(arguments['updates']):
            raise ValueError(f'incomplete updates: {run}')
        runs[condition] = {
            'arguments': arguments,
            'evaluation': summary['evaluation'],
            'updates': len(summary['updates']),
        }
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen3', type=Path, required=True)
    parser.add_argument('--molmo2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'protocol': 'protocol_chunk_v13.json',
              'qwen3': load(args.qwen3), 'molmo2': load(args.molmo2)}
    for backbone in ('qwen3', 'molmo2'):
        if len({tuple(run['arguments'][key] for key in MATCHED_ARGUMENTS)
                for run in report[backbone].values()}) != 1:
            raise ValueError(f'condition settings differ: {backbone}')
    report['paired'] = {}
    for condition in CONDITIONS:
        qwen = report['qwen3'][condition]['evaluation']
        molmo = report['molmo2'][condition]['evaluation']
        report['paired'][condition] = {
            'qwen3_successes': qwen['successes'],
            'molmo2_successes': molmo['successes'],
            'success_difference': qwen['successes'] - molmo['successes'],
            'qwen3_progress_m': qwen['mean_progress_m'],
            'molmo2_progress_m': molmo['mean_progress_m'],
            'progress_difference_m': qwen['mean_progress_m'] - molmo['mean_progress_m'],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Comparison written: {args.output}')


if __name__ == '__main__':
    main()
