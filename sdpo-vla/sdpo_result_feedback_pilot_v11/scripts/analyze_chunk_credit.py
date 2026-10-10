from pathlib import Path
import argparse
import json
import numpy as np


LABELS = 'ABCDE'


def analyze(run):
    run = Path(run)
    config = json.loads((run / 'run_config.json').read_text())['arguments']
    chunk_length = int(config['chunk_length'])
    progress_probabilities = [[] for _ in range(chunk_length)]
    other_probabilities = [[] for _ in range(chunk_length)]
    mixed = delayed = valid_chunks = executed_actions = 0
    for index in range(1, int(config['updates']) + 1):
        path = run / f'update_{index:03d}'
        rows = [json.loads(line) for line in (path / 'chunks.jsonl').read_text().splitlines()]
        update = json.loads((path / 'update.json').read_text())
        probabilities = np.exp(np.asarray(update['teacher_chunk_log_probs']))
        mask = np.asarray(update['valid_action_mask'], dtype=bool)
        for row_index, row in enumerate(rows):
            results = row['step_results']
            valid_chunks += 1
            executed_actions += len(results)
            signs = [result['goal_progress'] for result in results]
            if 'closer' in signs and any(sign != 'closer' for sign in signs):
                mixed += 1
            if (len(signs) > 1 and signs[0] != 'closer'
                    and 'closer' in signs[1:]
                    and row['result']['goal_progress'] == 'closer'):
                delayed += 1
            for position, result in enumerate(results):
                if not mask[row_index, position]:
                    continue
                action = LABELS.index(row['action_labels'][position])
                value = float(probabilities[row_index, position, action])
                bucket = (progress_probabilities if result['goal_progress'] == 'closer'
                          else other_probabilities)
                bucket[position].append(value)
    return {
        'chunks': valid_chunks,
        'executed_actions': executed_actions,
        'mixed_outcome_chunks': mixed,
        'delayed_progress_chunks': delayed,
        'position_credit': [
            {
                'position': position,
                'closer_count': len(progress_probabilities[position]),
                'other_count': len(other_probabilities[position]),
                'teacher_attempted_probability_after_closer':
                    float(np.mean(progress_probabilities[position]))
                    if progress_probabilities[position] else None,
                'teacher_attempted_probability_after_other':
                    float(np.mean(other_probabilities[position]))
                    if other_probabilities[position] else None,
            }
            for position in range(chunk_length)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    output = args.output or args.run / 'chunk_credit_analysis.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
