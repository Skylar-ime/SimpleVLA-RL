from pathlib import Path
import argparse
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def audit(root):
    root = Path(root)
    config = read(root / 'run_config.json')
    arguments = config['arguments']
    errors = []
    updates = int(arguments['updates'])
    chunk_length = int(arguments['chunk_length'])
    expected = int(arguments['transitions'])
    for index in range(1, updates + 1):
        path = root / f'update_{index:03d}'
        if not (path / 'update.json').exists():
            errors.append(f'missing update: {path}')
            continue
        rows = [json.loads(line) for line in (path / 'chunks.jsonl').read_text().splitlines()]
        if len(rows) != expected:
            errors.append(f'{path}: expected {expected} chunks, found {len(rows)}')
        if any(len(row['action_labels']) != chunk_length for row in rows):
            errors.append(f'{path}: inconsistent chunk length')
        update = read(path / 'update.json')
        masks = np.asarray(update['valid_action_mask'], dtype=bool)
        if masks.shape != (expected, chunk_length):
            errors.append(f'{path}: invalid action mask shape {masks.shape}')
        elif any(masks[row_index].tolist() != [
                action_index < len(row['step_results'])
                and row['safety_failure'] is None
                and (arguments['condition'].startswith('chunk_solution')
                     or not row['result']['success'])
                for action_index in range(chunk_length)]
                for row_index, row in enumerate(rows)):
            errors.append(f'{path}: action mask disagrees with executed steps')
        context = read(path / 'chunk_context.json')
        if context.get('solution_context_count', 0):
            if not arguments['condition'].startswith('chunk_solution'):
                errors.append(f'{path}: solution context in feedback-only condition')
            groups = {}
            for row in rows:
                groups.setdefault(row['rollout_group_id'], []).append(row)
            for row, solution in zip(rows, context['solution_actions']):
                if solution is None:
                    continue
                donors = [candidate for candidate in groups[row['rollout_group_id']]
                          if candidate['episode_id'] != row['episode_id']
                          and candidate['safety_failure'] is None
                          and candidate['result']['success']
                          and candidate['action_labels'] == solution]
                if not donors:
                    errors.append(f'{path}: solution has no same-state successful donor')
        if update['subtract_noop_baseline']:
            privileged = np.asarray(update['teacher_privileged_chunk_log_probs'])
            baseline = np.asarray(update['teacher_noop_chunk_log_probs'])
            residual = np.asarray(update['teacher_residual_chunk_log_probs'])
            target = np.asarray(update['teacher_chunk_log_probs'])
            shape = (expected, chunk_length, 5)
            if any(value.shape != shape for value in (privileged, baseline, residual, target)):
                errors.append(f'{path}: invalid residual tensor shape')
            if not np.allclose(residual, privileged - baseline, atol=1e-6):
                errors.append(f'{path}: residual mismatch')
            if not np.allclose(np.exp(target).sum(-1), 1., atol=1e-5):
                errors.append(f'{path}: target is not normalized')
    completed = read(root / 'completed.json') if (root / 'completed.json').exists() else {}
    status = 'passed' if not errors and completed.get('status') == 'complete' else 'failed'
    result = {'status': status, 'errors': errors}
    (root / 'chunk_v12_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return status == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    raise SystemExit(0 if audit(parser.parse_args().root) else 1)
