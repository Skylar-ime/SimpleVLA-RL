#!/usr/bin/env python3
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
    expected = int(arguments['transitions'])
    chunk_length = int(arguments['chunk_length'])
    suffix = arguments['condition'].startswith('chunk_suffix_')
    if config.get('protocol') != 'protocol_chunk_v13.json':
        errors.append('run does not identify protocol_chunk_v13.json')
    for index in range(1, updates + 1):
        path = root / f'update_{index:03d}'
        rows = [json.loads(line) for line in (path / 'chunks.jsonl').read_text().splitlines()]
        update = read(path / 'update.json')
        context = read(path / 'chunk_context.json')
        if len(rows) != expected:
            errors.append(f'{path}: expected {expected} chunks, found {len(rows)}')
        if any(len(row['action_labels']) != chunk_length for row in rows):
            errors.append(f'{path}: inconsistent chunk length')
        masks = np.asarray(update['valid_action_mask'], dtype=bool)
        if masks.shape != (expected, chunk_length):
            errors.append(f'{path}: invalid action mask shape {masks.shape}')
        if suffix:
            records = context.get('records_by_position')
            if records is None or len(records) != expected:
                errors.append(f'{path}: missing position-specific records')
            else:
                for row_index, position_records in enumerate(records):
                    if len(position_records) != chunk_length:
                        errors.append(f'{path}: invalid position record count')
                        break
                    for position, record in enumerate(position_records):
                        if position < chunk_length - 1:
                            interpretation = record.get('interpretation') or ''
                            if f'Step {position}' not in interpretation:
                                errors.append(f'{path}: suffix transcript missing current step')
                        if position_records[position].get('goal_progress') is None:
                            errors.append(f'{path}: suffix record missing progress')
                            break
        expected_scope = 'return_to_go_suffix' if suffix else 'aggregate_chunk'
        if context.get('position_specific_feedback') != suffix:
            errors.append(f'{path}: position feedback flag mismatch')
        if context.get('feedback_scope') != expected_scope:
            errors.append(f'{path}: feedback scope mismatch')
        if update['subtract_noop_baseline'] != arguments['condition'].endswith('_residual'):
            errors.append(f'{path}: residual flag mismatch')
        if update['subtract_noop_baseline']:
            shape = (expected, chunk_length, 5)
            privileged = np.asarray(update['teacher_privileged_chunk_log_probs'])
            baseline = np.asarray(update['teacher_noop_chunk_log_probs'])
            residual = np.asarray(update['teacher_residual_chunk_log_probs'])
            target = np.asarray(update['teacher_chunk_log_probs'])
            if any(value.shape != shape for value in (privileged, baseline, residual, target)):
                errors.append(f'{path}: invalid residual tensor shape')
            if not np.allclose(residual, privileged - baseline, atol=1e-6):
                errors.append(f'{path}: residual mismatch')
            if not np.allclose(np.exp(target).sum(-1), 1., atol=1e-5):
                errors.append(f'{path}: target is not normalized')
    completed = read(root / 'completed.json') if (root / 'completed.json').exists() else {}
    status = 'passed' if not errors and completed.get('status') == 'complete' else 'failed'
    result = {'status': status, 'errors': errors}
    (root / 'chunk_v13_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return status == 'passed'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    raise SystemExit(0 if audit(parser.parse_args().root) else 1)
