from dataclasses import replace
from collections import defaultdict

import numpy as np

from .feedback import (aggregate_feedback, binary_feedback, build_chunk_feedback_text,
                       causal_feedback, directional_feedback,
                       neutral_feedback, no_op_directional_feedback,
                       wrong_directional_feedback)


CHUNK_CONDITIONS = (
    'chunk_result', 'chunk_binary', 'chunk_causal',
    'chunk_directional', 'chunk_wrong_directional',
    'chunk_directional_residual', 'chunk_wrong_directional_residual',
    'chunk_suffix_result', 'chunk_suffix_directional',
    'chunk_suffix_directional_residual', 'chunk_suffix_wrong_directional',
    'chunk_suffix_wrong_directional_residual',
    'chunk_solution', 'chunk_solution_plus_directional',
)


def _with_chunk_transcript(chunk, record):
    transcript = build_chunk_feedback_text(chunk.action_labels, chunk.step_results,
                                           chunk.result)
    interpretation = ((record.interpretation + '\n' + transcript)
                      if record.interpretation else transcript)
    return replace(record, interpretation=interpretation)


def _with_suffix_transcript(chunk, record, start_index):
    transcript = build_chunk_feedback_text(chunk.action_labels, chunk.step_results,
                                           record, start_index=start_index)
    interpretation = ((record.interpretation + '\n' + transcript)
                      if record.interpretation else transcript)
    return replace(record, interpretation=interpretation)


def _feedback_record(chunk, condition, position):
    suffix = condition.startswith('chunk_suffix_')
    base_condition = ('chunk_' + condition.removeprefix('chunk_suffix_')
                      if suffix else condition)
    result = (aggregate_feedback(chunk.step_results[position:]) if suffix
              else chunk.result)
    action_text = ' '.join(chunk.action_labels[position:] if suffix
                           else chunk.action_labels)
    if base_condition in {'chunk_directional', 'chunk_directional_residual'}:
        record = directional_feedback(action_text, chunk, result=result)
    elif base_condition in {'chunk_wrong_directional', 'chunk_wrong_directional_residual'}:
        record = wrong_directional_feedback(action_text, chunk, result=result)
    elif base_condition == 'chunk_binary':
        record = binary_feedback(result)
    elif base_condition == 'chunk_causal':
        record = causal_feedback(action_text, result)
    else:
        record = result
    return (_with_suffix_transcript(chunk, record, position) if suffix
            else _with_chunk_transcript(chunk, record))


def build_chunk_context(chunks, condition):
    if condition not in CHUNK_CONDITIONS or not chunks:
        raise ValueError('Unknown chunk condition or empty batch')
    groups = defaultdict(list)
    for index, chunk in enumerate(chunks):
        if chunk.rollout_group_id is None:
            raise ValueError('Chunk group identity is required')
        groups[chunk.rollout_group_id].append(index)
    for members in groups.values():
        original = chunks[members[0]]
        for index in members[1:]:
            candidate = chunks[index]
            if (candidate.scene_id != original.scene_id
                    or not np.array_equal(candidate.image, original.image)
                    or not np.array_equal(candidate.proprio, original.proprio)):
                raise ValueError('Chunk group contains different initial states')

    records = []
    records_by_position = []
    solution_condition = condition in {'chunk_solution', 'chunk_solution_plus_directional'}
    solutions = ({group: next((chunks[index] for index in members
                               if chunks[index].safety_failure is None
                               and chunks[index].result.success), None)
                  for group, members in groups.items()} if solution_condition else {})
    solution_actions = []
    include_outcome_feedback = []
    for chunk in chunks:
        donor = solutions.get(chunk.rollout_group_id)
        solution = (tuple(donor.action_labels) if donor is not None
                    and donor.episode_id != chunk.episode_id else None)
        solution_actions.append(solution)
        include_outcome_feedback.append(
            solution is None or condition == 'chunk_solution_plus_directional')
        if condition == 'chunk_solution':
            record = neutral_feedback() if solution is not None else chunk.result
        elif condition == 'chunk_solution_plus_directional':
            record = directional_feedback(' '.join(chunk.action_labels), chunk)
        elif condition == 'chunk_binary':
            record = binary_feedback(chunk.result)
        elif condition == 'chunk_causal':
            record = causal_feedback(' '.join(chunk.action_labels), chunk.result)
        elif condition in {'chunk_directional', 'chunk_directional_residual'}:
            record = directional_feedback(' '.join(chunk.action_labels), chunk)
        elif condition in {'chunk_wrong_directional', 'chunk_wrong_directional_residual'}:
            record = wrong_directional_feedback(' '.join(chunk.action_labels), chunk)
        else:
            record = chunk.result
        if condition.startswith('chunk_suffix_'):
            record = _feedback_record(chunk, condition, 0)
        if condition not in {'chunk_binary', 'chunk_solution'} or solution is None:
            record = _with_chunk_transcript(chunk, record)
        records.append(record)
        records_by_position.append([
            _feedback_record(chunk, condition, position)
            if condition.startswith('chunk_suffix_') else record
            for position in range(chunk.chunk_length)
        ])
    subtract_noop_baseline = condition in {
        'chunk_directional_residual', 'chunk_wrong_directional_residual',
        'chunk_suffix_directional_residual',
        'chunk_suffix_wrong_directional_residual',
    }
    baseline_records = ([no_op_directional_feedback(record) for record in records]
                        if subtract_noop_baseline else None)
    baseline_records_by_position = (
        [[no_op_directional_feedback(record) for record in position_records]
         for position_records in records_by_position]
        if subtract_noop_baseline else None
    )
    feedback_only = not solution_condition
    valid_mask = np.asarray([
        chunk.safety_failure is None and (not feedback_only or not chunk.result.success)
        for chunk in chunks
    ], dtype=bool)
    return {
        'condition': condition,
        'position_specific_feedback': condition.startswith('chunk_suffix_'),
        'feedback_scope': ('return_to_go_suffix' if condition.startswith('chunk_suffix_')
                           else 'aggregate_chunk'),
        'records': records,
        'records_by_position': records_by_position,
        'valid_mask': valid_mask,
        'donor_indices': list(range(len(chunks))),
        'group_count': len(groups),
        'chunk_length': len(chunks[0].action_labels),
        'subtract_noop_baseline': subtract_noop_baseline,
        'baseline_records': baseline_records,
        'baseline_records_by_position': baseline_records_by_position,
        'solution_actions': solution_actions,
        'include_outcome_feedback': include_outcome_feedback,
        'solution_context_count': sum(action is not None for action in solution_actions),
        'feedback_only': feedback_only,
        'excluded_success_count': sum(chunk.result.success for chunk in chunks)
        if feedback_only else 0,
    }


def chunk_noop_records(chunks, records):
    return [no_op_directional_feedback(record) for record in records]
