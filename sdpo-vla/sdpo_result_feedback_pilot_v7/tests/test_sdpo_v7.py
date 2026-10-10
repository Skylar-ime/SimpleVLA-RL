from types import SimpleNamespace

import numpy as np

from sdpo_result_feedback.feedback import FeedbackRecord
from sdpo_result_feedback.sdpo_context import build_sdpo_context


def transition(episode, group, action, success):
    result = FeedbackRecord(True, 'detected', 'closer' if success else 'unchanged', success)
    return SimpleNamespace(episode_id=episode, rollout_group_id=group,
                           rollout_candidate_index=0, scene_id=group,
                           decision_index=0, instruction='task',
                           image=np.zeros((2, 2, 3), dtype=np.uint8),
                           proprio=np.zeros(3), action_label=action,
                           result=result, safety_failure=None)


def test_v7_uses_only_actual_same_state_success_as_solution():
    rows = [transition('g0a', 'g0', 'A', True), transition('g0b', 'g0', 'C', False),
            transition('g1a', 'g1', 'A', False), transition('g1b', 'g1', 'C', False)]
    context = build_sdpo_context(rows, 'sdpo_result')
    assert context['solution_actions'] == [None, 'A', None, None]
    assert context['solution_donor_indices'] == [None, 0, None, None]
    assert context['solution_context_count'] == 1
    assert context['include_outcome_feedback'] == [True, False, True, True]
    assert context['valid_mask'].tolist() == [True, True, True, True]


def test_v7_shuffled_donor_cannot_share_original_state():
    rows = [transition('g0a', 'g0', 'A', False), transition('g0b', 'g0', 'A', False),
            transition('g1a', 'g1', 'A', False), transition('g1b', 'g1', 'A', False)]
    context = build_sdpo_context(rows, 'sdpo_shuffled', seed=3)
    assert all(donor is None or rows[donor].rollout_group_id != rows[index].rollout_group_id
               for index, donor in enumerate(context['donor_indices']))
