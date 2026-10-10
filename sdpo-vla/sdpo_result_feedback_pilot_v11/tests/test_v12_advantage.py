from types import SimpleNamespace

import numpy as np
import pytest

from sdpo_result_feedback.feedback import (
    FeedbackRecord,
    build_feedback_text,
    counterfactual_advantage_feedback,
)
from sdpo_result_feedback.sdpo_context import build_sdpo_context


def transition(episode, group, action, progress, result=None):
    return SimpleNamespace(
        episode_id=episode,
        rollout_group_id=group,
        rollout_candidate_index=0,
        scene_id=group,
        decision_index=0,
        instruction='task',
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        proprio=np.zeros(3),
        action_label=action,
        result=result or FeedbackRecord(False, 'none', 'unchanged', False),
        safety_failure=None,
        measurements={
            'counterfactual_progress_m': progress,
            'counterfactual_contact': [False, False, False, False, False],
            'counterfactual_safety_failure': [False, False, False, False, False],
        },
    )


def test_action_relative_advantage_uses_same_state_hold_baseline():
    row = transition('a', 'g0', 'A', [0.01, -0.01, 0.02, 0.0, 0.003])
    feedback = counterfactual_advantage_feedback(row)
    assert feedback.counterfactual_advantage_m == pytest.approx(
        (0.007, -0.013, 0.017, -0.003, 0.0))
    assert feedback.action_advantage_m == pytest.approx(0.007)
    text = build_feedback_text('A', feedback)
    assert 'Same-state counterfactual action-relative feedback' in text
    assert 'A: advantage_vs_hold=0.007' in text
    assert 'Contact with the cube: not reported.' in text
    assert 'progress=' not in text
    assert 'unsafe=' not in text


def test_reversed_advantage_flips_only_counterfactual_values():
    row = transition('a', 'g0', 'C', [0.01, -0.01, 0.02, 0.0, 0.003])
    feedback = counterfactual_advantage_feedback(row, sign=-1.0)
    assert feedback.counterfactual_progress_m == pytest.approx((0.01, -0.01, 0.02, 0.0, 0.003))
    assert feedback.counterfactual_advantage_m == pytest.approx(
        (-0.007, 0.013, -0.017, 0.003, 0.0))
    assert feedback.action_advantage_m == pytest.approx(-0.017)
    text = build_feedback_text('C', feedback)
    assert 'C: advantage_vs_hold=-0.017' in text
    assert 'progress=' not in text


def test_v12_context_keeps_advantage_feedback_outcome_free():
    rows = [
        transition('g0a', 'g0', 'A', [0.01, -0.01, 0.02, 0.0, 0.003]),
        transition('g0b', 'g0', 'C', [0.01, -0.01, 0.02, 0.0, 0.003]),
        transition('g1a', 'g1', 'A', [0.0, 0.02, 0.0, -0.01, 0.0]),
        transition('g1b', 'g1', 'D', [0.0, 0.02, 0.0, -0.01, 0.0]),
    ]
    context = build_sdpo_context(rows, 'sdpo_advantage')
    assert context['valid_mask'].all()
    assert all(record.success is None for record in context['records'])
    assert context['records'][0].action_advantage_m == pytest.approx(0.007)


def test_v12_shuffled_advantage_uses_other_group():
    rows = [
        transition('g0a', 'g0', 'A', [0.01, -0.01, 0.02, 0.0, 0.003]),
        transition('g0b', 'g0', 'C', [0.01, -0.01, 0.02, 0.0, 0.003]),
        transition('g1a', 'g1', 'A', [0.0, 0.02, 0.0, -0.01, 0.0]),
        transition('g1b', 'g1', 'D', [0.0, 0.02, 0.0, -0.01, 0.0]),
    ]
    context = build_sdpo_context(rows, 'sdpo_advantage_shuffled', seed=7)
    assert all(donor is not None for donor in context['donor_indices'])
    assert all(rows[donor].rollout_group_id != rows[index].rollout_group_id
               for index, donor in enumerate(context['donor_indices']))
    assert context['records'][0].counterfactual_progress_m == pytest.approx(
        rows[2].measurements['counterfactual_progress_m'])
