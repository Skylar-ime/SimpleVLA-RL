from types import SimpleNamespace

import numpy as np
import torch

from sdpo_result_feedback.chunk_context import build_chunk_context
from sdpo_result_feedback.env_adapter import ChunkTransition
from sdpo_result_feedback.feedback import FeedbackRecord, aggregate_feedback
from sdpo_result_feedback.trainer import (ChunkRolloutCollector, TrainingConfig,
                                           sdpo_chunk_update)
from sdpo_result_feedback.scenes import create_directional_local_scene_splits


def result(progress='closer', success=False):
    return FeedbackRecord(True, 'detected', progress, success,
                          .01 if progress == 'closer' else 0., .01, .1, .09)


def make_chunk(group='g0', actions=('A', 'C')):
    return ChunkTransition(
        episode_id=group + '_c0', decision_index=0,
        image=np.zeros((2, 2, 3), dtype=np.uint8), instruction='task',
        proprio=np.zeros(3), action_labels=actions,
        step_results=(result(), result()), result=result(), terminated=False,
        truncated=True, scene_id=group, measurements={
            'object_before': [0., 0., .02], 'object_after': [.1, .0, .02],
            'goal_xy': [.2, .0],
        }, rollout_group_id=group, rollout_candidate_index=0)


def test_chunk_context_residual_preserves_outcome_and_removes_direction():
    chunk = make_chunk()
    context = build_chunk_context([chunk], 'chunk_directional_residual')
    assert context['records'][0].target_direction == '+x'
    assert context['baseline_records'][0].target_direction == 'hold'
    assert context['records'][0].goal_progress == context['baseline_records'][0].goal_progress


def test_suffix_feedback_uses_position_specific_return_to_go():
    chunk = make_chunk()
    chunk.step_results = (
        FeedbackRecord(False, 'none', 'farther', False, -.01, 0., .10, .11),
        FeedbackRecord(True, 'detected', 'closer', False, .01, .01, .11, .10),
    )
    chunk.result = aggregate_feedback(chunk.step_results)
    context = build_chunk_context([chunk], 'chunk_suffix_result')
    first = context['records_by_position'][0]
    second = context['records_by_position'][0]
    assert first[0].goal_progress == 'unchanged'
    assert second[1].goal_progress == 'closer'
    assert 'Step 0' in first[0].interpretation
    assert 'Step 1' in second[1].interpretation
    assert 'Step 0' not in second[1].interpretation


def test_suffix_residual_has_position_specific_noop_baseline():
    chunk = make_chunk()
    context = build_chunk_context([chunk], 'chunk_suffix_directional_residual')
    assert context['records_by_position'][0][0].target_direction == '+x'
    assert context['baseline_records_by_position'][0][0].target_direction == 'hold'
    assert context['records_by_position'][0][0].goal_progress == 'closer'
    assert context['records_by_position'][0][1].goal_progress == 'closer'


def test_chunk_solution_uses_same_state_success_only():
    success = make_chunk('g0', ('A', 'C'))
    success.result = result(success=True)
    success.episode_id = 'g0_c0'
    failure = make_chunk('g0', ('B', 'D'))
    failure.episode_id = 'g0_c1'
    unrelated = make_chunk('g1', ('B', 'D'))
    context = build_chunk_context([success, failure, unrelated], 'chunk_solution')
    assert context['solution_actions'] == [None, ('A', 'C'), None]
    assert context['include_outcome_feedback'] == [True, False, True]
    assert context['solution_context_count'] == 1
    rich = build_chunk_context([success, failure, unrelated],
                               'chunk_solution_plus_directional')
    assert rich['solution_actions'][1] == ('A', 'C')
    assert rich['include_outcome_feedback'] == [True, True, True]


def test_chunk_collector_reuses_exact_initial_state(tmp_path):
    scenes = create_directional_local_scene_splits(
        tmp_path / 'chunk_scenes.json', {'train': 1, 'dev': 0, 'test': 0})['train']

    class FixedPolicy:
        def sample_chunk(self, inputs, length, **kwargs):
            return [('A', 'C'), ('B', 'D'), ('E', 'E'), ('C', 'A')][:len(inputs)]

    collector = ChunkRolloutCollector(
        TrainingConfig(transitions_per_update=4, microbatch_size=2,
                       parallel_envs=1, max_decisions=2, seed=11, chunk_length=2),
        scenes, candidates_per_scene=4)
    try:
        chunks = collector.collect(FixedPolicy())
        assert len(chunks) == 4
        assert all(np.array_equal(chunk.image, chunks[0].image) for chunk in chunks)
        assert all(np.array_equal(chunk.proprio, chunks[0].proprio) for chunk in chunks)
        assert all(chunk.rollout_group_id == chunks[0].rollout_group_id for chunk in chunks)
        assert all(len(chunk.action_labels) == 2 for chunk in chunks)
    finally:
        collector.close()


class TinyChunkPolicy:
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = torch.nn.Linear(2, 5, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        self.config = SimpleNamespace(distillation_loss='js')
        self.teacher_calls = 0

    def chunk_log_probs(self, chunks, action_chunks, feedback=False, grad=False,
                        feedback_records=None, **kwargs):
        rows = []
        for chunk in chunks:
            x = torch.tensor([[float(chunk.rollout_candidate_index or 0), float(feedback)]])
            logits = self.model(x)
            row = torch.log_softmax(logits, dim=-1).repeat(len(action_chunks[0]), 1)
            rows.append(row)
        return torch.stack(rows)

    def teacher_chunk_log_probs(self, chunks, action_chunks, feedback=False,
                                feedback_records=None, **kwargs):
        self.teacher_calls += 1
        with torch.no_grad():
            return self.chunk_log_probs(chunks, action_chunks, feedback, False, feedback_records)

    def trainable_parameters(self):
        return self.model.parameters()

    def update_teacher_ema(self):
        pass


def test_chunk_update_accumulates_one_optimizer_step():
    chunks = [make_chunk('g0'), make_chunk('g1', ('B', 'D'))]
    context = build_chunk_context(chunks, 'chunk_directional_residual')
    policy = TinyChunkPolicy()
    stats = sdpo_chunk_update(policy, chunks, microbatch_size=1, teacher_context=context)
    assert stats['optimizer_steps'] == 1
    assert stats['valid_actions'] == 4
    assert stats['teacher_residual_chunk_log_probs'] is not None
    assert policy.teacher_calls == 2
