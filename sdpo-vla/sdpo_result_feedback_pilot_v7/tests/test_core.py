import torch
import numpy as np

from sdpo_result_feedback.actions import ACTION_LABELS, ACTION_VECTORS
from sdpo_result_feedback.feedback import FeedbackRecord, binary_feedback, build_feedback_text, causal_feedback, feedback_is_informative, interpreted_feedback
from sdpo_result_feedback.losses import reverse_kl_loss, symmetric_js_loss
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot


def test_action_contract():
    assert ACTION_LABELS == ("A", "B", "C", "D", "E")
    assert ACTION_VECTORS["A"][0] == -ACTION_VECTORS["B"][0]
    assert ACTION_VECTORS["C"][1] == -ACTION_VECTORS["D"][1]
    assert ACTION_VECTORS["E"].tolist() == [0.0, 0.0, 0.0]


def test_feedback_text_has_only_observed_fields():
    result = FeedbackRecord(True, "detected", "farther", False)
    text = build_feedback_text("C", result)
    assert "should move" not in text
    assert "farther" in text
    binary = binary_feedback(result)
    assert binary.success is False
    assert binary.goal_progress is None
    assert feedback_is_informative(result)
    assert not feedback_is_informative(FeedbackRecord(False,'none','unchanged',False))
    interpreted=interpreted_feedback('C',FeedbackRecord(True,'detected','closer',False))
    assert 'model-generated action C' in interpreted.interpretation
    assert 'expert' not in interpreted.interpretation.lower()
    causal=causal_feedback('D',FeedbackRecord(True,'detected','closer',False))
    assert 'OUTCOME_VERDICT=ACCEPT' in causal.interpretation
    assert 'preserve response D' in causal.interpretation
    assert 'expert' not in causal.interpretation.lower()


def test_js_zero_and_gradients():
    logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = symmetric_js_loss(log_probs, log_probs.detach())
    assert float(loss) < 1e-7
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-6)


def test_reverse_kl_matches_manual_and_detaches_teacher():
    student=torch.tensor([[.1,.2,.3,.15,.25]],requires_grad=True)
    teacher=torch.tensor([[.4,.1,.2,.2,.1]],requires_grad=True)
    expected=torch.sum(student.detach()*torch.log(student.detach()/teacher.detach()))
    loss=reverse_kl_loss(student.log(),teacher.log())
    assert torch.allclose(loss,expected)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_panda_push_transition_contract():
    env = PandaPushOutcomePilot(seed=11)
    observation = env.reset("smoke_0001")
    assert observation["image"].ndim == 3
    assert observation["image"].shape[-1] == 3
    _, transition = env.step("E")
    assert transition.result.goal_progress in {"closer", "unchanged", "farther"}
    assert transition.result.object_motion in {"none", "detected"}
    assert len(transition.actual_ee_delta_m) == 3
    assert np.isfinite(transition.distance_before_m)
    assert np.isfinite(transition.distance_after_m)
    env.close()
