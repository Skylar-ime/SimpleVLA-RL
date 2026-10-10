from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_teacher_residual(
    student_log_probs: torch.Tensor,
    privileged_teacher_log_probs: torch.Tensor,
    baseline_teacher_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add teacher-vs-no-op log-ratio to the current student distribution."""
    if not (student_log_probs.shape == privileged_teacher_log_probs.shape
            == baseline_teacher_log_probs.shape):
        raise ValueError("student, teacher, and baseline shapes must match")
    student_log_probs = student_log_probs.float().detach()
    privileged_teacher_log_probs = privileged_teacher_log_probs.float().detach()
    baseline_teacher_log_probs = baseline_teacher_log_probs.float().detach()
    residual = privileged_teacher_log_probs - baseline_teacher_log_probs
    corrected = F.log_softmax(student_log_probs + residual, dim=-1)
    return corrected, residual


def symmetric_js_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher log-probability shapes must match")
    student_log_probs = student_log_probs.float()
    teacher_log_probs = teacher_log_probs.float().detach()
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    mixture = 0.5 * (student_probs + teacher_probs)
    mixture_log_probs = mixture.clamp_min(torch.finfo(torch.float32).tiny).log()
    student_kl = torch.sum(student_probs * (student_log_probs - mixture_log_probs), dim=-1)
    teacher_kl = torch.sum(teacher_probs * (teacher_log_probs - mixture_log_probs), dim=-1)
    per_item = 0.5 * (student_kl + teacher_kl)
    if mask is not None:
        mask = mask.to(dtype=per_item.dtype, device=per_item.device)
        return (per_item * mask).sum() / mask.sum().clamp_min(1.0)
    return per_item.mean()


def reverse_kl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """SDPO alpha=1 loss: KL(student || teacher), with detached teacher."""
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher log-probability shapes must match")
    teacher_log_probs = teacher_log_probs.float().detach()
    student_log_probs = student_log_probs.float()
    per_item = F.kl_div(teacher_log_probs, student_log_probs, reduction='none', log_target=True).sum(-1)
    if mask is not None:
        mask = mask.to(dtype=per_item.dtype, device=per_item.device)
        return (per_item * mask).sum() / mask.sum().clamp_min(1.0)
    return per_item.mean()
