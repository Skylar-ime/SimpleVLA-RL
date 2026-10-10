from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FeedbackRecord:
    contact: bool | None
    object_motion: str | None
    goal_progress: str | None
    success: bool | None
    progress_m: float | None = None
    object_motion_m: float | None = None
    distance_before_m: float | None = None
    distance_after_m: float | None = None
    interpretation: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_feedback_text(action_label: str, result: FeedbackRecord) -> str:
    def value(value: Any) -> str:
        if value is None:
            return "not reported"
        if isinstance(value, bool):
            return "yes" if value else "no"
        return str(value)

    interpretation_section=(f"Outcome interpretation: {result.interpretation}.\n\n"
                            if result.interpretation is not None else '')
    return (
        f"Attempted action at the original observation: {action_label}.\n"
        "Observed result after execution:\n"
        f"Contact with the cube: {value(result.contact)}.\n"
        f"Cube motion: {value(result.object_motion)}.\n"
        f"Distance to the target: {value(result.goal_progress)}.\n"
        f"Signed distance progress in meters (positive means closer): {value(result.progress_m)}.\n"
        f"Cube displacement in meters: {value(result.object_motion_m)}.\n"
        f"Distance before action in meters: {value(result.distance_before_m)}.\n"
        f"Distance after action in meters: {value(result.distance_after_m)}.\n"
        f"Task completed: {value(result.success)}.\n\n"
        + interpretation_section +
        "Choose the action for the ORIGINAL image/state, not for the post-action state.\n"
        "Return exactly one action label."
    )


def feedback_is_informative(result: FeedbackRecord) -> bool:
    """Whether measured interaction produced non-empty SDPO environment feedback."""
    return bool(result.contact or result.object_motion=='detected' or result.goal_progress not in {None,'unchanged'})


def binary_feedback(result: FeedbackRecord) -> FeedbackRecord:
    return FeedbackRecord(
        contact=None,
        object_motion=None,
        goal_progress=None,
        success=result.success,
    )


def neutral_feedback() -> FeedbackRecord:
    return FeedbackRecord(None, None, None, None)


def interpreted_feedback(action_label: str, result: FeedbackRecord) -> FeedbackRecord:
    if result.goal_progress=='closer':
        interpretation=(f'The model-generated action {action_label} produced positive measured progress. '
                        f'Retain action {action_label} for the original state.')
    elif result.goal_progress=='farther':
        interpretation=(f'The model-generated action {action_label} produced negative measured progress. '
                        f'Do not repeat action {action_label} for the original state.')
    elif result.contact and result.object_motion=='none':
        interpretation=(f'The model-generated action {action_label} reached the cube but was ineffective. '
                        f'Reconsider action {action_label} for the original state.')
    else:
        interpretation=None
    return FeedbackRecord(**{**result.as_dict(),'interpretation':interpretation})


def select_feedback(transitions, mode: str, seed: int = 11):
    """Return feedback, validity mask, and auditable donor mapping without altering raw outcomes.

    Shuffled donors preserve the complete tuple, attempted action, and success. Donors
    must come from another episode. Rows without an eligible donor fall back to
    binary feedback, as specified in the protocol; their donor index is None.
    """
    import numpy as np
    if mode not in {'result','result_interpreted','binary','shuffled','hard_shuffled','neutral','no_feedback'}:
        raise ValueError(f'Unsupported feedback group: {mode}')
    if any(t.result is None for t in transitions):
        raise ValueError('Executed transitions require results')
    records=[t.result for t in transitions]
    valid=np.array([t.safety_failure is None for t in transitions],dtype=bool)
    donors=list(range(len(transitions)))
    if mode=='binary':
        records=[binary_feedback(f) for f in records]
    elif mode=='result_interpreted':
        records=[interpreted_feedback(t.action_label,f) for t,f in zip(transitions,records)]
    elif mode=='neutral':
        records=[neutral_feedback() for _ in records]
    elif mode=='no_feedback':
        records=[neutral_feedback() for _ in records]
    elif mode=='shuffled':
        rng=np.random.default_rng(seed)
        groups={}
        for i,t in enumerate(transitions):
            if valid[i]:
                groups.setdefault((t.action_label,t.result.success),[]).append(i)
        donors=[None]*len(transitions)
        shuffled=[neutral_feedback() for _ in records]
        for i,t in enumerate(transitions):
            candidates=[j for j in groups.get((t.action_label,t.result.success),[])
                        if transitions[j].episode_id!=t.episode_id]
            if not valid[i] or not candidates:
                shuffled[i]=binary_feedback(records[i])
                continue
            j=int(rng.choice(candidates))
            donors[i]=j
            shuffled[i]=records[j]
        records=shuffled
    elif mode=='hard_shuffled':
        rng=np.random.default_rng(seed)
        candidates=[i for i,t in enumerate(transitions) if valid[i]]
        donors=[None]*len(transitions)
        shuffled=[binary_feedback(records[i]) for i in range(len(records))]
        for i,t in enumerate(transitions):
            pool=[j for j in candidates if transitions[j].episode_id!=t.episode_id and records[j]!=records[i]]
            if not valid[i] or not pool:
                continue
            j=int(rng.choice(pool))
            donors[i]=j
            shuffled[i]=records[j]
        records=shuffled
    return records,valid,donors
