from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


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
    target_dx_m: float | None = None
    target_dy_m: float | None = None
    target_direction: str | None = None

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
    direction_section=(
        "Measured target displacement from the cube:\n"
        f"  x: {value(result.target_dx_m)} meters; y: {value(result.target_dy_m)} meters.\n"
        f"  Dominant remaining direction: {value(result.target_direction)}.\n\n"
        if result.target_direction is not None else '')
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
        + direction_section
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


def outcome_verdict(result: FeedbackRecord) -> str:
    if result.goal_progress == 'closer':
        return 'accept'
    if result.goal_progress == 'farther' or not result.contact:
        return 'reject'
    return 'reconsider'


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


def causal_feedback(action_label: str, result: FeedbackRecord) -> FeedbackRecord:
    """Convert measured outcome into a generic self-revision rule, never an expert action."""
    verdict=outcome_verdict(result)
    if verdict == 'accept':
        interpretation=(
            f'PREVIOUS_RESPONSE={action_label}; OUTCOME_VERDICT=ACCEPT. '
            f'Measured target distance decreased. For this same original state, preserve response {action_label}.'
        )
    elif result.goal_progress == 'farther':
        interpretation=(
            f'PREVIOUS_RESPONSE={action_label}; OUTCOME_VERDICT=REJECT. '
            'Measured target distance increased. Infer a different response from the original image.'
        )
    elif not result.contact:
        interpretation=(
            f'PREVIOUS_RESPONSE={action_label}; OUTCOME_VERDICT=REJECT. '
            'The action produced no cube contact. Infer a different response from the original image.'
        )
    elif result.object_motion == 'none':
        interpretation=(
            f'PREVIOUS_RESPONSE={action_label}; OUTCOME_VERDICT=RECONSIDER. '
            'The action contacted the cube but caused no measurable cube motion. Re-evaluate the original image.'
        )
    else:
        interpretation=(
            f'PREVIOUS_RESPONSE={action_label}; OUTCOME_VERDICT=RECONSIDER. '
            'Use only the measured outcome to preserve or revise the response for the original image.'
        )
    return FeedbackRecord(**{**result.as_dict(), 'interpretation': interpretation})


def directional_feedback(action_label: str, transition) -> FeedbackRecord:
    """Add measured target-relative direction without supplying an expert action label."""
    result = transition.result
    measurements = getattr(transition, 'measurements', {})
    if 'object_before' not in measurements or 'goal_xy' not in measurements:
        raise ValueError('Directional feedback requires object_before and goal_xy measurements')
    displacement = np.asarray(measurements['goal_xy'], dtype=float) - np.asarray(
        measurements['object_before'][:2], dtype=float)
    if abs(displacement[0]) >= abs(displacement[1]):
        direction = '+x' if displacement[0] > 0 else '-x'
    else:
        direction = '+y' if displacement[1] > 0 else '-y'
    return FeedbackRecord(**{
        **result.as_dict(),
        'target_dx_m': float(displacement[0]),
        'target_dy_m': float(displacement[1]),
        'target_direction': direction,
    })


def wrong_directional_feedback(action_label: str, transition) -> FeedbackRecord:
    """Return measured direction with signs flipped as a causal control."""
    feedback = directional_feedback(action_label, transition)
    direction = {
        '+x': '-x', '-x': '+x', '+y': '-y', '-y': '+y',
    }[feedback.target_direction]
    return FeedbackRecord(**{
        **feedback.as_dict(),
        'target_dx_m': -feedback.target_dx_m,
        'target_dy_m': -feedback.target_dy_m,
        'target_direction': direction,
    })


def select_feedback(transitions, mode: str, seed: int = 11):
    """Return feedback, validity mask, and auditable donor mapping without altering raw outcomes.

    Shuffled donors preserve the complete tuple, attempted action, and success. Donors
    must come from another episode. Rows without an eligible donor fall back to
    binary feedback, as specified in the protocol; their donor index is None.
    """
    import numpy as np
    if mode not in {'result','result_interpreted','causal','causal_shuffled','binary','shuffled','hard_shuffled','neutral','no_feedback','reflection'}:
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
    elif mode=='causal':
        records=[causal_feedback(t.action_label,f) for t,f in zip(transitions,records)]
    elif mode=='neutral':
        records=[neutral_feedback() for _ in records]
    elif mode=='no_feedback':
        records=[neutral_feedback() for _ in records]
    elif mode in {'shuffled','causal_shuffled'}:
        rng=np.random.default_rng(seed)
        groups={}
        for i,t in enumerate(transitions):
            if valid[i]:
                key=(t.result.success,) if mode=='causal_shuffled' else (t.action_label,t.result.success)
                groups.setdefault(key,[]).append(i)
        donors=[None]*len(transitions)
        shuffled=([causal_feedback(t.action_label,binary_feedback(records[i])) for i,t in enumerate(transitions)]
                  if mode=='causal_shuffled' else [neutral_feedback() for _ in records])
        for i,t in enumerate(transitions):
            key=(t.result.success,) if mode=='causal_shuffled' else (t.action_label,t.result.success)
            candidates=[j for j in groups.get(key,[])
                        if transitions[j].episode_id!=t.episode_id and
                        (mode!='causal_shuffled' or outcome_verdict(records[j])!=outcome_verdict(records[i]))]
            if not valid[i] or not candidates:
                shuffled[i]=(causal_feedback(t.action_label,binary_feedback(records[i]))
                             if mode=='causal_shuffled' else binary_feedback(records[i]))
                continue
            j=int(rng.choice(candidates))
            donors[i]=j
            shuffled[i]=causal_feedback(t.action_label,records[j]) if mode=='causal_shuffled' else records[j]
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
