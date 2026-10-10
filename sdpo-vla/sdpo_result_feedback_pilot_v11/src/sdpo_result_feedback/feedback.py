from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .actions import ACTION_LABELS


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
    counterfactual_progress_m: tuple[float, ...] | None = None
    counterfactual_advantage_m: tuple[float, ...] | None = None
    counterfactual_contact: tuple[bool, ...] | None = None
    counterfactual_safety_failure: tuple[bool, ...] | None = None
    action_advantage_m: float | None = None

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
    counterfactual_section = ''
    if result.counterfactual_advantage_m is not None:
        rows = []
        for label, advantage in zip(ACTION_LABELS, result.counterfactual_advantage_m):
            rows.append(f'  {label}: advantage_vs_hold={advantage}.')
        counterfactual_section = (
            'Same-state counterfactual action-relative feedback '
            '(positive means better than hold):\n'
            + '\n'.join(rows) + '\n'
            f'Attempted-action advantage: {result.action_advantage_m}.\n\n'
        )
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
        + counterfactual_section
        + interpretation_section +
        "Choose the action for the ORIGINAL image/state, not for the post-action state.\n"
        "Return exactly one action label."
    )


def build_chunk_feedback_text(action_labels, step_results, final_result, start_index=0) -> str:
    """Describe measured outcomes from a chunk suffix without prescribing an action."""
    if start_index < 0 or start_index >= len(step_results):
        raise ValueError('Chunk feedback start index must select an executed step')
    suffix = step_results[start_index:]
    lines = ['Observed result for each executed action in the chunk suffix:']
    for index, (action, result) in enumerate(zip(action_labels[start_index:], suffix),
                                             start=start_index):
        lines.append(
            f'  Step {index}: action {action}; contact={result.contact}; '
            f'cube_motion={result.object_motion}; goal_progress={result.goal_progress}; '
            f'signed_progress_m={result.progress_m}; success={result.success}.')
    return '\n'.join(lines)


def aggregate_feedback(step_results: tuple[FeedbackRecord, ...] | list[FeedbackRecord]) -> FeedbackRecord:
    """Aggregate measured return-to-go feedback for an executed chunk suffix."""
    if not step_results:
        raise ValueError('Cannot aggregate empty chunk feedback')
    first, last = step_results[0], step_results[-1]
    progress = float(first.distance_before_m - last.distance_after_m)
    if progress > 0.002:
        goal_progress = 'closer'
    elif progress < -0.002:
        goal_progress = 'farther'
    else:
        goal_progress = 'unchanged'
    return FeedbackRecord(
        contact=any(result.contact for result in step_results),
        object_motion=('detected' if any(result.object_motion == 'detected'
                                         for result in step_results) else 'none'),
        goal_progress=goal_progress,
        success=bool(last.success),
        progress_m=progress,
        object_motion_m=sum(result.object_motion_m or 0. for result in step_results),
        distance_before_m=first.distance_before_m,
        distance_after_m=last.distance_after_m,
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


def directional_feedback(action_label: str, transition, result=None) -> FeedbackRecord:
    """Add measured target-relative direction without supplying an expert action label."""
    result = transition.result if result is None else result
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


def wrong_directional_feedback(action_label: str, transition, result=None) -> FeedbackRecord:
    """Return measured direction with signs flipped as a causal control."""
    feedback = directional_feedback(action_label, transition, result=result)
    direction = {
        '+x': '-x', '-x': '+x', '+y': '-y', '-y': '+y',
    }[feedback.target_direction]
    return FeedbackRecord(**{
        **feedback.as_dict(),
        'target_dx_m': -feedback.target_dx_m,
        'target_dy_m': -feedback.target_dy_m,
        'target_direction': direction,
    })


def counterfactual_advantage_feedback(transition, sign: float = 1.0,
                                      result: FeedbackRecord | None = None,
                                      counterfactual_source: dict | None = None) -> FeedbackRecord:
    """Expose same-state action-relative progress without prescribing an action."""
    measurements = getattr(transition, 'measurements', {})
    source_measurements = counterfactual_source or measurements
    progress = source_measurements.get('counterfactual_progress_m')
    contacts = source_measurements.get('counterfactual_contact')
    failures = source_measurements.get('counterfactual_safety_failure')
    if progress is None or len(progress) != len(ACTION_LABELS):
        raise ValueError('Counterfactual advantage feedback requires five action outcomes')
    baseline = float(progress[ACTION_LABELS.index('E')])
    advantages = tuple(float(sign) * (float(value) - baseline) for value in progress)
    attempted = ACTION_LABELS.index(transition.action_label)
    source = result if result is not None else FeedbackRecord(None, None, None, None)
    return FeedbackRecord(
        **{**source.as_dict(),
           'counterfactual_progress_m': tuple(float(value) for value in progress),
           'counterfactual_advantage_m': advantages,
           'counterfactual_contact': (None if contacts is None else tuple(bool(value) for value in contacts)),
           'counterfactual_safety_failure': (
               None if failures is None else tuple(bool(value) for value in failures)),
           'action_advantage_m': advantages[attempted]})


def no_op_directional_feedback(result: FeedbackRecord) -> FeedbackRecord:
    if result.target_direction not in {'+x', '-x', '+y', '-y'}:
        raise ValueError('No-op baseline requires directional teacher feedback')
    return replace(result, target_dx_m=0.0, target_dy_m=0.0, target_direction='hold')


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
