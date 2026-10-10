from collections import defaultdict

import numpy as np

from .feedback import (binary_feedback, causal_feedback, directional_feedback,
                       counterfactual_advantage_feedback, neutral_feedback,
                       wrong_directional_feedback)


CONDITIONS = ('sdpo_result', 'sdpo_binary', 'sdpo_shuffled', 'sdpo_neutral',
              'sdpo_causal', 'sdpo_directional', 'sdpo_wrong_directional',
              'sdpo_directional_residual', 'sdpo_wrong_directional_residual',
              'sdpo_reflection', 'sdpo_advantage', 'sdpo_advantage_shuffled',
              'sdpo_advantage_reversed')


def build_sdpo_context(transitions, condition, seed=11):
    """Build the original SDPO sample mask and same-state solution context."""
    if condition not in CONDITIONS or not transitions:
        raise ValueError('Unknown condition or empty batch')
    groups = defaultdict(list)
    for index, transition in enumerate(transitions):
        if transition.rollout_group_id is None:
            raise ValueError('Same-state group identity is required')
        groups[transition.rollout_group_id].append(index)
    for members in groups.values():
        original = transitions[members[0]]
        for index in members[1:]:
            candidate = transitions[index]
            if (candidate.scene_id != original.scene_id
                    or candidate.decision_index != original.decision_index
                    or candidate.instruction != original.instruction
                    or not np.array_equal(candidate.image, original.image)
                    or not np.array_equal(candidate.proprio, original.proprio)):
                raise ValueError('Candidate group contains different original states')

    successes = {
        group: [index for index in members
                if transitions[index].safety_failure is None and transitions[index].result.success]
        for group, members in groups.items()
    }
    solutions, solution_donors = [], []
    for index, transition in enumerate(transitions):
        donor = next((other for other in successes[transition.rollout_group_id] if other != index), None)
        solution_donors.append(donor)
        solutions.append(None if donor is None else transitions[donor].action_label)

    records = [transition.result for transition in transitions]
    donors = list(range(len(transitions)))
    if condition == 'sdpo_binary':
        records = [binary_feedback(record) for record in records]
    elif condition == 'sdpo_neutral':
        records = [neutral_feedback() for _ in records]
    elif condition == 'sdpo_causal':
        records = [causal_feedback(transition.action_label, transition.result)
                   for transition in transitions]
    elif condition in {'sdpo_directional', 'sdpo_directional_residual'}:
        records = [directional_feedback(transition.action_label, transition)
                   for transition in transitions]
    elif condition in {'sdpo_wrong_directional', 'sdpo_wrong_directional_residual'}:
        records = [wrong_directional_feedback(transition.action_label, transition)
                   for transition in transitions]
    elif condition == 'sdpo_advantage':
        records = [counterfactual_advantage_feedback(transition)
                   for transition in transitions]
    elif condition == 'sdpo_advantage_reversed':
        records = [counterfactual_advantage_feedback(transition, sign=-1.0)
                   for transition in transitions]
    elif condition == 'sdpo_advantage_shuffled':
        rng = np.random.default_rng(seed)
        donors = [None] * len(transitions)
        for index, transition in enumerate(transitions):
            candidates = [other for other, candidate in enumerate(transitions)
                          if candidate.rollout_group_id != transition.rollout_group_id
                          and candidate.action_label == transition.action_label]
            if not candidates:
                candidates = [other for other, candidate in enumerate(transitions)
                              if candidate.rollout_group_id != transition.rollout_group_id]
            donors[index] = int(rng.choice(candidates)) if candidates else None
        records = [counterfactual_advantage_feedback(
            transition,
            counterfactual_source=(transitions[donor].measurements if donor is not None else None))
                   for transition, donor in zip(transitions, donors)]
    elif condition == 'sdpo_shuffled':
        rng = np.random.default_rng(seed)
        strata = defaultdict(list)
        for index, transition in enumerate(transitions):
            if transition.safety_failure is None:
                strata[(transition.action_label, transition.result.success)].append(index)
        donors = [None] * len(transitions)
        for members in strata.values():
            for index in members:
                candidates = [other for other in members
                              if transitions[other].rollout_group_id
                              != transitions[index].rollout_group_id]
                if candidates:
                    donors[index] = int(rng.choice(candidates))
        records = [transition.result if donor is None else transitions[donor].result
                   for transition, donor in zip(transitions, donors)]

    include_feedback = [solution is None for solution in solutions]
    valid = np.asarray([transition.safety_failure is None for transition in transitions], dtype=bool)
    return {
        'condition': condition,
        'records': records,
        'solution_actions': solutions,
        'solution_donor_indices': solution_donors,
        'include_outcome_feedback': include_feedback,
        'valid_mask': valid,
        'donor_indices': donors,
        'group_count': len(groups),
        'witness_count': sum(len(indices) for indices in successes.values()),
        'groups_with_witness': sum(bool(indices) for indices in successes.values()),
        'solution_context_count': sum(solution is not None for solution in solutions),
        'subtract_noop_baseline': condition in {
            'sdpo_directional_residual', 'sdpo_wrong_directional_residual'
        },
    }
