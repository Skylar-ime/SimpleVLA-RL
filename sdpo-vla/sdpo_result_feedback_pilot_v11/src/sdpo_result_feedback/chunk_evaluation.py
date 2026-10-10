from collections import Counter
import numpy as np

from .trainer import observation_transition
from .env_adapter import InvalidScene, PandaPushOutcomePilot


def evaluate_chunk(policy, scenes, chunk_length, max_decisions=12):
    episodes = []
    for scene in scenes:
        env = PandaPushOutcomePilot(max_decisions=max_decisions, seed=scene.seed)
        try:
            try:
                observation = env.reset(f'eval_chunk_{scene.scene_id}', scene)
            except InvalidScene as error:
                episodes.append({'scene_id': scene.scene_id, 'status': 'invalid_reset',
                                 'error': str(error)})
                continue
            initial_distance = env._distance()
            actions = []
            safety_failure = None
            while not env.done:
                transition = observation_transition(env, observation)
                chunk = policy.sample_chunk([transition], chunk_length, greedy=True)[0]
                for action in chunk:
                    actions.append(action)
                    observation, result = env.step(action)
                    if result.safety_failure is not None:
                        safety_failure = result.safety_failure
                    if env.done:
                        break
            final_distance = env._distance()
            episodes.append({
                'scene_id': scene.scene_id,
                'status': 'complete',
                'success': bool(final_distance < .02 and safety_failure is None),
                'progress_m': float(initial_distance - final_distance),
                'interactions': len(actions),
                'action_counts': dict(Counter(actions)),
                'safety_failure': safety_failure,
            })
        finally:
            env.close()
    valid = [episode for episode in episodes if episode['status'] == 'complete']
    successes = sum(episode['success'] for episode in valid)
    return {
        'episodes': episodes,
        'summary': {
            'scenes_requested': len(scenes),
            'scenes_valid': len(valid),
            'invalid_resets': sum(episode['status'] == 'invalid_reset' for episode in episodes),
            'successes': successes,
            'success_rate': successes / len(scenes) if scenes else 0.,
            'mean_progress_m': float(np.mean([episode['progress_m'] for episode in valid]))
            if valid else 0.,
            'safety_failures': sum(episode['safety_failure'] is not None for episode in valid),
            'interactions': sum(episode['interactions'] for episode in valid),
            'action_counts': dict(Counter(action for episode in valid
                                          for action in episode['action_counts']
                                          for _ in range(episode['action_counts'][action]))),
        },
    }
