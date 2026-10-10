from pathlib import Path
import argparse
import json

from sdpo_result_feedback.env_adapter import InvalidScene, PandaPushOutcomePilot
from sdpo_result_feedback.scenes import create_directional_scene_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    split_path = root / 'artifacts/directional_scenes_v2.json'
    split = create_directional_scene_splits(split_path)['dev'][:args.scenes]
    env = PandaPushOutcomePilot(seed=args.seed, max_decisions=12)
    report = {'requested_scenes': len(split), 'valid_scenes': 0, 'invalid_resets': 0,
              'action_probe_count': 0, 'action_counts': {}, 'contact_counts': {},
              'motion_counts': {}, 'progress_counts': {}}
    try:
        for scene in split:
            try:
                env.reset(f'physics_{scene.scene_id}', scene)
            except InvalidScene:
                report['invalid_resets'] += 1
                continue
            report['valid_scenes'] += 1
            snapshot = env.snapshot()
            for action in ('A', 'B', 'C', 'D', 'E'):
                env.restore(snapshot)
                _, transition = env.step(action)
                report['action_probe_count'] += 1
                report['action_counts'][action] = report['action_counts'].get(action, 0) + 1
                report['contact_counts'][action] = report['contact_counts'].get(action, 0) + int(transition.result.contact)
                report['motion_counts'][action] = report['motion_counts'].get(action, 0) + int(transition.result.object_motion == 'detected')
                bucket = report['progress_counts'].setdefault(action, {})
                bucket[transition.result.goal_progress] = bucket.get(transition.result.goal_progress, 0) + 1
            env.remove_snapshot(snapshot)
    finally:
        env.close()
    report['action_contact_rate'] = {k: report['contact_counts'][k] / report['action_counts'][k]
                                     for k in report['action_counts']}
    report['action_motion_rate'] = {k: report['motion_counts'][k] / report['action_counts'][k]
                                    for k in report['action_counts']}
    report['status'] = 'pass' if report['valid_scenes'] >= .95 * len(split) and any(report['contact_counts'].values()) else 'fail'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, sort_keys=True))
    if report['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
