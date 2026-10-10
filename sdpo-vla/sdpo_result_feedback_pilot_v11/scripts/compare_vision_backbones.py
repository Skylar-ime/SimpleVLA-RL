"""Create one comparable report for the two action policies and Molmo grounding."""
from pathlib import Path
import argparse
import json


def read_json(path):
    return json.loads(Path(path).read_text())


def action_row(name, root, evaluation_root):
    root = Path(root)
    audit = read_json(root / 'policy_audit.json')
    probe = read_json(root / 'probe' / 'report.json')['summary']
    evaluation = read_json(Path(evaluation_root) / 'evaluation.json')['summary']
    student = probe['groups']['no_feedback']
    return {
        'model': name,
        'model_id': audit['model_id'],
        'architecture': audit['architecture'],
        'model_type': audit['model_type'],
        'scenes': probe['scenes'],
        'one_step_oracle_accuracy': student['teacher_oracle_accuracy'],
        'one_step_action_counts': student['teacher_argmax_counts'],
        'one_step_mean_total_variation_vs_self': student['mean_total_variation'],
        'closed_loop_successes': evaluation['successes'],
        'closed_loop_success_rate': evaluation['success_rate'],
        'closed_loop_mean_progress_m': evaluation['mean_progress_m'],
        'closed_loop_safety_failures': evaluation['safety_failures'],
        'closed_loop_interactions': evaluation['interactions'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qwen3', type=Path, required=True)
    parser.add_argument('--qwen3-eval', type=Path, required=True)
    parser.add_argument('--internvl', type=Path, required=True)
    parser.add_argument('--internvl-eval', type=Path, required=True)
    parser.add_argument('--molmo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    action_models = [action_row('Qwen3-VL-8B-Instruct', args.qwen3, args.qwen3_eval),
                     action_row('InternVL3.5-8B-HF', args.internvl, args.internvl_eval)]
    molmo = read_json(args.molmo / 'report.json')
    report = {
        'protocol': {
            'scene_profile': 'directional_long',
            'dev_scenes': action_models[0]['scenes'],
            'action_probe': 'greedy argmax over A/B/C/D/E at initial state',
            'closed_loop': 'greedy action policy, max 12 decisions',
            'molmo_probe': 'independent image point grounding; no action labels',
        },
        'action_models': action_models,
        'molmo2_grounding': molmo['summary'],
        'limitations': molmo['limitations'] + [
            'Action-policy oracle accuracy and Molmo point accuracy are different tasks.',
            'This comparison does not claim Molmo can control the robot.',
            'Closed-loop results are development-scene results, not held-out test results.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'action_models': action_models,
                      'molmo2_grounding': molmo['summary']}), flush=True)


if __name__ == '__main__':
    main()
