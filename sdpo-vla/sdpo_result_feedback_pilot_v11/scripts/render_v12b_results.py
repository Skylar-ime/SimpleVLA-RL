from pathlib import Path
import argparse
import json


CONDITIONS = (
    'sdpo_result', 'sdpo_advantage',
    'sdpo_advantage_shuffled', 'sdpo_advantage_reversed',
)
BACKBONES = ('qwen3', 'molmo2')


def interval(stat, digits):
    bounds = stat['ci95']
    if bounds is None:
        return 'not available'
    return f"[{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


def render(report):
    if (report['protocol'] != 'protocol_v12b.json'
            or report['seeds'] != [12, 13, 14, 15, 16]
            or tuple(report['conditions']) != CONDITIONS):
        raise ValueError('Not the complete v12b protocol comparison')
    lines = [
        '# V12b: Same-State Action-Relative Advantage, Five-Seed Results',
        '',
        'This report summarizes the audited `protocol_v12b.json` development runs. '
        'The V12b teacher sees only five action-relative advantages; raw progress, '
        'contact, and safety outcomes remain audit data and are not in its prompt. '
        'The original V12 reversed control leaked the unmodified progress table and '
        'is not used as evidence here.',
        '',
        'Each backbone has five seeds (12–16), four conditions, 12 updates per run, '
        '128 one-step transitions per update, and 64 paired development scenes per '
        'seed. Confidence intervals below use a two-sided t interval over five seed '
        'means or paired seed differences, not 320 independent scenes. Teacher '
        'best-action accuracy is an in-training diagnostic on states whose '
        'counterfactual action-progress range exceeds 0.002 m; it is not a '
        'held-out performance metric.',
        '',
    ]
    for backbone in BACKBONES:
        data = report[backbone]
        lines.extend([
            f'## {backbone.upper()}',
            '',
            '| Condition | Successes / 64 by seed | Mean successes / 64 | Mean progress (m) | '
            'Mean safety failures | Teacher best-action accuracy |',
            '| --- | --- | ---: | ---: | ---: | ---: |',
        ])
        for condition in CONDITIONS:
            row = data[condition]
            successes = ', '.join(str(int(value)) for value in row['successes']['values'])
            lines.append(
                f"| `{condition}` | {successes} | {row['successes']['mean']:.2f} | "
                f"{row['mean_progress_m']['mean']:.5f} | "
                f"{row['safety_failures']['mean']:.2f} | "
                f"{row['teacher_alignment']['true_best_action_accuracy_by_seed']['mean']:.3f} |"
            )
        lines.extend([
            '',
            'Paired differences are `sdpo_advantage` minus comparator. Positive progress '
            'and success differences favor action-relative advantage.',
            '',
            '| Comparator | Success difference / 64 (95% CI) | '
            'Progress difference in m (95% CI) | Safety difference / 64 (95% CI) |',
            '| --- | ---: | ---: | ---: |',
        ])
        comparisons = {
            'sdpo_result': data['paired_vs_result']['sdpo_advantage'],
            **data['advantage_vs_controls'],
        }
        for comparator, difference in comparisons.items():
            success = difference['success_count_delta_by_seed']
            progress = difference['mean_progress_delta_m_by_seed']
            safety = difference['safety_failure_delta_by_seed']
            lines.append(
                f"| `{comparator}` | {success['mean']:+.2f} {interval(success, 2)} | "
                f"{progress['mean']:+.5f} {interval(progress, 5)} | "
                f"{safety['mean']:+.2f} {interval(safety, 2)} |"
            )
        lines.extend(['', 'First-update teacher total variation versus `sdpo_advantage` '
                      'on identical initial RGB, proprioception, scene, and sampled action:'])
        for comparator, statistic in data['first_update_teacher_tv_vs_advantage'].items():
            lines.append(f"- `{comparator}`: {statistic['mean']:.3f} {interval(statistic, 3)}")
        lines.append('')
        lines.append('Action counts across five development runs:')
        for condition in CONDITIONS:
            counts = data[condition]['action_counts']
            lines.append(f"- `{condition}`: " + ', '.join(
                f'{action}={counts[action]}' for action in 'ABCDE'))
        lines.append('')
    qwen = report['qwen3']
    molmo = report['molmo2']
    qwen_reversed = qwen['advantage_vs_controls']['sdpo_advantage_reversed']
    molmo_reversed = molmo['advantage_vs_controls']['sdpo_advantage_reversed']
    molmo_shuffled = molmo['advantage_vs_controls']['sdpo_advantage_shuffled']
    lines.extend([
        '## Conclusion',
        '',
        'The leakage-free advantage table gives a strongly aligned teacher on '
        'informative training states for both backbones (Qwen3 '
        f"{qwen['sdpo_advantage']['teacher_alignment']['true_best_action_accuracy_by_seed']['mean']:.3f}; "
        'Molmo2 '
        f"{molmo['sdpo_advantage']['teacher_alignment']['true_best_action_accuracy_by_seed']['mean']:.3f}). "
        'Reversing the table drops this alignment to '
        f"{qwen['sdpo_advantage_reversed']['teacher_alignment']['true_best_action_accuracy_by_seed']['mean']:.3f} "
        'and '
        f"{molmo['sdpo_advantage_reversed']['teacher_alignment']['true_best_action_accuracy_by_seed']['mean']:.3f}, "
        'respectively. This verifies that the teacher responds to the advantage sign '
        'after raw outcome leakage is removed.',
        '',
        'Closed-loop evidence is mixed. Qwen3 advantage improves mean progress over '
        'result and shuffled feedback, but its five-seed success difference versus '
        'all controls has a 95% interval crossing zero; progress versus reversed '
        f"also crosses zero ({interval(qwen_reversed['mean_progress_delta_m_by_seed'], 5)} m). "
        'Molmo2 advantage improves over result feedback and the reversed-sign control '
        'in both successes and progress: versus reversed, success difference is '
        f"{molmo_reversed['success_count_delta_by_seed']['mean']:+.2f}/64 "
        f"({interval(molmo_reversed['success_count_delta_by_seed'], 2)}) and progress difference "
        f"is {molmo_reversed['mean_progress_delta_m_by_seed']['mean']:+.5f} m "
        f"({interval(molmo_reversed['mean_progress_delta_m_by_seed'], 5)} m). "
        'Versus shuffled, Molmo2 success and progress intervals still cross zero '
        f"({interval(molmo_shuffled['success_count_delta_by_seed'], 2)} successes; "
        f"{interval(molmo_shuffled['mean_progress_delta_m_by_seed'], 5)} m). "
        'Shuffling also changes within-state table consistency, so this control '
        'cannot isolate state-table correspondence alone.',
        '',
        'Therefore V12b supports an identifiable one-step action-relative teacher '
        'and a Molmo2 closed-loop benefit over the reversed-sign control, not a '
        'backbone-independent success gain over both controls. Action coverage is '
        'narrow (Qwen3 never selects E; Molmo2 never selects A in these runs), '
        'and no held-out test or longer-horizon counterfactual credit assignment '
        'was evaluated.',
        '',
    ])
    lines.extend([
        '## Interpretation Limits',
        '',
        'This is a local calibrated PandaPush pilot with five discrete actions and '
        'one-step counterfactual credit. Simulator-measured progress is privileged '
        'feedback, not an expert action label. The 64 development scenes are reused '
        'across seeds, so the seed is the unit of uncertainty. These results do not '
        'establish cross-task or real-robot effectiveness. Inspect the paired control '
        'intervals and safety counts before claiming any benefit.',
        '',
        'Machine-readable statistics: '
        '`artifacts/v12b_qwen3_molmo2_multiseed_comparison_01.json`.',
        '',
    ])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text())
    args.output.write_text(render(report))
    print(f'Results written: {args.output}')


if __name__ == '__main__':
    main()
