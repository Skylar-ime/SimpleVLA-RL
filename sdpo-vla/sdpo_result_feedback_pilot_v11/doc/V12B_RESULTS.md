# V12b: Same-State Action-Relative Advantage, Five-Seed Results

This report summarizes the audited `protocol_v12b.json` development runs. The V12b teacher sees only five action-relative advantages; raw progress, contact, and safety outcomes remain audit data and are not in its prompt. The original V12 reversed control leaked the unmodified progress table and is not used as evidence here.

Each backbone has five seeds (12–16), four conditions, 12 updates per run, 128 one-step transitions per update, and 64 paired development scenes per seed. Confidence intervals below use a two-sided t interval over five seed means or paired seed differences, not 320 independent scenes. Teacher best-action accuracy is an in-training diagnostic on states whose counterfactual action-progress range exceeds 0.002 m; it is not a held-out performance metric.

## QWEN3

| Condition | Successes / 64 by seed | Mean successes / 64 | Mean progress (m) | Mean safety failures | Teacher best-action accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `sdpo_result` | 9, 7, 10, 9, 9 | 8.80 | 0.03817 | 0.00 | 0.825 |
| `sdpo_advantage` | 10, 7, 10, 10, 10 | 9.40 | 0.04296 | 0.00 | 0.991 |
| `sdpo_advantage_shuffled` | 9, 8, 8, 8, 7 | 8.00 | 0.03473 | 0.00 | 0.410 |
| `sdpo_advantage_reversed` | 9, 10, 9, 8, 10 | 9.20 | 0.03786 | 0.00 | 0.049 |

Paired differences are `sdpo_advantage` minus comparator. Positive progress and success differences favor action-relative advantage.

| Comparator | Success difference / 64 (95% CI) | Progress difference in m (95% CI) | Safety difference / 64 (95% CI) |
| --- | ---: | ---: | ---: |
| `sdpo_result` | +0.60 [-0.08, 1.28] | +0.00480 [0.00037, 0.00923] | +0.00 [0.00, 0.00] |
| `sdpo_advantage_shuffled` | +1.40 [-0.48, 3.28] | +0.00823 [0.00298, 0.01349] | +0.00 [0.00, 0.00] |
| `sdpo_advantage_reversed` | +0.20 [-2.19, 2.59] | +0.00510 [-0.00141, 0.01162] | +0.00 [0.00, 0.00] |

First-update teacher total variation versus `sdpo_advantage` on identical initial RGB, proprioception, scene, and sampled action:
- `sdpo_result`: 0.585 [0.394, 0.777]
- `sdpo_advantage_shuffled`: 0.630 [0.529, 0.730]
- `sdpo_advantage_reversed`: 0.935 [0.880, 0.990]

Action counts across five development runs:
- `sdpo_result`: A=184, B=299, C=1863, D=1209, E=0
- `sdpo_advantage`: A=509, B=3, C=1775, D=1249, E=0
- `sdpo_advantage_shuffled`: A=209, B=9, C=2246, D=1105, E=0
- `sdpo_advantage_reversed`: A=595, B=8, C=2299, D=639, E=0

## MOLMO2

| Condition | Successes / 64 by seed | Mean successes / 64 | Mean progress (m) | Mean safety failures | Teacher best-action accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| `sdpo_result` | 5, 4, 4, 5, 5 | 4.60 | 0.02471 | 2.00 | 0.393 |
| `sdpo_advantage` | 15, 13, 13, 15, 15 | 14.20 | 0.03481 | 1.80 | 0.987 |
| `sdpo_advantage_shuffled` | 15, 12, 12, 14, 15 | 13.60 | 0.03243 | 2.00 | 0.408 |
| `sdpo_advantage_reversed` | 10, 10, 10, 9, 12 | 10.20 | 0.03020 | 2.00 | 0.046 |

Paired differences are `sdpo_advantage` minus comparator. Positive progress and success differences favor action-relative advantage.

| Comparator | Success difference / 64 (95% CI) | Progress difference in m (95% CI) | Safety difference / 64 (95% CI) |
| --- | ---: | ---: | ---: |
| `sdpo_result` | +9.60 [8.92, 10.28] | +0.01010 [0.00736, 0.01284] | -0.20 [-0.76, 0.36] |
| `sdpo_advantage_shuffled` | +0.60 [-0.08, 1.28] | +0.00238 [-0.00005, 0.00481] | -0.20 [-0.76, 0.36] |
| `sdpo_advantage_reversed` | +4.00 [2.24, 5.76] | +0.00461 [0.00029, 0.00892] | -0.20 [-0.76, 0.36] |

First-update teacher total variation versus `sdpo_advantage` on identical initial RGB, proprioception, scene, and sampled action:
- `sdpo_result`: 0.628 [0.593, 0.663]
- `sdpo_advantage_shuffled`: 0.569 [0.436, 0.702]
- `sdpo_advantage_reversed`: 0.869 [0.814, 0.924]

Action counts across five development runs:
- `sdpo_result`: A=0, B=40, C=2658, D=185, E=806
- `sdpo_advantage`: A=0, B=30, C=2892, D=397, E=44
- `sdpo_advantage_shuffled`: A=0, B=4, C=2910, D=435, E=33
- `sdpo_advantage_reversed`: A=0, B=77, C=3034, D=302, E=74

## Conclusion

The leakage-free advantage table gives a strongly aligned teacher on informative training states for both backbones (Qwen3 0.991; Molmo2 0.987). Reversing the table drops this alignment to 0.049 and 0.046, respectively. This verifies that the teacher responds to the advantage sign after raw outcome leakage is removed.

Closed-loop evidence is mixed. Qwen3 advantage improves mean progress over result and shuffled feedback, but its five-seed success difference versus all controls has a 95% interval crossing zero; progress versus reversed also crosses zero ([-0.00141, 0.01162] m). Molmo2 advantage improves over result feedback and the reversed-sign control in both successes and progress: versus reversed, success difference is +4.00/64 ([2.24, 5.76]) and progress difference is +0.00461 m ([0.00029, 0.00892] m). Versus shuffled, Molmo2 success and progress intervals still cross zero ([-0.08, 1.28] successes; [-0.00005, 0.00481] m). Shuffling also changes within-state table consistency, so this control cannot isolate state-table correspondence alone.

Therefore V12b supports an identifiable one-step action-relative teacher and a Molmo2 closed-loop benefit over the reversed-sign control, not a backbone-independent success gain over both controls. Action coverage is narrow (Qwen3 never selects E; Molmo2 never selects A in these runs), and no held-out test or longer-horizon counterfactual credit assignment was evaluated.

## Interpretation Limits

This is a local calibrated PandaPush pilot with five discrete actions and one-step counterfactual credit. Simulator-measured progress is privileged feedback, not an expert action label. The 64 development scenes are reused across seeds, so the seed is the unit of uncertainty. These results do not establish cross-task or real-robot effectiveness. Inspect the paired control intervals and safety counts before claiming any benefit.

Machine-readable statistics: `artifacts/v12b_qwen3_molmo2_multiseed_comparison_01.json`.
