# V11 Results: Directional No-op-Baseline Residual

## Run

Formal run: `artifacts/stage_i_v11_v10_comparable_01`.

- Scene profile: `directional_long`
- Held-out development scenes: 64
- Seed: 12
- Updates: 12
- Transitions per update: 128
- Grouped candidates: 8
- One-step rollout: yes
- Teacher: separate EMA LoRA, update rate `0.05`
- Distillation: JSD, `alpha=0.5`
- Same-state success contexts: 0

V10 uses the unchanged raw directory `sdpo_result_feedback_pilot_v7/artifacts/stage_i_v10_directional_control_03`. V11 uses the same scene profile, seed, rollout dimensions, candidate count, and condition names for the five shared conditions.

## V10/V11 Comparison

| Condition | V10 success | V11 success | V10 progress (m) | V11 progress (m) |
|---|---:|---:|---:|---:|
| `sdpo_result` | 11/64 (17.2%) | 11/64 (17.2%) | 0.02620 | 0.02644 |
| `sdpo_binary` | 13/64 (20.3%) | 13/64 (20.3%) | 0.02829 | 0.02829 |
| `sdpo_causal` | 0/64 (0.0%) | 0/64 (0.0%) | 0.00184 | 0.00054 |
| `sdpo_directional` | 6/64 (9.4%) | 7/64 (10.9%) | 0.01658 | 0.01794 |
| `sdpo_wrong_directional` | 9/64 (14.1%) | 8/64 (12.5%) | 0.02214 | 0.02010 |

V11-only residual controls:

| Condition | Success | Progress (m) | Mean absolute residual |
|---|---:|---:|---:|
| `sdpo_directional_residual` | 8/64 (12.5%) | 0.02022 | 0.67749 |
| `sdpo_wrong_directional_residual` | 8/64 (12.5%) | 0.02119 | 0.65310 |

## Diagnostics

- `feedback_only_audit.json`: passed. All seven conditions have identical first rollout hash, 0 solution contexts, 1,536 feedback-used rows, and 12 valid optimizer steps.
- `v11_audit.json`: passed. Every residual update stores privileged teacher, no-op baseline, residual, and normalized corrected target distributions.
- Residual target is active only for the two residual conditions; raw directional condition remains available as the V10-style control.
- V11 shared-condition results remain within the expected single-seed pilot variation of V10. Residual correction does not show deployment improvement in this run: 8/64 versus 6/64 for raw directional, while binary remains strongest at 13/64.

## Interpretation

V11 verifies the proposed no-op-baseline residual mechanism and makes its privileged directional contribution measurable. It does not establish a closed-loop success improvement. Residual magnitude is substantial, but teacher distribution change and deployment success remain non-monotonic; report as mechanism validation, not efficacy proof.

Primary machine-readable artifacts:

- `artifacts/stage_i_v11_v10_comparable_01/stage_i_v11_comparison.json`
- `artifacts/stage_i_v11_v10_comparable_01/feedback_only_audit.json`
- `artifacts/stage_i_v11_v10_comparable_01/v11_audit.json`
- `../sdpo_result_feedback_pilot_v7/artifacts/stage_i_v10_directional_control_03/stage_i_v10_comparison.json`

Qwen 7B rerun and paired 3B/7B, V10/V11 comparison: `V10_V11_7B_RESULTS.md`.
