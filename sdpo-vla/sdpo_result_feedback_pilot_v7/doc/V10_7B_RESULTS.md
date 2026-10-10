# V10 Qwen 7B Rerun

The 7B rerun is stored in `artifacts/stage_i_v10_qwen25vl7b_01`. Original 3B V10 results remain in `artifacts/stage_i_v10_directional_control_03`.

Both use seed 12, the same `directional_long` scenes, 64 dev scenes, 12 updates, 128 transitions/update, 8 same-state candidates, one-step rollouts, JSD and EMA rate `0.05`. The 7B run uses the same microbatch size 4. Its checkpoint revision is `cc594898137f460bfe9f0759e9844b3ce807cfb5`.

| Condition | 3B success | 7B success | 7B mean progress (m) |
|---|---:|---:|---:|
| `sdpo_result` | 11/64 | 2/64 | 0.02069 |
| `sdpo_binary` | 13/64 | 0/64 | 0.00077 |
| `sdpo_causal` | 0/64 | 0/64 | 0.00077 |
| `sdpo_directional` | 6/64 | 2/64 | 0.00967 |
| `sdpo_wrong_directional` | 9/64 | 2/64 | 0.00574 |

The feedback-only audit passed. All five 7B conditions share the same first rollout hash and contain 1,536 feedback-used rows, zero solution contexts, and 12 optimizer steps. 7B deployment mostly selected `E`: 746/768 actions for binary and 753/768 for causal. Larger backbone alone did not improve this V10 protocol.

Artifacts: `artifacts/stage_i_v10_qwen25vl7b_01/stage_i_v10_comparison.json` and `artifacts/stage_i_v10_qwen25vl7b_01/feedback_only_audit.json`. Cross-version paired analysis: `../../sdpo_result_feedback_pilot_v11/doc/V10_V11_7B_RESULTS.md`.
