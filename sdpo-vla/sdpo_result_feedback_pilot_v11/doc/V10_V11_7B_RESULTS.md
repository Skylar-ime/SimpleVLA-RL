# V10/V11 7B Results

## Protocol

V10 and V11 rerun with local `Qwen2.5-VL-7B-Instruct`. Existing 3B artifacts remain unchanged.

- Same `directional_long` scene manifest: SHA-256 `af0d0acb...`
- Same seed: 12
- Same 64 dev scenes, 12 updates, 128 transitions/update
- Same 8 grouped candidates, one-step rollout, JSD, EMA rate `0.05`
- Same microbatch size: 4
- 7B revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- 7B trainable LoRA parameters: 5,046,272

V10 run: `sdpo_result_feedback_pilot_v7/artifacts/stage_i_v10_qwen25vl7b_01`  
V11 run: `sdpo_result_feedback_pilot_v11/artifacts/stage_i_v11_qwen25vl7b_01`

## 7B V10/V11

| Condition | V10 success | V11 success | V10 progress (m) | V11 progress (m) |
|---|---:|---:|---:|---:|
| `sdpo_result` | 2/64 (3.1%) | 2/64 (3.1%) | 0.02069 | 0.01951 |
| `sdpo_binary` | 0/64 (0.0%) | 0/64 (0.0%) | 0.00077 | 0.00077 |
| `sdpo_causal` | 0/64 (0.0%) | 0/64 (0.0%) | 0.00077 | 0.00000 |
| `sdpo_directional` | 2/64 (3.1%) | 2/64 (3.1%) | 0.00967 | 0.00872 |
| `sdpo_wrong_directional` | 2/64 (3.1%) | 2/64 (3.1%) | 0.00574 | 0.00574 |
| `sdpo_directional_residual` | not applicable | 2/64 (3.1%) | not applicable | 0.01920 |
| `sdpo_wrong_directional_residual` | not applicable | 2/64 (3.1%) | not applicable | 0.01572 |

## Backbone Comparison

| Version/condition | 3B | 7B |
|---|---:|---:|
| V10 `sdpo_result` | 11/64 | 2/64 |
| V10 `sdpo_binary` | 13/64 | 0/64 |
| V10 `sdpo_causal` | 0/64 | 0/64 |
| V10 `sdpo_directional` | 6/64 | 2/64 |
| V10 `sdpo_wrong_directional` | 9/64 | 2/64 |
| V11 `sdpo_result` | 11/64 | 2/64 |
| V11 `sdpo_binary` | 13/64 | 0/64 |
| V11 `sdpo_causal` | 0/64 | 0/64 |
| V11 `sdpo_directional` | 7/64 | 2/64 |
| V11 `sdpo_directional_residual` | 8/64 | 2/64 |
| V11 `sdpo_wrong_directional` | 8/64 | 2/64 |
| V11 `sdpo_wrong_directional_residual` | 8/64 | 2/64 |

## Validation

- V10 7B feedback-only audit: passed.
- V11 7B feedback-only audit: passed.
- V11 7B residual audit: passed for both residual conditions.
- Cross-condition first rollout hashes identical within each run.
- Shared V10/V11 first rollout hashes identical for each backbone.
- Machine-readable paired comparison: `artifacts/v10_v11_backbone_comparison_01.json`.
- Tests: V10 `30 passed`; V11 `32 passed`.

## Conclusion

Changing only backbone from 3B to 7B does not improve this SDPO pilot. 7B collapses toward `E` in binary/causal deployment and remains far below 3B on result and directional conditions. V11 residual runs are executable and auditable, but do not recover 7B closed-loop performance. Conclusions: backbone capacity alone is insufficient; prompt grounding, action prior, and long-horizon execution remain dominant constraints.
