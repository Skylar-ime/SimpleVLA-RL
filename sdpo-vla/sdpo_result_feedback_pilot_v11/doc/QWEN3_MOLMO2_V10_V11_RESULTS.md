# Qwen3/Molmo2 V10/V11 Results

## Protocol

- Backbones: local `Qwen3-VL-8B-Instruct` and `Molmo2-8B`.
- Scene profile: `directional_long`; same 64 development scenes, manifest SHA-256 `af0d0acb...`.
- Seed `12`; 12 updates; 128 transitions/update; 8 grouped candidates; one-step rollout; microbatch size 4.
- Teacher: EMA LoRA, update rate `0.05`; distillation: JSD, `alpha=0.5`.
- V10 conditions: result, binary, causal, directional, wrong-directional.
- V11 adds same-state no-op residual conditions for directional and wrong-directional feedback.
- Single repeat seed. Results are pilot evidence, not population estimates.

## V10

| Condition | Qwen3 success | Molmo2 success | Qwen3 progress (m) | Molmo2 progress (m) |
| --- | ---: | ---: | ---: | ---: |
| `sdpo_result` | 9/64 (14.1%) | 5/64 (7.8%) | 0.03750 | 0.02539 |
| `sdpo_binary` | 10/64 (15.6%) | 9/64 (14.1%) | 0.03485 | 0.02986 |
| `sdpo_causal` | 8/64 (12.5%) | 13/64 (20.3%) | 0.04003 | 0.03723 |
| `sdpo_directional` | 9/64 (14.1%) | 13/64 (20.3%) | 0.03943 | 0.03677 |
| `sdpo_wrong_directional` | 9/64 (14.1%) | 11/64 (17.2%) | 0.04611 | 0.03065 |

## V11

| Condition | Qwen3 success | Molmo2 success | Qwen3 progress (m) | Molmo2 progress (m) |
| --- | ---: | ---: | ---: | ---: |
| `sdpo_result` | 8/64 (12.5%) | 5/64 (7.8%) | 0.03649 | 0.02539 |
| `sdpo_binary` | 8/64 (12.5%) | 9/64 (14.1%) | 0.03289 | 0.02986 |
| `sdpo_causal` | 8/64 (12.5%) | 13/64 (20.3%) | 0.04056 | 0.03723 |
| `sdpo_directional` | 10/64 (15.6%) | 13/64 (20.3%) | 0.04143 | 0.03677 |
| `sdpo_wrong_directional` | 8/64 (12.5%) | 11/64 (17.2%) | 0.04666 | 0.03065 |
| `sdpo_directional_residual` | 12/64 (18.8%) | 10/64 (15.6%) | 0.05294 | 0.03098 |
| `sdpo_wrong_directional_residual` | 9/64 (14.1%) | 10/64 (15.6%) | 0.04509 | 0.02930 |

## Conclusions

1. Backbone effect is condition-dependent. Molmo2 obtains more binary/causal/directional successes in this single run, but Qwen3 has higher mean progress in every matched V10/V11 condition and no safety failures. Molmo2 reports 2 safety failures in every condition.
2. V11 residual correction helps Qwen3 directional feedback: `10 -> 12` successes and `0.04143 -> 0.05294 m` progress. Wrong-direction residual changes `8 -> 9` but progress slightly decreases.
3. V11 residual correction does not help Molmo2: directional `13 -> 10`, progress `0.03677 -> 0.03098 m`; wrong-directional `11 -> 10`, progress `0.03065 -> 0.02930 m`.
4. Directional feedback is not sufficient by itself. Wrong-direction controls remain competitive, especially for Molmo2. Report V11 as mechanism validation, not proof of general efficacy.
5. Practical choice for this pilot: Qwen3 is safer and more stable for action deployment; Molmo2 remains viable for selected feedback conditions but needs action-prior, safety, and multi-seed investigation.

## Audit and artifacts

- All four runs: 64 valid scenes/condition, 12 valid optimizer updates, no invalid resets.
- Qwen3 V10/V11 and Molmo2 V10/V11 feedback-only audits: passed.
- Molmo2 V11 residual audit: passed; all residual updates contain privileged teacher, no-op baseline, residual, and corrected target distributions.
- First rollout hashes match across conditions; all four runs share scene manifest and experiment dimensions.
- Machine-readable comparison: `artifacts/qwen3_molmo2_v10_v11_comparison_01.json`.
- Run roots: `artifacts/stage_i_v10_qwen3_01`, `artifacts/stage_i_v11_qwen3_01`, `artifacts/stage_i_v10_molmo2_01_retry`, `artifacts/stage_i_v11_molmo2_01_retry`.
