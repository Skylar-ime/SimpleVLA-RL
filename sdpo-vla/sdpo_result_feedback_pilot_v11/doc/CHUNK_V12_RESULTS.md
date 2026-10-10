# V12 Autoregressive Chunk SDPO Results

## Method

- Chunk length `K=2`; each chunk is an autoregressive action sequence, not a `5^K` macro-action class.
- Teacher/student distributions are computed at each action position conditioned on the actual action prefix.
- Primary training budget: 64 chunks/update × 2 primitive actions = 128 primitive actions/update, matching V11's 128 transitions/update.
- 12 updates, 8 same-state candidates/group, seed 12, 64 development scenes, `directional_long`.
- Qwen3 uses microbatch 4; Molmo2 uses microbatch 1 because its multimodal memory footprint otherwise OOMs. Optimizer/data dimensions remain matched.
- Every run passed chunk audit: complete updates, correct action masks, normalized targets, and residual identity checks.

## Main results

| Condition | Qwen3 success | Qwen3 progress (m) | Molmo2 success | Molmo2 progress (m) |
| --- | ---: | ---: | ---: | ---: |
| `chunk_result` | 0/64 | 0.03930 | 3/64 | 0.03408 |
| `chunk_binary` | 0/64 | 0.03320 | 5/64 | 0.03653 |
| `chunk_causal` | 2/64 | 0.05039 | 5/64 | 0.03825 |
| `chunk_directional` | 6/64 | 0.04507 | 4/64 | 0.03514 |
| `chunk_wrong_directional` | 4/64 | 0.03824 | 4/64 | 0.03509 |
| `chunk_directional_residual` | 10/64 | 0.04563 | 8/64 | 0.04002 |
| `chunk_wrong_directional_residual` | 8/64 | 0.04313 | 5/64 | 0.03160 |

All main runs have 64 valid development scenes, zero invalid resets, zero safety failures, and 1,536 valid action-position updates (64 chunks × 2 positions × 12 updates) before safety masking.

## Chunk credit diagnostics

Training chunks frequently contain mixed or delayed outcomes. Across 768 chunks per run, Qwen3 mixed-outcome counts range 50–69 and delayed-progress counts 42–61; Molmo2 ranges are 35–40 and 30–35. Thus K=2 exposes real temporal credit ambiguity, but aggregate chunk feedback alone does not identify which position caused progress.

Per-position teacher diagnostics show position 0 often separates after-closer from after-other outcomes, while position 1 can remain nearly saturated or non-separating. This confirms chunking exposes the credit problem; it does not solve fine-grained credit assignment automatically.

## Same-state solution control

Separate `directional_local` control avoids mixing solution-context supervision with feedback-only results:

| Condition | Success | Mean progress (m) | Same-state solution contexts |
| --- | ---: | ---: | ---: |
| `chunk_directional` | 44/64 (68.75%) | 0.011717 | 0 |
| `chunk_solution` | 43/64 (67.19%) | 0.011631 | 532 |
| `chunk_solution_plus_directional` | 43/64 (67.19%) | 0.011579 | 525 |

These local runs show same-state successful chunk demonstrations are executable and audited, but do not improve over directional feedback in this single run. They are a supervision control, not part of the main feedback-only claim.

## Conclusions

1. The implementation follows SDPO's language-model mechanism: a chunk is an autoregressive response and loss is applied per action position under the realized prefix.
2. V12 validates the engineering path and makes temporal credit measurable. It does not establish that chunking alone improves credit assignment: Qwen3 residual directional reaches 10/64, below the prior V11 one-step residual result 12/64; Molmo2 reaches 8/64 versus prior V11 10/64.
3. No-op residual remains strongest among main chunk conditions for both backbones. It preserves useful directional information after subtracting the no-op teacher, but wrong-direction residual is weaker, especially for Molmo2.
4. Aggregate chunk feedback is insufficient for precise temporal credit. Next method should add suffix/return-to-go feedback or leave-one-action-out counterfactual diagnostics, while retaining per-position prefix-conditioned SDPO.
5. Do not claim general chunk efficacy. Current evidence supports: chunk-level SDPO is correct and auditable; it exposes credit ambiguity; residual feedback is more useful than raw aggregate feedback in this pilot.

## Artifacts

- Qwen3 main runs: `artifacts/chunk_v12_qwen3_01/`
- Molmo2 main runs: `artifacts/chunk_v12_molmo2_04/`
- Cross-backbone JSON: `artifacts/chunk_v12_qwen3_molmo2_comparison_01.json`
- Same-state control: `artifacts/chunk_v12_solution_qwen3_01/`
- Credit diagnostics: each run's `chunk_credit_analysis.json`
- Protocol: `protocol_chunk_v12.json`
- Tests: `35 passed`; compile checks passed.
