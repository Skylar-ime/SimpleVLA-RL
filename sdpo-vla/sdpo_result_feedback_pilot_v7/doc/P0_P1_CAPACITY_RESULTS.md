# P0/P1 and Capacity Comparison

## Protocol

- Environment: frozen `directional_long` PandaPush scenes, 64 held-out scenes, greedy closed loop, 12 decisions.
- Base model: local `Qwen2.5-VL-3B-Instruct`, language-decoder LoRA, batch size 8.
- Capacity model: local `Qwen2.5-VL-7B-Instruct`, same policy code and protocol, batch size 1 to fit the RTX 4090 49 GiB GPU.
- All results use the same held-out scene set and simulator reset protocol.
- Results remain single-seed (`seed=11`); they are controlled diagnostics, not final benchmark estimates.

## P0: Frozen Direction Control

The missing causal baseline is now measured:

| Condition | Closed-loop success | Mean progress |
| --- | ---: | ---: |
| Qwen 3B, frozen RGB | 11/64 (17.2%) | 0.02561 m |
| Qwen 3B, frozen target direction | 42/64 (65.6%) | 0.09135 m |
| Qwen 3B, oracle-supervised target direction | 43/64 (67.2%) | 0.09613 m |

Training with the original simulator one-step oracle label adds only one successful scene: `42/64` to `43/64`. The earlier `67.2%` result therefore comes almost entirely from privileged target-direction input, not from LoRA training.

Artifact: `artifacts/fast_oracle_validation_v3/p0_frozen_target_direction_closed_loop.json`.

## P1: Clean Direction Labels

The target-direction condition was retrained with the deterministic `direction_action` label instead of the state-dependent one-step `oracle_action` label.

- Held-out one-step direction accuracy: `71.9% -> 100.0%`.
- Held-out oracle-label accuracy: `71.9% -> 65.6%`.
- Closed-loop success: `42/64 -> 57/64` (`65.6% -> 89.1%`).
- Mean progress: `0.09135 m -> 0.12802 m`.
- `-x` first-action accuracy becomes `18/18` with `B`.
- Per-direction success: `+x 13/14`, `+y 16/18`, `-y 14/14`, `-x 14/18`.

This isolates a major label-design problem. The original oracle label is not a clean target-direction label; it often selects a contact or progress action that differs from the dominant goal direction. Exact oracle accuracy decreases while behavior improves.

Artifact: `artifacts/fast_oracle_validation_v3/p1_direction_supervision/`.

## P1: Intermediate-State Data

An additional trajectory experiment collected 503 train and 469 held-out expert intermediate states, using repeated `direction_action` labels.

- Held-out intermediate-state direction accuracy after training: `100.0%`.
- Closed-loop success: `57/64` (`89.1%`).
- Result matches initial-state clean-label training exactly at this scene scale.

Interpretation: first-action direction recognition is no longer the main bottleneck after clean supervision. Remaining failures come from contact geometry, dynamic replanning, and task execution. More repeated direction examples alone do not improve the current policy.

Artifact: `artifacts/fast_oracle_validation_v3/p1_trajectory_direction/`.

For completeness, clean direction supervision on Qwen 3B does not fix RGB grounding:

- Held-out one-step RGB direction accuracy remains `21.9%`.
- Closed-loop RGB success changes from `11/64` to `8/64`.

Artifact: `artifacts/fast_oracle_validation_v3/p1_direction_supervision/closed_loop_rgb_only.json`.

## Capacity Comparison

### Frozen model

| Model | RGB closed loop | Target-direction closed loop | Frozen RGB action pattern |
| --- | ---: | ---: | --- |
| Qwen 3B | 11/64 (17.2%) | 42/64 (65.6%) | mostly `A` |
| Qwen 7B | 0/64 (0.0%) | 57/64 (89.1%) | `E` for all 64 scenes |

Capacity alone does not solve RGB grounding. In this prompt/action setup, the larger model collapses to `E` when no direction hint is provided. With privileged direction, the larger model maps all four directions correctly at the one-step level and reaches `57/64` closed-loop success.

### Same clean direction supervision

| Model | Held-out one-step RGB direction accuracy | RGB closed-loop success | Target-direction closed-loop success |
| --- | ---: | ---: | ---: |
| Qwen 3B | 21.9% | 8/64 (12.5%) | 57/64 (89.1%) |
| Qwen 7B | 100.0% | 25/64 (39.1%) | 57/64 (89.1%) |

The 7B model can learn visual direction from clean labels where the 3B model cannot. However, `25/64` closed-loop success despite `100%` initial direction accuracy shows sequence execution remains unsolved.

Artifacts: `artifacts/fast_oracle_validation_v3/capacity_qwen25vl7b_direction_supervision/` and `artifacts/fast_oracle_validation_v3/capacity_qwen25vl7b_frozen_probe/`.

### Same original oracle supervision

| Model | RGB closed-loop success | Target-direction closed-loop success |
| --- | ---: | ---: |
| Qwen 3B | 16/64 (25.0%) | 43/64 (67.2%) |
| Qwen 7B | 10/64 (15.6%) | 44/64 (68.8%) |

The larger backbone does not rescue the mismatched oracle objective. Clean label quality matters more than parameter count for this diagnostic.

Artifact: `artifacts/fast_oracle_validation_v3/capacity_qwen25vl7b_oracle_supervision/`.

## Conclusions

1. P0 confirms that the previous `67.2%` target-direction result was primarily an input-information effect, not an oracle-supervised learning effect.
2. P1 confirms that `oracle_action` is a poor training target for a dominant-direction diagnostic. Replacing it with `direction_action` raises Qwen 3B target-direction success to `89.1%`.
3. Intermediate-state direction training reaches perfect held-out direction classification but does not improve closed-loop success beyond `89.1%`. Execution and replanning now dominate.
4. Backbone capacity matters for RGB only when supervision is aligned: Qwen 7B reaches `100%` one-step direction accuracy and `39.1%` closed-loop success under clean direction labels; Qwen 3B does not.
5. Backbone capacity alone is insufficient: Qwen 7B frozen RGB scores `0/64`, and Qwen 7B with the mismatched oracle label scores only `10/64` on RGB.
6. These results do not yet validate OpenVLA or SimpleVLA. Their continuous/chunked action contracts differ from this five-token Qwen policy and require a separate same-environment adapter audit.

## Validation Status

- Python syntax checks: passed.
- Repository tests with local source path: `30 passed`.
- Default pytest plugin loading is broken by an unrelated ROS `launch_testing`/`lark` environment conflict; tests pass with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src`.
- No multi-seed estimate yet; current conclusions are controlled single-seed causal diagnostics.
