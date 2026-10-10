# Fast Oracle Validation

## Protocol

- Model: Qwen2.5-VL-3B-Instruct, language-decoder LoRA only.
- Frozen directional-long scenes: 64 train, 64 held-out dev.
- Each state enumerates A/B/C/D/E in PyBullet and assigns an oracle label by success, then safe one-step progress, then contact.
- RGB and target-direction conditions receive the same oracle action label through four epochs of LoRA cross-entropy training, batch size 8.
- `oracle_action` input is only an answer-following upper-bound probe. It is excluded from learned-action claims because it reveals the label.
- Closed-loop evaluation uses the same 64 held-out scenes, greedy actions, and at most 12 decisions.

## Results

| Condition | Held-out oracle-action accuracy after training | Closed-loop success |
| --- | ---: | ---: |
| Frozen RGB | 21.9% | 11/64 (17.2%) |
| RGB + oracle supervision | 45.3% | 16/64 (25.0%) |
| Target direction + oracle supervision | 70.3% | 43/64 (67.2%) |
| Oracle action input upper bound | 100.0% | not used for deployment claim |

All one-step candidate actions fail the strict task-success threshold in this long-horizon scene profile. The one-step diagnostic therefore uses oracle action accuracy and progress; closed-loop success is the deployment measure.

## Interpretation

1. LoRA can learn oracle action labels: RGB held-out accuracy rises from 21.9% to 45.3%; target-direction input reaches 70.3%.
2. Target-direction input is much easier than RGB, showing a substantial visual grounding or image-to-direction bottleneck.
3. Better action-label accuracy transfers to closed-loop behavior when target direction is available: 67.2% versus frozen RGB 17.2% on the paired held-out scenes.
4. RGB improvement is a weak pilot signal: 25.0% versus 17.2%, paired net change +7.8 percentage points, with a wide bootstrap interval of approximately -7.8 to +23.4 points.
5. This experiment does not prove that the original SDPO teacher is correct, because supervision uses simulator oracle labels. It isolates student learnability and input quality from teacher quality.

Artifacts: `artifacts/fast_oracle_validation_v3/results.json`, `artifacts/fast_oracle_validation_v3/closed_loop_evaluation.json`, and `artifacts/fast_oracle_validation_v3/oracle_dataset.json`.
