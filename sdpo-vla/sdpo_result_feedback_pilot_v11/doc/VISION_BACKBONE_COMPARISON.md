# Vision backbone comparison

This directory supports one reproducible comparison:

- `Qwen3-VL-8B-Instruct`: action-token policy, LoRA only on language attention.
- `InternVL3.5-8B-HF`: same action-token policy and same prompt, scene split, and evaluation.
- `Molmo2-8B`: independent point-grounding probe; never used as an action policy or teacher.

Run after all three local model directories are complete:

```bash
source /home/skylar/miniconda3/bin/activate sdpo-vla-pilot
PYTHONPATH=src scripts/run_vision_backbone_comparison.sh artifacts/vision_backbone_comparison_01
```

Comparison uses 64 fixed `directional_long` development scenes, seed `11`, greedy action selection, and at most 12 closed-loop decisions. Action models report initial-state oracle-direction accuracy and closed-loop success. Molmo reports parsed point rate, simulator-projected-center error, and 32-pixel hit rate for cube and target.

Action accuracy and point-grounding accuracy measure different interfaces. Molmo results diagnose visual localization only; they do not establish action-token or closed-loop control quality.

## Completed run

Artifact: `artifacts/vision_backbone_comparison_01/comparison.json`

| Model | Initial oracle action accuracy | Closed-loop success | Mean progress (m) |
| --- | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 20/64 (31.25%) | 15/64 (23.4375%) | 0.036760 |
| InternVL3.5-8B-HF | 16/64 (25.00%) | 9/64 (14.0625%) | 0.029283 |

Molmo2-8B point grounding on the same 64 scenes:

- Green cube: 51/64 parsed (79.6875%); parsed mean simulator-center error 3.472 px; 51/51 within 32 px.
- Blue target: 64/64 parsed against simulator centers; mean error 16.856 px; 61/64 within 32 px (95.3125%). Visible-mask parse rates were 59/64 for target and 51/64 for cube.

Interpretation: Qwen3 is stronger action policy in this frozen development protocol. Molmo shows strong visual localization, but this probe does not test action-token prediction, feedback learning, or closed-loop control. Molmo remote code requires PyTorch `>=2.6`; verified environment uses PyTorch `2.6.0+cu124`, torchvision `0.21.0+cu124`.
