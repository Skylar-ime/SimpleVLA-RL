#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot_v7")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA=/home/skylar/miniconda3/bin/conda
PYTHON=("$CONDA" run -n sdpo-vla-pilot --no-capture-output python)
RUN="$ROOT/artifacts/stage_d_v7_sdpo_aligned"
test ! -e "$RUN"
mkdir -p "$RUN"
for group in sdpo_result sdpo_binary sdpo_shuffled sdpo_neutral sdpo_causal; do
  "${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode train --scene-profile directional_local \
    --one-step-rollouts --feedback-mode "$group" --grouped-candidates 8 \
    --updates 24 --transitions 128 --seed 11 --dev-scenes 64 \
    --output "$RUN/$group" > "$RUN/$group.log" 2>&1
done
"${PYTHON[@]}" "$ROOT/scripts/compare_stage_d.py" "$RUN"
