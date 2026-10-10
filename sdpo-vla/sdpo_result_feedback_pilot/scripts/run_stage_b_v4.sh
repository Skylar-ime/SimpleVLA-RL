#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
CONDA=/home/skylar/miniconda3/bin/conda
PYTHON=("$CONDA" run -n sdpo-vla-pilot --no-capture-output python)
"${PYTHON[@]}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
RUN="$ROOT/artifacts/stage_b_v4_sdpo_aligned"
if [[ -e "$RUN" ]]; then
  printf 'Refusing existing run: %s\n' "$RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
for group in frozen result result_interpreted binary hard_shuffled; do
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$group"
  if [[ "$group" == frozen ]]; then
    "${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode eval --scene-profile directional_long --dev-scenes 64 --seed 11 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  else
    "${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode train --scene-profile directional_long --one-step-rollouts --feedback-mode "$group" --updates 6 --transitions 256 --seed 11 --dev-scenes 64 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  fi
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$group"
done
"${PYTHON[@]}" "$ROOT/scripts/summarize_stage_b.py" "$RUN" > "$RUN/comparison.log"
"${PYTHON[@]}" "$ROOT/scripts/audit_runs.py" "$RUN" > "$RUN/artifact-audit.log"
"${PYTHON[@]}" "$ROOT/scripts/grounding_audit.py" "$RUN" > "$RUN/grounding-audit.log"
printf '%s STAGE_B_V4_COMPLETE\n' "$(date --iso-8601=seconds)"
