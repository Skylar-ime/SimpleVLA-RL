#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python
# This script must be launched through conda run to apply the environment-only driver repair.
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Activate sdpo-vla-pilot first"'
RUN="$ROOT/artifacts/stage_b_v1"
mkdir -p "$RUN"
for group in frozen result binary shuffled; do
  if [[ -f "$RUN/$group/completed.json" ]]; then
    continue
  fi
  if [[ -d "$RUN/$group" ]]; then
    printf 'Incomplete existing run: %s\n' "$RUN/$group" >&2
    exit 1
  fi
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$group"
  if [[ "$group" == frozen ]]; then
    "$PYTHON" -u "$ROOT/run_pilot.py" --mode eval --dev-scenes 64 --seed 11 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  else
    "$PYTHON" -u "$ROOT/run_pilot.py" --mode train --feedback-mode "$group" --updates 10 --transitions 256 --seed 11 --dev-scenes 64 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  fi
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$group"
done
printf '%s STAGE_B_COMPLETE\n' "$(date --iso-8601=seconds)"
