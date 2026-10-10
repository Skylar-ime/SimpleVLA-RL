#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Activate sdpo-vla-pilot first"'
RUN="$ROOT/artifacts/stage_b_v2_near_contact"
mkdir -p "$RUN"
for group in frozen result binary shuffled neutral; do
  if [[ -f "$RUN/$group/completed.json" ]]; then
    continue
  fi
  if [[ -d "$RUN/$group" ]]; then
    printf 'Incomplete existing run: %s\n' "$RUN/$group" >&2
    exit 1
  fi
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$group"
  if [[ "$group" == frozen ]]; then
    "$PYTHON" -u "$ROOT/run_pilot.py" --mode eval --scene-profile mechanism_near_contact --dev-scenes 64 --seed 11 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  else
    "$PYTHON" -u "$ROOT/run_pilot.py" --mode train --scene-profile mechanism_near_contact --feedback-mode "$group" --updates 10 --transitions 256 --seed 11 --dev-scenes 64 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  fi
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$group"
done
"$PYTHON" "$ROOT/scripts/summarize_stage_b.py" "$RUN" > "$RUN/comparison.log"
"$PYTHON" "$ROOT/scripts/audit_runs.py" "$RUN" > "$RUN/artifact-audit.log"
"$PYTHON" "$ROOT/scripts/grounding_audit.py" "$RUN" > "$RUN/grounding-audit.log"
printf '%s STAGE_B_V2_COMPLETE\n' "$(date --iso-8601=seconds)"
