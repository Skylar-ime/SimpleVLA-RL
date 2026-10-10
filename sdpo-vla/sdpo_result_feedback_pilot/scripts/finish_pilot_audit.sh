#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python
TARGET_PID=1936533
# This is internal completion waiting, not repeated user-facing status checking.
while kill -0 "$TARGET_PID" 2>/dev/null; do sleep 60; done
[[ -f "$ROOT/artifacts/stage_b_v1/shuffled/completed.json" ]] || { echo 'SHUFFLED_NOT_COMPLETE'; exit 1; }
"$PYTHON" "$ROOT/scripts/summarize_stage_b.py" "$ROOT/artifacts/stage_b_v1" > "$ROOT/artifacts/stage_b_v1/comparison.log" 2>&1
"$PYTHON" "$ROOT/scripts/audit_runs.py" "$ROOT/artifacts/stage_b_v1" > "$ROOT/artifacts/stage_b_v1/artifact-audit.log" 2>&1
"$PYTHON" -u "$ROOT/scripts/grounding_audit.py" "$ROOT/artifacts/stage_b_v1" > "$ROOT/artifacts/stage_b_v1/grounding-audit.log" 2>&1
printf '%s PILOT_AUDITS_COMPLETE\n' "$(date --iso-8601=seconds)"
