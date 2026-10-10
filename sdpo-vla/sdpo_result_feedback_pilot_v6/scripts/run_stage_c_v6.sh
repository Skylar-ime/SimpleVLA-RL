#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot_v6")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA=/home/skylar/miniconda3/bin/conda
PYTHON=("$CONDA" run -n sdpo-vla-pilot --no-capture-output python)
RUN="$ROOT/artifacts/stage_c_v6_grouped"
PROBE="$ROOT/artifacts/grouped_mechanism_probe/report.json"
[[ -f "$PROBE" ]] || { echo 'GROUPED_MECHANISM_PROBE_MISSING' >&2; exit 1; }
python3 - "$PROBE" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get('gate',{}).get('status')!='passed':
    raise SystemExit('GROUPED_MECHANISM_GATE_NOT_PASSED')
PY
[[ ! -e "$RUN" ]] || { echo "Refusing existing run: $RUN" >&2; exit 1; }
mkdir -p "$RUN"
printf '%s START frozen\n' "$(date --iso-8601=seconds)"
"${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode eval --scene-profile directional_calibrated \
  --dev-scenes 64 --seed 11 --output "$RUN/frozen" > "$RUN/frozen.log" 2>&1
printf '%s COMPLETE frozen\n' "$(date --iso-8601=seconds)"
for group in solution_only solution_plus_binary solution_plus_causal solution_plus_causal_shuffled; do
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$group"
  "${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode train --scene-profile directional_calibrated \
    --one-step-rollouts --feedback-mode "$group" --grouped-candidates 8 \
    --updates 8 --transitions 256 --seed 11 --dev-scenes 64 \
    --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$group"
done
"${PYTHON[@]}" "$ROOT/scripts/summarize_stage_b.py" "$RUN" > "$RUN/comparison.log"
"${PYTHON[@]}" "$ROOT/scripts/audit_runs.py" "$RUN" > "$RUN/artifact-audit.log"
"${PYTHON[@]}" "$ROOT/scripts/grounding_audit.py" "$RUN" > "$RUN/grounding-audit.log"
printf '%s STAGE_C_V6_COMPLETE\n' "$(date --iso-8601=seconds)"
