#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?output directory required}"
UPDATES="${2:-24}"

if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite existing run: $RUN" >&2
  exit 2
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src"

mkdir -p "$RUN"
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python

for group in sdpo_result sdpo_binary sdpo_shuffled sdpo_neutral sdpo_causal; do
  "$PYTHON" -u "$ROOT/run_pilot.py" \
    --mode train \
    --scene-profile directional_long \
    --one-step-rollouts \
    --expect-feedback-only \
    --feedback-mode "$group" \
    --grouped-candidates 8 \
    --updates "$UPDATES" \
    --transitions 128 \
    --seed 11 \
    --dev-scenes 64 \
    --output "$RUN/$group" \
    > "$RUN/$group.log" 2>&1
done

"$PYTHON" "$ROOT/scripts/compare_stage_d.py" "$RUN"
"$PYTHON" "$ROOT/scripts/audit_feedback_only.py" "$RUN"
"$PYTHON" "$ROOT/scripts/compare_teacher_conditions.py" "$RUN" --update 1
"$PYTHON" "$ROOT/scripts/analyze_feedback_trajectory.py" "$RUN"
