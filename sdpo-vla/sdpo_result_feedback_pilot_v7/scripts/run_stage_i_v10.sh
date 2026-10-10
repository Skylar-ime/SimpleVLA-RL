#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?output directory required}"
SEED="${2:-12}"
UPDATES="${3:-12}"
MODEL_ID="${4:-}"
MICROBATCH_SIZE="${5:-4}"
MODEL_REVISION="${6:-}"
MODEL_ARGS=()
if [[ -n "$MODEL_ID" ]]; then MODEL_ARGS+=(--model-id "$MODEL_ID"); fi
if [[ -n "$MODEL_REVISION" ]]; then MODEL_ARGS+=(--model-revision "$MODEL_REVISION"); fi

if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite existing run: $RUN" >&2
  exit 2
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src"
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python

mkdir -p "$RUN"
for probe_seed in 11 "$SEED"; do
  "$PYTHON" -u "$ROOT/run_pilot.py" \
    --mode probe \
    --scene-profile directional_long \
    --protocol-file "$ROOT/protocol_v10.json" \
    --feedback-mode sdpo_directional \
    --seed "$probe_seed" \
    --microbatch-size "$MICROBATCH_SIZE" "${MODEL_ARGS[@]}" \
    --dev-scenes 64 \
    --output "$RUN/probe_seed_${probe_seed}" \
    > "$RUN/probe_seed_${probe_seed}.log" 2>&1
done

for group in sdpo_result sdpo_binary sdpo_causal sdpo_directional sdpo_wrong_directional; do
  "$PYTHON" -u "$ROOT/run_pilot.py" \
    --mode train \
    --scene-profile directional_long \
    --one-step-rollouts \
    --expect-feedback-only \
    --protocol-file "$ROOT/protocol_v10.json" \
    --feedback-mode "$group" \
    --grouped-candidates 8 \
    --updates "$UPDATES" \
    --transitions 128 \
    --seed "$SEED" \
    --microbatch-size "$MICROBATCH_SIZE" "${MODEL_ARGS[@]}" \
    --dev-scenes 64 \
    --output "$RUN/$group" \
    > "$RUN/$group.log" 2>&1
done

"$PYTHON" "$ROOT/scripts/compare_stage_i_v10.py" "$RUN"
"$PYTHON" "$ROOT/scripts/audit_feedback_only.py" "$RUN" \
  --conditions sdpo_result sdpo_binary sdpo_causal sdpo_directional sdpo_wrong_directional
