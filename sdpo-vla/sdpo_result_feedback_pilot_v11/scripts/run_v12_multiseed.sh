#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE="${1:?backbone required}"
OUTPUT_ROOT="${2:?output root required}"
SEEDS="${*:3}"
SEEDS="${SEEDS:-12 13 14 15 16}"
PROTOCOL_FILE="${SDPO_PROTOCOL_FILE:-protocol_v12.json}"
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python
MICROBATCH_SIZE="${SDPO_MICROBATCH_SIZE:-12}"
EVAL_MICROBATCH_SIZE="${SDPO_EVAL_MICROBATCH_SIZE:-16}"
if [[ "$BACKBONE" == molmo2 && -z "${SDPO_MICROBATCH_SIZE:-}" ]]; then
  MICROBATCH_SIZE=1
fi
if [[ "$BACKBONE" == molmo2 && -z "${SDPO_EVAL_MICROBATCH_SIZE:-}" ]]; then
  EVAL_MICROBATCH_SIZE=4
fi
CONDITIONS=(sdpo_result sdpo_advantage sdpo_advantage_shuffled sdpo_advantage_reversed)

export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"

for seed in $SEEDS; do
  for condition in "${CONDITIONS[@]}"; do
    run="$OUTPUT_ROOT/seed_${seed}/${condition}"
    if [[ -e "$run" ]]; then
      if [[ -f "$run/completed.json" && -f "$run/v12_audit.json" ]]; then
        "$PYTHON" "$ROOT/scripts/audit_v12.py" "$run" >/dev/null
        echo "Skipping audited run: $run"
        continue
      fi
      echo "Refusing to overwrite existing run: $run" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$run")"
    "$PYTHON" -u "$ROOT/run_pilot.py" \
      --mode train \
      --backbone "$BACKBONE" --feedback-mode "$condition" \
      --protocol-file "$ROOT/$PROTOCOL_FILE" \
      --scene-profile directional_long --one-step-rollouts \
      --grouped-candidates 8 --transitions 128 --updates 12 \
      --parallel-envs 1 --microbatch-size "$MICROBATCH_SIZE" \
      --eval-microbatch-size "$EVAL_MICROBATCH_SIZE" \
      --dev-scenes 64 --seed "$seed" --output "$run" \
      > "$OUTPUT_ROOT/seed_${seed}_${condition}.log" 2>&1
    "$PYTHON" "$ROOT/scripts/audit_v12.py" "$run"
  done
done
