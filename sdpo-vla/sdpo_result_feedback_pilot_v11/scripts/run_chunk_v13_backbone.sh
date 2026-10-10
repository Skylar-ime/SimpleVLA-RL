#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE="${1:?backbone required}"
OUTPUT_ROOT="${2:?output root required}"
PYTHON=/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python
MICROBATCH_SIZE=4
if [[ "$BACKBONE" == molmo2 ]]; then
  MICROBATCH_SIZE=1
fi
CONDITIONS=(chunk_suffix_result chunk_suffix_directional
  chunk_suffix_directional_residual chunk_suffix_wrong_directional
  chunk_suffix_wrong_directional_residual)

export HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_ROOT"

for condition in "${CONDITIONS[@]}"; do
  run="$OUTPUT_ROOT/${condition}"
  if [[ -e "$run" ]]; then
    echo "Refusing to overwrite existing run: $run" >&2
    exit 2
  fi
  "$PYTHON" -u "$ROOT/run_chunk_pilot.py" \
    --backbone "$BACKBONE" --condition "$condition" \
    --transitions 64 --updates 12 --chunk-length 2 \
    --grouped-candidates 8 --seed 12 --microbatch-size "$MICROBATCH_SIZE" \
    --dev-scenes 64 --expect-feedback-only --output "$run" \
    > "$OUTPUT_ROOT/${condition}.log" 2>&1
  "$PYTHON" "$ROOT/scripts/audit_chunk_v13.py" "$run"
done
