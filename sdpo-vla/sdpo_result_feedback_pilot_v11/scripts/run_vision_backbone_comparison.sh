#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?output directory required}"
PYTHON="${PYTHON:-/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python}"

if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite existing run: $RUN" >&2
  exit 2
fi

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/src"
mkdir -p "$RUN"

run_action_model() {
  local backbone="$1"
  local name="$2"
  "$PYTHON" -u "$ROOT/run_pilot.py" \
    --backbone "$backbone" --mode probe --scene-profile directional_long \
    --seed 11 --microbatch-size 4 --dev-scenes 64 \
    --output "$RUN/${name}_probe" > "$RUN/${name}_probe.log" 2>&1
  "$PYTHON" -u "$ROOT/run_pilot.py" \
    --backbone "$backbone" --mode eval --scene-profile directional_long \
    --seed 11 --microbatch-size 4 --dev-scenes 64 \
    --output "$RUN/${name}_eval" > "$RUN/${name}_eval.log" 2>&1
}

run_action_model qwen3 qwen3
run_action_model internvl3.5 internvl35

"$PYTHON" -u "$ROOT/scripts/molmo_grounding_probe.py" \
  --model-id "$ROOT/models/Molmo2-8B" --dev-scenes 64 --seed 11 \
  --output "$RUN/molmo2_grounding" > "$RUN/molmo2_grounding.log" 2>&1

"$PYTHON" "$ROOT/scripts/compare_vision_backbones.py" \
  --qwen3 "$RUN/qwen3_probe" \
  --qwen3-eval "$RUN/qwen3_eval" \
  --internvl "$RUN/internvl35_probe" \
  --internvl-eval "$RUN/internvl35_eval" \
  --molmo "$RUN/molmo2_grounding" \
  --output "$RUN/comparison.json"
