#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:?version required: v10 or v11}"
BACKBONE="${2:?backbone required: qwen3 or molmo2}"
RUN="${3:?output directory required}"
SEED="${4:-12}"
UPDATES="${5:-12}"
MICROBATCH_SIZE="${6:-4}"

case "$VERSION" in
  v10)
    PROTOCOL="$ROOT/protocol_v10.json"
    CONDITIONS=(sdpo_result sdpo_binary sdpo_causal sdpo_directional sdpo_wrong_directional)
    COMPARER="$ROOT/scripts/compare_stage_i_v10.py"
    ;;
  v11)
    PROTOCOL="$ROOT/protocol_v11.json"
    CONDITIONS=(sdpo_result sdpo_binary sdpo_causal sdpo_directional
      sdpo_directional_residual sdpo_wrong_directional sdpo_wrong_directional_residual)
    COMPARER="$ROOT/scripts/compare_stage_i_v11.py"
    ;;
  *)
    echo "Unsupported version: $VERSION" >&2
    exit 2
    ;;
esac

if [[ "$BACKBONE" != qwen3 && "$BACKBONE" != molmo2 ]]; then
  echo "Unsupported backbone: $BACKBONE" >&2
  exit 2
fi
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
    --backbone "$BACKBONE" --mode probe --scene-profile directional_long \
    --protocol-file "$PROTOCOL" --feedback-mode sdpo_directional \
    --seed "$probe_seed" --microbatch-size "$MICROBATCH_SIZE" \
    --dev-scenes 64 --output "$RUN/probe_seed_${probe_seed}" \
    > "$RUN/probe_seed_${probe_seed}.log" 2>&1
done

for condition in "${CONDITIONS[@]}"; do
  "$PYTHON" -u "$ROOT/run_pilot.py" \
    --backbone "$BACKBONE" --mode train --scene-profile directional_long \
    --one-step-rollouts --expect-feedback-only --protocol-file "$PROTOCOL" \
    --feedback-mode "$condition" --grouped-candidates 8 --updates "$UPDATES" \
    --transitions 128 --seed "$SEED" --microbatch-size "$MICROBATCH_SIZE" \
    --dev-scenes 64 --output "$RUN/$condition" \
    > "$RUN/$condition.log" 2>&1
done

"$PYTHON" "$COMPARER" "$RUN"
"$PYTHON" "$ROOT/scripts/audit_feedback_only.py" "$RUN" --conditions "${CONDITIONS[@]}"
if [[ "$VERSION" == v11 ]]; then
  "$PYTHON" "$ROOT/scripts/audit_v11.py" "$RUN"
fi
