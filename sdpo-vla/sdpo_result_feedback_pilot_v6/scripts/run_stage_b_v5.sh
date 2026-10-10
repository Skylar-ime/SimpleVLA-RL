#!/usr/bin/env bash
set -euo pipefail
ROOT=$(realpath "$(dirname "$0")/../../sdpo_result_feedback_pilot_v5")
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CONDA=/home/skylar/miniconda3/bin/conda
PYTHON=("$CONDA" run -n sdpo-vla-pilot --no-capture-output python)
DRIVER_ROOT=/home/skylar/miniconda3/envs/sdpo-vla-pilot/opt/nvidia-driver-580.173.02/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH="$DRIVER_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
while systemctl --user is-active --quiet sdpo-vla-stage-b-v4.service; do
  printf '%s WAITING_FOR_V4\n' "$(date --iso-8601=seconds)"
  sleep 60
done
"${PYTHON[@]}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
"${PYTHON[@]}" -c 'import os,pathlib,sdpo_result_feedback.scenes as s; assert pathlib.Path(s.__file__).resolve().is_relative_to(pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]))'
RUN="$ROOT/artifacts/stage_b_v5_calibrated"
if [[ -e "$RUN" ]]; then
  printf 'Refusing existing run: %s\n' "$RUN" >&2
  exit 1
fi
mkdir -p "$RUN"
printf '%s START mechanism_probe\n' "$(date --iso-8601=seconds)"
"${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode probe --scene-profile directional_calibrated \
  --dev-scenes 64 --seed 11 --output "$RUN/mechanism_probe" > "$RUN/mechanism_probe.log" 2>&1
"${PYTHON[@]}" "$ROOT/scripts/check_probe_gate.py" "$RUN/mechanism_probe/probe/report.json" \
  > "$RUN/mechanism_gate.log" 2>&1
printf '%s COMPLETE mechanism_probe\n' "$(date --iso-8601=seconds)"
for group in frozen result causal causal_shuffled binary; do
  printf '%s START %s\n' "$(date --iso-8601=seconds)" "$group"
  if [[ "$group" == frozen ]]; then
    "${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode eval --scene-profile directional_calibrated \
      --dev-scenes 64 --seed 11 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  else
    "${PYTHON[@]}" -u "$ROOT/run_pilot.py" --mode train --scene-profile directional_calibrated \
      --one-step-rollouts --feedback-mode "$group" --updates 6 --transitions 256 --seed 11 \
      --dev-scenes 64 --output "$RUN/$group" > "$RUN/$group.log" 2>&1
  fi
  printf '%s COMPLETE %s\n' "$(date --iso-8601=seconds)" "$group"
done
"${PYTHON[@]}" "$ROOT/scripts/summarize_stage_b.py" "$RUN" > "$RUN/comparison.log"
"${PYTHON[@]}" "$ROOT/scripts/audit_runs.py" "$RUN" > "$RUN/artifact-audit.log"
"${PYTHON[@]}" "$ROOT/scripts/grounding_audit.py" "$RUN" > "$RUN/grounding-audit.log"
printf '%s STAGE_B_V5_COMPLETE\n' "$(date --iso-8601=seconds)"
