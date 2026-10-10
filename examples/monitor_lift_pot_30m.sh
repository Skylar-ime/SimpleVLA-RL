#!/usr/bin/env bash
set -u

PID="${1:?training PID required}"
LOG="${2:?training log required}"
OUT="${3:?training output required}"
SNAP="${4:?snapshot log required}"

while kill -0 "$PID" 2>/dev/null; do
  {
    echo "===== $(date '+%F %T %Z') ====="
    echo "PID: $PID"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo "-- events --"
    grep -E 'step:[0-9]+ -|update_actor:|val/test_score|Accuracy distribution|collected|rollout batch size|Saving|checkpoint|Traceback|Error|RuntimeError|CUDA out|ZeroDivision' "$LOG" | tail -40 || true
    echo "-- checkpoints --"
    find "$OUT" -maxdepth 2 -type d -name 'global_step_*' -printf '%f %TH:%TM:%TS\n' | sort -V | tail -10
  } >> "$SNAP" 2>&1
  sleep 1800
done
echo "===== $(date '+%F %T %Z') training PID exited; watcher stopped =====" >> "$SNAP"
