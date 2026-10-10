#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

while systemctl --user is-active --quiet sdpo-v12b-qwen3.service; do
  sleep 60
done

check_runs() {
  local backbone="$1"
  root="artifacts/v12b_${backbone}_multiseed"
  completed=$(find "$root" -mindepth 3 -maxdepth 3 -name completed.json | wc -l)
  audited=$(find "$root" -mindepth 3 -maxdepth 3 -name v12_audit.json | wc -l)
  if [[ "$completed" -ne 20 || "$audited" -ne 20 ]]; then
    echo "$backbone incomplete: completed=$completed audited=$audited" >&2
    exit 1
  fi
}

check_runs qwen3
SDPO_PROTOCOL_FILE=protocol_v12b.json SDPO_MICROBATCH_SIZE=1 \
  SDPO_EVAL_MICROBATCH_SIZE=4 bash scripts/run_v12_multiseed.sh \
  molmo2 artifacts/v12b_molmo2_multiseed 12 13 14 15 16
check_runs molmo2

/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python scripts/compare_v12.py \
  --protocol protocol_v12b.json \
  --qwen3 artifacts/v12b_qwen3_multiseed \
  --molmo2 artifacts/v12b_molmo2_multiseed \
  --output artifacts/v12b_qwen3_molmo2_multiseed_comparison_01.json
/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/python scripts/render_v12b_results.py \
  --comparison artifacts/v12b_qwen3_molmo2_multiseed_comparison_01.json \
  --output doc/V12B_RESULTS.md
