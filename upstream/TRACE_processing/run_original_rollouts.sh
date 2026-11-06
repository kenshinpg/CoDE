#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source "${TRACE_ENV:-configs/environment.example.sh}"
FROZEN_INPUT="${FROZEN_INPUT:?Set to the authorized frozen CoDE rollout input package}"
STAGE="${1:?Choose generate, match-prepare, match-node, match-merge}"
shift
case "$STAGE" in
generate)
  "$PYTHON" scripts/trace_rollouts_full_v1/pipeline.py prepare --input "$FROZEN_INPUT" --out outputs/original_qwen4b --model-dir "$QWEN4B_DIR" --endpoint "$GENERATION_ENDPOINT"
  "$PYTHON" scripts/trace_rollouts_full_v1/pipeline.py run --run outputs/original_qwen4b --workers "${GENERATION_WORKERS:-64}" ;;
match-prepare)
  "$PYTHON" tools/prepare_match_source.py --generation outputs/original_qwen4b --input "$FROZEN_INPUT" --out outputs/match_source
  "$PYTHON" scripts/trace_match_full_v1/pipeline.py prepare --source outputs/match_source --root outputs/full_match ;;
match-node)
  "$PYTHON" scripts/trace_match_full_v1/pipeline.py run --root outputs/full_match --node "${1:?Pass node 0..7}" --workers "$WORKERS" --endpoint "$TRACE_ENDPOINT" --model Qwen3.8-27B ;;
match-merge)
  "$PYTHON" scripts/trace_match_full_v1/pipeline.py merge --root outputs/full_match ;;
*) exit 2 ;;
esac
