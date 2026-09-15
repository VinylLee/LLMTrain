#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ubuntu/LLMTrain/LLMTrain"
CONFIG="${PROJECT_ROOT}/RQ1/configs/rq1_snli_original_controls_gemma-3-4b-it_config.json"
STANDARD_ROOT="${PROJECT_ROOT}/RQ1/output/gemma-3-4b-it/snli_original_2033_multiseed_v1"
TOKENMATCHED_ROOT="${PROJECT_ROOT}/RQ1/output/gemma-3-4b-it/snli_original_2033_tokenmatched_multiseed_v1"
QUEUE_DIR="${PROJECT_ROOT}/RQ1/output/gemma-3-4b-it/snli_original_controls_test_queue_v1"
LOG_FILE="${QUEUE_DIR}/queue.log"
LOCK_FILE="${QUEUE_DIR}/queue.lock"
GPU_INDEX=2
MAX_MEMORY_MIB=20000
MAX_UTILIZATION=20
REQUIRED_IDLE_CHECKS=3

mkdir -p "${QUEUE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '%s duplicate launcher refused\n' "$(date --iso-8601=seconds)" >> "${LOG_FILE}"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "${LOG_FILE}"
}

tests_complete() {
  local progress_file="$1/_progress.json"
  /usr/bin/python3 - "${progress_file}" <<'PY_CHECK'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    progress = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
raise SystemExit(0 if len(progress) == 3 and all(
    stages.get("test_merged") == "completed"
    for stages in progress.values()
) else 1)
PY_CHECK
}

wait_for_gpu() {
  local streak=0
  local memory_used utilization
  while (( streak < REQUIRED_IDLE_CHECKS )); do
    IFS=',' read -r memory_used utilization < <(
      nvidia-smi --id="${GPU_INDEX}"         --query-gpu=memory.used,utilization.gpu         --format=csv,noheader,nounits
    )
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if (( memory_used <= MAX_MEMORY_MIB && utilization <= MAX_UTILIZATION )); then
      streak=$((streak + 1))
      log "gpu=${GPU_INDEX} idle_check=${streak}/${REQUIRED_IDLE_CHECKS} memory_mib=${memory_used} utilization=${utilization}"
    else
      streak=0
      log "waiting gpu=${GPU_INDEX} memory_mib=${memory_used} utilization=${utilization}"
    fi
    if (( streak < REQUIRED_IDLE_CHECKS )); then
      sleep 60
    fi
  done
}

run_group() {
  local dataset_alias="$1"
  local output_root="$2"
  local label="$3"
  if tests_complete "${output_root}"; then
    log "${label} already complete; skip"
    return
  fi
  wait_for_gpu
  log "starting ${label} merged tests on gpu=${GPU_INDEX}"
  cd "${PROJECT_ROOT}"
  /home/ubuntu/.conda/envs/llmtrain310/bin/python RQ1/run_rq1_nli.py     --model gemma-3-4b-it     --experiment original     --train-data "${dataset_alias}"     --seeds 42 43 44     --config "${CONFIG}"     --output-root "${output_root}"     --steps test_merged     --resume     --cuda "${GPU_INDEX}" >> "${LOG_FILE}" 2>&1
  log "finished ${label} merged tests"
}

log "launcher started pid=$$"
run_group   "snli_2033_gemma-3-4b-it"   "${STANDARD_ROOT}"   "original_2033"
run_group   "snli_2033_gemma-3-4b-it_tokenmatched"   "${TOKENMATCHED_ROOT}"   "original_2033_tokenmatched"
log "all queued merged tests complete"
