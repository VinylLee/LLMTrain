#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/ubuntu/LLMTrain/LLMTrain"
OUTPUT_ROOT="${PROJECT_ROOT}/RQ2/output/gemma3_4b_nli_full_seed42"
PROGRESS_FILE="${OUTPUT_ROOT}/_progress.json"
MONITOR_DIR="${OUTPUT_ROOT}/monitor"
LOG_FILE="${MONITOR_DIR}/hourly_status.tsv"

mkdir -p "${MONITOR_DIR}"

timestamp="$(date --iso-8601=seconds)"
finetune_completed=0
test_completed=0

if [[ -f "${PROGRESS_FILE}" ]]; then
  finetune_completed="$(grep -c "finetune.*completed" "${PROGRESS_FILE}" || true)"
  test_completed="$(grep -c "test.*completed" "${PROGRESS_FILE}" || true)"
fi

runner_count="$(pgrep -fc 'RQ2/run_rq2_snli.py --seeds 43 44' || true)"
trainer_count="$(pgrep -fc 'llamafactory.cli train .*rq2_snli_.*_seed(43|44)' || true)"

if [[ "${finetune_completed}" -eq 15 && "${test_completed}" -eq 15 ]]; then
  status="all_complete"
elif [[ "${finetune_completed}" -eq 15 ]]; then
  status="training_complete_awaiting_test"
elif [[ "${runner_count}" -gt 0 || "${trainer_count}" -gt 0 ]]; then
  status="training_running"
else
  status="training_not_detected"
fi

if [[ ! -f "${LOG_FILE}" ]]; then
  printf 'timestamp\tstatus\tfinetune_completed\ttest_completed\trunner_count\ttrainer_count\n' > "${LOG_FILE}"
fi

printf '%s\t%s\t%s/15\t%s/15\t%s\t%s\n' \
  "${timestamp}" "${status}" "${finetune_completed}" "${test_completed}" \
  "${runner_count}" "${trainer_count}" >> "${LOG_FILE}"

