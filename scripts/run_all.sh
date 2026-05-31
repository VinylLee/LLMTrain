#!/usr/bin/env bash
# 对指定任务下的所有数据集依次执行预测 + 评估
# 输出结构: output/<llm>/<dataset>/<mr>.jsonl
# Usage:
#   bash scripts/run_all.sh --llm phi-4 --delay 0.5
#   bash scripts/run_all.sh --llm qwen --task nli --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY
set -euo pipefail

LLM="deepseek-chat"
DELAY=0.3
OUTPUT_DIR="output"
TASK="nli"
API_URL_ENV=""
API_KEY_ENV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm) LLM="$2"; shift 2 ;;
    --delay) DELAY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir) TASK="$2"; shift 2 ;;  # --data-dir sets the task subdirectory under data/
    --task) TASK="$2"; shift 2 ;;
    --api-url-env) API_URL_ENV="$2"; shift 2 ;;
    --api-key-env) API_KEY_ENV="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

DATA_DIR="data/$TASK"
SAFE_LLM="${LLM// /_}"
echo "========================================"
echo "  Batch Run"
echo "  LLM:      $LLM"
echo "  Task:     $TASK"
echo "  Delay:    ${DELAY}s"
echo "  Data:     $DATA_DIR/"
echo "  Output:   $OUTPUT_DIR/$SAFE_LLM/"
echo "========================================"

# ---- Step 1: Predict ----
for ds_dir in "$DATA_DIR"/*_MR; do
  [ -d "$ds_dir" ] || continue
  ds_name=$(basename "$ds_dir" | sed 's/_test_MR//')
  count=$(find "$ds_dir" -name '*.json' -type f | wc -l)
  echo ">>> [predict] $ds_name ($count files)"

  conda run -n LLMTrain3.9 \
    python3 scripts/test_llm.py \
      --data-dir "$ds_dir" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$SAFE_LLM" \
      --delay "$DELAY" \
      --llm "$LLM" \
      ${API_URL_ENV:+--api-url-env "$API_URL_ENV"} \
      ${API_KEY_ENV:+--api-key-env "$API_KEY_ENV"} \
      2>&1
  echo
done

# ---- Step 2: Evaluate per dataset ----
echo "========================================"
echo "  Evaluating..."
echo "========================================"

for ds_dir in "$DATA_DIR"/*_MR; do
  [ -d "$ds_dir" ] || continue
  ds_name=$(basename "$ds_dir" | sed 's/_test_MR//')
  echo ">>> [eval] $ds_name"

  ds_output_dir="$OUTPUT_DIR/$SAFE_LLM/$ds_name"
  combined="$ds_output_dir/${ds_name}_all.jsonl"

  if [ -d "$ds_output_dir" ]; then
    find "$ds_output_dir" -maxdepth 1 -name '*.jsonl' ! -name '*_all.jsonl' \
      -exec cat {} + | sort -t'"' -k2,2n > "$combined" || true
  fi

  if [ -s "$combined" ]; then
    conda run -n LLMTrain3.9 \
      python3 scripts/evaluate.py "$combined" \
        --report "$ds_output_dir/${ds_name}_report.txt" \
        2>&1 | tail -3
  else
    echo "  (no output)"
  fi
  echo
done

# ---- Step 3: Evaluate all combined ----
echo ">>> [eval] ALL combined"
combined_all="$OUTPUT_DIR/$SAFE_LLM/all_combined.jsonl"
cat "$OUTPUT_DIR/$SAFE_LLM"/*/*_all.jsonl 2>/dev/null > "$combined_all" || true

if [ -s "$combined_all" ]; then
  conda run -n LLMTrain3.9 \
    python3 scripts/evaluate.py "$combined_all" \
      --report "$OUTPUT_DIR/$SAFE_LLM/all_report.txt" \
      --group-by _source mr_type \
      2>&1 | tail -3
fi

# ---- Summary ----
echo
echo "========================================"
echo "  Output directory:"
echo "    $OUTPUT_DIR/$SAFE_LLM/"
echo "  Reports:"
find "$OUTPUT_DIR/$SAFE_LLM" -name '*_report.txt' 2>/dev/null | sed 's/^/    /'
echo "========================================"
