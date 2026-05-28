#!/usr/bin/env bash
# 依次运行 data/ 下所有 MR 数据的预测 + 评估
# 每跑完一个数据集就评估该数据集，最后出全量汇总报告
# 输出结构: output/<run_id>/<llm>/<dataset>/<mr>.jsonl
# Usage: bash scripts/run_all.sh [--llm deepseek-chat] [--delay 0.3]
set -euo pipefail

LLM="deepseek-chat"
DELAY=0.3
OUTPUT_DIR="output"
DATA_DIR="data"
API_URL_ENV=""
API_KEY_ENV=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm) LLM="$2"; shift 2 ;;
    --delay) DELAY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --api-url-env) API_URL_ENV="$2"; shift 2 ;;
    --api-key-env) API_KEY_ENV="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAFE_LLM="${LLM// /_}"
echo "========================================"
echo "  Batch Run: $TIMESTAMP"
echo "  LLM:      $LLM"
echo "  Delay:    ${DELAY}s"
echo "  Data:     $DATA_DIR/"
echo "  Output:   $OUTPUT_DIR/$TIMESTAMP/$SAFE_LLM/"
echo "========================================"

# ---- Step 1: Predict ----
for ds_dir in "$DATA_DIR"/*_MR; do
  [ -d "$ds_dir" ] || continue
  ds_name=$(basename "$ds_dir" | sed 's/_test_MR//')
  count=$(find "$ds_dir" -name '*.json' -type f | wc -l)
  echo ">>> [predict] $ds_name ($count files)"

  conda run -n LLMTrain3.9 \
    python3 scripts/test_deepseek.py \
      --data-dir "$ds_dir" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$TIMESTAMP" \
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

  ds_output_dir="$OUTPUT_DIR/$TIMESTAMP/$SAFE_LLM/$ds_name"
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
combined_all="$OUTPUT_DIR/$TIMESTAMP/$SAFE_LLM/all_combined.jsonl"
cat "$OUTPUT_DIR/$TIMESTAMP/$SAFE_LLM"/*/*_all.jsonl 2>/dev/null > "$combined_all" || true

if [ -s "$combined_all" ]; then
  conda run -n LLMTrain3.9 \
    python3 scripts/evaluate.py "$combined_all" \
      --report "$OUTPUT_DIR/$TIMESTAMP/$SAFE_LLM/all_report.txt" \
      --group-by _source mr_type \
      2>&1 | tail -3
fi

# ---- Summary ----
echo
echo "========================================"
echo "  Output directory:"
echo "    $OUTPUT_DIR/$TIMESTAMP/$SAFE_LLM/"
echo "  Reports:"
find "$OUTPUT_DIR/$TIMESTAMP" -name '*_report.txt' 2>/dev/null | sed 's/^/    /'
echo "========================================"
