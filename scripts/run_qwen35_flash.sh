#!/usr/bin/env bash
# 从最小数据集开始串行跑 NLI 预测 + 评估
set -euo pipefail

LLM="qwen3.6-flash"
DELAY=0.1
OUTPUT_DIR="output"
TASK="nli"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm) LLM="$2"; shift 2 ;;
    --delay) DELAY="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

SAFE_LLM="${LLM// /_}"
DATA_DIR="data/$TASK"

echo "========================================"
echo "  Run"
echo "  LLM: $LLM"
echo "  Delay: ${DELAY}s"
echo "  Output: $OUTPUT_DIR/$SAFE_LLM/"
echo "========================================"

# 按数据量从小到大排列
DATASETS=(
  "rte_test_MR"
  "sick_test_MR"
  "mnlim_test_MR"
  "mnlimm_test_MR"
  "snli_test_MR"
)

# ---- Step 1: Predict ----
for ds_dir in "${DATASETS[@]}"; do
  full_path="$DATA_DIR/$ds_dir"
  [ -d "$full_path" ] || continue
  ds_name=$(basename "$full_path" | sed 's/_test_MR//')
  count=$(find "$full_path" -name '*.json' -type f | wc -l)
  total_lines=$(cat "$full_path"/*.json 2>/dev/null | wc -l)
  echo ">>> [predict] $ds_name ($count files, ~$total_lines lines)"

  conda run -n llmtrain310 \
    python3 scripts/test_llm.py \
      --data-dir "$full_path" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$SAFE_LLM" \
      --delay "$DELAY" \
      --llm "$LLM" \
      --api-url-env BAILIAN_BASE_URL \
      --api-key-env BAILIAN_API_KEY \
      2>&1

  echo "  [done] $ds_name"
  echo
done

# ---- Step 2: Evaluate per dataset ----
echo "========================================"
echo "  Evaluating..."
echo "========================================"

for ds_dir in "${DATASETS[@]}"; do
  full_path="$DATA_DIR/$ds_dir"
  [ -d "$full_path" ] || continue
  ds_name=$(basename "$full_path" | sed 's/_test_MR//')

  ds_output_dir="$OUTPUT_DIR/$SAFE_LLM/$ds_name"
  combined="$ds_output_dir/${ds_name}_all.jsonl"

  if [ -d "$ds_output_dir" ]; then
    find "$ds_output_dir" -maxdepth 1 -name '*.jsonl' ! -name '*_all.jsonl' \
      -exec cat {} + | sort -t'"' -k2,2n > "$combined" || true
  fi

  if [ -s "$combined" ]; then
    conda run -n llmtrain310 \
      python3 scripts/evaluate.py "$combined" \
        --report "$ds_output_dir/${ds_name}_report.txt" \
        2>&1 | tail -3
  fi
  echo
done

# ---- Step 3: Evaluate all combined ----
echo ">>> [eval] ALL combined"
combined_all="$OUTPUT_DIR/$SAFE_LLM/all_combined.jsonl"
cat "$OUTPUT_DIR/$SAFE_LLM"/*/*_all.jsonl 2>/dev/null > "$combined_all" || true

if [ -s "$combined_all" ]; then
  conda run -n llmtrain310 \
    python3 scripts/evaluate.py "$combined_all" \
      --report "$OUTPUT_DIR/$SAFE_LLM/all_report.txt" \
      --group-by _source mr_type \
      2>&1 | tail -5
fi

# ---- Summary ----
echo
echo "========================================"
echo "  Output: $OUTPUT_DIR/$SAFE_LLM/"
echo "  Reports:"
find "$OUTPUT_DIR/$SAFE_LLM" -name '*_report.txt' 2>/dev/null | sed 's/^/    /'
echo "========================================"
