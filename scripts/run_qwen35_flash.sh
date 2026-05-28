#!/usr/bin/env bash
# 使用 qwen3.5-flash-2026-02-23 (百炼) 串行运行所有数据集
# 从最小数据集开始跑，尽量在额度耗尽前多跑有效数据
set -euo pipefail

LLM="qwen3.5-flash-2026-02-23"
DELAY=0.1
OUTPUT_DIR="output"
RUN_ID=$(date +%Y%m%d_%H%M%S)
SAFE_LLM="${LLM// /_}"

echo "========================================"
echo "  Run: $RUN_ID"
echo "  LLM: $LLM"
echo "  Delay: ${DELAY}s"
echo "  Output: $OUTPUT_DIR/$RUN_ID/$SAFE_LLM/"
echo "========================================"

# 按数据量从小到大排列，优先跑小的
DATASETS=(
  "rte_test_MR"
  "sick_test_MR"
  "mnlim_test_MR"
  "mnlimm_test_MR"
  "snli_test_MR"
)

# ---- Step 1: Predict ----
for ds_dir in "${DATASETS[@]}";  do
  full_path="data/$ds_dir"
  [ -d "$full_path" ] || continue
  ds_name=$(basename "$full_path" | sed 's/_test_MR//')
  count=$(find "$full_path" -name '*.json' -type f | wc -l)
  total_lines=$(cat "$full_path"/*.json 2>/dev/null | wc -l)
  echo ">>> [predict] $ds_name ($count files, ~$total_lines lines)"

  conda run -n LLMTrain3.9 \
    python3 scripts/test_deepseek.py \
      --data-dir "$full_path" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$RUN_ID" \
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
  full_path="data/$ds_dir"
  [ -d "$full_path" ] || continue
  ds_name=$(basename "$full_path" | sed 's/_test_MR//')

  ds_output_dir="$OUTPUT_DIR/$RUN_ID/$SAFE_LLM/$ds_name"
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
  fi
  echo
done

# ---- Step 3: Evaluate all combined ----
echo ">>> [eval] ALL combined"
combined_all="$OUTPUT_DIR/$RUN_ID/$SAFE_LLM/all_combined.jsonl"
cat "$OUTPUT_DIR/$RUN_ID/$SAFE_LLM"/*/*_all.jsonl 2>/dev/null > "$combined_all" || true

if [ -s "$combined_all" ]; then
  conda run -n LLMTrain3.9 \
    python3 scripts/evaluate.py "$combined_all" \
      --report "$OUTPUT_DIR/$RUN_ID/$SAFE_LLM/all_report.txt" \
      --group-by _source mr_type \
      2>&1 | tail -5
fi

# ---- Summary ----
echo
echo "========================================"
echo "  Output: $OUTPUT_DIR/$RUN_ID/$SAFE_LLM/"
echo "  Reports:"
find "$OUTPUT_DIR/$RUN_ID" -name '*_report.txt' 2>/dev/null | sed 's/^/    /'
echo "========================================"
