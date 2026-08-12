#!/usr/bin/env bash
# 批量 NLI 实验入口：对指定 LLM 跑 SNLI/MNLIm/MNLImm/SICK 的 Original + MR 测试，
# 然后整理成逐用例数据并生成 Markdown 摘要。
#
# 输出结构:
#   output/<llm>/<OFFICIAL_DATASET>/original/<DATASET>_original.jsonl   # 每行=一个用例
#   output/<llm>/<OFFICIAL_DATASET>/MR/<mr>.jsonl                        # 每个 MR 一个文件
#   output/<llm>/<OFFICIAL_DATASET>/MR/<DATASET>_MR_all.jsonl            # MR 合并文件
#   output/<llm>/SUMMARY.md                                              # 逐数据集/逐MR准确率摘要
#
# Usage:
#   bash scripts/run_all.sh                                   # 默认 DeepSeek-v4-flash-0731
#   bash scripts/run_all.sh --llm DeepSeek-v4-flash-0731 --delay 0.1
#   bash scripts/run_all.sh --llm phi-4 --datasets snli,sick \
#       --api-url-env LMSTUDIO_BASE_URL --api-key-env LMSTUDIO_KEY
set -euo pipefail

LLM="DeepSeek-v4-flash-0731"
DELAY=0.1
OUTPUT_DIR="output"
DATASETS="snli,mnlim,mnlimm,sick"
ENV_FILE=".env"
API_URL_ENV=""
API_KEY_ENV=""
API_MODEL=""
MAX_SAMPLES=""
RESUME=""
REASONING_EFFORT=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm) LLM="$2"; shift 2 ;;
    --api-model) API_MODEL="$2"; shift 2 ;;
    --delay) DELAY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --env) ENV_FILE="$2"; shift 2 ;;
    --api-url-env) API_URL_ENV="$2"; shift 2 ;;
    --api-key-env) API_KEY_ENV="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --reasoning-effort) REASONING_EFFORT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SAFE_LLM="${LLM// /_}"
# 优先用绝对路径的 llmtrain310 python（无需激活 conda）；否则退回 conda run / python3
if [ -x /home/ubuntu/.conda/envs/llmtrain310/bin/python ]; then
  PY="/home/ubuntu/.conda/envs/llmtrain310/bin/python"
elif command -v conda >/dev/null 2>&1; then
  PY="conda run -n llmtrain310 python3"
else
  PY="python3"
fi

# 有 run-id 时输出位于 output/<run-id>/<llm>/...，否则直接 output/<llm>/...
if [ -n "$RUN_ID" ]; then
  OUT_LLM_DIR="$OUTPUT_DIR/$RUN_ID/$SAFE_LLM"
else
  OUT_LLM_DIR="$OUTPUT_DIR/$SAFE_LLM"
fi

echo "========================================"
echo "  NLI Batch Run (original + MR)"
echo "  LLM:       $LLM"
if [ -n "$API_MODEL" ]; then echo "  API model: $API_MODEL"; fi
echo "  Datasets:  $DATASETS"
echo "  Delay:     ${DELAY}s"
if [ -n "$MAX_SAMPLES" ]; then echo "  Max samples per file: $MAX_SAMPLES"; fi
if [ -n "$RUN_ID" ]; then echo "  Run id:    $RUN_ID"; fi
echo "  Output:    $OUT_LLM_DIR/"
echo "========================================"

# ---- Step 1: Original 测试 ----
echo ">>> [original] $DATASETS"
"$PY" scripts/test_llm.py \
  --data-dir data/nli/original_dataset \
  --test-type original \
  --datasets "$DATASETS" \
  --output-dir "$OUTPUT_DIR" \
  ${RUN_ID:+--run-id "$RUN_ID"} \
  --delay "$DELAY" \
  --llm "$LLM" \
  ${API_MODEL:+--api-model "$API_MODEL"} \
  ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
  ${REASONING_EFFORT:+--reasoning-effort "$REASONING_EFFORT"} \
  ${RESUME:+--resume} \
  --env "$ENV_FILE" \
  ${API_URL_ENV:+--api-url-env "$API_URL_ENV"} \
  ${API_KEY_ENV:+--api-key-env "$API_KEY_ENV"} \
  2>&1
echo

# ---- Step 2: MR 测试 ----
echo ">>> [mr] $DATASETS"
"$PY" scripts/test_llm.py \
  --data-dir data/nli/MR_testing \
  --test-type mr \
  --datasets "$DATASETS" \
  --output-dir "$OUTPUT_DIR" \
  ${RUN_ID:+--run-id "$RUN_ID"} \
  --delay "$DELAY" \
  --llm "$LLM" \
  ${API_MODEL:+--api-model "$API_MODEL"} \
  ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"} \
  ${REASONING_EFFORT:+--reasoning-effort "$REASONING_EFFORT"} \
  ${RESUME:+--resume} \
  --env "$ENV_FILE" \
  ${API_URL_ENV:+--api-url-env "$API_URL_ENV"} \
  ${API_KEY_ENV:+--api-key-env "$API_KEY_ENV"} \
  2>&1
echo

# ---- Step 3: 整理 + 合并 + 摘要 ----
echo ">>> [organize] 生成逐用例数据与 SUMMARY.md"
"$PY" scripts/organize_nli_experiment.py \
  --model "$SAFE_LLM" \
  --output-dir "$OUTPUT_DIR" \
  ${RUN_ID:+--run-id "$RUN_ID"} \
  --datasets "$DATASETS" \
  2>&1
echo

# ---- Step 4: 产物清单 ----
echo "========================================"
echo "  Output: $OUT_LLM_DIR/"
echo "========================================"
find "$OUT_LLM_DIR" -name '*.jsonl' -o -name 'SUMMARY.md' 2>/dev/null | sort
echo
echo "Summary: $OUT_LLM_DIR/SUMMARY.md"
