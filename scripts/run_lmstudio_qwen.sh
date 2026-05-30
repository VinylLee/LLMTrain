#!/bin/bash
# 使用本地LM Studio的Qwen3.6进行NLI测试

set -e

# 配置
LM_STUDIO_URL="http://localhost:1234"
MODEL_NAME="${1:-qwen2.5-3b}"
DATA_DIR="${2:-data}"
OUTPUT_DIR="output"
DELAY="${3:-0.5}"

echo "============================================================"
echo "LM Studio + Qwen Local Test"
echo "============================================================"
echo "URL:           $LM_STUDIO_URL"
echo "Model:         $MODEL_NAME"
echo "Data Dir:      $DATA_DIR"
echo "Output Dir:    $OUTPUT_DIR"
echo "Request Delay: ${DELAY}s"
echo ""

# 检查LM Studio连接
echo "1. Testing LM Studio connectivity..."
python3 test.py
if [ $? -ne 0 ]; then
    echo "❌ LM Studio not reachable. Please start LM Studio and load a model."
    exit 1
fi
echo ""

# 运行测试
echo "2. Starting NLI prediction tests..."
python scripts/test_deepseek.py \
    --llm "$MODEL_NAME" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --delay "$DELAY"

echo ""
echo "✅ Tests completed!"
echo "Results saved to: $OUTPUT_DIR"
