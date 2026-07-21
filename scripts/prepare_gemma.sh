#!/usr/bin/env bash
# Gemma-3-4B 下载与微调准备脚本
# 使用: bash scripts/prepare_gemma.sh <your_hf_token>

set -e

WORK_DIR="/home/ubuntu/LLMTrain/LLMTrain"
cd "$WORK_DIR"

TOKEN="$1"

if [ -z "$TOKEN" ]; then
    echo "================================================"
    echo "  📋 Gemma-3-4B 下载准备"
    echo "================================================"
    echo ""
    echo "  步骤 1: 访问 https://huggingface.co/google/gemma-3-4b-it"
    echo "  步骤 2: 点击 'Agree and access repository' 同意协议"
    echo "  步骤 3: 访问 https://huggingface.co/settings/tokens"
    echo "  步骤 4: 创建一个 token (或复制已有的)"
    echo "  步骤 5: 运行: bash scripts/prepare_gemma.sh <你的token>"
    echo ""
    exit 1
fi

echo "✅ Token 已提供: ${TOKEN:0:8}..."
echo ""

# 登录HF
echo "🔄 登录 HuggingFace..."
huggingface-cli login --token "$TOKEN"

# 下载模型
echo ""
echo "🔄 下载 Gemma-3-4B-it (约 8GB)..."
echo "    可能需要几分钟..."
huggingface-cli download google/gemma-3-4b-it --local-dir-use-symlinks False

# 更新配置文件
echo ""
echo "🔄 更新微调配置..."
cat > ft_config_gemma.yaml << 'YAML'
### LoRA Fine-tuning Config for Gemma-3-4B
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
dataset: snli_lr0.0037_gemma-3-4b-it-qat
cutoff_len: 512
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 0.0003
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
logging_steps: 10
save_steps: 500
eval_strategy: "no"
output_dir: output/ft_gemma_output
report_to: none
bf16: true
trust_remote_code: true
remove_unused_columns: false
model_name_or_path: google/gemma-3-4b-it
template: gemma
YAML

echo ""
echo "================================================"
echo "  ✅ 准备完成！"
echo "================================================"
echo "  模型: google/gemma-3-4b-it"
echo "  配置: ft_config_gemma.yaml"
echo ""
echo "  启动训练:"
echo "    conda activate llmtrain310"
echo "    CUDA_VISIBLE_DEVICES=0 python -m llamafactory.cli train ft_config_gemma.yaml"
echo ""
