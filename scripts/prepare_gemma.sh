#!/usr/bin/env bash
# Gemma-3-4B 下载与微调准备脚本
# 使用: bash scripts/prepare_gemma.sh

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"

if ! hf auth whoami >/dev/null 2>&1; then
    echo "================================================"
    echo "  📋 Gemma-3-4B 下载准备"
    echo "================================================"
    echo ""
    echo "  步骤 1: 访问 https://huggingface.co/google/gemma-3-4b-it"
    echo "  步骤 2: 点击 'Agree and access repository' 同意协议"
    echo "  步骤 3: 在当前环境运行: hf auth login"
    echo "  步骤 4: 重新运行: bash scripts/prepare_gemma.sh"
    echo ""
    exit 1
fi

# 下载模型
echo ""
echo "🔄 下载 Gemma-3-4B-it (约 8GB)..."
echo "    可能需要几分钟..."
python scripts/cache_hf_model.py --model google/gemma-3-4b-it

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
echo "    python scripts/run_finetune.py --model gemma --cuda 0 --run"
echo ""
