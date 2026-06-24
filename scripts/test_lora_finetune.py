#!/usr/bin/env python3
"""
LLaMA-Factory LoRA 微调测试脚本
"""
import json, os, sys, subprocess
from pathlib import Path

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")
os.chdir(WORK_DIR)

print("=" * 60)
print("Step 1: 创建测试数据集")
print("=" * 60)

samples = []
for f in sorted((WORK_DIR / "data/nli/rte_test_MR").glob("*.json"))[:2]:
    with open(f) as fh:
        for line in fh:
            if len(samples) >= 30: break
            obj = json.loads(line)
            samples.append({
                "instruction": f"Premise: {obj['premise']}\nHypothesis: {obj['hypothesis']}\nDoes the premise entail the hypothesis?",
                "output": "entailment" if obj["label"] == 0 else "not_entailment"
            })
    if len(samples) >= 30: break

train_path = WORK_DIR / "data" / "test_lora_train.json"
val_path = WORK_DIR / "data" / "test_lora_val.json"
json.dump(samples[:24], train_path.open("w"), ensure_ascii=False)
json.dump(samples[24:], val_path.open("w"), ensure_ascii=False)

# dataset_info.json
info = {"test_lora": {
    "file_name": "test_lora_train.json",
    "formatting": "alpaca",
    "columns": {"instruction": "instruction", "output": "output"}
}}
json.dump(info, (WORK_DIR / "data" / "dataset_info.json").open("w"), indent=2)
print(f"✅ 数据集: 24 train + 6 val (RTE NLI)")

print("\n" + "=" * 60)
print("Step 2: 创建训练配置")
print("=" * 60)

# 选 SmolLM2-135M 快速测试 (<300MB, 秒下)
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

yaml_content = f"""
model_name_or_path: {MODEL}
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,v_proj
lora_rank: 8
lora_alpha: 16
dataset: test_lora
template: default
cutoff_len: 256
max_samples: 24
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 5e-4
num_train_epochs: 2.0
logging_steps: 1
save_steps: 9999
output_dir: output/test_lora_output
report_to: none
overwrite_cache: true
ignore_data_skip: true
"""

yaml_path = WORK_DIR / "test_lora.yaml"
yaml_path.write_text(yaml_content)
print(f"✅ 配置已创建: {yaml_path}")
print(f"模型: {MODEL}")

print("\n" + "=" * 60)
print("Step 3: 启动训练 (约1-2分钟)...")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)],
    capture_output=True, text=True, timeout=600, cwd=WORK_DIR
)

# 打印最后的关键输出
lines = result.stdout.split("\n")
key_lines = [l for l in lines if any(k in l for k in
    ["100%", "train", "loss", "step", "saving", "Saved", " MODEL", "Checkpoint", "Best"])]

print("\n--- 训练摘要 ---")
for l in key_lines[-15:]:
    print(f"  {l.strip()}")

if result.returncode == 0:
    print("\n✅ LoRA 微调测试成功！训练流程正常！")
else:
    print(f"\n❌ 训练异常 (code={result.returncode})")
    stderr_lines = result.stderr.split("\n")[-10:]
    for l in stderr_lines:
        if "Error" in l or "error" in l or "Traceback" in l:
            print(f"  {l.strip()}")

# 清理
yaml_path.unlink(missing_ok=True)
train_path.unlink(missing_ok=True)
val_path.unlink(missing_ok=True)
print("\nDone!")
