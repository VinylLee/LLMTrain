#!/usr/bin/env python3
"""
统一实验入口：微调 → MR测试 → 原始数据测试
结果自动存入 output/experiments/{实验名}/

用法:
  # 微调 + 测试
  python scripts/run_experiment.py --name my_experiment --model gemma --dataset my_dataset

  # 仅测试已有模型
  python scripts/run_experiment.py --name mettrain_gemma3_4b --test-only

  # 指定微调参数
  python scripts/run_experiment.py --name test --model gemma --dataset snli_original_5340 --lr 5e-4 --epochs 5
"""
import subprocess, sys, os, json, argparse
from pathlib import Path

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")
os.chdir(WORK_DIR)

MODELS = {
    "gemma": {"name": "google/gemma-3-4b-it", "template": "gemma"},
    "qwen":  {"name": "Qwen/Qwen2.5-7B-Instruct", "template": "qwen"},
}

def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, cwd=WORK_DIR)
    if result.returncode != 0:
        print(f"  ❌ {desc} 失败 (code={result.returncode})")
        sys.exit(1)
    print(f"  ✅ {desc} 完成")

def detect_binary_from_dataset(dataset_name):
    """从数据集名自动检测是否为RTE二分类。"""
    name_lower = dataset_name.lower()
    return "rte" in name_lower


def main():
    parser = argparse.ArgumentParser(description="统一实验入口")
    parser.add_argument("--name", required=True, help="实验名 (例如 my_experiment)")
    parser.add_argument("--model", default="gemma", choices=list(MODELS.keys()))
    parser.add_argument("--dataset", default=None, help="数据集名 (dataset_info.json 中注册的)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--test-only", action="store_true", help="仅测试已有模型（跳过微调）")
    parser.add_argument("--skip-test", action="store_true", help="仅微调，跳过测试")
    parser.add_argument("--task-type", default=None, choices=["nli", "nli-binary"],
                        help="任务类型: nli(三分类) 或 nli-binary(RTE二分类)。默认从数据集名自动检测")
    args = parser.parse_args()

    exp_dir = WORK_DIR / "output" / "experiments" / args.name
    model_dir = exp_dir / "model"
    tests_mr_dir = exp_dir / "tests" / "mr"
    tests_orig_dir = exp_dir / "tests" / "original"

    # 任务类型: --task-type 优先，未指定则从数据集名自动检测
    task_type = args.task_type
    if task_type is None and args.dataset:
        task_type = "nli-binary" if detect_binary_from_dataset(args.dataset) else "nli"
    elif task_type is None:
        task_type = "nli"
    task_label = "二分类(RTE)" if task_type == "nli-binary" else "三分类"

    print(f"\n{'='*60}")
    print(f"  🔬 实验: {args.name}")
    print(f"  任务类型: {task_label}")
    print(f"{'='*60}\n")

    # 保存实验元信息
    exp_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": args.name,
        "model": MODELS[args.model]["name"] if not args.test_only else "N/A",
        "dataset": args.dataset,
        "task_type": task_type,
        "model_dir": str(model_dir),
    }
    with open(exp_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if not args.test_only:
        if not args.dataset:
            print("❌ 请指定 --dataset")
            sys.exit(1)

        mc = MODELS[args.model]
        yaml_content = f"""### LoRA Fine-tuning: {args.name}
# Task type: {task_type}
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {args.rank}
lora_alpha: {args.rank * 2}
lora_dropout: 0.05
dataset: {args.dataset}
cutoff_len: 512
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: {args.lr}
num_train_epochs: {args.epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
logging_steps: 10
save_steps: 9999
eval_strategy: "no"
output_dir: {model_dir}
report_to: none
bf16: true
trust_remote_code: true
remove_unused_columns: false
model_name_or_path: {mc['name']}
template: {mc['template']}
use_cache: false
"""
        yaml_path = WORK_DIR / f"ft_config_{args.name}.yaml"
        yaml_path.write_text(yaml_content)

        run_cmd(
            f"CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 "
            f"python -m llamafactory.cli train {yaml_path}",
            f"🏋️ 微调 {args.name} ({mc['name']} + {args.dataset}, {task_label})",
        )
        yaml_path.unlink(missing_ok=True)

    if args.skip_test:
        return

    # 测试原始数据
    tests_orig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 原始数据测试 → {tests_orig_dir}")
    run_cmd(
        f"CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 python scripts/test_ft_model.py "
        f"--experiment {args.name} "
        f"--base-model {MODELS[args.model]['name']} "
        f"{'--model-task binary' if task_type == 'nli-binary' else ''} 2>&1",
        f"🧪 测试原始数据 ({task_label})",
    )

    # 测试 MR 数据（通过 test_mettrain_experiment.py）
    tests_mr_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📊 MR 测试 → {tests_mr_dir}")
    run_cmd(
        f"CUDA_VISIBLE_DEVICES=0 TORCH_COMPILE_DISABLE=1 python scripts/test_mettrain_experiment.py "
        f"--experiment {args.name} "
        f"--base-model {MODELS[args.model]['name']}",
        f"🧪 测试MR数据 ({task_label})",
    )

    # 生成测试结果摘要
    print(f"\n✅ 实验 {args.name} 完成")
    print(f"  LoRA: {model_dir}")
    print(f"  任务类型: {task_label}")
    print(f"  MR测试: {tests_mr_dir}")
    print(f"  原始测试: {tests_orig_dir}")

if __name__ == "__main__":
    main()
