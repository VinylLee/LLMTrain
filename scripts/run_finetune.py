#!/usr/bin/env python3
"""
LLaMA-Factory LoRA 微调启动脚本
用法:
  # 最简模式（Qwen2.5-7B + 你的数据集）
  python scripts/run_finetune.py --model qwen

  # 指定模型和数据集
  python scripts/run_finetune.py --model gemma --dataset snli_lr0.0037_gemma-3-4b-it-qat

  # 高级：自定义参数
  python scripts/run_finetune.py --model qwen --lr 2e-4 --epochs 5 --rank 16
"""
import json, subprocess, sys, os
from pathlib import Path

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")
os.chdir(WORK_DIR)

# 预设模型配置
MODELS = {
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "template": "qwen",
        "note": "MR测试最强(76.6%)，综合推荐",
        "size": "~15GB",
    },
    "gemma": {
        "name": "google/gemma-3-4b-it",
        "template": "gemma",
        "note": "原始数据最强(53.5%)，速度最快",
        "size": "~8GB",
    },
    "llama8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "template": "llama",
        "note": "通用均衡之选",
        "size": "~16GB",
    },
    "llama3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "template": "llama",
        "note": "轻量快速",
        "size": "~6GB",
    },
}

# 默认训练参数
DEFAULT_CONFIG = {
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "lora_target": "all",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "dataset": "snli_lr0.0037_gemma-3-4b-it-qat",
    "cutoff_len": 512,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "num_train_epochs": 3.0,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "load_best_model_at_end": True,
    # output_dir 会被 --output 覆盖，默认自动生成
    # 格式: output/experiments/{实验名}/model/
    "output_dir": None,
    "report_to": "none",
    "bf16": True,
}


def detect_binary_from_dataset(dataset_name):
    """从数据集名自动检测是否为RTE二分类。"""
    name_lower = dataset_name.lower()
    return "rte" in name_lower

def print_header():
    print("=" * 65)
    print("  🏋️  LLaMA-Factory LoRA 微调启动器")
    print("=" * 65)
    print()
    print("  可选模型:")
    for key, cfg in MODELS.items():
        print(f"    {key:<12} {cfg['name']:<40} {cfg['note']}")
    print()

def show_config(model_key, custom_args, task_type="nli"):
    mc = MODELS[model_key]
    config = DEFAULT_CONFIG.copy()
    config.update(custom_args)
    config["model_name_or_path"] = mc["name"]
    config["template"] = mc["template"]

    task_label = "二分类(RTE)" if task_type == "nli-binary" else "三分类"
    print(f"  📋 训练配置")
    print(f"  {'模型':<20} {mc['name']}")
    print(f"  {'数据集':<20} {config['dataset']}")
    print(f"  {'任务类型':<20} {task_label}")
    print(f"  {'方法':<20} LoRA (rank={config['lora_rank']})")
    print(f"  {'学习率':<20} {config['learning_rate']}")
    print(f"  {'Epochs':<20} {config['num_train_epochs']}")
    print(f"  {'Batch size':<20} {config['per_device_train_batch_size']}")
    print(f"  {'Gradient accum':<20} {config['gradient_accumulation_steps']}")
    print(f"  {'有效batch':<20} {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}")
    print(f"  {'输出目录':<20} {config['output_dir']}")
    print(f"  {'测试结果':<20} {config['output_dir'].replace('/model','/tests/mr')}")
    print()

    # 保存YAML（包含任务类型注释，供下游脚本读取）
    yaml_path = WORK_DIR / "ft_config.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"### LoRA Fine-tuning Config\n")
        f.write(f"# Model: {mc['name']}\n")
        f.write(f"# Dataset: {config['dataset']}\n")
        f.write(f"# Task type: {task_type}\n")
        f.write(f"# Generated: auto\n\n")
        for k, v in config.items():
            if isinstance(v, bool):
                f.write(f"{k}: {'true' if v else 'false'}\n")
            elif isinstance(v, str):
                f.write(f"{k}: {v}\n")
            else:
                f.write(f"{k}: {v}\n")

    # 同时保存元信息 JSON 供下游脚本读取
    meta_path = Path(config["output_dir"]).parent / "experiment_meta.json" if config["output_dir"] else WORK_DIR / "experiment_meta.json"
    if config["output_dir"]:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": mc["name"],
        "dataset": config["dataset"],
        "task_type": task_type,
        "output_dir": config["output_dir"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ 元信息已保存: {meta_path}")

    print(f"  ✅ 配置已保存: {yaml_path}")
    print()
    print(f"  🚀 运行命令:")
    print(f"     python -m llamafactory.cli train ft_config.yaml")
    print()
    return yaml_path

def run_training(yaml_path):
    print("=" * 65)
    print("  🚀 开始训练...")
    print("=" * 65)
    result = subprocess.run(
        [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)],
        cwd=WORK_DIR
    )
    return result.returncode

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=list(MODELS.keys()),
                        help="模型选择")
    parser.add_argument("--dataset", default=DEFAULT_CONFIG["dataset"],
                        help="数据集名称（已在dataset_info.json注册的）")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="学习率")
    parser.add_argument("--epochs", type=float, default=DEFAULT_CONFIG["num_train_epochs"],
                        help="训练轮数")
    parser.add_argument("--rank", type=int, default=DEFAULT_CONFIG["lora_rank"],
                        help="LoRA rank")
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG["per_device_train_batch_size"],
                        help="per device batch size")
    parser.add_argument("--output", default=DEFAULT_CONFIG["output_dir"],
                        help="输出目录")
    parser.add_argument("--run", action="store_true",
                        help="直接开始训练（不预览）")
    parser.add_argument("--task-type", default=None, choices=["nli", "nli-binary"],
                        help="任务类型: nli(三分类) 或 nli-binary(RTE二分类)。默认从数据集名自动检测")
    args = parser.parse_args()

    print_header()

    # 任务类型: --task-type 优先，未指定则从数据集名自动检测
    task_type = args.task_type or ("nli-binary" if detect_binary_from_dataset(args.dataset) else "nli")

    custom = {
        "dataset": args.dataset,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "lora_rank": args.rank,
        "per_device_train_batch_size": args.batch,
        "output_dir": args.output,
    }
    yaml_path = show_config(args.model, custom, task_type)

    if args.run:
        sys.exit(run_training(yaml_path))
    else:
        print("  💡 加 --run 参数直接启动训练")
        print()
