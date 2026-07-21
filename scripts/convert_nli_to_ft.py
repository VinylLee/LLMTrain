#!/usr/bin/env python3
"""
将 NLI 数据集转换为 LLaMA-Factory 微调格式

用法:
  # 转换单个数据集目录
  python scripts/convert_nli_to_ft.py \
    --input data/nli/train/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json \
    --name snli_labeled \
    --split

  # 合并多个文件（推荐：labeled + augmented）
  python scripts/convert_nli_to_ft.py \
    --input data/nli/train/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json \
           data/nli/train/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json \
    --name snli_mixed \
    --split \
    --val-ratio 0.1

  # 指定输出目录
  python scripts/convert_nli_to_ft.py \
    --input data/nli/train/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json \
    --name snli_labeled \
    --output-dir data/ft_datasets

  # 指定prompt模板
  python scripts/convert_nli_to_ft.py \
    --input data/nli/train/snli_lr0.0037_gemma-3-4b-it-qat/labeled_data.json \
    --name snli_labeled \
    --prompt "Premise: {premise}\nHypothesis: {hypothesis}\nWhat is the relationship?"
"""
import json
import os
import argparse
import random
from pathlib import Path
from collections import Counter

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")

# 标签映射
LABEL_NAMES_3CLASS = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES_BINARY = {0: "entailment", 1: "not_entailment"}

# 默认instruction模板
INSTRUCTION_3CLASS = "Determine the natural language inference relation between the premise and hypothesis. Answer with exactly one label: entailment, neutral, or contradiction."
INSTRUCTION_BINARY = "Determine whether the premise entails the hypothesis. Answer with exactly one label: entailment or not_entailment."


def detect_binary_from_path(paths):
    """Auto-detect if inputs are RTE (binary) by checking paths for 'rte'."""
    for p in paths:
        parts = str(p).lower().replace('\\', '/').split('/')
        if any('rte' in part for part in parts):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="将NLI数据转为LLaMA-Factory微调格式")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="输入文件（JSONL格式，每行有premise, hypothesis, label字段）")
    parser.add_argument("--name", "-n", default="nli_dataset",
                        help="数据集名称（用于注册到dataset_info.json）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（默认: data/ft_datasets/<name>/）")
    parser.add_argument("--output-name", default=None,
                        help="输出文件名（不含扩展名；默认: full；拆分时用于生成 train/val 文件名前缀）")
    parser.add_argument("--split", "-s", action="store_true",
                        help="是否拆分为训练集和验证集")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例（默认0.1）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="是否打乱数据")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--instruction", type=str, default=None,
                        help="instruction文本（默认使用内置NLI指令）")
    parser.add_argument("--binary", action="store_true", default=None,
                        help="强制二分类模式（RTE风格: entailment/not_entailment）。默认自动检测")
    return parser.parse_args()


def load_jsonl(filepath):
    """加载JSONL文件"""
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "premise" in d and "hypothesis" in d and "label" in d:
                    samples.append(d)
            except json.JSONDecodeError:
                continue
    print(f"  📥 {Path(filepath).name}: {len(samples)} 条有效样本")
    return samples


def convert_to_alpaca(samples, instruction_text, label_names):
    """将NLI样本转换为Alpaca格式"""
    converted = []
    label_dist = Counter()

    for s in samples:
        label = s["label"]
        if isinstance(label, int):
            label_name = label_names.get(label, str(label))
        else:
            label_name = label  # 已经是字符串

        input_text = f"Premise: {s['premise'].strip()}\nHypothesis: {s['hypothesis'].strip()}"

        converted.append({
            "instruction": instruction_text,
            "input": input_text,
            "output": label_name
        })
        label_dist[label_name] += 1

    return converted, label_dist


def save_jsonl(samples, filepath):
    """保存为JSONL文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  💾 保存 {len(samples)} 条 → {filepath}")


def register_dataset(dataset_name, train_file, val_file=None, task_type="nli"):
    """注册数据集到dataset_info.json，同时保存task_type供下游使用"""
    info_path = WORK_DIR / "data" / "dataset_info.json"

    # 读取已有配置
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except:
            info = {}
    else:
        info = {}

    # 注册新数据集
    entry = {
        "file_name": str(train_file),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        },
    }
    if task_type != "nli":
        entry["task_type"] = task_type
    info[dataset_name] = entry

    if val_file:
        val_entry = {
            "file_name": str(val_file),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            },
        }
        if task_type != "nli":
            val_entry["task_type"] = task_type
        info[dataset_name + "_val"] = val_entry

    # 写回
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    ds_name = dataset_name
    task_label = "二分类(RTE)" if task_type == "nli-binary" else "三分类"
    print(f"  📝 已注册数据集 '{ds_name}' ({task_label}) → {info_path}")


def main():
    args = parse_args()
    random.seed(args.seed)

    # 判断是否RTE二分类模式：--binary 优先，未指定则自动检测
    is_binary = args.binary if args.binary is not None else detect_binary_from_path(args.input)
    task_type = "nli-binary" if is_binary else "nli"

    instruction_text = args.instruction or (INSTRUCTION_BINARY if is_binary else INSTRUCTION_3CLASS)
    label_names = LABEL_NAMES_BINARY if is_binary else LABEL_NAMES_3CLASS

    print(f"\n{'='*60}")
    print(f"  🔄 NLI → LLaMA-Factory 格式转换")
    print(f"  数据集: {args.name}")
    print(f"  任务类型: {'二分类(RTE)' if is_binary else '三分类'}")
    print(f"  标签: {', '.join(label_names.values())}")
    print(f"  Instruction: {instruction_text}")
    if args.output_name:
        print(f"  输出文件名: {args.output_name}")
    print(f"{'='*60}\n")

    # 1. 加载所有输入文件
    all_samples = []
    for filepath in args.input:
        path = Path(filepath)
        if not path.exists():
            print(f"  ⚠️ 文件不存在: {path}")
            continue
        samples = load_jsonl(path)
        all_samples.extend(samples)

    if not all_samples:
        print("  ❌ 没有有效样本")
        return

    print(f"\n  📊 总计: {len(all_samples)} 条")

    # 2. 可选限制样本数
    if args.max_samples and len(all_samples) > args.max_samples:
        all_samples = random.sample(all_samples, args.max_samples)
        print(f"  ✂️ 采样至 {args.max_samples} 条")

    # 3. 打乱
    if args.shuffle:
        random.shuffle(all_samples)

    # 4. 转换格式
    converted, label_dist = convert_to_alpaca(all_samples, instruction_text, label_names)
    print(f"\n  📈 标签分布: {dict(label_dist)}")

    # 5. 输出目录
    if args.output_dir:
        out_dir = WORK_DIR / args.output_dir / args.name
    else:
        out_dir = WORK_DIR / "data" / "ft_datasets" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6. 拆分或直接保存
    output_stem = args.output_name or "full"
    if args.split and len(converted) > 1:
        val_size = max(1, int(len(converted) * args.val_ratio))
        train_size = len(converted) - val_size

        train_data = converted[:train_size]
        val_data = converted[train_size:]

        train_path = out_dir / f"{output_stem}_train.json"
        val_path = out_dir / f"{output_stem}_val.json"

        save_jsonl(train_data, train_path)
        save_jsonl(val_data, val_path)

        print(f"\n  ✅ 拆分为: 训练集 {len(train_data)}条 + 验证集 {len(val_data)}条")

        # 注册数据集
        register_dataset(args.name, train_path, val_path, task_type)
    else:
        all_path = out_dir / f"{output_stem}.json"
        save_jsonl(converted, all_path)
        register_dataset(args.name, all_path, task_type=task_type)

    print(f"\n{'='*60}")
    print(f"  ✅ 完成！数据集: {args.name}")
    print(f"  路径: {out_dir}")
    print(f"  使用方式:")
    print(f"    llamafactory-cli train \\")
    print(f"      --dataset {args.name} \\")
    print(f"      --dataset_dir data/ft_datasets")
    if is_binary:
        print(f"  注意: 该数据集为RTE二分类格式（entailment/not_entailment）")
    else:
        print(f"  注意: 该数据集为三分类格式（entailment/neutral/contradiction）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
