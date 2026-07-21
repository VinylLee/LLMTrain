#!/usr/bin/env python3
"""
批量实验编排器：采样 → 转换 → 微调 → 测试全自动流水线

读取 experiments_config.json（或自定义配置文件），对其中定义的一个或多个
训练数据集依次完成完整实验流程，并在原始 + MR 测试集上评估。

用法:
  # 跑全部实验 × 3种子 = 24次
  python scripts/run_batch_experiments.py --config experiments_config.json --seeds 42 43 44

  # 先只跑2个实验验证
  python scripts/run_batch_experiments.py --config experiments_config.json --only mettrain_mnlim_4413_gemma3_4b
  original_snli_5340_gemma3_4b --seeds 42 43 44 --dry-run

  # 使用分层采样
  python scripts/run_batch_experiments.py --config experiments_config.json --seeds 42 43 44 --stratify

  # 跑全部实验
  python scripts/run_batch_experiments.py --config experiments_config.json

  # 只跑原始数据实验
  python scripts/run_batch_experiments.py --config experiments_config.json --only original_snli_5340_gemma3_4b
  original_mnlim_4413_gemma3_4b original_sick_4439_gemma3_4b

  # 只跑 MetTrain 数据实验
  python scripts/run_batch_experiments.py --config experiments_config.json --only mettrain_snli_5340_gemma3_4b
  mettrain_mnlim_4413_gemma3_4b mettrain_sick_4439_gemma3_4b

  # 指定部分实验
  python scripts/run_batch_experiments.py --config experiments_config.json --only mettrain_mnlim_4413_gemma3_4b

  # 干跑（只看命令不执行）
  python scripts/run_batch_experiments.py --config experiments_config.json --dry-run

  # 断点续跑
  python scripts/run_batch_experiments.py --config experiments_config.json --resume

  # 覆盖模型
  python scripts/run_batch_experiments.py --config experiments_config.json --model qwen
"""
import json
import subprocess
import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")
DEFAULT_OUTPUT_ROOT = WORK_DIR / "output" / "experiments"

# 步骤名称
STEPS = ["sample", "convert", "finetune", "test_original", "test_mr"]

# 跟踪已写入的 cohort manifest（同一 run 内的首批变体写入，后续复用）
_cohort_manifest_written = set()


def load_config(path):
    """加载 JSON 配置文件"""
    with open(path) as f:
        return json.load(f)


def resolve_workspace_path(path_value):
    """Resolve a config/CLI path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else WORK_DIR / path


def load_progress(progress_file):
    """加载进度文件（如果存在）"""
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {}


def save_progress(progress_file, progress):
    """保存进度文件"""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


def step_completed(progress, exp_name, step):
    """检查某实验的某步骤是否已完成"""
    return progress.get(exp_name, {}).get(step) == "completed"


def mark_step(progress_file, progress, exp_name, step, status):
    """标记某步骤状态"""
    if exp_name not in progress:
        progress[exp_name] = {}
    progress[exp_name][step] = status
    save_progress(progress_file, progress)


def run_cmd(cmd, desc, dry_run=False, cwd=None):
    """运行命令或打印（dry-run模式）"""
    print(f"\n  ▶ {'[DRY-RUN]' if dry_run else 'RUN'} {desc}")
    print(f"    {cmd}")
    if dry_run:
        return True
    result = subprocess.run(cmd, shell=True, cwd=cwd or WORK_DIR)
    if result.returncode != 0:
        print(f"    ❌ {desc} 失败 (code={result.returncode})")
        return False
    print(f"    ✅ {desc} 完成")
    return True


def build_yaml(exp, model_name, template, output_root):
    """生成微调 YAML 配置内容"""
    p = exp.get("ft_params", {})
    return f"""### LoRA Fine-tuning: {exp['name']}
# Task type: {exp['task_type']}
# Generated: {datetime.now().isoformat()}
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {p.get('rank', 8)}
lora_alpha: {p.get('rank', 8) * 2}
lora_dropout: 0.05
dataset: {exp['name']}
cutoff_len: 512
per_device_train_batch_size: {p.get('batch', 4)}
gradient_accumulation_steps: {p.get('grad_accum', 8)}
learning_rate: {p.get('lr', 3e-4)}
num_train_epochs: {p.get('epochs', 3.0)}
lr_scheduler_type: cosine
warmup_ratio: 0.1
logging_steps: 10
save_steps: 9999
eval_strategy: "no"
output_dir: {output_root / exp['name'] / 'model'}
report_to: none
bf16: true
trust_remote_code: true
remove_unused_columns: false
model_name_or_path: {model_name}
template: {template}
use_cache: false
"""


def save_experiment_meta(exp, model_name, template, output_root, config):
    """保存实验元信息 JSON"""
    meta_dir = output_root / exp["name"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": exp["name"],
        "model": model_name,
        "requested_model": config.get("requested_model", model_name),
        "model_source_note": config.get("model_source_note"),
        "template": template,
        "dataset": exp["train_data"],
        "target": exp["target"],
        "task_type": exp["task_type"],
        "seed": exp.get("seed"),
        "ft_params": exp.get("ft_params", {}),
        "seed_scope": config.get("seed_scope", "sampling"),
        "output_root": str(output_root),
        # MR-instruction 相关
        "mr_instruction_mode": exp.get("mr_instruction_mode", "none"),
        "cohort_id": exp.get("cohort_id"),
        "strict_pairing": exp.get("strict_pairing", False),
    }
    with open(meta_dir / "experiment_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_experiment(exp, config, output_root, progress_file, selected_steps,
                   dry_run=False, resume=False):
    """运行单个实验的完整流水线"""
    name = exp["name"]
    model_name = config["model"]
    template = config.get("template", "gemma")
    cuda = config.get("cuda", "2")
    task_type = exp["task_type"]
    is_binary = task_type == "nli-binary"
    progress = load_progress(progress_file)

    # 测试集范围由 config 的 test_sets 决定（统一 original+mr 的数据集键）
    ts = config.get("test_sets", {})
    test_ds = sorted(set(ts.get("original", {}).keys()) | set(ts.get("mr", {}).keys()))
    datasets_flag = f"--datasets {','.join(test_ds)}" if test_ds else ""
    batch_flag = f"--batch-size {config.get('batch_size', 32)}"

    print(f"\n{'='*70}")
    print(f"  📦 实验: {name}")
    print(f"  模型: {model_name}")
    print(f"  训练数据: {exp['train_data']}")
    print(f"  目标采样: {exp['target']}")
    print(f"  任务类型: {'二分类' if is_binary else '三分类'}")
    print(f"{'='*70}")

    # Step 0: 准备目录
    ft_data_dir = WORK_DIR / "data" / "ft_datasets" / name
    sampled_path = ft_data_dir / "sampled.json"
    exp_dir = output_root / name
    model_dir = exp_dir / "model"
    if not dry_run:
        ft_data_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["tests/original", "tests/mr"]:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        save_experiment_meta(exp, model_name, template, output_root, config)

    # ============================================================
    # Step 1: 采样
    # ============================================================
    print(f"\n  ── Step 1/5: 采样 ──")
    if "sample" not in selected_steps:
        print(f"    ⏩ 跳过（本次未选择该阶段）")
    elif resume and step_completed(progress, name, "sample"):
        print(f"    ⏩ 跳过（已完成）")
    else:
        stratify_flag = "--stratify" if exp.get("stratify") else ""
        ok = run_cmd(
            f"python scripts/sample_mettrain_pairid.py "
            f"--input {exp['train_data']} "
            f"--target {exp['target']} "
            f"--seed {exp.get('seed', 42)} "
            f"{stratify_flag} "
            f"--output {sampled_path}",
            f"采样 {name} (target={exp['target']})",
            dry_run,
        )
        if not dry_run:
            mark_step(progress_file, progress, name, "sample", "completed" if ok else "failed")
        if not ok and not dry_run:
            return False

    # ============================================================
    # Step 2: 转换 + 注册
    # ============================================================
    print(f"\n  ── Step 2/5: 转换 → Alpaca 格式 ──")
    if "convert" not in selected_steps:
        print(f"    ⏩ 跳过（本次未选择该阶段）")
    elif resume and step_completed(progress, name, "convert"):
        print(f"    ⏩ 跳过（已完成）")
    else:
        # MR-instruction mode
        mr_mode = exp.get("mr_instruction_mode", "none")
        mr_mode_flag = f"--mr-instruction-mode {mr_mode}"

        # Strict pairing
        strict_flag = "--strict-pairing" if exp.get("strict_pairing", False) else ""

        # Cohort-based split manifest sharing
        cohort_id = exp.get("cohort_id")
        manifest_flag = ""
        if cohort_id:
            # Manifest 放在 cohort 的 ft_datasets 目录下（以 cohort 中第一个实验名作为基础路径）
            cohort_first = cohort_id
            manifest_path = WORK_DIR / "data" / "ft_datasets" / cohort_first / "split_manifest.json"

            if cohort_id not in _cohort_manifest_written:
                # 第一个变体：写入 manifest
                manifest_flag = f"--write-split-manifest {manifest_path}"
                _cohort_manifest_written.add(cohort_id)
            else:
                # 后续变体：复用 manifest
                manifest_flag = f"--split-manifest {manifest_path}"

        binary_flag = "--binary" if is_binary else ""
        ok = run_cmd(
            f"python scripts/convert_nli_to_ft.py "
            f"--input {sampled_path} "
            f"--name {name} "
            f"--split "
            f"--val-ratio 0.05 "
            f"{binary_flag} "
            f"{mr_mode_flag} "
            f"{strict_flag} "
            f"{manifest_flag}",
            f"转换 {name} (mode={mr_mode})",
            dry_run,
        )
        if not dry_run:
            mark_step(progress_file, progress, name, "convert", "completed" if ok else "failed")
        if not ok and not dry_run:
            return False

    # ============================================================
    # Step 3: 微调
    # ============================================================
    print(f"\n  ── Step 3/5: LoRA 微调 ──")
    if "finetune" not in selected_steps:
        print(f"    ⏩ 跳过（本次未选择该阶段）")
    elif resume and step_completed(progress, name, "finetune"):
        print(f"    ⏩ 跳过（已完成）")
    else:
        yaml_content = build_yaml(exp, model_name, template, output_root)
        yaml_path = WORK_DIR / f"ft_config_{name}.yaml"
        if not dry_run:
            yaml_path.write_text(yaml_content)

        ok = run_cmd(
            f"CUDA_VISIBLE_DEVICES={cuda} TORCH_COMPILE_DISABLE=1 "
            f"python -m llamafactory.cli train {yaml_path}",
            f"微调 {name}",
            dry_run,
        )
        if not dry_run:
            yaml_path.unlink(missing_ok=True)
        if not dry_run:
            mark_step(progress_file, progress, name, "finetune", "completed" if ok else "failed")
        if not ok and not dry_run:
            return False

    # ============================================================
    # Step 4: 测试原始数据集
    # ============================================================
    print(f"\n  ── Step 4/5: 测试原始数据集 ──")
    if "test_original" not in selected_steps:
        print(f"    ⏩ 跳过（本次未选择该阶段）")
    elif resume and step_completed(progress, name, "test_original"):
        print(f"    ⏩ 跳过（已完成）")
    else:
        model_task_val = "binary" if task_type == "nli-binary" else "3class"
        model_task_flag = f"--model-task {model_task_val}" if is_binary else ""
        ok = run_cmd(
            f"CUDA_VISIBLE_DEVICES={cuda} TORCH_COMPILE_DISABLE=1 "
            f"python scripts/test_mettrain_experiment.py "
            f"--experiment {name} "
            f"--base-model {model_name} "
            f"--lora {model_dir} "
            f"--output-root {output_root} "
            f"{model_task_flag} "
            f"{datasets_flag} "
            f"{batch_flag} "
            f"--skip-mr",
            f"测试原始数据 {name}",
            dry_run,
        )
        if not dry_run:
            mark_step(progress_file, progress, name, "test_original", "completed" if ok else "failed")
        if not ok and not dry_run:
            return False

    # ============================================================
    # Step 5: 测试 MR 数据集
    # ============================================================
    print(f"\n  ── Step 5/5: 测试 MR 数据集 ──")
    if "test_mr" not in selected_steps:
        print(f"    ⏩ 跳过（本次未选择该阶段）")
    elif resume and step_completed(progress, name, "test_mr"):
        print(f"    ⏩ 跳过（已完成）")
    else:
        model_task_val = "binary" if task_type == "nli-binary" else "3class"
        model_task_flag = f"--model-task {model_task_val}" if is_binary else ""
        ok = run_cmd(
            f"CUDA_VISIBLE_DEVICES={cuda} TORCH_COMPILE_DISABLE=1 "
            f"python scripts/test_mettrain_experiment.py "
            f"--experiment {name} "
            f"--base-model {model_name} "
            f"--lora {model_dir} "
            f"--output-root {output_root} "
            f"{model_task_flag} "
            f"{datasets_flag} "
            f"{batch_flag} "
            f"--skip-original",
            f"测试MR数据 {name}",
            dry_run,
        )
        if not dry_run:
            mark_step(progress_file, progress, name, "test_mr", "completed" if ok else "failed")
        if not ok and not dry_run:
            return False

    print(f"\n  ✅ 实验 {name} 全部完成！")
    return True


def collect_summary(output_root, summary_file):
    """从已完成实验中收集测试结果，生成 SUMMARY.md"""
    exp_base = output_root
    summaries = []

    for exp_dir in sorted(exp_base.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        meta_file = exp_dir / "experiment_meta.json"
        if not meta_file.exists():
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        # 读取原始测试结果
        original_results = {}
        orig_dir = exp_dir / "tests" / "original"
        if orig_dir.exists():
            for ds_file in sorted(orig_dir.glob("*.jsonl")):
                ds_name = ds_file.stem
                total = correct = 0
                with open(ds_file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                r = json.loads(line)
                                total += 1
                                if r.get("correct"):
                                    correct += 1
                            except json.JSONDecodeError:
                                continue
                if total > 0:
                    original_results[ds_name] = {
                        "acc": f"{correct/total*100:.2f}%",
                        "correct": correct,
                        "total": total,
                    }

        # 读取 MR 测试结果
        mr_results = {}
        mr_dir = exp_dir / "tests" / "mr"
        if mr_dir.exists():
            for ds_file in sorted(mr_dir.glob("*.jsonl")):
                ds_name = ds_file.stem
                total = correct = 0
                with open(ds_file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                r = json.loads(line)
                                total += 1
                                if r.get("correct"):
                                    correct += 1
                            except json.JSONDecodeError:
                                continue
                if total > 0:
                    mr_results[ds_name] = {
                        "acc": f"{correct/total*100:.2f}%",
                        "correct": correct,
                        "total": total,
                    }

        if original_results or mr_results:
            summaries.append({
                "name": exp_dir.name,
                "task_type": meta.get("task_type", "nli"),
                "original": original_results,
                "mr": mr_results,
            })

    # 写 SUMMARY.md
    all_ds = ["mnlim", "mnlimm", "sick", "snli"]
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as f:
        f.write("# 批量实验测试结果汇总\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for s in summaries:
            f.write(f"## {s['name']} ({'二分类' if s['task_type'] == 'nli-binary' else '三分类'})\n\n")

            if s["original"]:
                f.write("### Original\n\n")
                f.write("| 数据集 | 正确/总数 | 准确率 |\n")
                f.write("|--------|----------|--------|\n")
                for ds in all_ds:
                    if ds in s["original"]:
                        r = s["original"][ds]
                        f.write(f"| {ds} | {r['correct']}/{r['total']} | {r['acc']} |\n")
                f.write("\n")

            if s["mr"]:
                f.write("### MR\n\n")
                f.write("| 数据集 | 正确/总数 | 准确率 |\n")
                f.write("|--------|----------|--------|\n")
                for ds in all_ds:
                    if ds in s["mr"]:
                        r = s["mr"][ds]
                        f.write(f"| {ds} | {r['correct']}/{r['total']} | {r['acc']} |\n")
                f.write("\n")

            f.write("---\n\n")

    print(f"\n  📊 汇总报告: {summary_file}")
    return summaries


def collect_test_results(exp_dir):
    """读取一个实验目录下所有测试 JSONL 文件，返回 {test_type: {ds: acc}}"""
    results = {}
    for test_type in ["original", "mr"]:
        results[test_type] = {}
        tdir = exp_dir / "tests" / test_type
        if tdir.exists():
            for ds_file in sorted(tdir.glob("*.jsonl")):
                ds_name = ds_file.stem
                total = correct = 0
                with open(ds_file) as f:
                    for line in f:
                        if line.strip():
                            try:
                                r = json.loads(line)
                                total += 1
                                if r.get("correct"):
                                    correct += 1
                            except json.JSONDecodeError:
                                continue
                if total > 0:
                    results[test_type][ds_name] = correct / total * 100
    return results


def aggregate_results(seeds, config, output_root, result_file, expected_names):
    """聚合多 seed 实验结果，生成 RESULTS.md 完整矩阵表格。

    对 original 和 mr 各输出一张表：列为全部实验（mettrain + original），
    行为该测试类型下的数据集（取自 config 的 test_sets），值为多 seed 均值±标准差。
    """
    exp_base = output_root
    import math

    # 收集所有 seed 实验目录
    seed_dirs = {}
    for d in exp_base.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        # 匹配 *_seed{seed} 模式
        for s in seeds:
            suffix = f"_seed{s}"
            if d.name.endswith(suffix):
                base_name = d.name[:-len(suffix)]
                if base_name not in expected_names:
                    break
                if base_name not in seed_dirs:
                    seed_dirs[base_name] = {}
                seed_dirs[base_name][s] = d
                break

    if not seed_dirs:
        print("  ⚠️  未找到多 seed 实验结果，跳过聚合")
        return

    # 实验列顺序：先 mettrain 后 original，各自按名排序
    exp_names = (
        sorted(n for n in seed_dirs if n.startswith("mettrain"))
        + sorted(n for n in seed_dirs if n.startswith("original"))
    )

    # 数据集行：从 config 的 test_sets 读取（排除 rte 等）
    ts = config.get("test_sets", {})
    original_ds = list(ts.get("original", {}).keys())
    mr_ds = list(ts.get("mr", {}).keys())

    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        f.write("# 实验结果汇总\n\n")
        seed_str = ", ".join(str(s) for s in sorted(seeds))
        f.write(f"种子: {seed_str}  ")
        f.write(f"模型: {config.get('model', 'google/gemma-3-4b-it')}\n\n")

        for section_title, ds_list, test_type_label in [
            ("Original (acc)", original_ds, "original"),
            ("MR (acc)", mr_ds, "mr"),
        ]:
            if not ds_list:
                continue
            f.write(f"## {section_title}\n\n")

            # Header row
            f.write("| 测试集 |")
            for en in exp_names:
                f.write(f" {en} |")
            f.write("\n")
            f.write("|--------|")
            for _ in exp_names:
                f.write("-------------|")
            f.write("\n")

            # Data rows
            for ds in ds_list:
                f.write(f"| {ds} |")
                for en in exp_names:
                    accs = []
                    for s in sorted(seeds):
                        exp_dir = seed_dirs[en].get(s)
                        if exp_dir:
                            results = collect_test_results(exp_dir)
                            if test_type_label in results and ds in results[test_type_label]:
                                accs.append(results[test_type_label][ds])
                    if accs:
                        mean = sum(accs) / len(accs)
                        if len(accs) > 1:
                            variance = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                            std = math.sqrt(variance)
                            f.write(f" {mean:.2f} ± {std:.2f}% |")
                        else:
                            f.write(f" {mean:.2f}% |")
                    else:
                        f.write(" — |")
                f.write("\n")
            f.write("\n")

        f.write("---\n\n")
        f.write("> 准确率格式: 均值 ± 样本标准差 (分母 n-1，多seed时)\n")

    print(f"\n  📊 聚合报告: {result_file}")


def main():
    parser = argparse.ArgumentParser(
        description="批量实验编排器：采样 → 转换 → 微调 → 测试"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="配置文件路径 (JSON)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="仅运行指定的实验名（空格分隔）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式：只打印命令，不执行",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：跳过已完成的实验和步骤",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖模型名称 (例如 qwen, gemma)",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="覆盖 chat template (例如 qwen, gemma, llama)",
    )
    parser.add_argument(
        "--cuda",
        default=None,
        help="覆盖 CUDA 设备号",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="跳过最终汇总",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="随机种子列表 (例如 42 43 44)，默认只跑 seed=42",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="分层采样：传递给 sample_mettrain_pairid.py 的 --stratify 参数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="测试阶段批量推理大小（0=逐条）；默认 32，可显著缩短测试时间",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=STEPS,
        default=STEPS,
        help="只执行指定阶段；默认执行全部五个阶段",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="覆盖实验输出根目录；默认读取 config.output_root 或 output/experiments",
    )
    parser.add_argument(
        "--progress-file",
        default=None,
        help="覆盖进度文件；默认使用 <output-root>/_progress.json",
    )
    parser.add_argument(
        "--result-file",
        default=None,
        help="覆盖多 seed 汇总；默认使用 <output-root>/RESULTS.md",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(args.config)

    # 模型已缓存在本地 HF hub，且当前环境无法访问 huggingface.co（SSL EOF）。
    # 强制离线模式，避免 finetune/test 子进程因联网校验 tokenizer 而失败。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # CLI 覆盖配置
    if args.model:
        config["model"] = args.model
    if args.template:
        config["template"] = args.template
    if args.cuda:
        config["cuda"] = args.cuda
    config["batch_size"] = args.batch_size

    output_root = resolve_workspace_path(
        args.output_root or config.get("output_root", DEFAULT_OUTPUT_ROOT)
    )
    progress_file = resolve_workspace_path(
        args.progress_file or config.get("progress_file", output_root / "_progress.json")
    )
    result_file = resolve_workspace_path(
        args.result_file or config.get("result_file", output_root / "RESULTS.md")
    )
    summary_file = resolve_workspace_path(
        config.get("summary_file", output_root / "SUMMARY.md")
    )

    experiments = config.get("experiments", [])
    if not experiments:
        print("❌ 配置文件中没有定义实验 (experiments)")
        sys.exit(1)

    # --only 过滤
    if args.only:
        filtered = [e for e in experiments if e["name"] in args.only]
        not_found = set(args.only) - {e["name"] for e in experiments}
        if not_found:
            print(f"⚠️  未找到指定实验: {', '.join(not_found)}")
        experiments = filtered
        if not experiments:
            print("❌ 没有匹配的实验")
            sys.exit(1)

    seeds = args.seeds if args.seeds else [42]

    print(f"\n{'='*70}")
    print(f"  🚀 批量实验启动")
    print(f"  模型: {config['model']}")
    print(f"  实验数: {len(experiments)}")
    print(f"  种子: {seeds}")
    print(f"  阶段: {args.steps}")
    print(f"  输出根目录: {output_root}")
    print(f"  进度文件: {progress_file}")
    print(f"  模式: {'干跑' if args.dry_run else '执行'}")
    if args.resume:
        print(f"  断点续跑: 是")
    if args.stratify:
        print(f"  分层采样: 是")
    print(f"{'='*70}\n")

    # 多 seed 循环
    results = []
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"  🌱 种子: {seed}")
        print(f"{'='*70}")
        for exp in experiments:
            # 复制 exp，添加 seed 信息
            exp_seed = dict(exp)
            exp_seed["name"] = f"{exp['name']}_seed{seed}"
            exp_seed["seed"] = seed

            # 分层采样
            if args.stratify:
                exp_seed["stratify"] = True

            ok = run_experiment(
                exp_seed,
                config,
                output_root,
                progress_file,
                set(args.steps),
                dry_run=args.dry_run,
                resume=args.resume,
            )
            results.append({"name": exp_seed["name"], "ok": ok, "seed": seed})

    # 种子间聚合（多 seed 时）
    ran_tests = bool({"test_original", "test_mr"} & set(args.steps))
    if not args.dry_run and not args.skip_summary and len(seeds) > 1 and ran_tests:
        print(f"\n{'='*70}")
        print(f"  📊 聚合 {len(seeds)} 个种子的实验结果...")
        print(f"{'='*70}")
        aggregate_results(
            seeds,
            config,
            output_root,
            result_file,
            {e["name"] for e in experiments},
        )

    # 单个 seed 时走旧汇总
    if not args.dry_run and not args.skip_summary and len(seeds) == 1:
        print(f"\n{'='*70}")
        print(f"  📊 生成测试结果汇总...")
        print(f"{'='*70}")
        collect_summary(output_root, summary_file)

    # 最终报告
    print(f"\n{'='*70}")
    print(f"  🏁 批量实验完成")
    print(f"{'='*70}")
    success = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"  ✅ 成功: {success}/{total}")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"    {status} {r['name']}")
    if not args.dry_run:
        print(f"\n  输出目录: {output_root}")
    print()


if __name__ == "__main__":
    main()
