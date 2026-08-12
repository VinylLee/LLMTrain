#!/usr/bin/env python3
"""
RQ1 NLI Experiment Pipeline
============================
在统一 standard SFT objective 下，系统比较 Original 训练与 MR-guided 训练对
NLI 模型在 4 个测试集（SNLI / MNLIm / MNLImm / SICK）的 Original 和 MR 测试表现。

用法：
    # 最小示例
    python RQ1/run_rq1_nli.py --model gemma-3-4b-it

    # 多 seed 统计
    python RQ1/run_rq1_nli.py --model gemma-3-4b-it --seeds 42 43 44

    # 只跑 MR 实验、仅 SNLI 训练集
    python RQ1/run_rq1_nli.py --model gemma-3-4b-it --experiment mr --train-data snli

    # Dry-run 预览命令
    python RQ1/run_rq1_nli.py --model gemma-3-4b-it --dry-run

流水线阶段：sample → convert → finetune → test_merged（默认 merged test）
"""
__test__ = False

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ── 项目根目录与脚本路径 ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from project_runtime import PROJECT_ROOT as PR, build_subprocess_env, resolve_model_reference

RQ1_DIR = PROJECT_ROOT / "RQ1"
DEFAULT_CONFIG = RQ1_DIR / "configs" / "rq1_nli_config.json"
DEFAULT_OUTPUT_ROOT = RQ1_DIR / "output"

STAGES = ["sample", "convert", "finetune", "test_original", "test_mr", "test_merged"]


# ══════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_config(config_path):
    """加载 RQ1 配置，将相对路径转为绝对路径。"""
    config = load_json(config_path)
    return config


def load_progress(progress_file):
    if Path(progress_file).exists():
        return load_json(progress_file)
    return {}


def save_progress(progress_file, progress):
    save_json(progress, progress_file)


def mark_step(progress_file, progress, exp_name, step, status):
    progress.setdefault(exp_name, {})[step] = status
    save_progress(progress_file, progress)


def step_completed(progress, exp_name, step):
    return progress.get(exp_name, {}).get(step) == "completed"


def run_cmd(cmd, desc, dry_run=False, cwd=None, env=None):
    """运行子进程命令；dry_run 时仅打印。"""
    cmd = [str(p) for p in cmd]
    print(f"\n  ▶ {'[DRY-RUN]' if dry_run else 'RUN'} {desc}")
    print(f"    {subprocess.list2cmdline(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print(f"    ❌ {desc} 失败 (code={result.returncode})")
        return False
    print(f"    ✅ {desc} 完成")
    return True


# ══════════════════════════════════════════════════════════════════════════
# 实验列表构建
# ══════════════════════════════════════════════════════════════════════════

def build_experiment_list(config, args):
    """根据 CLI 参数构建实验列表。"""
    model_key = args.model
    if model_key not in config["models"]:
        available = ", ".join(config["models"].keys())
        sys.exit(f"❌ 未知模型 '{model_key}'。可用: {available}")

    model_cfg = config["models"][model_key]
    defaults = config["experiment_defaults"]

    experiments = []
    for exp_type in args.experiment:
        for train_ds in args.train_data:
            ds_cfg = model_cfg["training_data"].get(train_ds)
            if ds_cfg is None:
                available_ds = ", ".join(model_cfg["training_data"].keys())
                print(f"⚠️  模型 '{model_key}' 无 '{train_ds}' 训练数据，跳过。可用: {available_ds}")
                continue

            # Resolve training data path per experiment type
            if exp_type == "original":
                train_path = ds_cfg.get("original", "")
            elif exp_type == "zaug":
                train_path = ds_cfg.get("zaug", "")
            else:  # mr
                train_path = ds_cfg.get("mr", "")

            if not train_path:
                print(f"⚠️  模型 '{model_key}' / '{train_ds}' 无 '{exp_type}' 训练数据，跳过")
                continue

            target = args.target if args.target is not None else ds_cfg["target"]

            # MR instruction mode: 只有 MR 实验使用 pair_operation
            mr_mode = args.mr_instruction_mode if exp_type == "mr" else "none"

            # Z-Aug: 预采样数据，跳过 sample 阶段（直接复制文件）
            skip_sample = (exp_type == "zaug")

            for seed in args.seeds:
                # Resolve zaug seed placeholder
                resolved_train_path = train_path.replace("{seed}", str(seed))
                name = f"rq1_{exp_type}_{train_ds}_seed{seed}"
                ft_params = dict(defaults["ft_params"])
                if args.lr is not None:
                    ft_params["lr"] = args.lr
                if args.epochs is not None:
                    ft_params["epochs"] = args.epochs
                if args.max_steps is not None:
                    ft_params["max_steps"] = args.max_steps
                if args.rank is not None:
                    ft_params["rank"] = args.rank
                if args.batch is not None:
                    ft_params["batch"] = args.batch
                if args.grad_accum is not None:
                    ft_params["grad_accum"] = args.grad_accum

                exp = {
                    "name": name,
                    "exp_type": exp_type,
                    "train_dataset": train_ds,
                    "train_data": resolved_train_path,
                    "target": target,
                    "task_type": defaults["task_type"],
                    "seed": seed,
                    "mr_instruction_mode": mr_mode,
                    "skip_sample": skip_sample,
                    "ft_params": ft_params,
                    "val_ratio": defaults["val_ratio"],
                    "cutoff_len": defaults["cutoff_len"],
                    "eval_strategy": defaults.get("eval_strategy", "no"),
                    "save_steps": defaults.get("save_steps", 9999),
                    "save_total_limit": defaults.get("save_total_limit", 2),
                }
                experiments.append(exp)

    return experiments, model_cfg


# ══════════════════════════════════════════════════════════════════════════
# YAML 生成（从 run_batch_experiments 复用）
# ══════════════════════════════════════════════════════════════════════════

def find_latest_complete_checkpoint(model_dir):
    """返回最新可恢复的 Trainer checkpoint。"""
    model_dir = Path(model_dir)
    candidates = []
    if not model_dir.is_dir():
        return None
    required = (
        "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
        "rng_state.pth", "trainer_state.json",
    )
    for path in sorted(model_dir.glob("checkpoint-*")):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if not path.is_dir() or any(not (path / name).is_file() for name in required):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("global_step") != step:
            continue
        candidates.append((step, path))
    return max(candidates, default=(None, None))[1]


def build_yaml(exp, model_name, template, output_root):
    """生成 LLaMA-Factory LoRA SFT YAML 配置文本。"""
    p = exp.get("ft_params", {})
    duration = (
        f"max_steps: {p['max_steps']}"
        if p.get("max_steps") is not None
        else f"num_train_epochs: {p.get('epochs', 3.0)}"
    )
    model_dir = output_root / exp["name"] / "model"
    resume_checkpoint = find_latest_complete_checkpoint(model_dir)
    resume_line = (
        f"resume_from_checkpoint: {resume_checkpoint.as_posix()}\n"
        if resume_checkpoint is not None
        else ""
    )
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
dataset_dir: {(PROJECT_ROOT / 'data').as_posix()}
cutoff_len: {exp.get('cutoff_len', 512)}
per_device_train_batch_size: {p.get('batch', 4)}
gradient_accumulation_steps: {p.get('grad_accum', 8)}
learning_rate: {p.get('lr', 3e-4)}
{duration}
lr_scheduler_type: cosine
warmup_ratio: 0.1
logging_steps: 10
save_steps: {exp.get('save_steps', 9999)}
save_total_limit: {exp.get('save_total_limit', 2)}
eval_dataset: {exp['name']}_val
eval_strategy: "{exp.get('eval_strategy', 'no')}"
eval_steps: {exp.get('eval_steps', 50)}
seed: {exp.get('seed', 42)}
data_seed: {exp.get('seed', 42)}
output_dir: {model_dir.as_posix()}
{resume_line}report_to: none
bf16: true
trust_remote_code: true
remove_unused_columns: false
model_name_or_path: {model_name}
template: {template}
use_cache: false
"""


def save_experiment_meta(exp, model_cfg, output_root):
    """保存 experiment_meta.json。"""
    meta_dir = output_root / exp["name"]
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment": exp["name"],
        "model": model_cfg.get("hub_id", ""),
        "template": model_cfg.get("template", ""),
        "train_data": exp["train_data"],
        "target": exp["target"],
        "task_type": exp["task_type"],
        "seed": exp["seed"],
        "ft_params": exp.get("ft_params", {}),
        "exp_type": exp["exp_type"],
        "train_dataset": exp["train_dataset"],
        "mr_instruction_mode": exp.get("mr_instruction_mode", "none"),
        "output_root": str(output_root),
        "conda_environment": "llmtrain310",
    }
    with open(meta_dir / "experiment_meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════
# 流水线阶段
# ══════════════════════════════════════════════════════════════════════════

STAGE_NAMES = {
    "sample": "Sampling",
    "convert": "Conversion",
    "finetune": "Fine-tuning",
    "test_original": "Test Original",
    "test_mr": "Test MR",
    "test_merged": "Test Merged",
}


def run_pipeline_stage(exp, model_cfg, output_root, progress, progress_file,
                       stage, args, model_name):
    """执行单个流水线阶段（子进程调用）。"""
    ft_data_dir = PROJECT_ROOT / "data" / "ft_datasets" / exp["name"]
    exp_dir = output_root / exp["name"]
    model_path = resolve_model_reference(
        model_cfg.get("hub_id", ""),
        model_cfg.get("local_path"),
    )
    tokenizer_path = resolve_model_reference(
        model_cfg.get("hub_id", ""),
        model_cfg.get("local_tokenizer_path") or model_cfg.get("local_path"),
    )
    env = build_subprocess_env(
        cuda=model_cfg.get("cuda", "0"),
        offline=True,
        torch_compile_disable=True,
    )

    # ── Stage 1: Sample ──────────────────────────────────────────────
    if stage == "sample":
        if exp.get("skip_sample"):
            # Z-Aug 等预采样数据：直接复制到 ft_datasets 目录
            import shutil
            sampled_path = ft_data_dir / "sampled.json"
            if not args.dry_run:
                ft_data_dir.mkdir(parents=True, exist_ok=True)
                source_path = PROJECT_ROOT / exp["train_data"]
                if not source_path.exists():
                    print(f"    ⚠️  Z-Aug 数据不存在: {source_path}")
                    return False
                shutil.copy2(source_path, sampled_path)
                print(f"    ✅ Z-Aug 预采样数据已复制: {sampled_path} ({sampled_path.stat().st_size} bytes)")
            else:
                print(f"    ▶ [DRY-RUN] copy {exp['train_data']} -> {sampled_path}")
            return True

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "sample_mettrain_pairid.py"),
            "--input", exp["train_data"],
            "--target", str(exp["target"]),
            "--seed", str(exp["seed"]),
            "--output", str(ft_data_dir / "sampled.json"),
            "--report-output", str(ft_data_dir / "sampling_report.json"),
        ]
        return run_cmd(cmd, f"Sample {exp['name']}", args.dry_run, cwd=PROJECT_ROOT, env=env)

    # ── Stage 2: Convert ─────────────────────────────────────────────
    if stage == "convert":
        sampled_path = ft_data_dir / "sampled.json"
        if not args.dry_run and not sampled_path.exists():
            print(f"    ⚠️  sampled.json 不存在: {sampled_path}，跳过 convert")
            return False

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "convert_nli_to_ft.py"),
            "--input", str(sampled_path),
            "--name", exp["name"],
            "--split",
            "--val-ratio", str(exp.get("val_ratio", 0.05)),
            "--seed", str(exp["seed"]),
            "--mr-instruction-mode", exp.get("mr_instruction_mode", "none"),
            "--instruction-template-version", "2",
            "--report-token-lengths",
            "--tokenizer-path", str(tokenizer_path),
            "--cutoff-len", str(exp.get("cutoff_len", 512)),
        ]
        return run_cmd(cmd, f"Convert {exp['name']}", args.dry_run, cwd=PROJECT_ROOT, env=env)

    # ── Stage 3: Finetune ────────────────────────────────────────────
    if stage == "finetune":
        conversion_report = ft_data_dir / "conversion_report.json"
        if not args.dry_run and not conversion_report.exists():
            print(f"    ⚠️  conversion_report.json 不存在: {conversion_report}，跳过 finetune")
            return False

        yaml_content = build_yaml(exp, model_name, model_cfg["template"], output_root)
        yaml_path = PROJECT_ROOT / f"ft_config_{exp['name']}.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        save_experiment_meta(exp, model_cfg, output_root)

        try:
            result = run_cmd(
                [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)],
                f"Fine-tune {exp['name']}", args.dry_run, cwd=PROJECT_ROOT, env=env,
            )
            return result
        finally:
            if not args.dry_run and yaml_path.exists():
                yaml_path.unlink()

    # ── Stage 4 & 5: Test ────────────────────────────────────────────
    if stage in ("test_original", "test_mr"):
        lora_path = str(exp_dir / "model")
        if not args.dry_run and not Path(lora_path).is_dir():
            print(f"    ⚠️  LoRA 模型不存在: {lora_path}，跳过 {stage}")
            return False

        test_datasets = ",".join(args.test_datasets) if args.test_datasets else "snli,mnlim,mnlimm,sick"
        skip_flag = "--skip-mr" if stage == "test_original" else "--skip-original"

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "test_mettrain_experiment.py"),
            "--experiment", exp["name"],
            "--base-model", model_path,
            "--lora", lora_path,
            "--output-root", str(output_root),
            "--datasets", test_datasets,
            "--batch-size", str(args.test_batch_size),
            skip_flag,
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])

        return run_cmd(cmd, f"Test {exp['name']} ({stage})",
                       args.dry_run, cwd=PROJECT_ROOT, env=env)

    # ── Stage 6: Test Merged ──────────────────────────────────────────
    if stage == "test_merged":
        lora_path = str(exp_dir / "model")
        if not args.dry_run and not Path(lora_path).is_dir():
            print(f"    ⚠️  LoRA 模型不存在: {lora_path}，跳过 {stage}")
            return False

        test_datasets = ",".join(args.test_datasets) if args.test_datasets else "snli,mnlim,mnlimm,sick"

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "test_mettrain_experiment.py"),
            "--experiment", exp["name"],
            "--base-model", model_path,
            "--lora", lora_path,
            "--output-root", str(output_root),
            "--datasets", test_datasets,
            "--batch-size", str(args.test_batch_size),
            "--merged",
            # 只跑 merged 单次推理；显式跳过 original/mr，避免测试脚本重复跑两遍无用结果
            "--skip-original", "--skip-mr",
        ]
        if args.max_samples is not None:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.merged_data_path:
            cmd.extend(["--merged-path", args.merged_data_path])

        return run_cmd(cmd, f"Test {exp['name']} ({stage})",
                       args.dry_run, cwd=PROJECT_ROOT, env=env)

    return True


# ══════════════════════════════════════════════════════════════════════════
# 结果汇总
# ══════════════════════════════════════════════════════════════════════════

def read_test_results(exp_dir):
    """读取单个实验的测试结果，返回 {test_type: {ds: {correct, total, acc}}}。"""
    results = {}
    for test_type in ("original", "mr", "merged"):
        tests_dir = exp_dir / "tests" / test_type
        if not tests_dir.is_dir():
            continue
        if test_type == "merged":
            # Merged 数据：按 is_source 分别统计 source/MR accuracy，并计算 MSR
            results[test_type] = {}
            for jsonl in sorted(tests_dir.glob("*.jsonl")):
                ds_name = jsonl.stem
                src_correct = src_total = 0
                mr_correct = mr_total = 0
                rows = []
                try:
                    with open(jsonl, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            rows.append(row)
                            if row.get("is_source"):
                                src_total += 1
                                if row.get("correct") is True:
                                    src_correct += 1
                            else:
                                mr_total += 1
                                if row.get("correct") is True:
                                    mr_correct += 1
                except Exception as e:
                    print(f"  ⚠️  读取 {jsonl} 出错: {e}")
                    continue
                src_acc = (src_correct / src_total * 100) if src_total > 0 else 0.0
                mr_acc = (mr_correct / mr_total * 100) if mr_total > 0 else 0.0
                # MSR：复用 test_mettrain_experiment.compute_msr（按 pair_id 分组核对 MR 关系）
                # 延迟导入，避免启动时加载 torch/transformers
                from test_mettrain_experiment import compute_msr
                msr = compute_msr(rows) if rows else {}
                results[test_type][ds_name] = {
                    "source": {"correct": src_correct, "total": src_total, "acc": src_acc},
                    "mr": {"correct": mr_correct, "total": mr_total, "acc": mr_acc},
                    # Overall (source + MR combined)
                    "correct": src_correct + mr_correct,
                    "total": src_total + mr_total,
                    "acc": ((src_correct + mr_correct) / (src_total + mr_total) * 100) if (src_total + mr_total) > 0 else 0.0,
                    "msr": msr,
                }
        else:
            results[test_type] = {}
            for jsonl in sorted(tests_dir.glob("*.jsonl")):
                ds_name = jsonl.stem
                correct = 0
                total = 0
                try:
                    with open(jsonl, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            total += 1
                            if row.get("correct") is True:
                                correct += 1
                except Exception as e:
                    print(f"  ⚠️  读取 {jsonl} 出错: {e}")
                    continue
                acc = (correct / total * 100) if total > 0 else 0.0
                results[test_type][ds_name] = {"correct": correct, "total": total, "acc": acc}
    return results


def generate_summary(output_root, experiments, model_key, seeds):
    """生成 SUMMARY.md（单 seed）或 RESULTS.md（多 seed）。"""
    is_multi_seed = len(seeds) > 1
    report_path = output_root / ("RESULTS.md" if is_multi_seed else "SUMMARY.md")

    # 收集所有实验结果
    all_data = {}  # {experiment_base: {seed: results}}
    for exp in experiments:
        base = exp["name"]  # full name includes seed
        exp_dir = output_root / exp["name"]
        results = read_test_results(exp_dir) if exp_dir.is_dir() else {}
        all_data[base] = results

    lines = []
    lines.append(f"# RQ1 NLI Experiment Results")
    lines.append(f"")
    lines.append(f"**Model**: {model_key}  |  **Seeds**: {', '.join(str(s) for s in seeds)}  |  **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if not all_data:
        lines.append("*(No results found)*")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    test_datasets = ["snli", "mnlim", "mnlimm", "sick"]

    # ── 按 experiment × test_type 聚合 ──────────────────────────────
    # Group by (exp_type, train_dataset) ignoring seed
    import math

    groups = {}  # key=(exp_type, train_ds) → {test_type: {ds: [acc_list]}}
    for exp in experiments:
        key = (exp["exp_type"], exp["train_dataset"])
        groups.setdefault(key, {})
        results = all_data.get(exp["name"], {})
        for test_type in ("original", "mr", "merged"):
            groups[key].setdefault(test_type, {})
            for ds in test_datasets:
                groups[key][test_type].setdefault(ds, [])
                ds_results = results.get(test_type, {}).get(ds, {})
                if test_type == "merged":
                    # Merged 有 source.acc、mr.acc 和 msr.overall.rate 三个子指标
                    src_acc = ds_results.get("source", {}).get("acc")
                    mr_acc = ds_results.get("mr", {}).get("acc")
                    msr_rate = (ds_results.get("msr", {}) or {}).get("overall", {}).get("rate")
                    if src_acc is not None and mr_acc is not None:
                        groups[key][test_type][ds].append((src_acc, mr_acc, msr_rate))
                else:
                    acc = ds_results.get("acc")
                    if acc is not None:
                        groups[key][test_type][ds].append(acc)

    # ── Original Test Sets 表格 ─────────────────────────────────────
    lines.append("## Original Test Sets (Accuracy)")
    if is_multi_seed:
        lines.append("")
        lines.append("| Experiment | Type | Train DS | " + " | ".join(d.upper() for d in test_datasets) + " |")
        lines.append("|" + "---|" * (4 + len(test_datasets)) + "")
        for key, data in sorted(groups.items()):
            exp_type, train_ds = key
            test_data = data.get("original", {})
            row = f"| rq1_{exp_type}_{train_ds} | {exp_type} | {train_ds} |"
            for ds in test_datasets:
                accs = test_data.get(ds, [])
                if len(accs) >= 2:
                    mean = sum(accs) / len(accs)
                    var = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                    std = math.sqrt(var)
                    row += f" {mean:.2f} ± {std:.2f}% |"
                elif len(accs) == 1:
                    row += f" {accs[0]:.2f}% |"
                else:
                    row += " — |"
            lines.append(row)
    else:
        lines.append("")
        lines.append("| Experiment | Type | Train DS | " + " | ".join(d.upper() for d in test_datasets) + " |")
        lines.append("|" + "---|" * (4 + len(test_datasets)) + "")
        for key, data in sorted(groups.items()):
            exp_type, train_ds = key
            test_data = data.get("original", {})
            row = f"| rq1_{exp_type}_{train_ds} | {exp_type} | {train_ds} |"
            for ds in test_datasets:
                accs = test_data.get(ds, [])
                row += f" {accs[0]:.2f}% |" if accs else " — |"
            lines.append(row)

    lines.append("")

    # ── MR Test Sets 表格 ───────────────────────────────────────────
    lines.append("## MR Test Sets (Accuracy)")
    if is_multi_seed:
        lines.append("")
        lines.append("| Experiment | Type | Train DS | " + " | ".join(d.upper() for d in test_datasets) + " |")
        lines.append("|" + "---|" * (4 + len(test_datasets)) + "")
        for key, data in sorted(groups.items()):
            exp_type, train_ds = key
            test_data = data.get("mr", {})
            row = f"| rq1_{exp_type}_{train_ds} | {exp_type} | {train_ds} |"
            for ds in test_datasets:
                accs = test_data.get(ds, [])
                if len(accs) >= 2:
                    mean = sum(accs) / len(accs)
                    var = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
                    std = math.sqrt(var)
                    row += f" {mean:.2f} ± {std:.2f}% |"
                elif len(accs) == 1:
                    row += f" {accs[0]:.2f}% |"
                else:
                    row += " — |"
            lines.append(row)
    else:
        lines.append("")
        lines.append("| Experiment | Type | Train DS | " + " | ".join(d.upper() for d in test_datasets) + " |")
        lines.append("|" + "---|" * (4 + len(test_datasets)) + "")
        for key, data in sorted(groups.items()):
            exp_type, train_ds = key
            test_data = data.get("mr", {})
            row = f"| rq1_{exp_type}_{train_ds} | {exp_type} | {train_ds} |"
            for ds in test_datasets:
                accs = test_data.get(ds, [])
                row += f" {accs[0]:.2f}% |" if accs else " — |"
            lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Merged Test Sets (Source + MR accuracy) ─────────────────────
    has_merged = any(groups[key].get("merged") for key in groups)
    if has_merged:
        lines.append("## Merged Test Results (Source Accuracy / MR Accuracy)")
        lines.append("")
        lines.append("| Experiment | Type | Train DS | " + " | ".join(d.upper() for d in test_datasets) + " |")
        lines.append("|" + "---|" * (4 + len(test_datasets)) + "")
        for key, data in sorted(groups.items()):
            exp_type, train_ds = key
            test_data = data.get("merged", {})
            if not test_data:
                continue
            row = f"| rq1_{exp_type}_{train_ds} | {exp_type} | {train_ds} |"
            for ds in test_datasets:
                pairs = test_data.get(ds, [])
                if pairs:
                    src_accs = [p[0] for p in pairs]
                    mr_accs = [p[1] for p in pairs]
                    if len(pairs) >= 2:
                        src_mean = sum(src_accs) / len(src_accs)
                        mr_mean = sum(mr_accs) / len(mr_accs)
                        row += f" {src_mean:.1f}/{mr_mean:.1f}% |"
                    else:
                        row += f" {src_accs[0]:.1f}/{mr_accs[0]:.1f}% |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")
        lines.append("> 格式: source准确率 / MR准确率。Source = is_source=true 的原始数据；MR = is_source=false 的 MR 变体。")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── Merged MSR (Metamorphic Satisfaction Rate) ──────────────────
    if has_merged:
        lines.append("## Merged Test Results — MSR (Metamorphic Satisfaction Rate, overall)")
        lines.append("")
        lines.append("> MSR overall = compute_msr() 整体 satisfaction rate（%）")
        lines.append("")
        lines.append("| Experiment | Type | Train DS | " + " | ".join(d.upper() for d in test_datasets) + " |")
        lines.append("|" + "---|" * (4 + len(test_datasets)) + "")
        for key, data in sorted(groups.items()):
            exp_type, train_ds = key
            test_data = data.get("merged", {})
            if not test_data:
                continue
            row = f"| rq1_{exp_type}_{train_ds} | {exp_type} | {train_ds} |"
            for ds in test_datasets:
                rates = [p[2] for p in test_data.get(ds, []) if len(p) > 2 and p[2] is not None]
                if rates:
                    if len(rates) >= 2:
                        mean = sum(rates) / len(rates)
                        var = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
                        std = math.sqrt(var)
                        row += f" {mean:.2f} ± {std:.2f}% |"
                    else:
                        row += f" {rates[0]:.2f}% |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── RQ1 Key Comparison: Δ table (merged test results) ──────────
    lines.append("## RQ1 Key Comparison: Training Method Δ (Merged Test)")
    lines.append("")
    lines.append("*(Source accuracy = accuracy on is_source=true original test data;")
    lines.append("MR accuracy = accuracy on is_source=false MR test data)*")
    lines.append("")

    for metric, metric_label in [("source", "Source Accuracy (Original Test)"), ("mr", "MR Accuracy")]:
        lines.append(f"### {metric_label}")
        lines.append("")
        # Header: Train DS | Original | MR | Z-Aug
        lines.append("| Train DS | Original | MR | Z-Aug | MR-Orig Δ | Z-Aug-Orig Δ |")
        lines.append("|" + "---|" * 7 + "")

        for train_ds in sorted(set(exp["train_dataset"] for exp in experiments)):
            row = f"| {train_ds} |"
            orig_vals = []
            for etype in ["original", "mr", "zaug"]:
                key = (etype, train_ds)
                acc_pairs = groups.get(key, {}).get("merged", {}).get(train_ds, [])
                # Or if no merged data, try original/mr test types
                if not acc_pairs:
                    acc_pairs_fb = groups.get(key, {}).get("merged", {}).get("mnlim", groups.get(key, {}).get("merged", {}).get("snli", groups.get(key, {}).get("merged", {}).get("sick", [])))
                if not acc_pairs:
                    # Fallback to non-merged test types
                    for fb_type in ("original", "mr"):
                        vals = groups.get(key, {}).get(fb_type, {}).get(train_ds, [])
                        if vals:
                            acc_pairs = [(v, 0) if metric == "source" else (0, v) for v in vals]
                            break

                if acc_pairs:
                    vals_list = [p[0] for p in acc_pairs] if metric == "source" else [p[1] for p in acc_pairs]
                    if is_multi_seed and len(vals_list) >= 2:
                        mean = sum(vals_list) / len(vals_list)
                        row += f" {mean:.2f}% |"
                    else:
                        row += f" {vals_list[0]:.2f}% |"
                    orig_vals.append(vals_list)
                else:
                    row += " — |"
                    orig_vals.append(None)

            # Deltas: MR-Orig and Z-Aug-Orig
            if orig_vals[0] and orig_vals[1]:  # original and mr both present
                o_mean = sum(orig_vals[0]) / len(orig_vals[0]) if is_multi_seed else orig_vals[0][0]
                m_mean = sum(orig_vals[1]) / len(orig_vals[1]) if is_multi_seed else orig_vals[1][0]
                d1 = m_mean - o_mean
                row += f" {'+' if d1 >= 0 else ''}{d1:.2f}% |"
            else:
                row += " — |"
            if orig_vals[0] and orig_vals[2]:  # original and zaug both present
                o_mean = sum(orig_vals[0]) / len(orig_vals[0]) if is_multi_seed else orig_vals[0][0]
                z_mean = sum(orig_vals[2]) / len(orig_vals[2]) if is_multi_seed else orig_vals[2][0]
                d2 = z_mean - o_mean
                row += f" {'+' if d2 >= 0 else ''}{d2:.2f}% |"
            else:
                row += " — |"
            lines.append(row)
        lines.append("")

    if is_multi_seed:
        lines.append(f"> 准确率格式: 均值 ± 样本标准差 (分母 n-1, {len(seeds)} seeds)")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 报告已保存: {report_path}")
    return report_path


def generate_json_report(output_root, experiments, model_key):
    """生成机器可读的 rq1_report.json。"""
    report = {
        "model": model_key,
        "generated": datetime.now().isoformat(),
        "experiments": {},
    }
    for exp in experiments:
        exp_dir = output_root / exp["name"]
        results = read_test_results(exp_dir) if exp_dir.is_dir() else {}
        report["experiments"][exp["name"]] = {
            "train_data": exp["train_data"],
            "type": exp["exp_type"],
            "train_dataset": exp["train_dataset"],
            "target": exp["target"],
            "seed": exp["seed"],
            "results": results,
        }

    json_path = output_root / "rq1_report.json"
    save_json(report, json_path)
    print(f"📄 JSON 报告已保存: {json_path}")
    return json_path


# ══════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="RQ1 NLI Experiment Pipeline — MR-guided vs Original fine-tuning 对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python RQ1/run_rq1_nli.py --model gemma-3-4b-it
  python RQ1/run_rq1_nli.py --model gemma-3-4b-it --seeds 42 43 44
  python RQ1/run_rq1_nli.py --model gemma-3-4b-it --experiment mr --train-data snli
  python RQ1/run_rq1_nli.py --model gemma-3-4b-it --dry-run
        """,
    )

    # 核心参数
    p.add_argument("--model", required=True,
                   help="模型 key（如 gemma-3-4b-it, llama-3.2-3b）")
    p.add_argument("--experiment", nargs="+", default=["original", "mr"],
                   choices=["original", "mr", "zaug"],
                   help="实验类型 (default: original mr)")
    p.add_argument("--train-data", nargs="+", default=None,
                   help="训练数据集 (default: 配置中的 default_train_datasets)")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="随机种子列表 (default: [42])")
    p.add_argument("--mr-instruction-mode", default="pair_operation",
                   choices=["none", "operation_only", "pair_only", "pair_operation",
                            "shuffled_operation", "relation_aware", "full_oracle"],
                   help="MR instruction 模式，仅 MR 实验生效 (default: pair_operation)")

    # 可选覆盖
    p.add_argument("--target", type=int, default=None,
                   help="覆盖训练样本目标数 (default: 使用配置中的 target)")
    p.add_argument("--lr", type=float, default=None, help="覆盖 learning rate")
    p.add_argument("--epochs", type=float, default=None, help="覆盖 epochs")
    p.add_argument("--max-steps", type=int, default=None, help="覆盖 max_steps（优先级高于 epochs）")
    p.add_argument("--rank", type=int, default=None, help="覆盖 LoRA rank")
    p.add_argument("--batch", type=int, default=None, help="覆盖 per-device batch size")
    p.add_argument("--grad-accum", type=int, default=None, help="覆盖 gradient accumulation steps")
    p.add_argument("--test-batch-size", type=int, default=32, help="推理 batch size (default: 32)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="测试时每个数据集最大样本数（冒烟测试用）")

    # 流程控制
    p.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    p.add_argument("--resume", action="store_true", help="跳过已完成的阶段")
    p.add_argument("--no-summary", action="store_true", help="跳过 SUMMARY/RESULTS 生成")
    p.add_argument("--output-root", default=None,
                   help="输出根目录 (default: RQ1/output/<model_key>)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"配置文件路径 (default: {DEFAULT_CONFIG})")
    p.add_argument("--steps", nargs="+", default=STAGES,
                   choices=STAGES,
                   help="流水线阶段子集 (default: 全部 6 个阶段)")
    p.add_argument("--test-datasets", nargs="+", default=None,
                   help="测试数据集子集 (default: 全部 4 个)")
    p.add_argument("--no-merged-test", action="store_true",
                   help="禁用 merged test，使用传统的 test_original+test_mr 分开测试")
    p.add_argument("--merged-data-path", default=None,
                   help="覆盖 Merged 数据目录路径 (default: data/nli/mr_test_data_merged)")
    p.add_argument("--cuda", default=None, help="覆盖 CUDA 设备号")
    p.add_argument("--only", nargs="+", default=None,
                   help="只运行指定实验名（用于选择性重跑）")

    return p.parse_args()


def main():
    args = parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        sys.exit(f"❌ 配置文件不存在: {config_path}")
    config = load_config(config_path)

    # 默认训练数据集
    if args.train_data is None:
        args.train_data = config.get("default_train_datasets", ["snli", "mnlim", "sick"])
    # 默认测试数据集
    if args.test_datasets is None:
        test_sets_cfg = config.get("test_sets", {}).get("original", {})
        args.test_datasets = args.test_datasets or list(test_sets_cfg.keys())

    # 输出根目录
    if args.output_root:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
    else:
        output_root = DEFAULT_OUTPUT_ROOT / args.model
    output_root.mkdir(parents=True, exist_ok=True)

    # 构建实验列表
    experiments, model_cfg = build_experiment_list(config, args)
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
        if not experiments:
            sys.exit(f"❌ --only 未匹配任何实验。可用: " +
                      ", ".join(e["name"] for e in build_experiment_list(config, args)[0]))

    if not experiments:
        sys.exit("❌ 没有要运行的实验。请检查 --experiment / --train-data 参数。")

    # 覆盖 CUDA
    if args.cuda is not None:
        model_cfg = dict(model_cfg)
        model_cfg["cuda"] = args.cuda

    # 模型名称
    model_name = resolve_model_reference(
        model_cfg.get("hub_id", ""),
        model_cfg.get("local_path"),
    )

    # ── 打印实验概览 ──────────────────────────────────────────────────
    print("=" * 70)
    print("  RQ1 NLI Experiment Pipeline")
    print("=" * 70)
    print(f"  Model:        {args.model} ({model_cfg.get('hub_id', '')})")
    print(f"  Template:     {model_cfg.get('template', '')}")
    print(f"  Model path:   {model_name}")
    print(f"  Experiments:  {len(experiments)}")
    print(f"  Types:        {args.experiment}")
    print(f"  Train data:   {args.train_data}")
    print(f"  Test data:    {args.test_datasets}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Output root:  {output_root}")
    if args.dry_run:
        print(f"  Mode:         DRY-RUN (仅预览)")
    if args.resume:
        print(f"  Mode:         RESUME (跳过已完成)")
    print("-" * 70)
    for exp in experiments:
        print(f"  • {exp['name']}")
        print(f"    train={exp['train_data']}  target={exp['target']}  "
              f"seed={exp['seed']}  mr_mode={exp['mr_instruction_mode']}")
    print("=" * 70)

    # ── 进度文件 ──────────────────────────────────────────────────────
    progress_file = output_root / "_progress.json"
    progress = load_progress(progress_file) if args.resume else {}

    # ── Merged test 为默认模式：替换 test_original + test_mr ─────────
    if not args.no_merged_test:
        args.steps = [s for s in args.steps if s not in ("test_original", "test_mr")]
        if "test_merged" not in args.steps:
            args.steps.append("test_merged")
        print(f"  📊 Test mode: merged (source+MR 单次推理). 使用 --no-merged-test 切换为传统模式")
    else:
        # 传统模式：移除 test_merged，保留 test_original + test_mr
        if "test_merged" in args.steps:
            args.steps.remove("test_merged")
        print(f"  📊 Test mode: legacy (test_original + test_mr 分开测试)")

    # ── 运行流水线 ────────────────────────────────────────────────────
    total = len(experiments)
    failed = 0
    t_start = time.time()

    for i, exp in enumerate(experiments):
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/{total}] {exp['name']}")
        print(f"{'─' * 70}")

        stages = args.steps
        for stage in stages:
            if args.resume and step_completed(progress, exp["name"], stage):
                print(f"  ⏭️  {STAGE_NAMES[stage]} — 已完成，跳过")
                continue

            ok = run_pipeline_stage(
                exp, model_cfg, output_root, progress, progress_file,
                stage, args, model_name,
            )
            if ok:
                mark_step(progress_file, progress, exp["name"], stage, "completed")
            else:
                mark_step(progress_file, progress, exp["name"], stage, "failed")
                if not args.dry_run:
                    failed += 1
                    print(f"  ⛔ {exp['name']} 在 {STAGE_NAMES[stage]} 阶段失败，跳过后续阶段")
                    break

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline finished in {elapsed/60:.1f} min")
    print(f"  Success: {total - failed}/{total}  |  Failed: {failed}")
    print(f"{'=' * 70}")

    # ── 汇总 ─────────────────────────────────────────────────────────
    if not args.no_summary and not args.dry_run:
        print(f"\n📊 生成结果汇总...")
        generate_summary(output_root, experiments, args.model, args.seeds)
        generate_json_report(output_root, experiments, args.model)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
