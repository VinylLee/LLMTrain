#!/usr/bin/env python3
"""
Z_AUG_baseline.py
=================
用 data/nli/zaug/rq1/ 下 5 个种子的 Z-Aug MNLI 采样数据（zaug_sample_s{42..46}.jsonl，
各 4413 条，字段 premise/hypothesis/label/type）微调 Gemma-3-4B（LoRA SFT），并在
original 模式下评测 4 个 NLI 测试集（snli / mnlim / mnlimm / sick）的
original test.json 准确率。

复用现有脚本：
  - convert : scripts/convert_nli_to_ft.py（Alpaca 转换 + 注册 dataset_info.json）
  - finetune : RQ1/run_rq1_nli.py 的 build_yaml + `python -m llamafactory.cli train`
  - test     : scripts/test_mettrain_experiment.py（--skip-mr，读 original_dataset/<ds>/test.json）
  - 汇总     : RQ1/run_rq1_nli.py 的 read_test_results（逐条 correct 字段统计）

用法（在 llmtrain310 环境）：
  python RQ1/Z_AUG_baseline.py --dry-run                   # 预览所有命令
  python RQ1/Z_AUG_baseline.py --steps convert --seeds 42  # 冒烟：只转换一个种子
  python RQ1/Z_AUG_baseline.py                             # 完整跑 5 个种子
"""
__test__ = False

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RQ1_DIR = PROJECT_ROOT / "RQ1"
sys.path.insert(0, str(RQ1_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from project_runtime import build_subprocess_env, resolve_model_reference
from run_rq1_nli import (
    build_yaml,
    load_config,
    load_progress,
    mark_step,
    read_test_results,
    run_cmd,
    save_experiment_meta,
    step_completed,
)

DEFAULT_CONFIG = RQ1_DIR / "configs" / "rq1_nli_config.json"
Z_AUG_DATA_DIR = PROJECT_ROOT / "data" / "nli" / "zaug" / "rq1"
FT_DATASETS_DIR = PROJECT_ROOT / "data" / "ft_datasets"
DEFAULT_OUTPUT_ROOT = RQ1_DIR / "output" / "zaug_baseline"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TEST_DATASETS = ["snli", "mnlim", "mnlimm", "sick"]
STAGES = ["convert", "finetune", "test", "summarize"]


def build_experiment(model_cfg, defaults, seed):
    """构造单个种子的实验 dict（结构与 RQ1 exp 对齐，供 build_yaml/read_test_results 复用）。"""
    ft_params = dict(defaults.get("ft_params", {}))
    ft_params.setdefault("rank", 8)
    ft_params.setdefault("lr", 3e-4)
    ft_params.setdefault("epochs", 3.0)
    ft_params.setdefault("batch", 4)
    ft_params.setdefault("grad_accum", 8)
    zaug_path = Z_AUG_DATA_DIR / f"zaug_sample_s{seed}.jsonl"
    return {
        "name": f"zaug_rq1_s{seed}",
        "exp_type": "original",           # original 模式：仅评测 standard 测试集
        "train_dataset": "zaug",
        "train_data": str(zaug_path),
        "target": 0,                      # zaug 数据已按 target 采样完成，无需 sample 阶段
        "seed": seed,
        "task_type": defaults.get("task_type", "nli"),
        "ft_params": ft_params,
        "val_ratio": defaults.get("val_ratio", 0.05),
        "cutoff_len": defaults.get("cutoff_len", 512),
        "eval_strategy": defaults.get("eval_strategy", "no"),
        "save_steps": defaults.get("save_steps", 9999),
        "save_total_limit": defaults.get("save_total_limit", 2),
        "mr_instruction_mode": "none",
    }


def stage_convert(exp, tokenizer_path, progress, progress_file, args, env):
    if not args.force and step_completed(progress, exp["name"], "convert"):
        print(f"  • [convert] {exp['name']} 已完成，跳过")
        return True
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "convert_nli_to_ft.py"),
        "--input", exp["train_data"],
        "--name", exp["name"],
        "--split",
        "--val-ratio", str(exp["val_ratio"]),
        "--seed", str(exp["seed"]),
        "--mr-instruction-mode", "none",
        "--instruction-template-version", "2",
        "--tokenizer-path", str(tokenizer_path),
        "--cutoff-len", str(exp["cutoff_len"]),
    ]
    ok = run_cmd(cmd, f"Convert {exp['name']}", args.dry_run, cwd=PROJECT_ROOT, env=env)
    if ok and not args.dry_run:
        mark_step(progress_file, progress, exp["name"], "convert", "completed")
    return ok


def stage_finetune(exp, model_cfg, model_path, template, output_root,
                   progress, progress_file, args, env):
    if not args.force and step_completed(progress, exp["name"], "finetune"):
        print(f"  • [finetune] {exp['name']} 已完成，跳过")
        return True
    if not args.dry_run and not (FT_DATASETS_DIR / exp["name"] / "conversion_report.json").exists():
        print(f"    ⚠️  conversion_report.json 不存在: {FT_DATASETS_DIR / exp['name']}，跳过 finetune")
        return False

    yaml_content = build_yaml(exp, model_path, template, output_root)
    yaml_path = PROJECT_ROOT / f"ft_config_{exp['name']}.yaml"
    if not args.dry_run:
        yaml_path.write_text(yaml_content, encoding="utf-8")
        save_experiment_meta(exp, model_cfg, output_root)
    try:
        ok = run_cmd(
            [sys.executable, "-m", "llamafactory.cli", "train", str(yaml_path)],
            f"Fine-tune {exp['name']}", args.dry_run, cwd=PROJECT_ROOT, env=env,
        )
    finally:
        if not args.dry_run and yaml_path.exists():
            yaml_path.unlink()
    if ok and not args.dry_run:
        mark_step(progress_file, progress, exp["name"], "finetune", "completed")
    return ok


def stage_test(exp, model_path, output_root, progress, progress_file, args, env):
    if not args.force and step_completed(progress, exp["name"], "test"):
        print(f"  • [test] {exp['name']} 已完成，跳过")
        return True
    lora_path = output_root / exp["name"] / "model"
    if not args.dry_run and not lora_path.is_dir():
        print(f"    ⚠️  LoRA 模型不存在: {lora_path}，跳过 test")
        return False
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "test_mettrain_experiment.py"),
        "--experiment", exp["name"],
        "--base-model", model_path,
        "--lora", str(lora_path),
        "--output-root", str(output_root),
        "--datasets", ",".join(args.test_datasets),
        "--batch-size", str(args.test_batch_size),
        "--skip-mr",
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    ok = run_cmd(cmd, f"Test {exp['name']} (original)", args.dry_run, cwd=PROJECT_ROOT, env=env)
    if ok and not args.dry_run:
        mark_step(progress_file, progress, exp["name"], "test", "completed")
    return ok


def stage_summarize(experiments, output_root, test_datasets, args):
    """读取各种子逐条测试结果，打印表格 + 写 RESULTS.md / rq1_report.json。"""
    per_seed = {}
    for exp in experiments:
        res = read_test_results(output_root / exp["name"])
        original = res.get("original", {})
        per_seed[exp["seed"]] = {
            ds: (original[ds]["acc"] if ds in original else None)
            for ds in test_datasets
        }

    print("\n" + "=" * 70)
    print("Z-AUG Baseline 结果汇总（original 测试集准确率 %）")
    print("=" * 70)
    print(f"{'':>6} " + " ".join(f"{d:>10}" for d in test_datasets))
    for seed in sorted(per_seed):
        cells = []
        for ds in test_datasets:
            v = per_seed[seed][ds]
            cells.append(f"{v:8.2f}" if v is not None else "     n/a")
        print(f"{seed:>6} " + " ".join(f"{c:>10}" for c in cells))

    summary = {}
    print("\nmean±std（跨种子）:")
    for ds in test_datasets:
        vals = [per_seed[s][ds] for s in sorted(per_seed) if per_seed[s][ds] is not None]
        if vals:
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            summary[ds] = {"mean": round(mean, 2), "std": round(sd, 2), "n": len(vals)}
            print(f"  {ds:>10}: {mean:6.2f} ± {sd:5.2f}")
        else:
            summary[ds] = {"mean": None, "std": None, "n": 0}
            print(f"  {ds:>10}: n/a")

    # 写 RESULTS.md 与 rq1_report.json
    lines = [
        "# Z-AUG Baseline — original 测试集准确率",
        "",
        f"- 模型: `{args.model}`（LoRA SFT, 超参沿用 RQ1 默认）",
        f"- 训练数据: `data/nli/zaug/rq1/zaug_sample_s{{42..46}}.jsonl`（各 4413 条, Z-Aug MNLI 采样）",
        f"- 测试集: original test (`data/nli/original_dataset/<ds>/test.json`): {', '.join(test_datasets)}",
        f"- 种子: {', '.join(str(s) for s in sorted(per_seed))}",
        "",
        "| seed | " + " | ".join(test_datasets) + " |",
        "|---|" + "---|" * len(test_datasets),
    ]
    for seed in sorted(per_seed):
        cells = [str(seed)] + [
            f"{per_seed[seed][ds]:.2f}" if per_seed[seed][ds] is not None else "n/a"
            for ds in test_datasets
        ]
        lines.append("| " + " | ".join(cells) + " |")
    mean_cells = ["mean±std"] + [
        f"{summary[ds]['mean']:.2f}±{summary[ds]['std']:.2f}" if summary[ds]["mean"] is not None else "n/a"
        for ds in test_datasets
    ]
    lines.append("| " + " | ".join(mean_cells) + " |")
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "model": args.model,
        "test_datasets": test_datasets,
        "seeds": sorted(per_seed),
        "per_seed": per_seed,
        "summary": summary,
    }
    (output_root / "rq1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n已写入: {output_root / 'RESULTS.md'} 和 {output_root / 'rq1_report.json'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Z-AUG baseline: 用 5 种子 Z-Aug 数据微调 Gemma-3-4B 并评测 original 测试集准确率"
    )
    parser.add_argument("--model", default="gemma-3-4b-it", help="RQ1 配置中的模型 key")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="随机种子列表")
    parser.add_argument("--test-datasets", default=",".join(DEFAULT_TEST_DATASETS),
                        help="original 测试集（逗号分隔）")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--steps", nargs="+", choices=STAGES, default=STAGES, help="运行阶段子集")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="RQ1 配置文件")
    parser.add_argument("--test-batch-size", type=int, default=32, help="推理 batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="测试时每数据集最大样本数（冒烟）")
    parser.add_argument("--cuda", default=None, help="覆盖 CUDA 设备（默认取配置）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不执行")
    parser.add_argument("--force", action="store_true", help="重跑已完成阶段")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.model not in config["models"]:
        sys.exit(f"❌ 未知模型 '{args.model}'。可用: {', '.join(config['models'])}")

    model_cfg = config["models"][args.model]
    defaults = config["experiment_defaults"]

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    test_datasets = [d.strip() for d in args.test_datasets.split(",") if d.strip()]
    args.test_datasets = test_datasets  # 归一化后回写，供 stage_test 使用

    model_path = resolve_model_reference(model_cfg.get("hub_id", ""), model_cfg.get("local_path"))
    tokenizer_path = resolve_model_reference(
        model_cfg.get("hub_id", ""),
        model_cfg.get("local_tokenizer_path") or model_cfg.get("local_path"),
    )
    template = model_cfg.get("template", "gemma")
    cuda = args.cuda if args.cuda is not None else model_cfg.get("cuda", "0")
    env = build_subprocess_env(cuda=cuda, offline=True, torch_compile_disable=True)

    experiments = [build_experiment(model_cfg, defaults, seed) for seed in args.seeds]
    for exp in experiments:
        if not Path(exp["train_data"]).exists():
            sys.exit(f"❌ 训练数据不存在: {exp['train_data']}")
        print(f"  • {exp['name']}  <- {exp['train_data']}")

    progress_file = output_root / "_progress.json"
    progress = load_progress(progress_file)

    for stage in args.steps:
        if stage == "summarize":
            if args.dry_run:
                print("  • [summarize] (dry-run) 跳过结果汇总")
                continue
            stage_summarize(experiments, output_root, test_datasets, args)
            continue
        print(f"\n{'=' * 70}\n[{stage}] ({len(experiments)} 个实验)\n{'=' * 70}")
        for exp in experiments:
            if stage == "convert":
                ok = stage_convert(exp, tokenizer_path, progress, progress_file, args, env)
            elif stage == "finetune":
                ok = stage_finetune(exp, model_cfg, model_path, template, output_root,
                                    progress, progress_file, args, env)
            elif stage == "test":
                ok = stage_test(exp, model_path, output_root, progress, progress_file, args, env)
            if not ok:
                print(f"    ❌ [{stage}] {exp['name']} 失败，中止")
                sys.exit(1)

    print("\n完成 ✅")


if __name__ == "__main__":
    main()
