#!/usr/bin/env python3
"""RQ2 controlled SNLI pipeline.

Stages are explicit so audit/conversion can be reviewed before GPU work:
    audit -> sample -> convert -> finetune -> test -> summarize
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RQ2_DIR = PROJECT_ROOT / "RQ2"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DEFAULT_CONFIG = RQ2_DIR / "configs" / "rq2_snli_config.json"
STAGES = ["audit", "sample", "convert", "finetune", "test", "summarize"]

sys.path.insert(0, str(SCRIPTS_DIR))
from project_runtime import build_subprocess_env, resolve_model_reference  # noqa: E402
from mr_instruction_design import (  # noqa: E402
    ALL_MODE_NAMES,
    CORE_MODES,
    INSTRUCTION_DESIGN_VERSION,
    mode_design_meta,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_cmd(cmd, desc, dry_run=False, env=None):
    cmd = [str(item) for item in cmd]
    print(f"\n▶ {'[DRY-RUN] ' if dry_run else ''}{desc}")
    print("  " + subprocess.list2cmdline(cmd))
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode:
        print(f"❌ {desc} failed with code {result.returncode}")
        return False
    return True


def load_progress(path):
    return load_json(path) if Path(path).exists() else {}


def mark(progress, path, name, stage, status="completed"):
    progress.setdefault(name, {})[stage] = status
    save_json(path, progress)


def completed(progress, name, stage):
    return progress.get(name, {}).get(stage) == "completed"


def copy_test_snapshot(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    elif destination.read_bytes() != Path(source).read_bytes():
        raise ValueError(f"RQ2 test snapshot differs from source: {destination}")


def build_yaml(exp, cfg, output_root, smoke):
    training = cfg["training"]
    max_steps = 20 if smoke else None
    save_steps = 20 if smoke else training["save_steps"]
    model_dir = output_root / exp["name"] / "model"
    dataset_dir = RQ2_DIR / "data"
    duration = f"max_steps: {max_steps}" if max_steps is not None else f"num_train_epochs: {training['epochs']}"
    return f"""### RQ2 SNLI LoRA: {exp['name']}
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: {training['rank']}
lora_alpha: {training['rank'] * 2}
lora_dropout: 0.05
dataset: {exp['dataset_name']}
dataset_dir: {dataset_dir.as_posix()}
cutoff_len: {cfg['data']['cutoff_len']}
per_device_train_batch_size: {training['batch']}
gradient_accumulation_steps: {training['grad_accum']}
learning_rate: {training['learning_rate']}
{duration}
lr_scheduler_type: cosine
warmup_ratio: 0.1
logging_steps: 10
save_steps: {save_steps}
save_total_limit: {training['save_total_limit']}
eval_dataset: {exp['dataset_name']}_val
eval_strategy: "no"
seed: {exp['seed']}
data_seed: {exp['seed']}
output_dir: {model_dir.as_posix()}
report_to: none
bf16: true
trust_remote_code: true
remove_unused_columns: false
model_name_or_path: {cfg['model']['local_path']}
template: {cfg['model']['template']}
use_cache: false
"""


def rq2_converted_dir(cfg, data_root):
    """Where converted datasets go for this design version.

    The v3 block wording differs from the frozen v2 templates, so v3 runs must
    not overwrite the historical v2 conversions.  Configs may pin the directory
    explicitly via ``converted_dir``; otherwise the v3 default is used.
    """
    configured = cfg.get("converted_dir")
    if configured:
        return project_path(configured)
    return data_root / "converted"


def experiments(seeds, modes):
    return [
        {
            "name": f"rq2_snli_{mode}_seed{seed}",
            "dataset_name": f"rq2_snli_{mode}_seed{seed}",
            "mode": mode,
            "seed": seed,
        }
        for seed in seeds for mode in modes
    ]


def main():
    parser = argparse.ArgumentParser(description="RQ2 controlled SNLI experiment")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--modes", nargs="+", choices=sorted(ALL_MODE_NAMES),
                        help=f"核心 2x2x2: {list(CORE_MODES)}；其余为 controls/diagnostic")
    parser.add_argument("--steps", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--smoke", action="store_true", help="limit each training run to 20 steps")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_json(project_path(args.config))
    seeds = args.seeds or cfg["seeds"]
    modes = args.modes or cfg["modes"]
    output_root = project_path(args.output_root or cfg["output_root"])
    data_root = RQ2_DIR / "data"
    test_source = project_path(cfg["data"]["merged_test_input"])
    test_snapshot = data_root / "test" / "mr_test_data_merged" / "snli.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / ("_progress_smoke.json" if args.smoke else "_progress.json")
    progress = load_progress(progress_path)
    env = build_subprocess_env(cuda=cfg["model"]["cuda"], offline=True, torch_compile_disable=True)
    model_path = resolve_model_reference(cfg["model"]["hub_id"], cfg["model"].get("local_path"))
    exps = experiments(seeds, modes)

    if "audit" in args.steps:
        copy_test_snapshot(test_source, test_snapshot)
        audit_path = data_root / "audits" / "snli_preflight.json"
        ok = run_cmd([
            sys.executable, str(RQ2_DIR / "scripts" / "audit_snli_rq2.py"),
            "--train", project_path(cfg["data"]["training_input"]),
            "--test", test_snapshot,
            "--output", audit_path,
        ], "SNLI source-disjoint preflight", args.dry_run, env)
        if not ok:
            return 1

    cohorts_root = project_path(cfg["cohorts_dir"]) if cfg.get("cohorts_dir") else data_root / "cohorts"

    for exp in exps:
        name = exp["name"]
        cohort_dir = cohorts_root / f"snli_seed{exp['seed']}"
        sampled = cohort_dir / "sampled.json"
        sampling_report = cohort_dir / "sampling_report.json"
        manifest = cohort_dir / "split_manifest.json"
        converted_dir = rq2_converted_dir(cfg, data_root)
        exp_dir = output_root / name

        if "sample" in args.steps and (not args.resume or not sampled.exists()):
            ok = run_cmd([
                sys.executable, str(SCRIPTS_DIR / "sample_mettrain_pairid.py"),
                "--input", project_path(cfg["data"]["training_input"]),
                "--target", cfg["data"]["target_samples"],
                "--seed", exp["seed"],
                "--output", sampled,
                "--report-output", sampling_report,
            ], f"Sample shared cohort seed {exp['seed']}", args.dry_run, env)
            if not ok:
                return 1

        if "convert" in args.steps and (not args.resume or not (converted_dir / name / "conversion_report.json").exists()):
            first_mode = exp["mode"] == modes[0]
            cmd = [
                sys.executable, str(RQ2_DIR / "scripts" / "convert_snli_rq2.py"),
                "--input", sampled,
                "--name", name,
                "--mode", exp["mode"],
                "--seed", exp["seed"],
                "--output-dir", str(converted_dir.relative_to(PROJECT_ROOT)),
                "--manifest", manifest,
                "--val-ratio", cfg["data"]["val_ratio"],
                "--cutoff-len", cfg["data"]["cutoff_len"],
                "--tokenizer-path", model_path,
                "--instruction-template-version",
                cfg.get("instruction_template_version", INSTRUCTION_DESIGN_VERSION),
            ]
            if cfg.get("require_composite_provenance"):
                cmd.append("--require-composite-provenance")
            if first_mode:
                cmd.append("--write-manifest")
            if not run_cmd(cmd, f"Convert {name}", args.dry_run, env):
                return 1

        if "finetune" in args.steps:
            if args.resume and completed(progress, name, "finetune"):
                pass
            else:
                yaml_path = data_root / "generated_configs" / f"{name}{'_smoke' if args.smoke else ''}.yaml"
                yaml_path.parent.mkdir(parents=True, exist_ok=True)
                yaml_path.write_text(build_yaml(exp, cfg, output_root, args.smoke), encoding="utf-8")
                save_json(exp_dir / "experiment_meta.json", {
                    "experiment": name,
                    "model": cfg["model"]["hub_id"],
                    "model_path": model_path,
                    "mode": exp["mode"],
                    # P/O/R/L design metadata: recorded explicitly so downstream
                    # analysis never re-derives the design from the mode name.
                    "mr_design": {
                        **mode_design_meta(exp["mode"]),
                        "template_version": cfg.get(
                            "instruction_template_version", INSTRUCTION_DESIGN_VERSION
                        ),
                    },
                    "seed": exp["seed"],
                    "train_input": cfg["data"]["training_input"],
                    "target_samples": cfg["data"]["target_samples"],
                    "training": cfg["training"],
                    "smoke": args.smoke,
                    "created": datetime.now().isoformat(),
                })
                ok = run_cmd([sys.executable, "-m", "llamafactory.cli", "train", yaml_path], f"Fine-tune {name}", args.dry_run, env)
                if not args.dry_run and yaml_path.exists():
                    yaml_path.unlink()
                if not ok:
                    return 1
                if not args.dry_run:
                    mark(progress, progress_path, name, "finetune")

        if "test" in args.steps:
            cmd = [
                sys.executable, str(SCRIPTS_DIR / "test_mettrain_experiment.py"),
                "--experiment", name,
                "--base-model", model_path,
                "--lora", exp_dir / "model",
                "--output-root", output_root,
                "--datasets", "snli",
                "--batch-size", cfg["training"]["test_batch_size"],
                "--merged", "--skip-original", "--skip-mr",
                "--merged-path", data_root / "test",
            ]
            if args.smoke:
                cmd.extend(["--max-samples", "128"])
            ok = run_cmd(cmd, f"Merged SNLI test {name}", args.dry_run, env)
            if not ok:
                return 1
            if not args.dry_run:
                mark(progress, progress_path, name, "test")

    if "summarize" in args.steps:
        if not run_cmd([
            sys.executable, str(RQ2_DIR / "scripts" / "summarize_rq2_snli.py"),
            "--output-root", output_root,
        ], "Summarize RQ2 SNLI metrics", args.dry_run, env):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
