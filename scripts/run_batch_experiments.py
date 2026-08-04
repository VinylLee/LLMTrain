#!/usr/bin/env python3
"""
批量实验编排器：采样 → 转换 → 微调 → 测试全自动流水线

读取 experiments/configs/experiments_config.json（或自定义配置文件），对其中定义的一个或多个
训练数据集依次完成完整实验流程，并在原始 + MR 测试集上评估。

用法:
  # 跑全部实验 × 3种子 = 24次
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --seeds 42 43 44

  # 先只跑2个实验验证
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --only mettrain_mnlim_4413_gemma3_4b
  original_snli_5340_gemma3_4b --seeds 42 43 44 --dry-run

  # 使用分层采样
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --seeds 42 43 44 --stratify

  # 跑全部实验
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json

  # 只跑原始数据实验
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --only original_snli_5340_gemma3_4b
  original_mnlim_4413_gemma3_4b original_sick_4439_gemma3_4b

  # 只跑 MetTrain 数据实验
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --only mettrain_snli_5340_gemma3_4b
  mettrain_mnlim_4413_gemma3_4b mettrain_sick_4439_gemma3_4b

  # 指定部分实验
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --only mettrain_mnlim_4413_gemma3_4b

  # 干跑（只看命令不执行）
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --dry-run

  # 断点续跑
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --resume

  # 覆盖模型
  python scripts/run_batch_experiments.py --config experiments/configs/experiments_config.json --model qwen
"""
import json
import subprocess
import sys
import os
import argparse
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

from project_runtime import (
    PROJECT_ROOT,
    apply_offline_mode,
    build_subprocess_env,
    configure_console_encoding,
    resolve_model_reference,
)

from sample_mettrain_pairid import (
    build_sampling_report,
    compute_data_signature as compute_sampling_data_signature,
)

WORK_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_ROOT = WORK_DIR / "output" / "experiments"

# 步骤名称
STEPS = ["sample", "convert", "finetune", "test_original", "test_mr"]

# 跟踪已写入的 cohort manifest（同一 run 内的首批变体写入，后续复用）
_cohort_manifest_written = set()
_cohort_sample_planned = set()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl_strict(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def workspace_relative(path):
    return Path(path).resolve().relative_to(WORK_DIR.resolve()).as_posix()


def pair_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def group_rows_by_pair_id(rows, label):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if "pair_id" not in row:
            raise ValueError(f"{label} row {index} 缺少 pair_id，不能安全复用 cohort")
        groups[pair_key(row["pair_id"])].append(row)
    return groups


def validate_sampled_groups_against_source(source_rows, sampled_rows, pair_ids):
    """Require sampled rows to contain every row of each selected source group."""
    normalized_pair_ids = [pair_key(value) for value in pair_ids]
    if len(normalized_pair_ids) != len(set(normalized_pair_ids)):
        raise ValueError("sampled pair manifest 含重复 pair_id")

    source_groups = group_rows_by_pair_id(source_rows, "source")
    sampled_groups = group_rows_by_pair_id(sampled_rows, "sampled")
    selected_ids = set(normalized_pair_ids)
    if set(sampled_groups) != selected_ids:
        raise ValueError(
            "sampled groups 与 pair manifest 不一致: "
            f"sampled_only={sorted(set(sampled_groups) - selected_ids)[:5]} "
            f"manifest_only={sorted(selected_ids - set(sampled_groups))[:5]}"
        )
    missing_source = selected_ids - set(source_groups)
    if missing_source:
        raise ValueError(f"pair manifest 含 source 中不存在的 group: {sorted(missing_source)[:5]}")

    for group_id in sorted(selected_ids):
        source_counter = Counter(canonical_sha256(row) for row in source_groups[group_id])
        sampled_counter = Counter(canonical_sha256(row) for row in sampled_groups[group_id])
        if source_counter != sampled_counter:
            raise ValueError(f"sampled group 未完整复用 source group: {group_id}")
    return {
        "source_group_count": len(source_groups),
        "selected_group_count": len(sampled_groups),
        "selected_sample_count": len(sampled_rows),
    }


def initialize_existing_cohort_metadata(exp, cohort_dir, git_sha):
    """Explicitly bootstrap metadata only after validating a legacy cohort."""
    cohort_id = exp.get("cohort_id")
    seed = exp.get("seed", 42)
    source_path = resolve_workspace_path(exp["train_data"])
    sampled_path = cohort_dir / "sampled.json"
    pair_manifest_path = sampled_path.with_suffix(".pair_ids.json")
    sampling_report_path = cohort_dir / "sampling_report.json"
    cohort_meta_path = cohort_dir / "cohort_meta.json"

    for required in (source_path, sampled_path, pair_manifest_path):
        if not required.exists():
            raise FileNotFoundError(f"初始化 cohort metadata 缺少文件: {required}")

    source_rows = load_jsonl_strict(source_path)
    sampled_rows = load_jsonl_strict(sampled_path)
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    if pair_manifest.get("seed") != seed:
        raise ValueError("sampled pair manifest seed 与 experiment 不一致")
    if pair_manifest.get("target") != exp["target"]:
        raise ValueError("sampled pair manifest target 与 experiment 不一致")
    pair_ids = pair_manifest.get("pair_ids")
    if not isinstance(pair_ids, list):
        raise ValueError("sampled pair manifest 缺少 pair_ids list")
    validate_sampled_groups_against_source(source_rows, sampled_rows, pair_ids)

    sampling_report = build_sampling_report(
        source_path=source_path,
        sampled_path=sampled_path,
        source_samples=source_rows,
        selected_samples=sampled_rows,
        selected_pair_ids=pair_ids,
        seed=seed,
        target=exp["target"],
        stratify=exp.get("stratify", False),
    )
    sampling_report_path.write_text(
        json.dumps(sampling_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    dirty = bool(subprocess.check_output(
        ["git", "status", "--short"], cwd=WORK_DIR, text=True
    ).strip())
    cohort_meta = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "seed": seed,
        "source_dataset": workspace_relative(source_path),
        "target_samples": exp["target"],
        "stratify": bool(exp.get("stratify", False)),
        "sampled_file": workspace_relative(sampled_path),
        "sample_manifest": workspace_relative(pair_manifest_path),
        "sampling_report": workspace_relative(sampling_report_path),
        "source_file_sha256": file_sha256(source_path),
        "sampled_file_sha256": file_sha256(sampled_path),
        "sample_manifest_sha256": file_sha256(pair_manifest_path),
        "sampling_report_sha256": file_sha256(sampling_report_path),
        "sample_data_signature": compute_sampling_data_signature(sampled_rows),
        "created_by_commit": git_sha,
        "working_tree_dirty": dirty,
    }
    cohort_meta_path.write_text(
        json.dumps(cohort_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return validate_cohort_artifacts(exp, cohort_dir)


def validate_cohort_artifacts(exp, cohort_dir):
    """Validate metadata, hashes, pair manifest, and full-group sampling."""
    cohort_id = exp.get("cohort_id")
    seed = exp.get("seed", 42)
    source_path = resolve_workspace_path(exp["train_data"])
    sampled_path = cohort_dir / "sampled.json"
    pair_manifest_path = sampled_path.with_suffix(".pair_ids.json")
    sampling_report_path = cohort_dir / "sampling_report.json"
    cohort_meta_path = cohort_dir / "cohort_meta.json"
    for required in (
        source_path,
        sampled_path,
        pair_manifest_path,
        sampling_report_path,
        cohort_meta_path,
    ):
        if not required.exists():
            raise FileNotFoundError(f"cohort 复用缺少审计文件: {required}")

    source_rows = load_jsonl_strict(source_path)
    sampled_rows = load_jsonl_strict(sampled_path)
    pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
    sampling_report = json.loads(sampling_report_path.read_text(encoding="utf-8"))
    cohort_meta = json.loads(cohort_meta_path.read_text(encoding="utf-8"))
    pair_ids = pair_manifest.get("pair_ids")
    if not isinstance(pair_ids, list):
        raise ValueError("sampled pair manifest 缺少 pair_ids list")
    validation = validate_sampled_groups_against_source(
        source_rows, sampled_rows, pair_ids
    )

    expected_meta = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "seed": seed,
        "source_dataset": workspace_relative(source_path),
        "target_samples": exp["target"],
        "stratify": bool(exp.get("stratify", False)),
        "sampled_file": workspace_relative(sampled_path),
        "sample_manifest": workspace_relative(pair_manifest_path),
        "sampling_report": workspace_relative(sampling_report_path),
        "source_file_sha256": file_sha256(source_path),
        "sampled_file_sha256": file_sha256(sampled_path),
        "sample_manifest_sha256": file_sha256(pair_manifest_path),
        "sampling_report_sha256": file_sha256(sampling_report_path),
        "sample_data_signature": compute_sampling_data_signature(sampled_rows),
    }
    mismatches = {
        key: {"expected": value, "actual": cohort_meta.get(key)}
        for key, value in expected_meta.items()
        if cohort_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(f"cohort_meta 不匹配: {mismatches}")
    if not cohort_meta.get("created_by_commit"):
        raise ValueError("cohort_meta 缺少 created_by_commit")

    expected_report = {
        "schema_version": 1,
        "seed": seed,
        "target_samples": exp["target"],
        "stratify": bool(exp.get("stratify", False)),
        "source_file_sha256": expected_meta["source_file_sha256"],
        "source_data_signature": compute_sampling_data_signature(source_rows),
        "source_sample_count": len(source_rows),
        "sampled_file_sha256": expected_meta["sampled_file_sha256"],
        "sample_data_signature": expected_meta["sample_data_signature"],
        "selected_sample_count": len(sampled_rows),
        "selected_group_count": validation["selected_group_count"],
        "selected_pair_ids_sha256": canonical_sha256(pair_ids),
    }
    report_mismatches = {
        key: {"expected": value, "actual": sampling_report.get(key)}
        for key, value in expected_report.items()
        if sampling_report.get(key) != value
    }
    if report_mismatches:
        raise ValueError(f"sampling_report 不匹配: {report_mismatches}")
    if pair_manifest.get("seed") != seed or pair_manifest.get("target") != exp["target"]:
        raise ValueError("sampled pair manifest seed/target 不匹配")
    if len(sampled_rows) < exp["target"]:
        raise ValueError("sampled rows 未达到 target")
    return {
        **validation,
        "sample_data_signature": expected_meta["sample_data_signature"],
        "sampled_file_sha256": expected_meta["sampled_file_sha256"],
        "sampling_report_sha256": expected_meta["sampling_report_sha256"],
    }


def enforce_stage2_training_gate(config, exp, conversion_report, output_root=None):
    """Return an auditable gate decision or refuse finetune.

    A normal run still requires Stage 2 PASS/OPEN.  A deliberately configured
    exploratory override may admit a narrow seed/mode scope while preserving
    the report's FAIL/BLOCKED status and recording that human validation did
    not pass.
    """
    gate = config.get("stage2_gate") or {}
    if not gate.get("required", False):
        return None
    report_value = gate.get("report")
    if not report_value:
        raise RuntimeError("Stage 2 gate required but report path is not configured")
    report_path = resolve_workspace_path(report_value)
    if not report_path.exists():
        raise RuntimeError(f"Stage 2 gate report 不存在: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mode = exp.get("mr_instruction_mode", "none")
    mode_report = (report.get("mode_reports") or {}).get(mode) or {}
    failures = []
    expected = {
        "automated_checks_status": "PASS",
        "seed": exp.get("seed", 42),
        "cohort_id": exp.get("cohort_id"),
        "data_signature": conversion_report.get("data_signature"),
        "manifest_hash": conversion_report.get("manifest_hash"),
        "ordered_sample_signature": conversion_report.get("ordered_sample_signature"),
        "instruction_template_version": config.get("instruction_template_version", 2),
    }
    for key, value in expected.items():
        if report.get(key) != value:
            failures.append(f"{key}: expected={value!r} actual={report.get(key)!r}")
    for key in ("converted_train_sha256", "converted_val_sha256"):
        if mode_report.get(key) != conversion_report.get(key):
            failures.append(
                f"{mode}.{key}: expected={conversion_report.get(key)!r} "
                f"actual={mode_report.get(key)!r}"
            )
    if failures:
        raise RuntimeError("Stage 2 training gate blocked:\n  " + "\n  ".join(failures))

    report_identity = {
        "report": workspace_relative(report_path),
        "report_sha256": file_sha256(report_path),
        "stage2_status": report.get("stage2_status"),
        "training_gate": report.get("training_gate"),
        "automated_checks_status": report.get("automated_checks_status"),
    }
    if report.get("stage2_status") == "PASS" and report.get("training_gate") == "OPEN":
        decision = {
            "schema_version": 1,
            "decision": "STAGE2_PASS",
            "seed": exp.get("seed", 42),
            "mr_instruction_mode": mode,
            "stage2_report": report_identity,
        }
        decision["sha256"] = canonical_sha256(decision)
        return decision

    override = gate.get("exploratory_override") or {}
    research_status = config.get("research_status") or {}
    override_failures = []
    if not override.get("enabled", False):
        override_failures.append("exploratory override is not enabled")
    if report.get("stage2_status") != "FAIL" or report.get("training_gate") != "BLOCKED":
        override_failures.append(
            "exploratory override requires the original Stage 2 status to remain FAIL/BLOCKED"
        )
    if research_status.get("classification") != "exploratory_pilot":
        override_failures.append("research_status.classification must be exploratory_pilot")
    if research_status.get("human_validation_status") != "QUICK_AUDIT_ONLY_NOT_PASSED":
        override_failures.append(
            "research_status.human_validation_status must be QUICK_AUDIT_ONLY_NOT_PASSED"
        )
    if research_status.get("confirmatory_use_allowed") is not False:
        override_failures.append("research_status.confirmatory_use_allowed must be false")
    if exp.get("seed", 42) not in override.get("allowed_seeds", []):
        override_failures.append(f"seed {exp.get('seed', 42)} is outside exploratory scope")
    if mode not in override.get("allowed_modes", []):
        override_failures.append(f"mode {mode!r} is outside exploratory scope")
    required_output_root = override.get("required_output_root")
    if not required_output_root:
        override_failures.append("exploratory required_output_root is not configured")
    else:
        actual_output_root = (
            workspace_relative(output_root)
            if output_root is not None
            else Path(config.get("output_root", "")).as_posix()
        )
        if actual_output_root != Path(required_output_root).as_posix():
            override_failures.append(
                "exploratory output root mismatch: "
                f"expected={required_output_root!r} actual={actual_output_root!r}"
            )
    ft_params = exp.get("ft_params") or {}
    max_training_steps = override.get("max_training_steps")
    max_training_epochs = override.get("max_training_epochs")
    configured_steps = ft_params.get("max_steps")
    configured_epochs = ft_params.get("epochs")
    if max_training_steps is not None and max_training_epochs is not None:
        override_failures.append(
            "exploratory override must configure exactly one training budget"
        )
        budget_decision_fields = {}
    elif max_training_steps is not None:
        if (
            not isinstance(max_training_steps, int)
            or isinstance(max_training_steps, bool)
            or max_training_steps < 1
            or not isinstance(configured_steps, int)
            or isinstance(configured_steps, bool)
            or configured_steps < 1
            or configured_steps > max_training_steps
        ):
            override_failures.append(
                "exploratory max_steps invalid: "
                f"configured={configured_steps!r} limit={max_training_steps!r}"
            )
        # Keep the original decision shape for existing smoke-run provenance.
        budget_decision_fields = {"max_training_steps": configured_steps}
    elif max_training_epochs is not None:
        numeric_limit = (
            isinstance(max_training_epochs, (int, float))
            and not isinstance(max_training_epochs, bool)
        )
        numeric_epochs = (
            isinstance(configured_epochs, (int, float))
            and not isinstance(configured_epochs, bool)
        )
        if (
            not numeric_limit
            or max_training_epochs <= 0
            or not numeric_epochs
            or configured_epochs <= 0
            or configured_epochs > max_training_epochs
            or configured_steps is not None
        ):
            override_failures.append(
                "exploratory epochs invalid: "
                f"configured={configured_epochs!r} limit={max_training_epochs!r} "
                f"max_steps={configured_steps!r}"
            )
        budget_decision_fields = {"max_training_epochs": configured_epochs}
    else:
        override_failures.append("exploratory training budget is not configured")
        budget_decision_fields = {}

    evidence_value = override.get("evidence")
    evidence_path = resolve_workspace_path(evidence_value) if evidence_value else None
    if evidence_path is None or not evidence_path.is_file():
        override_failures.append(f"exploratory evidence is missing: {evidence_path}")
        evidence_sha256 = None
    else:
        evidence_sha256 = file_sha256(evidence_path)
        if evidence_sha256 != override.get("evidence_sha256"):
            override_failures.append(
                "exploratory evidence SHA-256 mismatch: "
                f"expected={override.get('evidence_sha256')!r} actual={evidence_sha256!r}"
            )
    if override_failures:
        raise RuntimeError(
            "Stage 2 training gate blocked; exploratory override invalid:\n  "
            + "\n  ".join(override_failures)
        )

    decision = {
        "schema_version": 1,
        "decision": "EXPLORATORY_OVERRIDE",
        "classification": "exploratory_pilot",
        "human_validation_status": "QUICK_AUDIT_ONLY_NOT_PASSED",
        "confirmatory_use_allowed": False,
        **budget_decision_fields,
        "output_root": required_output_root,
        "seed": exp.get("seed", 42),
        "mr_instruction_mode": mode,
        "stage2_report": report_identity,
        "evidence": {
            "file": workspace_relative(evidence_path),
            "sha256": evidence_sha256,
        },
    }
    decision["sha256"] = canonical_sha256(decision)
    return decision


def build_run_signature(exp, config, git_sha, data_signature=None,
                        manifest_hash=None, test_signatures=None,
                        gate_decision=None):
    """Immutable identity for safe resume/reuse of a scientific run."""
    payload = {
        "schema_version": 1,
        "git_sha": git_sha,
        "mode": exp.get("mr_instruction_mode", "none"),
        "seeds": {
            "sampling": exp.get("seed"), "split": exp.get("seed"),
            "shuffle": exp.get("seed"), "trainer": exp.get("seed"),
            "data": exp.get("seed"),
        },
        "data_signature": data_signature,
        "manifest_hash": manifest_hash,
        "template": {"name": config.get("template"),
                     "version": config.get("template_version", 1)},
        "instruction_template": {
            "version": config.get("instruction_template_version", 2),
        },
        "model": {"path": config.get("model"),
                  "revision": config.get("model_revision")},
        "tokenizer": {"path": config.get("tokenizer", config.get("model")),
                      "revision": config.get("tokenizer_revision")},
        "training": exp.get("ft_params", {}),
        "generation": config.get("generation", {"do_sample": False, "max_new_tokens": 10}),
        "test_sets": test_signatures or {},
        "research_status": config.get("research_status"),
        "stage2_gate_decision_sha256": (
            gate_decision.get("sha256") if gate_decision else None
        ),
    }
    return {"run_signature": canonical_sha256(payload), "run_signature_payload": payload}


def load_config(path):
    """加载 JSON 配置文件"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_workspace_path(path_value):
    """Resolve a config/CLI path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else WORK_DIR / path


def load_progress(progress_file):
    """加载进度文件（如果存在）"""
    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress_file, progress):
    """保存进度文件"""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def step_completed(progress, exp_name, step):
    """检查某实验的某步骤是否已完成"""
    return progress.get(exp_name, {}).get(step) == "completed"


def mark_step(progress_file, progress, exp_name, step, status):
    """标记某步骤状态"""
    if exp_name not in progress:
        progress[exp_name] = {}
    progress[exp_name][step] = status
    save_progress(progress_file, progress)


def run_cmd(cmd, desc, dry_run=False, cwd=None, env=None):
    """运行命令或打印（dry-run模式）"""
    cmd = [str(part) for part in cmd]
    print(f"\n  ▶ {'[DRY-RUN]' if dry_run else 'RUN'} {desc}")
    print(f"    {subprocess.list2cmdline(cmd)}")
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=cwd or WORK_DIR, env=env)
    if result.returncode != 0:
        print(f"    ❌ {desc} 失败 (code={result.returncode})")
        return False
    print(f"    ✅ {desc} 完成")
    return True


def find_latest_complete_checkpoint(model_dir):
    """Return the newest resumable Trainer checkpoint, ignoring partial saves."""
    candidates = []
    if not model_dir.is_dir():
        return None
    required = (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    )
    for path in model_dir.glob("checkpoint-*"):
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
    """生成微调 YAML 配置内容"""
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
dataset_dir: {(WORK_DIR / 'data').as_posix()}
cutoff_len: 512
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
output_dir: {model_dir}
{resume_line}report_to: none
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
        "instruction_template_version": config.get("instruction_template_version", 2),
        "research_status": config.get("research_status"),
        "conda_environment": "llmtrain310",
    }
    with open(
        meta_dir / "experiment_meta.json", "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def run_experiment(exp, config, output_root, progress_file, selected_steps,
                   dry_run=False, resume=False,
                   initialize_cohort_metadata=False):
    """运行单个实验的完整流水线"""
    name = exp["name"]
    model_name = resolve_model_reference(
        config["model"], config.get("local_model_path")
    )
    tokenizer_name = resolve_model_reference(
        config.get("tokenizer", config["model"]),
        config.get("local_tokenizer_path", config.get("local_model_path")),
    )
    template = config.get("template", "gemma")
    cuda = config.get("cuda", "0")
    runtime_env = build_subprocess_env(
        cuda=cuda,
        offline=True,
        torch_compile_disable=True,
    )
    task_type = exp["task_type"]
    is_binary = task_type == "nli-binary"
    progress = load_progress(progress_file)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORK_DIR,
                                      text=True).strip()

    # 测试集范围由 config 的 test_sets 决定（统一 original+mr 的数据集键）
    ts = config.get("test_sets", {})
    test_ds = sorted(set(ts.get("original", {}).keys()) | set(ts.get("mr", {}).keys()))
    dataset_args = ["--datasets", ",".join(test_ds)] if test_ds else []
    batch_args = ["--batch-size", str(config.get("batch_size", 32))]

    print(f"\n{'='*70}")
    print(f"  📦 实验: {name}")
    print(f"  模型: {model_name}")
    print(f"  训练数据: {exp['train_data']}")
    print(f"  目标采样: {exp['target']}")
    print(f"  任务类型: {'二分类' if is_binary else '三分类'}")
    print(f"{'='*70}")

    # Step 0: 准备目录
    ft_data_dir = WORK_DIR / "data" / "ft_datasets" / name
    cohort_id = exp.get("cohort_id")
    cohort_dir = (WORK_DIR / "data" / "ft_datasets" / "_cohorts" /
                  cohort_id / f"seed_{exp.get('seed', 42)}") if cohort_id else ft_data_dir
    sampled_path = cohort_dir / "sampled.json"
    cohort_key = (cohort_id, exp.get("seed", 42)) if cohort_id else None
    exp_dir = output_root / name
    model_dir = exp_dir / "model"
    if not dry_run:
        ft_data_dir.mkdir(parents=True, exist_ok=True)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["tests/original", "tests/mr"]:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        save_experiment_meta(exp, model_name, template, output_root, config)

    if cohort_id and sampled_path.exists() and cohort_key not in _cohort_sample_planned:
        metadata_missing = not (
            (cohort_dir / "cohort_meta.json").exists() and
            (cohort_dir / "sampling_report.json").exists()
        )
        if dry_run:
            if metadata_missing and not initialize_cohort_metadata:
                raise RuntimeError(
                    "已有 sampled file 但缺少 cohort_meta.json/sampling_report.json；"
                    "即使 dry-run 也拒绝复用。请先显式使用 --initialize-cohort-metadata。"
                )
            if metadata_missing:
                print(f"    [DRY-RUN] 将初始化并验证 cohort metadata/signatures: {cohort_dir}")
            else:
                validate_cohort_artifacts(exp, cohort_dir)
                print(f"    [DRY-RUN] cohort metadata/signatures 验证通过: {cohort_dir}")
            _cohort_sample_planned.add(cohort_key)
        else:
            if metadata_missing:
                if not initialize_cohort_metadata:
                    raise RuntimeError(
                        "已有 sampled file 但缺少 cohort_meta.json/sampling_report.json；"
                        "拒绝复用。请先显式使用 --initialize-cohort-metadata。"
                    )
                initialize_existing_cohort_metadata(exp, cohort_dir, git_sha)
            else:
                validate_cohort_artifacts(exp, cohort_dir)
            _cohort_sample_planned.add(cohort_key)

    # ============================================================
    # Step 1: 采样
    # ============================================================
    print(f"\n  ── Step 1/5: 采样 ──")
    if "sample" not in selected_steps:
        print(f"    ⏩ 跳过（本次未选择该阶段）")
    elif cohort_id and (cohort_key in _cohort_sample_planned or sampled_path.exists()):
        print(f"    ⏩ 复用 cohort sampled file: {sampled_path}")
        ok = True
    elif resume and step_completed(progress, name, "sample"):
        print(f"    ⏩ 跳过（已完成）")
    else:
        sample_cmd = [
            sys.executable,
            WORK_DIR / "scripts" / "sample_mettrain_pairid.py",
            "--input", exp["train_data"],
            "--target", exp["target"],
            "--seed", exp.get("seed", 42),
            "--output", sampled_path,
            "--report-output", cohort_dir / "sampling_report.json",
        ]
        if exp.get("stratify"):
            sample_cmd.append("--stratify")
        ok = run_cmd(
            sample_cmd,
            f"采样 {name} (target={exp['target']})",
            dry_run,
            env=runtime_env,
        )
        if not dry_run:
            mark_step(progress_file, progress, name, "sample", "completed" if ok else "failed")
        if cohort_id and ok:
            if not dry_run:
                initialize_existing_cohort_metadata(exp, cohort_dir, git_sha)
            _cohort_sample_planned.add(cohort_key)
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
        mr_mode = exp.get("mr_instruction_mode", "none")

        # Cohort-based split manifest sharing
        cohort_id = exp.get("cohort_id")
        manifest_args = []
        if cohort_id:
            # Manifest 放在 cohort 的 ft_datasets 目录下（以 cohort 中第一个实验名作为基础路径）
            cohort_key = (cohort_id, exp.get("seed", 42))
            manifest_path = cohort_dir / "split_manifest.json"

            if manifest_path.exists():
                manifest_args = ["--split-manifest", str(manifest_path)]
            elif cohort_key not in _cohort_manifest_written:
                # 第一个变体：写入 manifest
                manifest_args = ["--write-split-manifest", str(manifest_path)]
                _cohort_manifest_written.add(cohort_key)
            else:
                # 后续变体：复用 manifest
                manifest_args = ["--split-manifest", str(manifest_path)]

        convert_cmd = [
            sys.executable,
            WORK_DIR / "scripts" / "convert_nli_to_ft.py",
            "--input", sampled_path,
            "--name", name,
            "--split",
            "--val-ratio", exp.get("val_ratio", 0.05),
            "--seed", exp.get("seed", 42),
            "--report-token-lengths",
            "--tokenizer-path", tokenizer_name,
            "--cutoff-len", exp.get("cutoff_len", 512),
            "--mr-instruction-mode", mr_mode,
            "--instruction-template-version",
            config.get("instruction_template_version", 2),
        ]
        if is_binary:
            convert_cmd.append("--binary")
        if exp.get("strict_pairing", False):
            convert_cmd.append("--strict-pairing")
        convert_cmd.extend(manifest_args)
        ok = run_cmd(
            convert_cmd,
            f"转换 {name} (mode={mr_mode})",
            dry_run,
            env=runtime_env,
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
        report_path = ft_data_dir / "conversion_report.json"
        manifest_path = cohort_dir / "split_manifest.json"
        _has_manifest = manifest_path.exists()
        if not report_path.exists():
            if dry_run:
                raise RuntimeError("训练 dry-run 前仍需现有 conversion report")
            raise RuntimeError("训练前缺少 conversion report")
        if cohort_id and not _has_manifest:
            if dry_run:
                raise RuntimeError("训练 dry-run 前仍需现有 conversion report 和 split manifest")
            raise RuntimeError("训练前缺少 conversion report 或 split manifest")
        conversion = json.loads(report_path.read_text(encoding="utf-8"))
        gate_decision = enforce_stage2_training_gate(
            config, exp, conversion, output_root=output_root
        )
        if gate_decision and gate_decision.get("decision") == "EXPLORATORY_OVERRIDE":
            print("    ⚠️  EXPLORATORY PILOT：Human Validation 未通过")
            print("    ⚠️  Stage 2 仍为 FAIL/BLOCKED；结果不得作为确认性证据")
        elif dry_run:
            print("    [DRY-RUN] Stage 2 gate 已验证为 PASS/OPEN")
        if dry_run:
            print("    [DRY-RUN] run_signature 将使用现有 conversion 产物计算并校验")
        test_signatures = {}
        for test_kind, mapping in config.get("test_sets", {}).items():
            for ds_name, value in mapping.items():
                path = resolve_workspace_path(value)
                if path.is_file():
                    test_signatures[f"{test_kind}:{ds_name}"] = file_sha256(path)
                elif path.is_dir():
                    test_signatures[f"{test_kind}:{ds_name}"] = canonical_sha256(
                        [(str(p.relative_to(path)), file_sha256(p))
                         for p in sorted(path.rglob("*.json"))])
        if not dry_run:
            signature = build_run_signature(
                exp, config, git_sha, conversion.get("data_signature"),
                conversion.get("manifest_hash"), test_signatures,
                gate_decision)
        signature_path = exp_dir / "run_signature.json"
        gate_decision_path = exp_dir / "stage2_gate_decision.json"
        if not dry_run and signature_path.exists():
            existing = json.loads(signature_path.read_text(encoding="utf-8"))
            if existing.get("run_signature") != signature["run_signature"]:
                raise RuntimeError("run_signature 不匹配，拒绝 resume/reuse")
        elif not dry_run:
            signature_path.write_text(
                json.dumps(signature, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if not dry_run and gate_decision is not None:
            gate_decision_path.write_text(
                json.dumps(gate_decision, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        yaml_content = build_yaml(exp, model_name, template, output_root)
        yaml_path = WORK_DIR / f"ft_config_{name}.yaml"
        if not dry_run:
            yaml_path.write_text(yaml_content, encoding="utf-8")

        ok = run_cmd(
            [sys.executable, "-m", "llamafactory.cli", "train", yaml_path],
            f"微调 {name}",
            dry_run,
            env=runtime_env,
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
        test_cmd = [
            sys.executable,
            WORK_DIR / "scripts" / "test_mettrain_experiment.py",
            "--experiment", name,
            "--base-model", model_name,
            "--lora", model_dir,
            "--output-root", output_root,
            *dataset_args,
            *batch_args,
            "--skip-mr",
        ]
        if is_binary:
            test_cmd.extend(["--model-task", "binary"])
        ok = run_cmd(
            test_cmd,
            f"测试原始数据 {name}",
            dry_run,
            env=runtime_env,
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
        test_cmd = [
            sys.executable,
            WORK_DIR / "scripts" / "test_mettrain_experiment.py",
            "--experiment", name,
            "--base-model", model_name,
            "--lora", model_dir,
            "--output-root", output_root,
            *dataset_args,
            *batch_args,
            "--skip-original",
        ]
        if is_binary:
            test_cmd.extend(["--model-task", "binary"])
        ok = run_cmd(
            test_cmd,
            f"测试MR数据 {name}",
            dry_run,
            env=runtime_env,
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

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        # 读取原始测试结果
        original_results = {}
        orig_dir = exp_dir / "tests" / "original"
        if orig_dir.exists():
            for ds_file in sorted(orig_dir.glob("*.jsonl")):
                ds_name = ds_file.stem
                total = correct = 0
                with open(ds_file, encoding="utf-8") as f:
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
                with open(ds_file, encoding="utf-8") as f:
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
                "research_status": meta.get("research_status"),
                "original": original_results,
                "mr": mr_results,
            })

    # 写 SUMMARY.md
    all_ds = ["mnlim", "mnlimm", "sick", "snli"]
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 批量实验测试结果汇总\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if any(
            (summary.get("research_status") or {}).get("classification")
            == "exploratory_pilot"
            for summary in summaries
        ):
            f.write(
                "> **EXPLORATORY PILOT — Human Validation 未通过。** "
                "Stage 2 仍为 FAIL/BLOCKED；以下结果不得作为确认性证据。\n\n"
            )

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
                with open(ds_file, encoding="utf-8") as f:
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
    with open(result_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 实验结果汇总\n\n")
        seed_str = ", ".join(str(s) for s in sorted(seeds))
        f.write(f"种子: {seed_str}  ")
        f.write(f"模型: {config.get('model', 'google/gemma-3-4b-it')}\n\n")
        research_status = config.get("research_status") or {}
        if research_status.get("classification") == "exploratory_pilot":
            f.write(
                "> **EXPLORATORY PILOT — Human Validation 未通过。** "
                "Stage 2 仍为 FAIL/BLOCKED；以下结果不得作为确认性证据。\n\n"
            )

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
    configure_console_encoding()
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
    parser.add_argument(
        "--initialize-cohort-metadata",
        action="store_true",
        help="显式验证并为旧 cohort 创建缺失的 cohort_meta/sampling_report；默认拒绝缺元数据复用",
    )
    args = parser.parse_args()

    config_path = resolve_workspace_path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # 模型已缓存在本地 HF hub，且当前环境无法访问 huggingface.co（SSL EOF）。
    # 强制离线模式，避免 finetune/test 子进程因联网校验 tokenizer 而失败。
    apply_offline_mode(True)

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
    defaults = config.get("experiment_defaults", {})
    experiments = [{**defaults, **experiment} for experiment in experiments]
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
    research_status = config.get("research_status") or {}
    if research_status.get("classification") == "exploratory_pilot":
        print("  ⚠️  研究口径: EXPLORATORY PILOT")
        print("  ⚠️  Human Validation: QUICK AUDIT ONLY — NOT PASSED")
        print("  ⚠️  禁止将本轮结果作为确认性证据")
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
                initialize_cohort_metadata=args.initialize_cohort_metadata,
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
    if success != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
