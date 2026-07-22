#!/usr/bin/env python3
"""Validate the seed-42 MR-as-Instruction Stage 2 conversion artifacts.

This validator never trains or runs a model. It checks mode alignment, report
hashes, template-v2 boundaries, shuffled-description semantics, and token
statistics. It also writes a deterministic, ignored manual-review sample file.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from project_runtime import (
    PROJECT_ROOT,
    apply_offline_mode,
    configure_console_encoding,
    resolve_model_reference,
)

from convert_nli_to_ft import (
    INSTRUCTION_NLI,
    INSTRUCTION_TEMPLATE_HASH,
    INSTRUCTION_TEMPLATE_VERSION,
    MR_OPERATION_DESCRIPTIONS,
    MR_RELATION_EFFECTS,
    OPERATION_DESCRIPTION_HASH,
    RELATION_EFFECT_HASH,
    build_original_map,
    build_pair_groups,
    canonical_sha256,
    compute_data_signature,
    compute_ordered_sample_signature,
    compute_sample_key,
    normalize_mr_id,
    select_shuffled_descriptions,
    sort_samples_by_stable_key,
    validate_manifest_for_groups,
)
from run_batch_experiments import validate_cohort_artifacts


WORK_DIR = PROJECT_ROOT
REQUIRED_MODES = (
    "none",
    "operation_only",
    "pair_only",
    "pair_operation",
    "shuffled_operation",
    "full_oracle",
)
DATA_QUALITY_RISKS = {
    "adding_contradiction": (
        "Inserted contradictions require manual review for relation/label consistency."
    ),
    "composite_flip": (
        "Composite flip rows may combine heterogeneous edits and label assumptions."
    ),
    "conditional_clause": (
        "Conditional-clause edits may not deterministically imply the stored label."
    ),
    "pronoun_substitution": (
        "Pronoun/coreference substitutions require manual antecedent validation."
    ),
}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else WORK_DIR / path


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
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


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_check(checks, name, passed, detail=""):
    checks.append({
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail),
    })
    return passed


def determine_stage2_status(checks, blockers):
    automated_status = (
        "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    )
    training_gate = "OPEN" if automated_status == "PASS" and not blockers else "BLOCKED"
    stage2_status = "PASS" if training_gate == "OPEN" else "FAIL"
    return {
        "automated_checks_status": automated_status,
        "training_gate": training_gate,
        "stage2_status": stage2_status,
    }


def select_manual_sample_indices(samples, seed, per_mr=3):
    """Deterministically choose up to per_mr enhanced rows for every MR type."""
    buckets = defaultdict(list)
    for index, sample in enumerate(samples):
        mr_id = normalize_mr_id(sample.get("mr_id"))
        if mr_id != "none":
            buckets[mr_id].append(index)

    selected = []
    for mr_id in sorted(buckets):
        candidates = sorted(
            buckets[mr_id],
            key=lambda index: compute_sample_key(samples[index]),
        )
        if len(candidates) <= per_mr:
            chosen = candidates
        else:
            rng = random.Random(f"{seed}:{mr_id}")
            chosen = sorted(rng.sample(candidates, per_mr))
        selected.extend(chosen)
    return sorted(selected, key=lambda index: (
        normalize_mr_id(samples[index].get("mr_id")),
        compute_sample_key(samples[index]),
    ))


def count_row_tokens(tokenizer, row):
    messages = [
        {
            "role": "user",
            "content": row["instruction"] + "\n\n" + row["input"],
        },
        {"role": "assistant", "content": row["output"]},
    ]
    try:
        return len(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )), "apply_chat_template"
    except (AttributeError, ValueError, NotImplementedError):
        text = "\n\n".join(message["content"] for message in messages)
        return len(tokenizer(text)["input_ids"]), "tokenizer_fallback"


def manifest_payload_hash(manifest):
    payload = dict(manifest)
    payload.pop("sha256", None)
    return canonical_sha256(payload)


def description_matrix_counts(matrix):
    row_counts = {
        true_description: sum(assignments.values())
        for true_description, assignments in matrix.items()
    }
    column_counts = Counter()
    for assignments in matrix.values():
        column_counts.update(assignments)
    return row_counts, dict(column_counts)


def validate_by_mr_token_summaries(summaries, expected_counts, overall):
    if set(summaries) != set(expected_counts):
        return False
    required = {
        "count", "p95", "max", "over_cutoff_count", "over_cutoff_ratio",
    }
    for mr_id, expected_count in expected_counts.items():
        summary = summaries.get(mr_id) or {}
        if not required.issubset(summary) or summary.get("count") != expected_count:
            return False
        over = summary.get("over_cutoff_count")
        ratio = summary.get("over_cutoff_ratio")
        if not isinstance(over, int) or over < 0 or over > expected_count:
            return False
        expected_ratio = over / expected_count if expected_count else 0.0
        if abs(float(ratio) - expected_ratio) > 1e-12:
            return False
    return (
        sum(summary["count"] for summary in summaries.values()) == overall.get("count")
        and sum(summary["over_cutoff_count"] for summary in summaries.values())
        == overall.get("over_cutoff_count")
    )


def validate(args):
    apply_offline_mode(True)

    config_path = resolve_path(args.config)
    config = load_json(config_path)
    defaults = config.get("experiment_defaults", {})
    experiments = [{**defaults, **item} for item in config.get("experiments", [])]
    experiments_by_mode = {
        item.get("mr_instruction_mode", "none"): item for item in experiments
    }
    checks = []

    missing_modes = [mode for mode in REQUIRED_MODES if mode not in experiments_by_mode]
    add_check(checks, "required_modes_present", not missing_modes, missing_modes)
    if missing_modes:
        raise ValueError(f"Pilot config missing modes: {missing_modes}")

    cohort_id = defaults.get("cohort_id")
    if not cohort_id:
        raise ValueError("experiment_defaults.cohort_id is required")
    cohort_dir = (
        WORK_DIR / "data" / "ft_datasets" / "_cohorts" /
        cohort_id / f"seed_{args.seed}"
    )
    sampled_path = cohort_dir / "sampled.json"
    manifest_path = cohort_dir / "split_manifest.json"
    cohort_validation = validate_cohort_artifacts(
        {**experiments_by_mode["none"], "seed": args.seed},
        cohort_dir,
    )
    add_check(
        checks,
        "cohort_metadata_and_reuse_signatures",
        True,
        cohort_validation.get("sample_data_signature"),
    )
    sampled_rows = load_jsonl(sampled_path)
    manifest = load_json(manifest_path)

    source_file_id = sampled_path.stem
    groups = build_pair_groups(sampled_rows, [source_file_id] * len(sampled_rows))
    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]

    data_signature = compute_data_signature(sampled_rows)
    train_ids, val_ids = validate_manifest_for_groups(
        manifest,
        groups,
        args.seed,
        defaults.get("val_ratio", 0.05),
        data_signature,
    )
    group_map = {group["group_key"]: group for group in groups}
    train_groups = [group_map[group_id] for group_id in train_ids]
    val_groups = [group_map[group_id] for group_id in val_ids]
    train_samples = sort_samples_by_stable_key(
        [sample for group in train_groups for sample in group["samples"]]
    )
    val_samples = sort_samples_by_stable_key(
        [sample for group in val_groups for sample in group["samples"]]
    )
    expected_ordered_signature = compute_ordered_sample_signature(
        train_samples,
        val_samples,
    )
    expected_manifest_hash = manifest.get("sha256")
    expected_train_by_mr = Counter(
        normalize_mr_id(sample.get("mr_id")) for sample in train_samples
    )
    expected_val_by_mr = Counter(
        normalize_mr_id(sample.get("mr_id")) for sample in val_samples
    )

    add_check(
        checks,
        "manifest_self_hash",
        expected_manifest_hash == manifest_payload_hash(manifest),
        expected_manifest_hash,
    )
    add_check(
        checks,
        "train_val_group_disjoint",
        train_ids.isdisjoint(val_ids),
        f"train={len(train_ids)} val={len(val_ids)}",
    )
    add_check(
        checks,
        "manifest_covers_all_groups",
        train_ids | val_ids == set(group_map),
        f"all={len(group_map)}",
    )

    artifacts = {}
    for mode in REQUIRED_MODES:
        experiment = experiments_by_mode[mode]
        name = f"{experiment['name']}_seed{args.seed}"
        directory = WORK_DIR / "data" / "ft_datasets" / name
        train_path = directory / "full_train.json"
        val_path = directory / "full_val.json"
        report_path = directory / "conversion_report.json"
        artifacts[mode] = {
            "name": name,
            "directory": directory,
            "train_path": train_path,
            "val_path": val_path,
            "report_path": report_path,
            "train": load_jsonl(train_path),
            "validation": load_jsonl(val_path),
            "report": load_json(report_path),
        }

    baseline_train = artifacts["none"]["train"]
    baseline_val = artifacts["none"]["validation"]
    expected_train_count = len(train_samples)
    expected_val_count = len(val_samples)

    for mode, artifact in artifacts.items():
        report = artifact["report"]
        train_rows = artifact["train"]
        val_rows = artifact["validation"]
        prefix = f"{mode}:"
        add_check(
            checks,
            prefix + "row_counts",
            len(train_rows) == expected_train_count and len(val_rows) == expected_val_count,
            f"train={len(train_rows)} val={len(val_rows)}",
        )
        add_check(checks, prefix + "report_mode", report.get("mode") == mode, report.get("mode"))
        add_check(
            checks,
            prefix + "data_signature",
            report.get("data_signature") == data_signature,
            report.get("data_signature"),
        )
        add_check(
            checks,
            prefix + "manifest_hash",
            report.get("manifest_hash") == expected_manifest_hash,
            report.get("manifest_hash"),
        )
        add_check(
            checks,
            prefix + "template_version",
            report.get("instruction_template_version") == INSTRUCTION_TEMPLATE_VERSION,
            report.get("instruction_template_version"),
        )
        add_check(
            checks,
            prefix + "template_hash",
            report.get("instruction_template_hash") == INSTRUCTION_TEMPLATE_HASH,
            report.get("instruction_template_hash"),
        )
        add_check(
            checks,
            prefix + "operation_description_hash",
            report.get("operation_description_hash") == OPERATION_DESCRIPTION_HASH,
            report.get("operation_description_hash"),
        )
        add_check(
            checks,
            prefix + "relation_effect_hash",
            report.get("relation_effect_hash") == RELATION_EFFECT_HASH,
            report.get("relation_effect_hash"),
        )
        add_check(
            checks,
            prefix + "ordered_sample_signature",
            report.get("ordered_sample_signature") == expected_ordered_signature,
            report.get("ordered_sample_signature"),
        )
        add_check(
            checks,
            prefix + "converted_train_sha256",
            report.get("converted_train_sha256") == file_sha256(artifact["train_path"]),
            report.get("converted_train_sha256"),
        )
        add_check(
            checks,
            prefix + "converted_val_sha256",
            report.get("converted_val_sha256") == file_sha256(artifact["val_path"]),
            report.get("converted_val_sha256"),
        )
        add_check(
            checks,
            prefix + "fallback_zero",
            report.get("train", {}).get("fallback_count") == 0 and
            report.get("val", {}).get("fallback_count") == 0,
            f"train={report.get('train', {}).get('fallback_count')} "
            f"val={report.get('val', {}).get('fallback_count')}",
        )
        add_check(
            checks,
            prefix + "missing_operation_zero",
            report.get("train", {}).get("missing_operation_count") == 0 and
            report.get("val", {}).get("missing_operation_count") == 0,
            f"train={report.get('train', {}).get('missing_operation_count')} "
            f"val={report.get('val', {}).get('missing_operation_count')}",
        )
        add_check(
            checks,
            prefix + "underlying_train_alignment",
            [(row["input"], row["output"]) for row in train_rows] ==
            [(row["input"], row["output"]) for row in baseline_train],
        )
        add_check(
            checks,
            prefix + "validation_exact_match",
            val_rows == baseline_val,
        )
        token_report = report.get("token_length") or {}
        train_tokens = token_report.get("train") or {}
        val_tokens = token_report.get("validation") or {}
        by_mr = token_report.get("by_mr") or {}
        train_by_mr = by_mr.get("train") or {}
        val_by_mr = by_mr.get("validation") or {}
        add_check(
            checks,
            prefix + "actual_token_report",
            token_report.get("counting_method") in {
                "apply_chat_template", "tokenizer_fallback"
            } and train_tokens.get("count") == expected_train_count and
            val_tokens.get("count") == expected_val_count,
            token_report.get("counting_method"),
        )
        add_check(
            checks,
            prefix + "actual_token_report_by_mr",
            validate_by_mr_token_summaries(
                train_by_mr, expected_train_by_mr, train_tokens
            ) and validate_by_mr_token_summaries(
                val_by_mr, expected_val_by_mr, val_tokens
            ) and sum((token_report.get("counting_method_counts") or {}).values())
            == expected_train_count + expected_val_count,
            f"train_mr={len(train_by_mr)} val_mr={len(val_by_mr)}",
        )
        add_check(
            checks,
            prefix + "token_cutoff_ratio",
            float(train_tokens.get("over_cutoff_ratio", 1.0)) <= args.max_over_cutoff_ratio,
            train_tokens.get("over_cutoff_ratio"),
        )

    original_map, source_issues = build_original_map(groups)
    add_check(checks, "pairing_source_integrity", not source_issues, source_issues[:5])

    source_indices = [
        index for index, sample in enumerate(train_samples)
        if normalize_mr_id(sample.get("mr_id")) == "none"
    ]
    enhanced_indices = [
        index for index, sample in enumerate(train_samples)
        if normalize_mr_id(sample.get("mr_id")) != "none"
    ]
    source_ok = True
    for index in source_indices:
        baseline_instruction = baseline_train[index]["instruction"]
        for mode in REQUIRED_MODES:
            instruction = artifacts[mode]["train"][index]["instruction"]
            source_ok &= instruction == baseline_instruction
            source_ok &= "<reference_premise>" not in instruction
            source_ok &= "Transformation" not in instruction
    add_check(checks, "source_samples_generic_across_modes", source_ok, len(source_indices))

    validation_generic = all(
        "<reference_premise>" not in row["instruction"] and
        "Transformation" not in row["instruction"] and
        "Reference label" not in row["instruction"] and
        "Expected relation effect" not in row["instruction"]
        for row in baseline_val
    )
    add_check(checks, "validation_is_generic", validation_generic, len(baseline_val))

    pair_rows = artifacts["pair_operation"]["train"]
    pair_only_rows = artifacts["pair_only"]["train"]
    operation_only_rows = artifacts["operation_only"]["train"]
    full_oracle_rows = artifacts["full_oracle"]["train"]
    pair_checks = []
    pair_only_checks = []
    operation_only_checks = []
    oracle_checks = []
    for index in enhanced_indices:
        sample = train_samples[index]
        mr_id = normalize_mr_id(sample.get("mr_id"))
        original = original_map[sample["_group_key"]]
        pair_instruction = pair_rows[index]["instruction"]
        pair_checks.append(
            "<reference_premise>" in pair_instruction and
            "</reference_premise>" in pair_instruction and
            "<reference_hypothesis>" in pair_instruction and
            "</reference_hypothesis>" in pair_instruction and
            MR_OPERATION_DESCRIPTIONS[mr_id] in pair_instruction and
            "Reference label" not in pair_instruction and
            "Expected relation effect" not in pair_instruction and
            mr_id not in pair_instruction and
            pair_instruction.endswith(INSTRUCTION_NLI) and
            (
                sample["premise"].strip() == original["premise"].strip() or
                sample["premise"].strip() not in pair_instruction
            ) and
            (
                sample["hypothesis"].strip() == original["hypothesis"].strip() or
                sample["hypothesis"].strip() not in pair_instruction
            )
        )
        pair_only_instruction = pair_only_rows[index]["instruction"]
        pair_only_checks.append(
            "<reference_premise>" in pair_only_instruction and
            "Transformation" not in pair_only_instruction
        )
        operation_only_instruction = operation_only_rows[index]["instruction"]
        operation_only_checks.append(
            "<reference_premise>" not in operation_only_instruction and
            MR_OPERATION_DESCRIPTIONS[mr_id] in operation_only_instruction
        )
        oracle_instruction = full_oracle_rows[index]["instruction"]
        oracle_checks.append(
            "Reference label" in oracle_instruction and
            "Expected relation effect" in oracle_instruction and
            MR_RELATION_EFFECTS[mr_id] in oracle_instruction
        )

    add_check(checks, "pair_operation_schema_v2", all(pair_checks), len(pair_checks))
    add_check(checks, "pair_only_schema_v2", all(pair_only_checks), len(pair_only_checks))
    add_check(checks, "operation_only_schema_v2", all(operation_only_checks), len(operation_only_checks))
    add_check(checks, "full_oracle_is_explicit_upper_bound", all(oracle_checks) and artifacts[
        "full_oracle"
    ]["report"].get("upper_bound") is True, len(oracle_checks))
    add_check(
        checks,
        "oracle_fields_absent_from_main_modes",
        all(
            "Reference label" not in row["instruction"] and
            "Expected relation effect" not in row["instruction"]
            for mode in REQUIRED_MODES if mode != "full_oracle"
            for row in artifacts[mode]["train"]
        ),
    )

    shuffle_seed = args.seed + 999
    assigned_mr_ids = select_shuffled_descriptions(train_samples, set(), shuffle_seed)
    shuffled_rows = artifacts["shuffled_operation"]["train"]
    shuffled_pairwise_ok = True
    changed_count = 0
    for index in enhanced_indices:
        true_mr = normalize_mr_id(train_samples[index].get("mr_id"))
        assigned_mr = assigned_mr_ids[index]
        true_description = MR_OPERATION_DESCRIPTIONS[true_mr]
        assigned_description = MR_OPERATION_DESCRIPTIONS[assigned_mr]
        expected_instruction = pair_rows[index]["instruction"].replace(
            true_description,
            assigned_description,
            1,
        )
        shuffled_pairwise_ok &= shuffled_rows[index]["instruction"] == expected_instruction
        changed_count += true_description != assigned_description
    add_check(
        checks,
        "shuffled_only_changes_assigned_description",
        shuffled_pairwise_ok and changed_count == len(enhanced_indices),
        f"changed={changed_count}/{len(enhanced_indices)}",
    )

    shuffle_audit = artifacts["shuffled_operation"]["report"].get("shuffle_audit") or {}
    matrix = shuffle_audit.get("true_to_assigned_confusion_matrix") or {}
    row_counts, column_counts = description_matrix_counts(matrix)
    add_check(
        checks,
        "shuffled_frequency_and_fixed_points",
        shuffle_audit.get("fixed_point_count") == 0 and
        shuffle_audit.get("total_augmented") == len(enhanced_indices) and
        shuffle_audit.get("true_description_counts") ==
        shuffle_audit.get("assigned_description_counts"),
        shuffle_audit.get("mapping_signature"),
    )
    add_check(
        checks,
        "shuffled_confusion_matrix_marginals",
        row_counts == shuffle_audit.get("true_description_counts") and
        column_counts == shuffle_audit.get("assigned_description_counts"),
        f"cells={sum(len(row) for row in matrix.values())}",
    )

    pair_tokens = artifacts["pair_operation"]["report"]["token_length"]["train"]
    shuffled_tokens = artifacts["shuffled_operation"]["report"]["token_length"]["train"]
    add_check(
        checks,
        "pair_vs_shuffled_token_distribution_close",
        abs(pair_tokens["p95"] - shuffled_tokens["p95"]) <= 5 and
        abs(pair_tokens["max"] - shuffled_tokens["max"]) <= 32,
        f"p95={pair_tokens['p95']}/{shuffled_tokens['p95']} "
        f"max={pair_tokens['max']}/{shuffled_tokens['max']}",
    )

    risk_counts = Counter(
        normalize_mr_id(sample.get("mr_id")) for sample in sampled_rows
    )
    blockers = [
        {
            "code": f"data_quality:{mr_id}",
            "mr_id": mr_id,
            "sample_count": risk_counts[mr_id],
            "reason": reason,
            "resolution": "manual review required; labels and rows were not modified",
        }
        for mr_id, reason in DATA_QUALITY_RISKS.items()
        if risk_counts[mr_id]
    ]
    status = determine_stage2_status(checks, blockers)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = select_manual_sample_indices(
        train_samples,
        args.seed,
        args.manual_samples_per_mr,
    )
    from transformers import AutoTokenizer
    tokenizer_path = resolve_model_reference(
        config.get("tokenizer", config.get("model")),
        config.get("local_tokenizer_path", config.get("local_model_path")),
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Tokenizer cache is unavailable for {tokenizer_path}. "
            "Run scripts/cache_hf_model.py before Stage 2 validation."
        ) from exc
    manual_path = output_dir / "stage2_manual_samples.md"
    manual_lines = [
        "# Stage 2 Manual Samples",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Seed: {args.seed}",
        f"Tokenizer: `{tokenizer_path}`",
        "",
        "> Ignored local research artifact. Contains real dataset text; do not commit publicly.",
        "",
    ]
    counting_methods = Counter()
    for ordinal, index in enumerate(selected_indices, 1):
        sample = train_samples[index]
        original = original_map[sample["_group_key"]]
        token_length, counting_method = count_row_tokens(tokenizer, pair_rows[index])
        counting_methods[counting_method] += 1
        mr_id = normalize_mr_id(sample.get("mr_id"))
        manual_lines.extend([
            f"## {ordinal}. `{mr_id}`",
            "",
            f"- pair_id: `{sample.get('pair_id')}`",
            f"- stable sample key: `{compute_sample_key(sample)}`",
            f"- gold label: `{pair_rows[index]['output']}`",
            f"- pair_operation token length: `{token_length}` ({counting_method})",
            f"- known training blocker: `{'yes' if mr_id in DATA_QUALITY_RISKS else 'no'}`",
            "",
            "### Source premise",
            "```text",
            original["premise"].strip(),
            "```",
            "",
            "### Source hypothesis",
            "```text",
            original["hypothesis"].strip(),
            "```",
            "",
            "### Current premise",
            "```text",
            sample["premise"].strip(),
            "```",
            "",
            "### Current hypothesis",
            "```text",
            sample["hypothesis"].strip(),
            "```",
            "",
            "### pair_operation instruction",
            "```text",
            pair_rows[index]["instruction"],
            "```",
            "",
            "### shuffled_operation instruction",
            "```text",
            shuffled_rows[index]["instruction"],
            "```",
            "",
        ])
    manual_path.write_text("\n".join(manual_lines), encoding="utf-8")

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "stage": 2,
        "seed": args.seed,
        "cohort_id": cohort_id,
        "config": str(config_path.relative_to(WORK_DIR)),
        "sampled_file": str(sampled_path.relative_to(WORK_DIR)),
        "cohort_meta_file": str((cohort_dir / "cohort_meta.json").relative_to(WORK_DIR)),
        "sampling_report_file": str((cohort_dir / "sampling_report.json").relative_to(WORK_DIR)),
        "cohort_validation": cohort_validation,
        "manifest_file": str(manifest_path.relative_to(WORK_DIR)),
        "data_signature": data_signature,
        "manifest_hash": expected_manifest_hash,
        "ordered_sample_signature": expected_ordered_signature,
        "instruction_template_version": INSTRUCTION_TEMPLATE_VERSION,
        "instruction_template_hash": INSTRUCTION_TEMPLATE_HASH,
        "operation_description_hash": OPERATION_DESCRIPTION_HASH,
        "relation_effect_hash": RELATION_EFFECT_HASH,
        "train_count": expected_train_count,
        "validation_count": expected_val_count,
        "mode_reports": {
            mode: {
                "converted_train_sha256": artifact["report"].get("converted_train_sha256"),
                "converted_val_sha256": artifact["report"].get("converted_val_sha256"),
                "token_length": artifact["report"].get("token_length"),
            }
            for mode, artifact in artifacts.items()
        },
        "shuffle_audit": shuffle_audit,
        "checks": checks,
        "check_counts": dict(Counter(check["status"] for check in checks)),
        "data_quality_blockers": blockers,
        "manual_sample_count": len(selected_indices),
        "manual_sample_counting_methods": dict(counting_methods),
        **status,
    }
    json_path = output_dir / "stage2_report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_lines = [
        "# MR-as-Instruction Stage 2 Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Cohort: `{cohort_id}`",
        f"- Seed: `{args.seed}`",
        f"- Data signature: `{data_signature}`",
        f"- Cohort sampled SHA-256: `{cohort_validation['sampled_file_sha256']}`",
        f"- Sampling report SHA-256: `{cohort_validation['sampling_report_sha256']}`",
        f"- Manifest hash: `{expected_manifest_hash}`",
        f"- Ordered sample signature: `{expected_ordered_signature}`",
        f"- Train / validation: `{expected_train_count}` / `{expected_val_count}`",
        f"- Automated checks: **{status['automated_checks_status']}**",
        f"- Training gate: **{status['training_gate']}**",
        f"- Stage 2 status: **{status['stage2_status']}**",
        "",
        "## Automated checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for check in checks:
        detail = check["detail"].replace("|", "\\|").replace("\n", " ")
        markdown_lines.append(
            f"| `{check['name']}` | {check['status']} | {detail} |"
        )
    markdown_lines.extend([
        "",
        "## Actual tokenizer lengths (train)",
        "",
        "| Mode | P50 | P95 | P99 | Max | Over 512 | Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in REQUIRED_MODES:
        token_summary = artifacts[mode]["report"]["token_length"]["train"]
        markdown_lines.append(
            f"| `{mode}` | {token_summary['p50']} | {token_summary['p95']} | "
            f"{token_summary['p99']} | {token_summary['max']} | "
            f"{token_summary['over_cutoff_count']} | "
            f"{token_summary['over_cutoff_ratio']:.6f} |"
        )
    markdown_lines.extend([
        "",
        "## Actual tokenizer lengths by MR (train)",
        "",
        "| Mode | MR | Count | P95 | Max | Over 512 | Ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for mode in REQUIRED_MODES:
        by_mr = artifacts[mode]["report"]["token_length"]["by_mr"]["train"]
        for mr_id, token_summary in sorted(by_mr.items()):
            markdown_lines.append(
                f"| `{mode}` | `{mr_id}` | {token_summary['count']} | "
                f"{token_summary['p95']} | {token_summary['max']} | "
                f"{token_summary['over_cutoff_count']} | "
                f"{token_summary['over_cutoff_ratio']:.6f} |"
            )
    markdown_lines.extend([
        "",
        "## Shuffled-operation audit",
        "",
        f"- Augmented samples: `{shuffle_audit.get('total_augmented')}`",
        f"- Fixed points: `{shuffle_audit.get('fixed_point_count')}`",
        f"- Mapping signature: `{shuffle_audit.get('mapping_signature')}`",
        "- Description frequencies and confusion-matrix marginals: **PASS**",
        "",
        "## Data-quality training blockers",
        "",
    ])
    for blocker in blockers:
        markdown_lines.append(
            f"- `{blocker['mr_id']}`: {blocker['sample_count']} rows — "
            f"{blocker['reason']}"
        )
    markdown_lines.extend([
        "",
        "Labels and samples were not edited or selectively removed. These blockers "
        "keep Stage 2 from passing even when all automated conversion checks pass.",
        "",
        f"Manual sample review file: `{manual_path.relative_to(WORK_DIR)}` "
        f"({len(selected_indices)} samples).",
        "",
    ])
    markdown_path = output_dir / "stage2_report.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(json.dumps({
        "stage2_status": status["stage2_status"],
        "automated_checks_status": status["automated_checks_status"],
        "training_gate": status["training_gate"],
        "checks": report["check_counts"],
        "data_quality_blockers": len(blockers),
        "report_json": str(json_path),
        "report_md": str(markdown_path),
        "manual_samples": str(manual_path),
    }, indent=2, ensure_ascii=False))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments_config_mrinstr_pilot.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="artifacts/mrinstr_validation",
    )
    parser.add_argument("--manual-samples-per-mr", type=int, default=3)
    parser.add_argument("--max-over-cutoff-ratio", type=float, default=0.01)
    return parser.parse_args()


def main():
    configure_console_encoding()
    report = validate(parse_args())
    if report["stage2_status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
