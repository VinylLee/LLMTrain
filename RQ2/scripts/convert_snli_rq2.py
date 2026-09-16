#!/usr/bin/env python3
"""Convert one frozen SNLI cohort into an isolated RQ2 dataset registry.

The actual instruction construction is shared with ``scripts/convert_nli_to_ft.py``.
This wrapper keeps RQ2's converted files and dataset registry under ``RQ2/data``
without changing the project-wide RQ1 registry.
"""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from convert_nli_to_ft import (  # noqa: E402
    INSTRUCTION_TEMPLATE_VERSION,
    LEGACY_INSTRUCTION_TEMPLATE_VERSIONS,
    build_grounding_control_audit,
    build_matched_control_audit,
    build_matched_control_rows_meta,
    select_matched_shuffled_operation_donors_for,
    select_shuffled_relation_kinds_v4,
    uses_ordered_operation_trace,
    build_mismatched_pair_map,
    build_original_map,
    build_pair_groups,
    build_token_length_report,
    canonical_sha256,
    compute_data_signature,
    compute_ordered_sample_signature,
    convert_to_alpaca,
    descriptions_for_version,
    flatten_groups,
    load_jsonl,
    load_split_manifest,
    mode_design_meta,
    mode_spec,
    normalize_mr_id,
    resolve_mode_name,
    save_jsonl,
    save_split_manifest,
    select_shuffled_descriptions,
    select_shuffled_operation_mr_ids,
    select_shuffled_relation_types,
    sort_samples_by_stable_key,
    validate_manifest_for_groups,
    validate_mr_samples,
)

from mr_instruction_design import (  # noqa: E402
    ALL_MODE_NAMES,
    INSTRUCTION_DESIGN_VERSION,
    MODE_ALIASES,
)


def write_registry(path, dataset_name, train_path, val_path):
    """Write a LLaMA-Factory registry rooted at RQ2/data."""
    path = Path(path)
    root = (PROJECT_ROOT / "RQ2" / "data").resolve()

    def relative(value):
        return Path(value).resolve().relative_to(root).as_posix()

    registry = {}
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    entry = {
        "file_name": relative(train_path),
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    registry[dataset_name] = entry
    registry[dataset_name + "_val"] = {
        **entry,
        "file_name": relative(val_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert frozen RQ2 SNLI cohort")
    parser.add_argument("--input", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(ALL_MODE_NAMES),
                        help="RQ2 MR-information mode（含 deprecated 别名）")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instruction-template-version", type=int,
                        default=INSTRUCTION_DESIGN_VERSION)
    parser.add_argument("--require-composite-provenance", action="store_true",
                        help="v4 正式 run：任何 composite 行缺少 component_mrs 直接失败")
    parser.add_argument("--allow-partial-control-coverage", action="store_true",
                        help="仅输出 strict wrong Operation 的 eligible subset；不跨 relation kind fallback")
    parser.add_argument("--output-dir", required=True,
                        help="RQ2 data directory; one <name> subdirectory is created")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--cutoff-len", type=int, default=512)
    args = parser.parse_args()

    input_path = Path(args.input)
    samples = load_jsonl(input_path)
    if not samples:
        raise ValueError(f"没有有效输入样本: {input_path}")
    validate_mr_samples(samples, strict=True)
    source_ids = [input_path.stem] * len(samples)
    groups = build_pair_groups(samples, source_ids)
    data_signature = compute_data_signature(samples)

    if args.write_manifest:
        from convert_nli_to_ft import split_pair_groups
        train_groups, val_groups = split_pair_groups(groups, args.val_ratio, args.seed + 999)
        manifest = save_split_manifest(
            args.manifest, args.seed, args.val_ratio,
            [g["group_key"] for g in train_groups],
            [g["group_key"] for g in val_groups],
            [str(input_path)], data_signature,
            sum(len(g["samples"]) for g in train_groups),
            sum(len(g["samples"]) for g in val_groups),
        )
    else:
        manifest = load_split_manifest(args.manifest)
        train_ids, val_ids = validate_manifest_for_groups(
            manifest, groups, args.seed, args.val_ratio, data_signature
        )
        group_map = {g["group_key"]: g for g in groups}
        train_groups = [group_map[key] for key in train_ids]
        val_groups = [group_map[key] for key in val_ids]

    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]

    original_map, missing_sources = build_original_map(train_groups)
    if missing_sources:
        raise ValueError(f"训练组缺少唯一 source: {missing_sources[:3]}")

    template_version = args.instruction_template_version
    # v2 is the only per-mode hand-written path; v3 and v4 both compose blocks.
    legacy = template_version == 2
    ordered_trace = uses_ordered_operation_trace(template_version)
    op_map, rel_map = descriptions_for_version(template_version)
    if args.require_composite_provenance and not ordered_trace:
        raise ValueError(
            "--require-composite-provenance 只对 v4（ordered trace）有意义"
        )

    mode = args.mode
    canonical_mode = resolve_mode_name(mode)
    spec = mode_spec(canonical_mode)

    train_samples = sort_samples_by_stable_key(flatten_groups(train_groups))
    val_samples = sort_samples_by_stable_key(flatten_groups(val_groups))

    shuffle_seed = args.seed + 999
    shuffled = None
    shuffled_relation_types = None
    shuffled_operation_donors = None
    mismatched_pair_map = None
    shuffle_audit = None
    grounding_control_audit = None
    matched_control_audit = None

    if legacy and spec["operation_source"] == "shuffled":
        all_mr_ids = {normalize_mr_id(s.get("mr_id")) for s in samples}
        shuffled = select_shuffled_descriptions(
            train_samples, all_mr_ids, shuffle_seed, descriptions=op_map
        )
    elif ordered_trace:
        if spec["operation_source"] == "shuffled":
            shuffled_operation_donors, rows_meta = select_matched_shuffled_operation_donors_for(
                train_samples, shuffle_seed
            )
        else:
            rows_meta = build_matched_control_rows_meta(train_samples)
        if spec["relation_source"] == "shuffled":
            shuffled_relation_types = select_shuffled_relation_kinds_v4(train_samples, shuffle_seed)
        if spec["pair_source"] == "mismatched":
            mismatched_pair_map = build_mismatched_pair_map(train_groups)
        if shuffled_operation_donors is not None or shuffled_relation_types is not None:
            matched_control_audit = build_matched_control_audit(
                rows_meta,
                shuffled_operation_donors if shuffled_operation_donors is not None
                else [None] * len(train_samples),
                shuffled_relation_types,
                shuffle_seed,
            )
            # Hard requirements: the shuffled Operation must never render the
            # row's own trace or its own text.
            for field in (
                "operation_trace_identity_collision_count",
                "operation_text_unchanged_count",
            ):
                if matched_control_audit.get(field):
                    raise ValueError(
                        f"v4 shuffled 控制不满足约束: {field}="
                        f"{matched_control_audit[field]}"
                    )
            # The Relation control is capped by Hall's condition on this cohort
            # (one relation kind dominates).  Reported, not fatal.
            if matched_control_audit.get("shuffled_relation_identity_collision_count"):
                print(
                    "  ⚠️  shuffled_relation 存在无法 derange 的行: "
                    f"{matched_control_audit['shuffled_relation_identity_collision_count']}"
                    f"/{matched_control_audit.get('total_relation_eligible')} "
                    "（kind marginals 保持匹配，见 conversion report）"
                )
    else:
        if spec["operation_source"] == "shuffled":
            shuffled = select_shuffled_operation_mr_ids(train_samples, shuffle_seed)
        if spec["relation_source"] == "shuffled":
            shuffled_relation_types = select_shuffled_relation_types(train_samples, shuffle_seed)
        if spec["pair_source"] == "mismatched":
            mismatched_pair_map = build_mismatched_pair_map(train_groups)
        if shuffled is not None or shuffled_relation_types is not None:
            grounding_control_audit = build_grounding_control_audit(
                train_samples,
                shuffled if shuffled is not None else [None] * len(train_samples),
                shuffled_relation_types if shuffled_relation_types is not None else [None] * len(train_samples),
                shuffle_seed,
            )
            for field in (
                "operation_identity_collision_count",
                "relation_identity_collision_count",
            ):
                if grounding_control_audit[field]:
                    raise ValueError(
                        f"shuffled 控制出现 identity collision: {field}="
                        f"{grounding_control_audit[field]}"
                    )

    train_rows, train_report = convert_to_alpaca(
        train_samples,
        mode=canonical_mode if not legacy else mode,
        original_map=original_map if spec["pair"] else None,
        operation_descriptions=op_map,
        relation_effects=rel_map if spec["relation"] else None,
        shuffled_descriptions=shuffled,
        shuffled_relation_types=shuffled_relation_types,
        shuffled_operation_donors=shuffled_operation_donors,
        control_seed=args.seed,
        allow_partial_control_coverage=args.allow_partial_control_coverage,
        mismatched_pair_map=mismatched_pair_map,
        require_composite_provenance=args.require_composite_provenance,
        strict=True,
        template_version=template_version,
    )
    val_rows, val_report = convert_to_alpaca(
        val_samples, mode="none", strict=True, template_version=template_version
    )

    out_dir = PROJECT_ROOT / args.output_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "full_train.json"
    val_path = out_dir / "full_val.json"
    save_jsonl(train_rows, train_path)
    save_jsonl(val_rows, val_path)
    registry_path = PROJECT_ROOT / "RQ2" / "data" / "dataset_info.json"
    write_registry(registry_path, args.name, train_path, val_path)

    token_report = None
    if args.tokenizer_path:
        token_report = build_token_length_report(
            train_rows, val_rows, args.tokenizer_path, args.cutoff_len,
            train_samples=train_samples, val_samples=val_samples,
        )
    report = {
        "schema_version": 2,
        "name": args.name,
        "mode": args.mode,
        "canonical_mode": canonical_mode,
        "deprecated_alias": args.mode if args.mode in MODE_ALIASES else None,
        "seed": args.seed,
        "source_files": [str(input_path)],
        "source_sha256": canonical_sha256(input_path.read_bytes().decode("utf-8")),
        "data_signature": data_signature,
        "manifest_sha256": manifest.get("sha256"),
        "instruction_template_version": template_version,
        "operation_description_hash": canonical_sha256(op_map),
        "relation_effect_hash": canonical_sha256(rel_map),
        # Explicit P/O/R/L design metadata (never re-derived from the mode name).
        "mr_design": {
            **mode_design_meta(mode),
            "template_version": template_version,
        },
        "design_integrity": {
            "pair_fallback_count": train_report.get("pair_fallback_count", 0),
            "missing_operation_count": train_report.get("missing_operation_count", 0),
            "missing_relation_count": train_report.get("missing_relation_count", 0),
            "missing_source_label_count": train_report.get("missing_source_label_count", 0),
            "composite_operation_fallback_count": train_report.get(
                "composite_operation_fallback_count", 0
            ),
            "shuffled_operation_identity_collision_count": train_report.get(
                "shuffled_operation_identity_collision_count", 0
            ),
            "shuffled_relation_identity_collision_count": train_report.get(
                "shuffled_relation_identity_collision_count", 0
            ),
            "relation_type_mismatch_count": train_report.get("relation_type_mismatch_count", 0),
            "wrong_operation_relation_matched_total": train_report.get(
                "wrong_operation_relation_matched_total", 0
            ),
            "wrong_operation_relation_matched_eligible": train_report.get(
                "wrong_operation_relation_matched_eligible", 0
            ),
            "wrong_operation_relation_matched_ineligible": train_report.get(
                "wrong_operation_relation_matched_ineligible", 0
            ),
            "strict_relation_matched_wrong_operation_coverage": train_report.get(
                "strict_relation_matched_wrong_operation_coverage", 0.0
            ),
            "wrong_operation_trace_identity_collision_count": train_report.get(
                "wrong_operation_trace_identity_collision_count", 0
            ),
            "wrong_operation_text_unchanged_count": train_report.get(
                "wrong_operation_text_unchanged_count", 0
            ),
            "wrong_operation_same_relation_kind_rate": train_report.get(
                "wrong_operation_same_relation_kind_rate", 0.0
            ),
            "wrong_operation_same_arity_rate": train_report.get(
                "wrong_operation_same_arity_rate", 0.0
            ),
            "wrong_relation_total": train_report.get("wrong_relation_total", 0),
            "wrong_relation_identity_collision_count": train_report.get(
                "wrong_relation_identity_collision_count", 0
            ),
            "wrong_relation_text_unchanged_count": train_report.get(
                "wrong_relation_text_unchanged_count", 0
            ),
            "wrong_relation_deranged_rate": train_report.get(
                "wrong_relation_deranged_rate", 0.0
            ),
            "operation_provenance_distribution": train_report.get(
                "operation_provenance_distribution", {}
            ),
            "relation_type_distribution": train_report.get("relation_type_distribution", {}),
            "operation_text_unchanged_count": (
                grounding_control_audit or {}
            ).get("operation_text_unchanged_count", 0),
            "relation_text_unchanged_count": (
                grounding_control_audit or {}
            ).get("relation_text_unchanged_count", 0),
        },
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "train_sample_count": len(train_rows),
        "val_sample_count": len(val_rows),
        "train": train_report,
        "val": val_report,
        "ordered_sample_signature": compute_ordered_sample_signature(train_samples, val_samples),
        "train_json_sha256": canonical_sha256(train_path.read_text(encoding="utf-8")),
        "val_json_sha256": canonical_sha256(val_path.read_text(encoding="utf-8")),
        "token_length": token_report,
        "shuffle_audit": shuffle_audit,
        "grounding_control_audit": grounding_control_audit,
        "mismatched_pair_used": mismatched_pair_map is not None,
    }
    if ordered_trace:
        # v4-only audit fields.  Emitted conditionally so that regenerating a
        # frozen v2/v3 conversion reproduces its report byte-for-byte.
        report["matched_control_audit"] = matched_control_audit
        report["require_composite_provenance"] = args.require_composite_provenance
        report["wrong_operation_relation_matched_audit"] = train_report.get(
            "wrong_operation_relation_matched_audit"
        )
        report["wrong_relation_audit"] = {
            key: train_report.get(key)
            for key in (
                "wrong_relation_total",
                "wrong_relation_identity_collision_count",
                "wrong_relation_text_unchanged_count",
                "wrong_relation_deranged_rate",
            )
        }
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "name": args.name, "mode": args.mode, "canonical_mode": canonical_mode,
        "train": len(train_rows), "val": len(val_rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
