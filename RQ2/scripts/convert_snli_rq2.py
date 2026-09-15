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
    LABEL_NAMES_3CLASS,
    MODES_REQUIRING_OPERATION,
    MODES_REQUIRING_RELATION_EFFECT,
    MR_OPERATION_DESCRIPTIONS,
    MR_RELATION_EFFECTS,
    build_original_map,
    build_pair_groups,
    build_token_length_report,
    canonical_sha256,
    compute_data_signature,
    compute_ordered_sample_signature,
    convert_to_alpaca,
    flatten_groups,
    load_jsonl,
    load_split_manifest,
    save_jsonl,
    save_split_manifest,
    select_shuffled_descriptions,
    sort_samples_by_stable_key,
    validate_manifest_for_groups,
    validate_mr_samples,
    normalize_mr_id,
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
    parser.add_argument("--mode", required=True, choices=[
        "none", "operation_only", "pair_only", "pair_operation", "shuffled_operation",
    ])
    parser.add_argument("--seed", type=int, required=True)
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

    train_samples = sort_samples_by_stable_key(flatten_groups(train_groups))
    val_samples = sort_samples_by_stable_key(flatten_groups(val_groups))
    shuffled = None
    shuffle_audit = None
    if args.mode == "shuffled_operation":
        all_mr_ids = {normalize_mr_id(s.get("mr_id")) for s in samples}
        shuffle_seed = args.seed + 999
        shuffled = select_shuffled_descriptions(train_samples, all_mr_ids, shuffle_seed)
        from convert_nli_to_ft import build_shuffle_audit
        shuffle_audit = build_shuffle_audit(train_samples, shuffled, shuffle_seed)

    train_rows, train_report = convert_to_alpaca(
        train_samples,
        mode=args.mode,
        original_map=original_map if args.mode not in ("none", "operation_only") else None,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
        relation_effects=MR_RELATION_EFFECTS if args.mode in MODES_REQUIRING_RELATION_EFFECT else None,
        shuffled_descriptions=shuffled,
        strict=True,
    )
    val_rows, val_report = convert_to_alpaca(val_samples, mode="none", strict=True)

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
        "schema_version": 1,
        "name": args.name,
        "mode": args.mode,
        "seed": args.seed,
        "source_files": [str(input_path)],
        "source_sha256": canonical_sha256(input_path.read_bytes().decode("utf-8")),
        "data_signature": data_signature,
        "manifest_sha256": manifest.get("sha256"),
        "instruction_template_version": INSTRUCTION_TEMPLATE_VERSION,
        "operation_description_hash": canonical_sha256(MR_OPERATION_DESCRIPTIONS),
        "relation_effect_hash": canonical_sha256(MR_RELATION_EFFECTS),
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
    }
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"name": args.name, "mode": args.mode, "train": len(train_rows), "val": len(val_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
