#!/usr/bin/env python3
"""Render one follow-up sample under every RQ2 MR-information mode.

Purpose: let a human eyeball, *before* spending GPU time, that the only thing
that changes across the core 2x2x2 cells is the presence of the P/O/R blocks.

No model, tokenizer or GPU is needed.

用法:
  # 列出某个文件里可用的 (pair_id, mr_id) 组合
  python scripts/inspect_rq2_instructions.py --input data/.../augmented_data.json --list

  # 渲染八个核心条件 + controls + diagnostic
  python scripts/inspect_rq2_instructions.py \
      --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat/augmented_data.json \
      --pair-id 358 --mr-id adding_contradiction

  # 机器可读输出
  python scripts/inspect_rq2_instructions.py --input ... --pair-id 358 --mr-id composite_flip --json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_nli_to_ft import (  # noqa: E402
    INSTRUCTION_NLI,
    LABEL_NAMES_3CLASS,
    build_pair_groups,
    build_instruction_for_sample,
    build_mismatched_pair_map,
    compute_sample_key,
    flatten_groups,
    load_jsonl,
    mode_design_meta,
    normalize_mr_id,
    resolve_mode_name,
    select_shuffled_operation_mr_ids,
    select_shuffled_relation_types,
    sort_samples_by_stable_key,
)
from mr_instruction_design import (  # noqa: E402
    CONTROL_MODES,
    CORE_MODES,
    DIAGNOSTIC_MODES,
    RELATION_TYPE_DESCRIPTIONS,
    mode_spec,
    relation_type_for,
    resolve_operation_description,
    resolve_relation_description,
)

ALL_INSPECT_MODES = list(CORE_MODES) + list(CONTROL_MODES) + list(DIAGNOSTIC_MODES)


def list_pairs(samples):
    counter = {}
    for row in samples:
        mr_id = normalize_mr_id(row.get("mr_id"))
        if mr_id == "none":
            continue
        counter.setdefault(row.get("pair_id"), []).append(mr_id)
    for pair_id in sorted(counter, key=str):
        print(f"  pair_id={pair_id}: {sorted(set(counter[pair_id]))}")


def pick_sample(samples, pair_id, mr_id):
    for row in samples:
        if str(row.get("pair_id")) == str(pair_id) and normalize_mr_id(row.get("mr_id")) == mr_id:
            return row
    return None


def render(sample, mode, context):
    """Render one mode for ``sample``; ``context`` carries the pair/control payloads."""
    spec = mode_spec(mode)
    is_source_row = normalize_mr_id(sample.get("mr_id")) == "none"
    true_mr_id = normalize_mr_id(sample.get("mr_id"))

    operation_description = ""
    if spec["operation"] and not is_source_row:
        if spec["operation_source"] == "shuffled":
            operation_description, _ = resolve_operation_description(
                context["shuffled_operation_mr_id"]
            )
        else:
            operation_description, _ = resolve_operation_description(
                true_mr_id, sample.get("component_mrs")
            )

    relation_description = ""
    if spec["relation"] and not is_source_row:
        if spec["relation_source"] == "shuffled":
            relation_description = RELATION_TYPE_DESCRIPTIONS[
                context["shuffled_relation_type"]
            ]
        else:
            relation_description = resolve_relation_description(true_mr_id)

    anchor_source = context["source"]
    if spec["pair_source"] == "mismatched":
        anchor_source = context["mismatched_source"]
    original_label = ""
    if spec["label_anchor"] and anchor_source is not None:
        raw_label = anchor_source.get("label")
        original_label = (
            LABEL_NAMES_3CLASS.get(raw_label, str(raw_label))
            if isinstance(raw_label, int) else str(raw_label)
        )

    return build_instruction_for_sample(
        sample=sample,
        original_sample=anchor_source,
        mode=mode,
        operation_description=operation_description,
        relation_description=relation_description,
        original_label=original_label,
        nli_instruction=INSTRUCTION_NLI,
        is_original=is_source_row,
        template_version=3,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect RQ2 MR-information instructions")
    parser.add_argument("--input", required=True, help="JSONL 数据文件")
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--mr-id", default=None)
    parser.add_argument("--seed", type=int, default=42, help="shuffled controls 的 seed")
    parser.add_argument("--list", action="store_true", help="列出可用的 (pair_id, mr_id)")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--source-rows-only", action="store_true",
                        help="演示 source row 在所有 mode 下都是普通 NLI instruction")
    args = parser.parse_args()

    samples = load_jsonl(args.input)
    if not samples:
        raise SystemExit(f"没有有效样本: {args.input}")
    if args.list:
        list_pairs(samples)
        return 0

    groups = build_pair_groups(samples, [Path(args.input).stem] * len(samples))
    for group in groups:
        for row in group["samples"]:
            row["_group_key"] = group["group_key"]
    ordered = sort_samples_by_stable_key(samples)

    shuffle_seed = args.seed + 999
    shuffled_ops = select_shuffled_operation_mr_ids(ordered, shuffle_seed)
    shuffled_rels = select_shuffled_relation_types(ordered, shuffle_seed)

    mismatched_map = build_mismatched_pair_map(groups) if len(groups) > 1 else {}
    source_by_group = {}
    for group in groups:
        for row in group["samples"]:
            if normalize_mr_id(row.get("mr_id")) == "none":
                source_by_group[group["group_key"]] = row

    if args.mr_id is None or args.pair_id is None:
        raise SystemExit("需要 --pair-id 与 --mr-id（或用 --list 查看可用组合）")

    sample = pick_sample(samples, args.pair_id, args.mr_id)
    if sample is None:
        raise SystemExit(f"未找到 pair_id={args.pair_id} mr_id={args.mr_id} 的样本")

    if args.source_rows_only:
        sample = dict(sample, mr_id="none")

    source = source_by_group.get(sample.get("_group_key"))
    mismatched_group = mismatched_map.get(sample.get("_group_key"))

    sample_key = compute_sample_key(sample)
    index = next(
        (i for i, row in enumerate(ordered) if compute_sample_key(row) == sample_key),
        0,
    )
    context = {
        "source": source,
        "mismatched_source": source_by_group.get(mismatched_group),
        "shuffled_operation_mr_id": shuffled_ops[index],
        "shuffled_relation_type": shuffled_rels[index],
    }

    results = {}
    for mode in ALL_INSPECT_MODES:
        results[mode] = {
            "design": mode_design_meta(mode),
            **render(sample, mode, context),
        }

    if args.json:
        print(json.dumps({
            "input": args.input,
            "pair_id": sample.get("pair_id"),
            "mr_id": normalize_mr_id(sample.get("mr_id")),
            "relation_type": relation_type_for(normalize_mr_id(sample.get("mr_id"))),
            "target_label": sample.get("label"),
            "source_label": source.get("label") if source else None,
            "shuffled_operation_mr_id": context["shuffled_operation_mr_id"],
            "shuffled_relation_type": context["shuffled_relation_type"],
            "modes": results,
        }, indent=2, ensure_ascii=False))
        return 0

    print("=" * 78)
    print(f"  input        : {args.input}")
    print(f"  pair_id      : {sample.get('pair_id')}")
    print(f"  mr_id        : {normalize_mr_id(sample.get('mr_id'))} "
          f"(relation={relation_type_for(normalize_mr_id(sample.get('mr_id')))})")
    print(f"  target label : {sample.get('label')}")
    print(f"  source label : {source.get('label') if source else 'N/A'}  "
          f"(only full_oracle may show this)")
    print(f"  shuffled op  : {context['shuffled_operation_mr_id']}   "
          f"shuffled relation: {context['shuffled_relation_type']}")
    print("=" * 78)
    print()
    print("  CURRENT SAMPLE (identical in every mode -- lives in the `input` field)")
    print(f"    Premise:    {sample.get('premise')}")
    print(f"    Hypothesis: {sample.get('hypothesis')}")
    print()

    for mode in ALL_INSPECT_MODES:
        design = results[mode]["design"]
        flags = "".join(
            letter if design[key] else "-"
            for letter, key in (("P", "pair"), ("O", "operation"),
                                ("R", "relation"), ("L", "label_anchor"))
        )
        print("-" * 78)
        print(f"  MODE {mode}   [{flags}]  {design['grounding']} / {design['role']}")
        print("-" * 78)
        print(results[mode]["instruction"])
        print(f"\n  >> output: {results[mode]['output']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
