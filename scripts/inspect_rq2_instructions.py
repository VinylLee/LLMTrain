#!/usr/bin/env python3
"""Render one follow-up sample under every RQ2 MR-information mode.

Purpose: let a human eyeball, *before* spending GPU time, that the only thing
that changes across the core 2x2x2 cells is the presence of the P/O/R blocks,
and that the v4 Operation block shows the real ordered transformation trace.

No model, tokenizer or GPU is needed.

用法:
  # 列出某个文件里可用的 (pair_id, mr_id) 组合
  python scripts/inspect_rq2_instructions.py --input data/.../augmented_data.json --list

  # 渲染八个核心条件 + controls + diagnostic（v4，默认）
  python scripts/inspect_rq2_instructions.py \
      --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json \
      --pair-id 358 --mr-id composite_flip

  # 对比冻结的 v3 设计
  python scripts/inspect_rq2_instructions.py --input ... --pair-id 358 --mr-id composite_flip \
      --template-version 3

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
    INSTRUCTION_TEMPLATE_VERSION,
    LABEL_NAMES_3CLASS,
    build_instruction_for_sample,
    build_mismatched_pair_map,
    build_pair_groups,
    compute_sample_key,
    load_jsonl,
    mode_design_meta,
    normalize_mr_id,
    select_matched_shuffled_operation_donors_for,
    select_shuffled_operation_mr_ids,
    select_shuffled_relation_kinds_v4,
    select_shuffled_relation_types,
    sort_samples_by_stable_key,
    uses_ordered_operation_trace,
)
from mr_instruction_design import (  # noqa: E402
    CONTROL_MODES,
    CORE_MODES,
    DIAGNOSTIC_MODES,
    RELATION_TYPE_DESCRIPTIONS,
    V4_ATOMIC_OPERATION_STEPS,
    V4_RELATION_KIND_DESCRIPTIONS,
    mode_spec,
    relation_kind_v4,
    relation_type_for,
    render_operation_v4,
    render_relation_v4,
    resolve_operation_description,
    resolve_relation_description,
    resolve_operation_trace_v4,
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


def describe_row(row):
    """Trace-level description of one transformed row (no rendering)."""
    mr_id = normalize_mr_id(row.get("mr_id"))
    trace, provenance = resolve_operation_trace_v4(mr_id, row.get("component_mrs"))
    return {
        "mr_id": mr_id,
        "mr_type": row.get("mr_type"),
        "relation_kind_v4": relation_kind_v4(mr_id),
        "relation_type_v3": relation_type_for(mr_id),
        "component_mrs": list(row.get("component_mrs") or []),
        "operation_trace_id": list(trace),
        "operation_arity": len(trace),
        "operation_provenance": provenance,
    }


def render(sample, mode, context, version):
    spec = mode_spec(mode)
    is_source_row = normalize_mr_id(sample.get("mr_id")) == "none"
    true_mr_id = normalize_mr_id(sample.get("mr_id"))

    operation_description = ""
    if spec["operation"] and not is_source_row:
        if uses_ordered_operation_trace(version):
            source_row = sample
            if spec["operation_source"] == "shuffled" and context["shuffled_operation_donor"] is not None:
                source_row = context["samples"][context["shuffled_operation_donor"]] \
                    if context.get("samples") else sample
            trace, provenance = resolve_operation_trace_v4(
                normalize_mr_id(source_row.get("mr_id")), source_row.get("component_mrs"))
            operation_description = render_operation_v4(trace, provenance)
        elif spec["operation_source"] == "shuffled":
            operation_description, _ = resolve_operation_description(
                context["shuffled_operation_mr_id"])
        else:
            operation_description, _ = resolve_operation_description(
                true_mr_id, sample.get("component_mrs"))

    relation_description = ""
    if spec["relation"] and not is_source_row:
        if uses_ordered_operation_trace(version):
            if spec["relation_source"] == "shuffled" and context["shuffled_relation_kind"]:
                relation_description = V4_RELATION_KIND_DESCRIPTIONS[
                    context["shuffled_relation_kind"]]
            else:
                relation_description = render_relation_v4(true_mr_id)
        elif spec["relation_source"] == "shuffled":
            relation_description = RELATION_TYPE_DESCRIPTIONS[
                context["shuffled_relation_type"]]
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
        template_version=version,
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect RQ2 MR-information instructions")
    parser.add_argument("--input", required=True, help="JSONL 数据文件")
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--mr-id", default=None)
    parser.add_argument("--seed", type=int, default=42, help="shuffled controls 的 seed")
    parser.add_argument("--template-version", type=int, default=INSTRUCTION_TEMPLATE_VERSION,
                        choices=[2, 3, 4], help="instruction 设计版本（默认当前版本）")
    parser.add_argument("--list", action="store_true", help="列出可用的 (pair_id, mr_id)")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()
    version = args.template_version

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
    if uses_ordered_operation_trace(version):
        shuffled_ops, _ = select_matched_shuffled_operation_donors_for(ordered, shuffle_seed)
        shuffled_rels = select_shuffled_relation_kinds_v4(ordered, shuffle_seed)
    else:
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

    index = next(
        (i for i, row in enumerate(ordered) if compute_sample_key(row) == compute_sample_key(sample)),
        0,
    )
    context = {
        "samples": ordered,
        "source": source_by_group.get(sample.get("_group_key")),
        "mismatched_source": source_by_group.get(mismatched_map.get(sample.get("_group_key"))),
        "shuffled_operation_donor": shuffled_ops[index],
        "shuffled_relation_kind": shuffled_rels[index],
        "shuffled_operation_mr_id": shuffled_ops[index],
        "shuffled_relation_type": shuffled_rels[index],
    }

    description = describe_row(sample)
    results = {}
    for mode in ALL_INSPECT_MODES:
        results[mode] = {
            "design": mode_design_meta(mode),
            **render(sample, mode, context, version),
        }

    if args.json:
        print(json.dumps({
            "input": args.input,
            "template_version": version,
            "pair_id": sample.get("pair_id"),
            "target_label": sample.get("label"),
            "source_label": context["source"].get("label") if context["source"] else None,
            "row": description,
            "shuffled_operation_donor_mr_id": (
                normalize_mr_id(ordered[shuffled_ops[index]].get("mr_id"))
                if shuffled_ops[index] is not None else None
            ),
            "shuffled_relation_kind": shuffled_rels[index],
            "modes": results,
        }, indent=2, ensure_ascii=False))
        return 0

    print("=" * 78)
    print(f"  input            : {args.input}")
    print(f"  template version : {version}")
    print(f"  pair_id          : {sample.get('pair_id')}")
    print(f"  MR               : {description['mr_id']}   (legacy mr_type={description['mr_type']})")
    print(f"  Relation kind    : {description['relation_kind_v4']}"
          f"   (v3 relation type={description['relation_type_v3']})")
    if description["component_mrs"]:
        print("  Components, in execution order:")
        for i, comp in enumerate(description["component_mrs"], 1):
            known = "ok" if comp in V4_ATOMIC_OPERATION_STEPS else "UNKNOWN"
            print(f"      {i}. {comp}   [{known}]")
    else:
        print("  Components       : (atomic row -- no component_mrs)")
    print(f"  operation_trace_id : {description['operation_trace_id']}")
    print(f"  operation_arity    : {description['operation_arity']}")
    print(f"  operation_provenance: {description['operation_provenance']}")
    print(f"  target label     : {sample.get('label')}")
    print(f"  source label     : {context['source'].get('label') if context['source'] else 'N/A'}"
          f"  (only full_oracle may show this)")
    if shuffled_ops[index] is not None:
        donor = ordered[shuffled_ops[index]]
        print(f"  shuffled op donor: {normalize_mr_id(donor.get('mr_id'))}")
    print(f"  shuffled relation: {shuffled_rels[index]}")
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
