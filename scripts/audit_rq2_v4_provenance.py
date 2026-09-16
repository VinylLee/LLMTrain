#!/usr/bin/env python3
"""Audit composite ``component_mrs`` provenance for the RQ2 v4 design.

Reports, without needing a GPU:

* composite provenance coverage (with / missing / empty / unsupported),
* the ordered transformation traces and their frequencies,
* operation arity distribution,
* relation-kind distribution per trace (does one trace span several kinds?),
* whether the cohort is fit for a formal v4 run
  (``composite_operation_fallback_count == 0``).

用法:
  python scripts/audit_rq2_v4_provenance.py \
      --input data/nli/mettrain/snli_lr0.0037_gemma-3-4b-it-qat-v3_3/augmented_data_all_label_mrs_v3_3_full.json

  # 只看某个 split（用 --manifest 复用 RQ2 的 split manifest）
  python scripts/audit_rq2_v4_provenance.py --input <cohort.json> \
      --manifest RQ2/data/cohorts_v4/snli_seed42/split_manifest.json --seed 42 --val-ratio 0.05

  python scripts/audit_rq2_v4_provenance.py --input <cohort.json> --json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert_nli_to_ft import (  # noqa: E402
    build_original_map,
    select_relation_matched_wrong_operation_donors_for,
    build_relation_matched_wrong_operation_audit,
    build_pair_groups,
    flatten_groups,
    load_jsonl,
    normalize_mr_id,
    split_pair_groups,
)
from mr_instruction_design import (  # noqa: E402
    COMPOSITE_MR_IDS,
    V4_ATOMIC_OPERATION_STEPS,
    operation_arity,
    relation_kind_v4,
    resolve_operation_trace_v4,
)


def audit(rows, seed=0):
    composites = [r for r in rows if normalize_mr_id(r.get("mr_id")) in COMPOSITE_MR_IDS]

    with_provenance = 0
    missing = 0
    empty = 0
    unsupported = 0
    trace_counts = Counter()
    trace_kinds = defaultdict(Counter)
    arity_counts = Counter()
    kind_counts = Counter()
    trace_rows = 0

    for row in rows:
        mr_id = normalize_mr_id(row.get("mr_id"))
        if mr_id == "none":
            continue
        trace, provenance = resolve_operation_trace_v4(mr_id, row.get("component_mrs"))
        kind = relation_kind_v4(mr_id)

        if mr_id in COMPOSITE_MR_IDS:
            components = row.get("component_mrs")
            if not components:
                missing += 1
            elif not any(str(c).strip() for c in components):
                empty += 1
            elif provenance == "composite_components":
                with_provenance += 1
            else:
                unsupported += 1

        if trace:
            trace_rows += 1
            trace_counts[trace] += 1
            trace_kinds[trace][kind] += 1
            arity_counts[operation_arity(trace)] += 1
        if kind:
            kind_counts[kind] += 1

    total_composites = len(composites)
    coverage = (with_provenance / total_composites) if total_composites else 0.0
    cross_kind = {
        " -> ".join(trace): dict(kinds)
        for trace, kinds in sorted(trace_kinds.items())
        if len(kinds) > 1
    }
    wrong_donors, wrong_meta = select_relation_matched_wrong_operation_donors_for(rows, seed)
    wrong_operation_audit = build_relation_matched_wrong_operation_audit(
        wrong_meta, wrong_donors, seed
    )

    return {
        "total_rows": len(rows),
        "augmented_rows": sum(1 for r in rows if normalize_mr_id(r.get("mr_id")) != "none"),
        "composite_total": total_composites,
        "composite_with_ordered_provenance": with_provenance,
        "composite_missing_provenance": missing,
        "composite_empty_provenance": empty,
        "composite_unsupported_provenance": unsupported,
        "composite_provenance_coverage": coverage,
        "operation_trace_count": trace_rows,
        "unique_operation_trace_count": len(trace_counts),
        "operation_arity_distribution": dict(sorted(arity_counts.items())),
        "relation_kind_distribution_v4": dict(sorted(kind_counts.items())),
        "composite_operation_fallback_count": missing + unsupported,
        "traces": {
            " -> ".join(trace): {
                "count": count,
                "arity": len(trace),
                "relation_kinds": dict(sorted(trace_kinds[trace].items())),
            }
            for trace, count in sorted(trace_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "traces_spanning_multiple_relation_kinds": cross_kind,
        "formal_v4_ready": (missing + unsupported) == 0,
        "strict_wrong_operation_feasibility": wrong_operation_audit,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit RQ2 v4 composite operation provenance")
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", default=None, help="可选：复用 split manifest，只审计 train split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input, quiet=args.json)
    if not rows:
        raise SystemExit(f"没有有效样本: {args.input}")

    scope = "all rows"
    if args.manifest:
        from convert_nli_to_ft import load_split_manifest, validate_manifest_for_groups, compute_data_signature
        groups = build_pair_groups(rows, [Path(args.input).stem] * len(rows))
        manifest = load_split_manifest(args.manifest)
        train_ids, _ = validate_manifest_for_groups(
            manifest, groups, args.seed, args.val_ratio, compute_data_signature(rows)
        )
        group_map = {g["group_key"]: g for g in groups}
        rows = flatten_groups([group_map[key] for key in train_ids])
        scope = "train split"

    result = audit(rows, args.seed)
    result["input"] = args.input
    result["scope"] = scope

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    r = result
    print("=" * 78)
    print(f"  input : {args.input}")
    print(f"  scope : {scope}")
    print("=" * 78)
    print(f"  rows total / augmented : {r['total_rows']} / {r['augmented_rows']}")
    print()
    print("  -- composite provenance --")
    print(f"    composite_total                 : {r['composite_total']}")
    print(f"    composite_with_ordered_provenance: {r['composite_with_ordered_provenance']}")
    print(f"    composite_missing_provenance    : {r['composite_missing_provenance']}")
    print(f"    composite_empty_provenance      : {r['composite_empty_provenance']}")
    print(f"    composite_unsupported_provenance: {r['composite_unsupported_provenance']}")
    print(f"    composite_provenance_coverage   : {r['composite_provenance_coverage']:.4f}")
    print(f"    composite_operation_fallback    : {r['composite_operation_fallback_count']}")
    print()
    print("  -- operation traces --")
    print(f"    operation_trace_count      : {r['operation_trace_count']}")
    print(f"    unique_operation_trace_count: {r['unique_operation_trace_count']}")
    print(f"    operation_arity_distribution: {r['operation_arity_distribution']}")
    print()
    print("  -- relation kinds (v4) --")
    print("  -- strict relation-matched wrong Operation feasibility --")
    f = r["strict_wrong_operation_feasibility"]
    print("    relation_kind                    arity rows unique_traces eligible ineligible coverage")
    for item in f["strata"]:
        print(
            f"    {item['relation_kind']:32s} {item['arity']:5d} "
            f"{item['row_count']:4d} {item['unique_trace_count']:13d} "
            f"{item['eligible_row_count']:8d} {item['ineligible_row_count']:9d} {item['coverage']:.4f}"
        )
    print(f"    TOTAL eligible / transformed: {f['wrong_operation_relation_matched_eligible']} / {f['wrong_operation_relation_matched_total']} "
          f"({f['strict_relation_matched_wrong_operation_coverage']:.4f})")
    print("    ordered trace frequencies:")
    for stratum, frequencies in f["trace_frequency"].items():
        print(f"      {stratum}: {frequencies}")
    print()
    for kind, count in sorted(r["relation_kind_distribution_v4"].items(), key=lambda kv: -kv[1]):
        print(f"    {kind:28s} {count}")
    print()
    print("  -- ordered traces, by frequency --")
    for trace, info in r["traces"].items():
        print(f"    {trace:70s} arity={info['arity']} n={info['count']:5d}")
    if r["traces_spanning_multiple_relation_kinds"]:
        print()
        print("  -- traces spanning multiple relation kinds --")
        for trace, kinds in r["traces_spanning_multiple_relation_kinds"].items():
            print(f"    {trace}: {kinds}")
    print()
    ready = "READY" if r["formal_v4_ready"] else "BLOCKED"
    print(f"  formal v4 readiness: {ready}")
    if not r["formal_v4_ready"]:
        print("    -> 该 cohort 存在无 provenance 的 composite 行，不能作为正式 v4 输入。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
