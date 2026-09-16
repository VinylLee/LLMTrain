"""Relation-level metrics for binary sentiment-analysis MR tests.

``scripts/metamorphic_metrics.py`` stays untouched: its flip map is hardcoded for
the three-way NLI label space (``entailment``/``neutral``/``contradiction``), so
reusing it for binary sentiment would silently produce empty denominators.  This
module owns the SA half instead and keeps the NLI runners bit-identical.

The row contract is shared with the NLI evaluator -- ``pair_id``, ``is_source``,
``mr_type`` in ``{inv, flip}``, plus ``pred``/``correct`` on prediction rows.
SA rows never carry ``mr_type == "neutral"``.

Every rate is returned as ``{correct|satisfied, total, rate}`` with the raw
counts kept alongside the percentage, per section 13.2 of the RQ1 SA route
document.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from metamorphic_metrics import RELATION_TYPES, compute_joint_correctness

SA_RELATION_TYPES: tuple[str, ...] = ("inv", "flip")
SA_FLIP_MAP = {"positive": "negative", "negative": "positive"}


def _pair_key(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _group_pairs(results: Sequence[Mapping[str, Any]]):
    pairs: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {"sources": [], "followups": []}
    )
    missing = 0
    for row in results:
        key = _pair_key(row.get("pair_id"))
        if key is None:
            missing += 1
            continue
        pairs[key]["sources" if row.get("is_source") else "followups"].append(row)
    return pairs, missing


def compute_sa_msr(
    results: Sequence[Mapping[str, Any]],
    flip_map: Mapping[str, str] = SA_FLIP_MAP,
) -> dict[str, Any]:
    """Metamorphic Satisfaction Rate for binary sentiment relations.

    ``inv`` is satisfied when the follow-up prediction equals the source
    prediction; ``flip`` is satisfied when it equals the mapped opposite.  A
    prediction outside the binary label space can never satisfy a flip.
    """
    pairs, missing = _group_pairs(results)
    counters = {relation: {"satisfied": 0, "total": 0} for relation in SA_RELATION_TYPES}
    overall = {"satisfied": 0, "total": 0}
    diagnostics = {
        "pair_groups": len(pairs),
        "missing_pair_id_rows": missing,
        "orphan_followups": 0,
        "ambiguous_source_followups": 0,
        "source_only_groups": 0,
    }

    for group in pairs.values():
        sources, followups = group["sources"], group["followups"]
        if not followups:
            diagnostics["source_only_groups"] += 1
            continue
        if not sources:
            diagnostics["orphan_followups"] += len(followups)
            continue
        if len(sources) != 1:
            diagnostics["ambiguous_source_followups"] += len(followups)
            continue
        source_pred = sources[0].get("pred")
        relation = sources[0].get("mr_type")
        for followup in followups:
            followup_pred = followup.get("pred")
            if relation == "inv":
                satisfied = followup_pred == source_pred
            elif relation == "flip":
                expected = flip_map.get(source_pred) if isinstance(source_pred, str) else None
                satisfied = expected is not None and followup_pred == expected
            else:
                satisfied = False
            overall["total"] += 1
            overall["satisfied"] += int(satisfied)
            if relation in counters:
                counters[relation]["total"] += 1
                counters[relation]["satisfied"] += int(satisfied)

    report: dict[str, Any] = {
        "overall": _finish(overall, "satisfied", empty_rate=0.0),
        "diagnostics": diagnostics,
    }
    for relation in SA_RELATION_TYPES:
        report[relation] = _finish(counters[relation], "satisfied", empty_rate=0.0)
    return report


def _finish(counter: Mapping[str, int], value_name: str, *, empty_rate: float | None) -> dict[str, Any]:
    total = counter["total"]
    rate = (counter[value_name] / total) if total else empty_rate
    return {value_name: counter[value_name], "total": total, "rate": rate}


def _percent_to_fraction(value: Any) -> Any:
    """The shared NLI evaluator reports rates as percentages; SA uses fractions.

    Converting once, here, keeps every rate in an SA report on the same 0-1 scale
    instead of silently mixing two conventions in one JSON document.
    """
    return None if value is None else value / 100.0


def compute_sa_joint_correctness(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Delegate to the shared evaluator, normalised to SA's fraction convention."""
    report = compute_joint_correctness(list(results))
    report.pop("neutral", None)
    for bucket in ("overall", *SA_RELATION_TYPES):
        entry = report.get(bucket)
        if isinstance(entry, dict) and "rate" in entry:
            entry["rate"] = _percent_to_fraction(entry["rate"])
    return report


def _group_mr_id(rows: Mapping[str, list[Mapping[str, Any]]]) -> str | None:
    for row in rows["followups"]:
        if row.get("mr_id"):
            return str(row["mr_id"])
    for row in rows["sources"]:
        if row.get("mr_id"):
            return str(row["mr_id"])
    return None


def _group_family(rows: Mapping[str, list[Mapping[str, Any]]]) -> str | None:
    for bucket in ("followups", "sources"):
        for row in rows[bucket]:
            if row.get("property_family"):
                return str(row["property_family"])
    return None


def compute_sa_relation_report(
    results: Sequence[Mapping[str, Any]],
    *,
    flip_map: Mapping[str, str] = SA_FLIP_MAP,
    source_correct_only: bool = False,
) -> dict[str, Any]:
    """Full SA relation report: per-MR, per-family, micro and macro aggregates.

    ``source_correct_only`` restricts every count to groups whose source was
    predicted correctly, which separates "two errors happened to satisfy the
    relation" from a genuine violation after the source was understood.
    """
    rows = list(results)
    pairs, missing = _group_pairs(rows)

    per_mr: dict[str, dict[str, int]] = defaultdict(lambda: {"satisfied": 0, "total": 0})
    per_family: dict[str, dict[str, int]] = defaultdict(lambda: {"satisfied": 0, "total": 0})
    overall = {"satisfied": 0, "total": 0}
    excluded = 0

    for group in pairs.values():
        sources, followups = group["sources"], group["followups"]
        if len(sources) != 1 or not followups:
            continue
        source = sources[0]
        if source_correct_only and source.get("correct") is not True:
            excluded += len(followups)
            continue
        source_pred = source.get("pred")
        relation = source.get("mr_type")
        mr_id = _group_mr_id(group)
        family = _group_family(group)
        for followup in followups:
            followup_pred = followup.get("pred")
            if relation == "inv":
                satisfied = followup_pred == source_pred
            elif relation == "flip":
                expected = flip_map.get(source_pred) if isinstance(source_pred, str) else None
                satisfied = expected is not None and followup_pred == expected
            else:
                satisfied = False
            overall["total"] += 1
            overall["satisfied"] += int(satisfied)
            if mr_id:
                per_mr[mr_id]["total"] += 1
                per_mr[mr_id]["satisfied"] += int(satisfied)
            if family:
                per_family[family]["total"] += 1
                per_family[family]["satisfied"] += int(satisfied)

    def _rates(buckets: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
        return {
            name: _finish(counter, "satisfied", empty_rate=None)
            for name, counter in sorted(buckets.items())
        }

    mr_rates = _rates(per_mr)
    family_rates = _rates(per_family)
    return {
        "scope": "source_correct_only" if source_correct_only else "all_groups",
        "excluded_groups": excluded,
        "msr": compute_sa_msr(rows, flip_map),
        "joint_correctness": compute_sa_joint_correctness(rows),
        "overall_micro": _finish(overall, "satisfied", empty_rate=None),
        "mr_macro": _macro(mr_rates),
        "family_macro": _macro(family_rates),
        "per_mr": mr_rates,
        "per_family": family_rates,
        "diagnostics": {
            "pair_groups": len(pairs),
            "missing_pair_id_rows": missing,
            "relation_types_seen": sorted(
                {str(row.get("mr_type")) for row in rows if row.get("mr_type")}
            ),
            "unsupported_relation_types": sorted(
                {
                    str(row.get("mr_type"))
                    for row in rows
                    if row.get("mr_type") and row.get("mr_type") not in SA_RELATION_TYPES
                }
            ),
        },
    }


def _macro(rates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Equal-weight mean of per-bucket rates, so one MR cannot dominate."""
    usable = [entry["rate"] for entry in rates.values() if entry.get("rate") is not None]
    if not usable:
        return None
    return {"macro_rate": sum(usable) / len(usable), "buckets": len(usable)}


def check_metric_contract(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Preflight the row contract before computing anything."""
    rows = list(rows)
    mr_types = Counter(str(row.get("mr_type")) for row in rows)
    unknown = {name: count for name, count in mr_types.items() if name not in RELATION_TYPES and name != "None"}
    missing_pred = sum(1 for row in rows if "pred" not in row)
    missing_id = sum(1 for row in rows if _pair_key(row.get("pair_id")) is None)
    return {
        "rows": len(rows),
        "mr_type_counts": dict(mr_types),
        "unknown_mr_types": unknown,
        "rows_missing_pred": missing_pred,
        "rows_missing_pair_id": missing_id,
        "ok": not unknown and missing_pred == 0 and missing_id == 0,
    }
