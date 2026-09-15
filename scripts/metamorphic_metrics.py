"""Metrics for paired source/follow-up metamorphic-test predictions.

The functions in this module intentionally have no model-framework dependencies,
so saved merged JSONL predictions can be evaluated without loading a model.
"""

from collections import defaultdict


RELATION_TYPES = ("inv", "flip", "neutral")


def _pair_key(value):
    """Normalize numeric/string pair IDs while rejecting missing IDs."""
    if value is None or value == "":
        return None
    return str(value)


def _group_pairs(results):
    """Group rows by pair_id and retain every source for validation."""
    pairs = defaultdict(lambda: {"sources": [], "followups": []})
    missing_pair_id_rows = 0

    for row in results:
        key = _pair_key(row.get("pair_id"))
        if key is None:
            missing_pair_id_rows += 1
            continue
        bucket = pairs[key]
        if row.get("is_source"):
            bucket["sources"].append(row)
        else:
            bucket["followups"].append(row)

    return pairs, missing_pair_id_rows


def _new_counter(value_name):
    return {value_name: 0, "total": 0}


def _finish_rates(counters, value_name, empty_rate):
    for counter in counters.values():
        total = counter["total"]
        counter["rate"] = (
            counter[value_name] / total * 100 if total > 0 else empty_rate
        )


def compute_joint_correctness(results):
    """Compute correctness jointly over each valid source/follow-up pair.

    Each follow-up is one evaluation unit. It is jointly correct only when its
    group contains exactly one source and both the source and follow-up have
    ``correct is True``. Source-only groups do not enter the denominator.

    Returns overall and relation-type counts plus pairing diagnostics. A rate is
    ``None`` when no valid source/follow-up pair is available.
    """
    pairs, missing_pair_id_rows = _group_pairs(results)
    counters = {"overall": _new_counter("correct")}
    counters.update({relation: _new_counter("correct") for relation in RELATION_TYPES})

    orphan_followups = 0
    ambiguous_source_followups = 0
    source_only_groups = 0

    for pair_data in pairs.values():
        sources = pair_data["sources"]
        followups = pair_data["followups"]

        if not followups:
            if sources:
                source_only_groups += 1
            continue
        if not sources:
            orphan_followups += len(followups)
            continue
        if len(sources) != 1:
            ambiguous_source_followups += len(followups)
            continue

        source_correct = sources[0].get("correct") is True
        for followup in followups:
            relation = followup.get("mr_type", "")
            jointly_correct = source_correct and followup.get("correct") is True

            counters["overall"]["total"] += 1
            if jointly_correct:
                counters["overall"]["correct"] += 1

            if relation in counters:
                counters[relation]["total"] += 1
                if jointly_correct:
                    counters[relation]["correct"] += 1

    _finish_rates(counters, "correct", empty_rate=None)
    counters["diagnostics"] = {
        "pair_groups": len(pairs),
        "missing_pair_id_rows": missing_pair_id_rows,
        "orphan_followups": orphan_followups,
        "ambiguous_source_followups": ambiguous_source_followups,
        "source_only_groups": source_only_groups,
    }
    return counters


def compute_msr(results):
    """Compute Metamorphic Satisfaction Rate for valid paired predictions.

    The output shape remains compatible with the previous implementation in
    ``test_mettrain_experiment.py``.
    """
    pairs, _ = _group_pairs(results)
    counters = {"overall": _new_counter("satisfied")}
    counters.update(
        {relation: _new_counter("satisfied") for relation in RELATION_TYPES}
    )

    for pair_data in pairs.values():
        sources = pair_data["sources"]
        if len(sources) != 1:
            continue
        source_pred = sources[0].get("pred", "")

        for followup in pair_data["followups"]:
            relation = followup.get("mr_type", "")
            followup_pred = followup.get("pred", "")
            satisfied = False

            if relation == "inv":
                satisfied = followup_pred == source_pred
            elif relation == "flip":
                expected = {
                    "entailment": "contradiction",
                    "contradiction": "entailment",
                    "neutral": "neutral",
                }.get(source_pred)
                satisfied = expected is not None and followup_pred == expected
            elif relation == "neutral":
                satisfied = followup_pred == "neutral"

            counters["overall"]["total"] += 1
            if satisfied:
                counters["overall"]["satisfied"] += 1

            if relation in counters:
                counters[relation]["total"] += 1
                if satisfied:
                    counters[relation]["satisfied"] += 1

    # Preserve the legacy MSR convention: an empty denominator reports 0.0%.
    _finish_rates(counters, "satisfied", empty_rate=0.0)
    return counters
