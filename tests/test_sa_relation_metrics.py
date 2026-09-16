"""Offline tests for SA relation-level metrics over merged MR test rows."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sa_relation_metrics as metrics


def row(pair_id, is_source, pred, mr_type, *, mr_id=None, family=None, correct=None):
    record = {
        "pair_id": pair_id,
        "is_source": is_source,
        "pred": pred,
        "mr_type": mr_type,
    }
    if mr_id is not None:
        record["mr_id"] = mr_id
    if family is not None:
        record["property_family"] = family
    if correct is not None:
        record["correct"] = correct
    return record


# --------------------------------------------------------------------------- #
# MSR
# --------------------------------------------------------------------------- #

def test_inv_is_satisfied_when_predictions_match():
    rows = [
        row("g1", True, "positive", "inv"),
        row("g1", False, "positive", "inv"),
    ]
    report = metrics.compute_sa_msr(rows)
    assert report["inv"] == {"satisfied": 1, "total": 1, "rate": 1.0}
    assert report["overall"]["rate"] == 1.0


def test_inv_is_violated_when_predictions_differ():
    rows = [
        row("g1", True, "positive", "inv"),
        row("g1", False, "negative", "inv"),
    ]
    assert metrics.compute_sa_msr(rows)["inv"]["rate"] == 0.0


def test_flip_is_satisfied_only_on_the_binary_opposite():
    rows = [
        row("g1", True, "positive", "flip"),
        row("g1", False, "negative", "flip"),
        row("g2", True, "negative", "flip"),
        row("g2", False, "positive", "flip"),
    ]
    report = metrics.compute_sa_msr(rows)
    assert report["flip"] == {"satisfied": 2, "total": 2, "rate": 1.0}


def test_flip_is_not_satisfied_by_an_out_of_space_prediction():
    """A three-way NLI label can never satisfy a binary flip."""
    rows = [
        row("g1", True, "positive", "flip"),
        row("g1", False, "neutral", "flip"),
    ]
    assert metrics.compute_sa_msr(rows)["flip"]["rate"] == 0.0


def test_msr_reports_counts_not_just_rates():
    rows = [
        row("g1", True, "positive", "inv"),
        row("g1", False, "positive", "inv"),
        row("g2", True, "positive", "inv"),
        row("g2", False, "negative", "inv"),
    ]
    report = metrics.compute_sa_msr(rows)["inv"]
    assert report["satisfied"] == 1
    assert report["total"] == 2
    assert report["rate"] == 0.5


def test_msr_diagnostics_flag_pairing_problems():
    rows = [
        row("g1", False, "positive", "inv"),          # orphan follow-up
        {"is_source": True, "pred": "positive", "mr_type": "inv"},  # missing pair_id
        row("g3", True, "positive", "inv"),           # source-only group
    ]
    diagnostics = metrics.compute_sa_msr(rows)["diagnostics"]
    assert diagnostics["orphan_followups"] == 1
    assert diagnostics["missing_pair_id_rows"] == 1
    assert diagnostics["source_only_groups"] == 1


def test_msr_empty_input_returns_zero_rate():
    report = metrics.compute_sa_msr([])
    assert report["overall"] == {"satisfied": 0, "total": 0, "rate": 0.0}


def test_custom_flip_map_is_honoured():
    rows = [
        row("g1", True, "positive", "flip"),
        row("g1", False, "positive", "flip"),
    ]
    assert metrics.compute_sa_msr(rows)["flip"]["rate"] == 0.0
    assert metrics.compute_sa_msr(rows, flip_map={"positive": "positive"})["flip"]["rate"] == 1.0


# --------------------------------------------------------------------------- #
# Joint correctness
# --------------------------------------------------------------------------- #

def test_joint_correctness_requires_both_rows_correct():
    rows = [
        row("g1", True, "positive", "inv", correct=True),
        row("g1", False, "positive", "inv", correct=True),
        row("g2", True, "positive", "inv", correct=True),
        row("g2", False, "positive", "inv", correct=False),
    ]
    report = metrics.compute_sa_joint_correctness(rows)
    assert report["overall"]["correct"] == 1
    assert report["overall"]["total"] == 2
    assert report["overall"]["rate"] == 0.5


def test_joint_correctness_never_reports_the_neutral_bucket():
    rows = [row("g1", True, "positive", "inv", correct=True),
            row("g1", False, "positive", "inv", correct=True)]
    assert "neutral" not in metrics.compute_sa_joint_correctness(rows)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _tagged_rows():
    """Two MRs in two families with known per-MR satisfaction rates."""
    rows = []
    # sa_tense_shift / structural_invariance: 3 of 4 pairs satisfied
    for index in range(4):
        satisfied = index < 3
        rows.append(row(f"t{index}", True, "positive", "inv",
                        mr_id="sa_tense_shift", family="structural_invariance"))
        rows.append(row(f"t{index}", False, "positive" if satisfied else "negative", "inv",
                        mr_id="sa_tense_shift", family="structural_invariance"))
    # sa_antonym_flip / label_transition: 1 of 2 pairs satisfied
    for index in range(2):
        satisfied = index < 1
        rows.append(row(f"a{index}", True, "positive", "flip",
                        mr_id="sa_sentiment_antonym_flip", family="task_specific_label_transition"))
        rows.append(row(f"a{index}", False, "negative" if satisfied else "positive", "flip",
                        mr_id="sa_sentiment_antonym_flip", family="task_specific_label_transition"))
    return rows


def test_per_mr_rates_are_reported_with_counts():
    report = metrics.compute_sa_relation_report(_tagged_rows())
    assert report["per_mr"]["sa_tense_shift"] == {"satisfied": 3, "total": 4, "rate": 0.75}
    assert report["per_mr"]["sa_sentiment_antonym_flip"] == {
        "satisfied": 1, "total": 2, "rate": 0.5
    }


def test_mr_macro_is_the_unweighted_mean_of_per_mr_rates():
    report = metrics.compute_sa_relation_report(_tagged_rows())
    assert report["mr_macro"]["macro_rate"] == pytest.approx((0.75 + 0.5) / 2)
    assert report["mr_macro"]["buckets"] == 2


def test_micro_overall_is_dominated_by_the_larger_mr():
    report = metrics.compute_sa_relation_report(_tagged_rows())
    assert report["overall_micro"]["satisfied"] == 4
    assert report["overall_micro"]["total"] == 6
    assert report["overall_micro"]["rate"] == pytest.approx(4 / 6)


def test_family_macro_aggregates_by_property_family():
    report = metrics.compute_sa_relation_report(_tagged_rows())
    assert set(report["per_family"]) == {"structural_invariance", "task_specific_label_transition"}
    assert report["per_family"]["structural_invariance"]["rate"] == 0.75


def test_source_correct_only_excludes_groups_with_a_wrong_source():
    rows = [
        row("g1", True, "positive", "inv", mr_id="m", family="f", correct=True),
        row("g1", False, "positive", "inv", mr_id="m", family="f", correct=True),
        row("g2", True, "negative", "inv", mr_id="m", family="f", correct=False),
        row("g2", False, "negative", "inv", mr_id="m", family="f", correct=True),
    ]
    full = metrics.compute_sa_relation_report(rows)
    conditional = metrics.compute_sa_relation_report(rows, source_correct_only=True)
    assert full["overall_micro"]["total"] == 2
    assert conditional["overall_micro"]["total"] == 1
    assert conditional["excluded_groups"] == 1
    assert conditional["scope"] == "source_correct_only"


def test_ambiguous_source_groups_are_excluded_from_aggregation():
    rows = [
        row("g1", True, "positive", "inv", mr_id="m", family="f"),
        row("g1", True, "negative", "inv", mr_id="m", family="f"),
        row("g1", False, "positive", "inv", mr_id="m", family="f"),
    ]
    report = metrics.compute_sa_relation_report(rows)
    assert report["overall_micro"]["total"] == 0
    assert report["diagnostics"]["pair_groups"] == 1


def test_report_flags_unsupported_relation_types():
    rows = [
        row("g1", True, "positive", "neutral"),
        row("g1", False, "positive", "neutral"),
    ]
    report = metrics.compute_sa_relation_report(rows)
    assert report["diagnostics"]["unsupported_relation_types"] == ["neutral"]


def test_numeric_pair_ids_are_coerced_to_strings():
    rows = [
        row(1, True, "positive", "inv"),
        row("1", False, "positive", "inv"),
    ]
    assert metrics.compute_sa_msr(rows)["overall"]["total"] == 1


# --------------------------------------------------------------------------- #
# Contract preflight
# --------------------------------------------------------------------------- #

def test_contract_check_passes_on_well_formed_rows():
    rows = [row("g1", True, "positive", "inv"), row("g1", False, "positive", "inv")]
    assert metrics.check_metric_contract(rows)["ok"] is True


def test_contract_check_flags_missing_pred_and_pair_id():
    rows = [{"is_source": True, "mr_type": "inv"}]
    result = metrics.check_metric_contract(rows)
    assert result["ok"] is False
    assert result["rows_missing_pred"] == 1
    assert result["rows_missing_pair_id"] == 1


def test_contract_check_flags_unknown_relation_types():
    rows = [row("g1", True, "positive", "sideways")]
    result = metrics.check_metric_contract(rows)
    assert "sideways" in result["unknown_mr_types"]
    assert result["ok"] is False


def test_joint_correctness_is_normalised_from_percent_to_fraction():
    """The shared evaluator returns 0-100; SA reports 0-1. Guard the boundary."""
    rows = [
        row("g1", True, "positive", "inv", correct=True),
        row("g1", False, "positive", "inv", correct=True),
        row("g2", True, "positive", "inv", correct=True),
        row("g2", False, "positive", "inv", correct=False),
    ]
    from metamorphic_metrics import compute_joint_correctness as nli_impl

    shared = nli_impl(rows)["overall"]["rate"]
    assert shared == 50.0, "upstream convention changed; update the boundary conversion"
    assert metrics.compute_sa_joint_correctness(rows)["overall"]["rate"] == 0.5


def test_all_rates_in_a_report_share_one_scale():
    rows = [
        row("g1", True, "positive", "inv", mr_id="m", family="f", correct=True),
        row("g1", False, "positive", "inv", mr_id="m", family="f", correct=True),
    ]
    report = metrics.compute_sa_relation_report(rows)
    assert report["msr"]["overall"]["rate"] == 1.0
    assert report["joint_correctness"]["overall"]["rate"] == 1.0
    assert report["overall_micro"]["rate"] == 1.0


def test_sa_metrics_never_import_the_nli_msr_function():
    """The NLI evaluator hardcodes a three-way flip map and must stay out of SA paths."""
    source = (SCRIPTS_DIR / "sa_relation_metrics.py").read_text(encoding="utf-8")
    assert "import compute_msr" not in source
    assert metrics.SA_FLIP_MAP == {"positive": "negative", "negative": "positive"}
