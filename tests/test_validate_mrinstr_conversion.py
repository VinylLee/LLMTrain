from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_mrinstr_conversion import (
    add_check,
    description_matrix_counts,
    determine_stage2_status,
    select_manual_sample_indices,
    validate_by_mr_token_summaries,
)


def sample(pair_id, mr_id, suffix):
    return {
        "pair_id": pair_id,
        "mr_id": mr_id,
        "premise": f"premise-{suffix}",
        "hypothesis": f"hypothesis-{suffix}",
        "label": 0,
        "idx": suffix,
    }


def test_stage2_status_requires_checks_and_no_blockers():
    checks = []
    add_check(checks, "ok", True, "fine")
    assert determine_stage2_status(checks, []) == {
        "automated_checks_status": "PASS",
        "training_gate": "OPEN",
        "stage2_status": "PASS",
    }
    blocked = determine_stage2_status(checks, [{"code": "risk"}])
    assert blocked["automated_checks_status"] == "PASS"
    assert blocked["training_gate"] == "BLOCKED"
    assert blocked["stage2_status"] == "FAIL"


def test_stage2_status_fails_when_an_automated_check_fails():
    checks = []
    add_check(checks, "bad", False, "broken")
    status = determine_stage2_status(checks, [])
    assert status["automated_checks_status"] == "FAIL"
    assert status["training_gate"] == "BLOCKED"
    assert status["stage2_status"] == "FAIL"


def test_manual_selection_is_deterministic_and_stratified_by_mr():
    rows = [sample(0, "none", 0)]
    rows.extend(sample(1, "synonym_replacement", i) for i in range(1, 6))
    rows.extend(sample(2, "conditional_clause", i) for i in range(6, 11))
    first = select_manual_sample_indices(rows, seed=42, per_mr=3)
    second = select_manual_sample_indices(list(rows), seed=42, per_mr=3)
    assert first == second
    selected_mrs = [rows[index]["mr_id"] for index in first]
    assert selected_mrs.count("synonym_replacement") == 3
    assert selected_mrs.count("conditional_clause") == 3
    assert "none" not in selected_mrs


def test_description_matrix_counts_returns_both_marginals():
    matrix = {
        "true-a": {"assigned-b": 2, "assigned-c": 1},
        "true-b": {"assigned-a": 2},
    }
    rows, columns = description_matrix_counts(matrix)
    assert rows == {"true-a": 3, "true-b": 2}
    assert columns == {"assigned-b": 2, "assigned-c": 1, "assigned-a": 2}


def test_validate_by_mr_token_summaries_checks_counts_and_cutoff_totals():
    summaries = {
        "none": {
            "count": 2, "p95": 8, "max": 8,
            "over_cutoff_count": 0, "over_cutoff_ratio": 0.0,
        },
        "adding_contradiction": {
            "count": 1, "p95": 12, "max": 12,
            "over_cutoff_count": 1, "over_cutoff_ratio": 1.0,
        },
    }
    overall = {"count": 3, "over_cutoff_count": 1}
    assert validate_by_mr_token_summaries(
        summaries, {"none": 2, "adding_contradiction": 1}, overall
    )
    summaries["none"]["count"] = 1
    assert not validate_by_mr_token_summaries(
        summaries, {"none": 2, "adding_contradiction": 1}, overall
    )
