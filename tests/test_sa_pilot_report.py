"""Offline tests for the SA pilot quality report and review artifacts."""

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sa_pilot_report as report


def verdict(generator_pass=None, classifier_pass=None, *, source="positive",
            followup=None, gate_mode="both"):
    validators = {}
    if generator_pass is not None:
        validators["generator"] = {
            "ran": True,
            "pass": generator_pass,
            "source_predicted": source,
            "followup_predicted": followup or ("negative" if generator_pass else "positive"),
        }
    if classifier_pass is not None:
        validators["classifier"] = {
            "ran": True,
            "pass": classifier_pass,
            "source_predicted": source,
            "followup_predicted": followup or ("negative" if classifier_pass else "positive"),
        }
    return {"gate_mode": gate_mode, "validators": validators, "passed": bool(generator_pass)}


def accepted_row(group_id, mr_id, verdict_obj, *, label="negative", source_label="positive"):
    return {
        "group_id": group_id,
        "mr_id": mr_id,
        "dataset": "imdb",
        "is_source": False,
        "text": "A reversed text of a reasonable length here.",
        "label": label,
        "source_label": source_label,
        "source_text": "An original text of a reasonable length here.",
        "property_family": "structural_invariance",
        "relation_type": "preserve",
        "automatic_checks": {"relation_verdict": verdict_obj},
    }


def pool_rows(n_positive=4, n_negative=4):
    rows = []
    for index in range(n_positive):
        rows.append({"source_id": f"p{index}", "text": f"good {index}", "label": "positive"})
    for index in range(n_negative):
        rows.append({"source_id": f"n{index}", "text": f"bad {index}", "label": "negative"})
    return rows


# --------------------------------------------------------------------------- #
# Gate re-evaluation
# --------------------------------------------------------------------------- #

def test_gate_modes_read_the_recorded_verdicts():
    both_pass = verdict(True, True, followup="negative")
    assert report.gate_passes(both_pass, "generator") is True
    assert report.gate_passes(both_pass, "classifier") is True
    assert report.gate_passes(both_pass, "both") is True
    assert report.gate_passes(both_pass, "any") is True

    split = verdict(True, False, followup="negative")
    assert report.gate_passes(split, "generator") is True
    assert report.gate_passes(split, "classifier") is False
    assert report.gate_passes(split, "both") is False
    assert report.gate_passes(split, "any") is True


def test_gate_returns_none_when_a_validator_did_not_run():
    only_generator = verdict(True, None, followup="negative")
    assert report.gate_passes(only_generator, "both") is None
    assert report.gate_passes(only_generator, "generator") is True
    assert report.gate_passes(None, "generator") is None


def test_gate_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown gate mode"):
        report.gate_passes(verdict(True, True), "sideways")


# --------------------------------------------------------------------------- #
# Domain summary
# --------------------------------------------------------------------------- #

def test_summary_counts_accepted_and_gate_risk_per_mr():
    good = accepted_row("g1", "sa_tense_shift", verdict(True, True))
    risky = accepted_row("g2", "sa_tense_shift", verdict(True, False))
    payload = report.summarize_domain(
        "imdb",
        accepted_rows=[good, risky],
        reject_rows=[],
        pool_rows=pool_rows(),
    )
    entry = payload["per_mr"]["sa_tense_shift"]
    assert entry["n_accepted"] == 2
    assert entry["n_applicable"] == 2
    assert entry["acceptance_rate"] == 1.0
    assert entry["gate_risk"]["generator_gate_acceptances"] == 2
    assert entry["gate_risk"]["classifier_disagreements"] == 1
    assert entry["gate_risk"]["generator_only_false_accept_estimate"] == 0.5


def test_summary_separates_declines_from_failures():
    rejected = [
        {"group_id": "d1", "mr_id": "sa_tense_shift", "stage": "generator_declined",
         "reason": "generator_declined"},
        {"group_id": "f1", "mr_id": "sa_tense_shift", "stage": "validation",
         "reason": "unchanged_text"},
        {"group_id": "s1", "mr_id": "sa_tense_shift", "stage": "applicability",
         "reason": "text_too_short_for_tense_rewrite"},
    ]
    payload = report.summarize_domain(
        "imdb", accepted_rows=[], reject_rows=rejected, pool_rows=pool_rows()
    )
    entry = payload["per_mr"]["sa_tense_shift"]
    assert entry["n_generator_declined"] == 1
    assert entry["n_failed"] == 1
    # Only applicability-passing groups count as attempted.
    assert entry["n_applicable"] == 2
    assert entry["applicability_rate"] == 2 / 8


def test_summary_reports_every_mr_even_with_no_data():
    payload = report.summarize_domain(
        "imdb", accepted_rows=[], reject_rows=[], pool_rows=pool_rows()
    )
    assert len(payload["per_mr"]) == 8
    assert payload["per_mr"]["sa_case_reversal"]["n_accepted"] == 0
    assert payload["per_mr"]["sa_case_reversal"]["acceptance_rate"] is None


def test_harmonized_gate_preview_shows_the_counterfactual():
    rows = [
        accepted_row("g1", "sa_tense_shift", verdict(True, True)),
        accepted_row("g2", "sa_tense_shift", verdict(True, False)),
    ]
    payload = report.summarize_domain(
        "imdb", accepted_rows=rows, reject_rows=[], pool_rows=pool_rows()
    )
    preview = payload["per_mr"]["sa_tense_shift"]["harmonized_gate_preview"]
    assert preview["generator"]["acceptance_rate"] == 1.0
    assert preview["classifier"]["acceptance_rate"] == 0.5
    assert preview["both"]["acceptance_rate"] == 0.5


def test_validator_label_confusion_is_recorded():
    rows = [accepted_row("g1", "sa_tense_shift", verdict(True, True, followup="negative"))]
    payload = report.summarize_domain(
        "imdb", accepted_rows=rows, reject_rows=[], pool_rows=pool_rows()
    )
    confusion = payload["per_mr"]["sa_tense_shift"]["validator_label_confusion"]
    assert confusion == {"negative|negative": 1}


def test_overall_gate_risk_aggregates_across_domains():
    domains = {
        "imdb": report.summarize_domain(
            "imdb",
            accepted_rows=[
                accepted_row("g1", "sa_tense_shift", verdict(True, True)),
                accepted_row("g2", "sa_tense_shift", verdict(True, False)),
            ],
            reject_rows=[],
            pool_rows=pool_rows(),
        )
    }
    overall = report.overall_gate_risk(domains)
    assert overall["generator_gate_acceptances"] == 2
    assert overall["classifier_disagreements"] == 1
    assert overall["generator_only_false_accept_estimate"] == 0.5


def test_classifier_vs_gold_is_surfaced_when_present():
    payload = report.summarize_domain(
        "imdb",
        accepted_rows=[],
        reject_rows=[],
        pool_rows=pool_rows(),
        classifier_gold={"decided": 8, "correct": 6, "undecided": 0,
                         "total": 8, "agreement_on_decided": 0.75},
    )
    assert payload["classifier_vs_gold"]["agreement_on_decided"] == 0.75


# --------------------------------------------------------------------------- #
# Agreement statistics
# --------------------------------------------------------------------------- #

def test_cohen_kappa_perfect_agreement_is_one():
    result = report.cohen_kappa(["yes", "no", "yes"], ["yes", "no", "yes"])
    assert result["observed_agreement"] == 1.0
    assert result["kappa"] == 1.0


def test_cohen_kappa_chance_agreement_is_near_zero():
    first = ["yes"] * 10 + ["no"] * 10
    second = ["yes"] * 10 + ["no"] * 10
    second[0] = "no"
    second[10] = "yes"
    result = report.cohen_kappa(first, second)
    assert result["kappa"] < 0.9


def test_cohen_kappa_rejects_mismatched_lengths():
    with pytest.raises(report.PilotReportError):
        report.cohen_kappa(["yes"], ["yes", "no"])


def test_cohen_kappa_handles_empty_input():
    assert report.cohen_kappa([], [])["kappa"] is None


def test_agreement_from_sheets_uses_only_completed_judgements(tmp_path):
    header = "group_id,relation_valid\n"
    sheet_a = tmp_path / "a.csv"
    sheet_b = tmp_path / "b.csv"
    sheet_a.write_text(header + "g1,yes\ng2,no\ng3,\n", encoding="utf-8")
    sheet_b.write_text(header + "g1,yes\ng2,yes\ng3,yes\n", encoding="utf-8")
    result = report.agreement_from_sheets(sheet_a, sheet_b, "relation_valid")
    assert result["n"] == 2
    assert result["observed_agreement"] == 0.5


def test_agreement_reads_a_sheet_that_has_a_utf8_bom(tmp_path):
    """Regression: spreadsheet exports add a BOM, which used to hide 'group_id'."""
    sheet_a = tmp_path / "a.csv"
    sheet_b = tmp_path / "b.csv"
    sheet_a.write_text("﻿group_id,relation_valid\ng1,yes\n", encoding="utf-8")
    sheet_b.write_text("group_id,relation_valid\ng1,yes\n", encoding="utf-8")
    result = report.agreement_from_sheets(sheet_a, sheet_b, "relation_valid")
    assert result["n"] == 1
    assert result["observed_agreement"] == 1.0


def test_agreement_rejects_a_sheet_without_a_group_id_column(tmp_path):
    bad = tmp_path / "bad.csv"
    good = tmp_path / "good.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    good.write_text("group_id,relation_valid\ng1,yes\n", encoding="utf-8")
    with pytest.raises(report.PilotReportError, match="group_id"):
        report.agreement_from_sheets(bad, good, "relation_valid")


def test_agreement_mode_does_not_require_a_pilot_root():
    """The documented --agreement invocation must work without --pilot-root."""
    args = report.build_parser().parse_args(["--agreement", "a.csv", "b.csv"])
    assert args.pilot_root is None
    assert args.agreement == ["a.csv", "b.csv"]


def test_agreement_from_sheets_raises_when_nothing_is_filled(tmp_path):
    header = "group_id,relation_valid\n"
    sheet_a = tmp_path / "a.csv"
    sheet_b = tmp_path / "b.csv"
    sheet_a.write_text(header + "g1,\n", encoding="utf-8")
    sheet_b.write_text(header + "g1,\n", encoding="utf-8")
    with pytest.raises(report.PilotReportError, match="no completed"):
        report.agreement_from_sheets(sheet_a, sheet_b, "relation_valid")


# --------------------------------------------------------------------------- #
# Review artifacts
# --------------------------------------------------------------------------- #

def test_select_review_groups_balances_labels_per_mr():
    rows = [
        accepted_row(f"g{index}", "sa_tense_shift", verdict(True, True), label=label)
        for index, label in enumerate(["negative"] * 10 + ["positive"] * 10)
    ]
    for row in rows:
        row["source_label"] = row["label"]
    selected = report.select_review_groups(rows, per_mr=6, seed=42)
    assert len(selected) <= 6
    labels = [row["label"] for row in selected]
    assert labels.count("positive") == labels.count("negative")


def test_write_review_artifacts_creates_packet_sheet_and_codebook(tmp_path):
    rows = [accepted_row("g1", "sa_tense_shift", verdict(True, True))]
    written = report.write_review_artifacts(
        tmp_path, rows, {"transformation_validity": 0.95, "relation_validity": 0.95}
    )
    for key in ("samples_for_review", "review_sheet", "codebook"):
        assert Path(written[key]).is_file()

    packet = Path(written["samples_for_review"]).read_text(encoding="utf-8")
    assert "An original text of a reasonable length here." in packet
    assert "A reversed text of a reasonable length here." in packet

    sheet = Path(written["review_sheet"]).read_text(encoding="utf-8")
    assert "transformation_valid" in sheet
    assert "g1" in sheet

    codebook = Path(written["codebook"]).read_text(encoding="utf-8")
    assert "0.95" in codebook


def test_review_sheet_has_blank_judgement_columns(tmp_path):
    rows = [accepted_row("g1", "sa_tense_shift", verdict(True, True))]
    written = report.write_review_artifacts(tmp_path, rows, {})
    import csv

    with Path(written["review_sheet"]).open(encoding="utf-8") as handle:
        rows_out = list(csv.DictReader(handle))
    assert rows_out[0]["relation_valid"] == ""
    assert rows_out[0]["notes"] == ""


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def test_markdown_renders_the_headline_gate_risk():
    payload = report.summarize_domain(
        "imdb",
        accepted_rows=[accepted_row("g1", "sa_tense_shift", verdict(True, False))],
        reject_rows=[],
        pool_rows=pool_rows(),
    )
    markdown = report.render_markdown(
        {
            "method": "SA-MR-Pilot",
            "catalog_version": "v1",
            "catalog_sha256": "a" * 64,
            "domains": {"imdb": payload},
            "validator_ids": {"generator": "generator:model", "classifier": "classifier:model"},
            "preserve_gate": "both",
            "flip_gate": "both",
            "overall_gate_risk": report.overall_gate_risk({"imdb": payload}),
        }
    )
    assert "Gate risk (headline)" in markdown
    assert "sa_tense_shift" in markdown
    assert "100.0%" in markdown
