import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "RQ1"))

from metamorphic_metrics import compute_joint_correctness, compute_msr
import run_rq1_nli as rq1


def result(pair_id, is_source, correct, mr_type="inv", pred="entailment"):
    return {
        "pair_id": pair_id,
        "is_source": is_source,
        "correct": correct,
        "mr_type": mr_type,
        "pred": pred,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_joint_correctness_uses_each_valid_followup_as_one_unit():
    rows = [
        result("p1", True, True),
        result("p1", False, True, "inv"),
        result("p1", False, False, "flip"),
        result("p2", True, False, pred="neutral"),
        result("p2", False, True, "neutral", pred="neutral"),
        result("p2", False, False, "inv", pred="neutral"),
        result("source-only", True, True),
        result("orphan", False, True),
        result("ambiguous", True, True),
        result("ambiguous", True, True),
        result("ambiguous", False, True),
        result("", False, True),
    ]

    metric = compute_joint_correctness(rows)

    assert metric["overall"] == {"correct": 1, "total": 4, "rate": 25.0}
    assert metric["inv"] == {"correct": 1, "total": 2, "rate": 50.0}
    assert metric["flip"] == {"correct": 0, "total": 1, "rate": 0.0}
    assert metric["neutral"] == {"correct": 0, "total": 1, "rate": 0.0}
    assert metric["diagnostics"] == {
        "pair_groups": 5,
        "missing_pair_id_rows": 1,
        "orphan_followups": 1,
        "ambiguous_source_followups": 1,
        "source_only_groups": 1,
    }


def test_joint_correctness_matches_numeric_and_string_pair_ids():
    metric = compute_joint_correctness([
        result(7, True, True),
        result("7", False, True),
    ])
    assert metric["overall"] == {"correct": 1, "total": 1, "rate": 100.0}


def test_joint_correctness_is_not_calculable_without_complete_pairs():
    metric = compute_joint_correctness([
        result("source-only", True, True),
        result("orphan", False, True),
    ])
    assert metric["overall"] == {"correct": 0, "total": 0, "rate": None}


def test_shared_msr_keeps_existing_relation_semantics():
    rows = [
        result("p1", True, True, pred="entailment"),
        result("p1", False, True, "inv", pred="entailment"),
        result("p1", False, True, "flip", pred="contradiction"),
        result("p1", False, True, "neutral", pred="neutral"),
    ]
    metric = compute_msr(rows)
    assert metric["overall"] == {"satisfied": 3, "total": 3, "rate": 100.0}


def test_rq1_reader_and_reports_include_joint_correctness(tmp_path):
    exp_name = "rq1_original_snli_seed42"
    rows = [
        result("p1", True, True, pred="entailment"),
        result("p1", False, True, "inv", pred="entailment"),
        result("p2", True, False, pred="neutral"),
        result("p2", False, True, "inv", pred="neutral"),
    ]
    write_jsonl(tmp_path / exp_name / "tests" / "merged" / "snli.jsonl", rows)

    parsed = rq1.read_test_results(tmp_path / exp_name)
    joint = parsed["merged"]["snli"]["joint_correctness"]["overall"]
    assert joint == {"correct": 1, "total": 2, "rate": 50.0}

    experiment = {
        "name": exp_name,
        "exp_type": "original",
        "train_dataset": "snli",
        "train_data": "unused.jsonl",
        "target": 2,
        "seed": 42,
    }
    summary_path = rq1.generate_summary(tmp_path, [experiment], "test-model", [42])
    summary = summary_path.read_text(encoding="utf-8")
    assert "Joint Correctness" in summary
    assert "## Merged Test Results — Joint Correctness (overall)" in summary
    assert "50.00%" in summary

    json_path = rq1.generate_json_report(tmp_path, [experiment], "test-model")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    report_joint = report["experiments"][exp_name]["results"]["merged"]["snli"]
    assert report_joint["joint_correctness"]["overall"]["rate"] == 50.0


def test_rq1_multiseed_joint_correctness_uses_sample_standard_deviation(tmp_path):
    experiments = []
    for seed, source_correct in ((42, True), (43, False)):
        exp_name = f"rq1_original_snli_seed{seed}"
        rows = [
            result("p1", True, source_correct),
            result("p1", False, True),
        ]
        write_jsonl(tmp_path / exp_name / "tests" / "merged" / "snli.jsonl", rows)
        experiments.append({
            "name": exp_name,
            "exp_type": "original",
            "train_dataset": "snli",
            "train_data": "unused.jsonl",
            "target": 2,
            "seed": seed,
        })

    summary_path = rq1.generate_summary(
        tmp_path, experiments, "test-model", [42, 43]
    )
    summary = summary_path.read_text(encoding="utf-8")
    assert "50.00 ± 70.71%" in summary
