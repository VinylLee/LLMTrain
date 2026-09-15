#!/usr/bin/env python3
"""Recompute the four RQ2 metrics from saved merged JSONL predictions."""

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from metamorphic_metrics import compute_joint_correctness, compute_msr  # noqa: E402


def load_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accuracy(rows, is_source):
    selected = [row for row in rows if bool(row.get("is_source")) == is_source]
    correct = sum(row.get("correct") is True for row in selected)
    return {"correct": correct, "total": len(selected), "rate": correct / len(selected) * 100 if selected else None}


def summarize_one(path):
    rows = load_rows(path)
    msr = compute_msr(rows)
    joint = compute_joint_correctness(rows)
    return {
        "prediction_file": str(path),
        "rows": len(rows),
        "source_accuracy": accuracy(rows, True),
        "mr_accuracy": accuracy(rows, False),
        "msr": msr.get("overall", {}),
        "joint_correctness": joint.get("overall", {}),
        "joint_diagnostics": joint.get("diagnostics", {}),
    }


def display_rate(value):
    return "—" if value is None else f"{value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    records = []
    for prediction in sorted(root.glob("*/tests/merged/snli.jsonl")):
        records.append(summarize_one(prediction))

    report = {
        "schema_version": 1,
        "metric_definitions": {
            "source_accuracy": "correct source rows / all source rows",
            "mr_accuracy": "correct follow-up rows / all follow-up rows",
            "msr": "metamorphic relation satisfaction over valid paired predictions",
            "joint_correctness": "source and follow-up both correct per valid follow-up unit",
        },
        "results": records,
    }
    (root / "rq2_snli_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# RQ2 SNLI Results", "", "Primary evaluation: SNLI merged test with the standard NLI inference prompt.", "", "| Experiment | Source accuracy | MR accuracy | MSR | Joint correctness |", "|---|---:|---:|---:|---:|"]
    for record in records:
        source = record["source_accuracy"]
        mr = record["mr_accuracy"]
        msr = record["msr"]
        joint = record["joint_correctness"]
        name = Path(record["prediction_file"]).parents[2].name
        lines.append(
            f"| {name} | {source['correct']}/{source['total']} ({display_rate(source['rate'])}) | "
            f"{mr['correct']}/{mr['total']} ({display_rate(mr['rate'])}) | "
            f"{msr.get('satisfied', 0)}/{msr.get('total', 0)} ({display_rate(msr.get('rate'))}) | "
            f"{joint.get('correct', 0)}/{joint.get('total', 0)} ({display_rate(joint.get('rate'))}) |"
        )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summarized {len(records)} result files into {root}")


if __name__ == "__main__":
    main()
