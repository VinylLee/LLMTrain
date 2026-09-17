#!/usr/bin/env python3
"""Quality report for the SA MR pilot, plus human-review artifacts.

Runs with no GPU: it reads the pilot's ``accepted.jsonl`` / rejects files and
recomputes everything from the recorded validator verdicts.  ``run_sa_mr_pilot.py``
calls it after generation, so there is a single report implementation and the
numbers can be re-derived later from the frozen artifacts.

The headline analysis is the *gate risk*: the pilot runs both the generator's
blind self-verification and an independent local classifier, so we can estimate
what fraction of the generator-only gate's acceptances the classifier would
disagree with.  That number decides whether the final generator-only pipeline can
be trusted, and it is the main reason the pilot exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project_runtime import PROJECT_ROOT, configure_console_encoding

import sa_mr_catalog as catalog


METHOD_NAME = "SA-MR-Pilot"
SA_LABELS = ("negative", "positive")
REVIEW_COLUMNS = (
    "group_id",
    "dataset",
    "mr_id",
    "property_family",
    "relation_type",
    "source_label",
    "expected_followup_label",
    "transformation_valid",
    "source_label_correct",
    "followup_label_correct",
    "relation_valid",
    "notes",
)


class PilotReportError(RuntimeError):
    """Raised when pilot artifacts are missing or inconsistent."""


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Verdict extraction and gate re-evaluation
# --------------------------------------------------------------------------- #

def relation_verdict_of(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    checks = row.get("automatic_checks") or {}
    return checks.get("relation_verdict")


def validator_pass(verdict: Mapping[str, Any] | None, name: str) -> bool | None:
    """Whether ``name`` passed, or ``None`` when that validator did not run."""
    if not verdict:
        return None
    entry = (verdict.get("validators") or {}).get(name)
    if not isinstance(entry, Mapping) or not entry.get("ran"):
        return None
    return bool(entry.get("pass"))


def gate_passes(verdict: Mapping[str, Any] | None, mode: str) -> bool | None:
    """Re-evaluate a stored verdict under a different gate mode.

    Returns ``None`` when the mode cannot be evaluated from what was recorded.
    """
    if not verdict:
        return None
    if mode == "generator":
        return validator_pass(verdict, "generator")
    if mode == "classifier":
        return validator_pass(verdict, "classifier")
    names = ("generator", "classifier")
    values = [validator_pass(verdict, name) for name in names]
    if mode == "both":
        if any(value is None for value in values):
            return None
        return all(values)
    if mode == "any":
        present = [value for value in values if value is not None]
        return any(present) if present else None
    raise ValueError(f"unknown gate mode {mode!r}")


def validator_label(verdict: Mapping[str, Any] | None, name: str) -> str | None:
    if not verdict:
        return None
    entry = (verdict.get("validators") or {}).get(name)
    if not isinstance(entry, Mapping):
        return None
    return entry.get("followup_predicted")


# --------------------------------------------------------------------------- #
# Per-domain summary
# --------------------------------------------------------------------------- #

def _word_stats(texts: Sequence[str]) -> dict[str, Any]:
    if not texts:
        return {"n": 0}
    counts = sorted(len(text.split()) for text in texts)
    return {
        "n": len(counts),
        "mean": round(statistics.fmean(counts), 1),
        "p50": counts[len(counts) // 2],
        "p90": counts[int(0.9 * (len(counts) - 1))],
        "max": counts[-1],
    }


def summarize_domain(
    dataset: str,
    *,
    accepted_rows: Sequence[Mapping[str, Any]],
    reject_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    classifier_gold: Mapping[str, Any] | None = None,
    gate_modes: Sequence[str] = ("generator", "classifier", "both", "any"),
) -> dict[str, Any]:
    """Aggregate one domain's pilot run into per-MR quality and gate statistics."""
    accepted_groups: dict[str, dict[str, Any]] = {}
    for row in accepted_rows:
        if row.get("is_source"):
            continue
        accepted_groups[str(row.get("group_id"))] = row

    applicable_by_mr: Counter = Counter()
    accepted_by_mr: Counter = Counter()
    declined_by_mr: Counter = Counter()
    failed_by_mr: Counter = Counter()
    reject_reasons: dict[str, Counter] = defaultdict(Counter)
    verdicts_by_mr: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in accepted_rows:
        if row.get("is_source"):
            continue
        mr_id = str(row.get("mr_id"))
        accepted_by_mr[mr_id] += 1
        applicable_by_mr[mr_id] += 1
        verdict = relation_verdict_of(row)
        if verdict:
            verdicts_by_mr[mr_id].append(verdict)

    for row in reject_rows:
        mr_id = str(row.get("mr_id"))
        stage = str(row.get("stage"))
        reason = str(row.get("reason"))
        reject_reasons[mr_id][f"{stage}:{reason}"] += 1
        if stage == "generator_declined":
            declined_by_mr[mr_id] += 1
            applicable_by_mr[mr_id] += 1
        elif stage == "validation":
            failed_by_mr[mr_id] += 1
            applicable_by_mr[mr_id] += 1
        verdict = row.get("relation_verdict")
        if verdict:
            verdicts_by_mr[mr_id].append(verdict)

    per_source_label = Counter(str(row.get("label")) for row in pool_rows)
    per_mr: dict[str, Any] = {}
    for mr_id in catalog.MR_ORDER:
        definition = catalog.get_mr(mr_id)
        attempted = applicable_by_mr[mr_id]
        accepted = accepted_by_mr[mr_id]
        verdicts = verdicts_by_mr[mr_id]

        gate_preview: dict[str, Any] = {}
        for mode in gate_modes:
            evaluable = [gate_passes(verdict, mode) for verdict in verdicts]
            evaluable = [value for value in evaluable if value is not None]
            gate_preview[mode] = {
                "evaluable_groups": len(evaluable),
                "would_accept": sum(1 for value in evaluable if value),
                "acceptance_rate": (
                    sum(1 for value in evaluable if value) / len(evaluable)
                    if evaluable
                    else None
                ),
            }

        generator_accepted = [
            verdict for verdict in verdicts if gate_passes(verdict, "generator") is True
        ]
        disagreements = [
            verdict
            for verdict in generator_accepted
            if gate_passes(verdict, "classifier") is False
        ]
        gate_risk = {
            "generator_gate_acceptances": len(generator_accepted),
            "classifier_disagreements": len(disagreements),
            "generator_only_false_accept_estimate": (
                len(disagreements) / len(generator_accepted) if generator_accepted else None
            ),
        }

        confusion: Counter = Counter()
        for verdict in verdicts:
            generated = validator_label(verdict, "generator")
            classified = validator_label(verdict, "classifier")
            if generated in SA_LABELS and classified in SA_LABELS:
                confusion[f"{generated}|{classified}"] += 1

        accepted_texts = [
            str(row.get("text"))
            for row in accepted_rows
            if not row.get("is_source") and str(row.get("mr_id")) == mr_id
        ]
        per_mr[mr_id] = {
            "property_family": definition.property_family,
            "relation_type": definition.relation_type,
            "mechanism": definition.mechanism,
            "template_id": definition.template_id,
            "n_pool": len(pool_rows),
            "n_applicable": attempted,
            "n_accepted": accepted,
            "n_generator_declined": declined_by_mr[mr_id],
            "n_failed": failed_by_mr[mr_id],
            "applicability_rate": (attempted / len(pool_rows)) if pool_rows else None,
            "acceptance_rate": (accepted / attempted) if attempted else None,
            "end_to_end_yield": (accepted / len(pool_rows)) if pool_rows else None,
            "accepted_followup_word_stats": _word_stats(accepted_texts),
            "reject_reasons": dict(reject_reasons[mr_id].most_common()),
            "harmonized_gate_preview": gate_preview,
            "gate_risk": gate_risk,
            "validator_label_confusion": dict(confusion),
        }

    return {
        "dataset": dataset,
        "pool_size": len(pool_rows),
        "pool_label_counts": dict(per_source_label),
        "accepted_groups": len(accepted_groups),
        "classifier_vs_gold": dict(classifier_gold) if classifier_gold else None,
        "per_mr": per_mr,
    }


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #

def macro_over(payload: Mapping[str, Any], field: str) -> dict[str, Any] | None:
    """Equal-weight mean over MRs, so one high-yield MR cannot dominate."""
    values = [
        entry[field]
        for entry in payload["per_mr"].values()
        if entry.get(field) is not None
    ]
    if not values:
        return None
    return {"macro": statistics.fmean(values), "mrs": len(values)}


def overall_gate_risk(domains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    accepted_total = 0
    disagreements_total = 0
    for payload in domains.values():
        for entry in payload["per_mr"].values():
            accepted_total += entry["gate_risk"]["generator_gate_acceptances"]
            disagreements_total += entry["gate_risk"]["classifier_disagreements"]
    return {
        "generator_gate_acceptances": accepted_total,
        "classifier_disagreements": disagreements_total,
        "generator_only_false_accept_estimate": (
            disagreements_total / accepted_total if accepted_total else None
        ),
    }


# --------------------------------------------------------------------------- #
# Human review artifacts
# --------------------------------------------------------------------------- #

def select_review_groups(
    accepted_rows: Sequence[Mapping[str, Any]], per_mr: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Pick a label-balanced review sample per MR from the accepted follow-ups."""
    import random

    rng = random.Random(seed)
    by_mr: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in accepted_rows:
        if not row.get("is_source"):
            by_mr[str(row.get("mr_id"))].append(row)

    selected: list[dict[str, Any]] = []
    for mr_id in catalog.MR_ORDER:
        rows = by_mr.get(mr_id, [])
        if not rows:
            continue
        by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_label[str(row.get("source_label"))].append(row)
        take = max(1, per_mr // len(SA_LABELS))
        chosen: list[Mapping[str, Any]] = []
        for label in SA_LABELS:
            bucket = by_label.get(label, [])
            chosen.extend(rng.sample(bucket, min(take, len(bucket))))
        selected.extend(chosen[:per_mr])
    return [dict(row) for row in selected]


def write_review_artifacts(
    out_root: Path,
    review_groups: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, str]:
    """Write the markdown packet, the two-annotator CSV and the codebook."""
    lines = [
        "# SA MR pilot — manual review packet",
        "",
        "Every block shows one accepted metamorphic group. Judge the follow-up against",
        "the source and record your answers in `review_sheet.csv`.",
        "",
        f"Pre-registered thresholds: "
        + ", ".join(f"{name} >= {value:.2f}" for name, value in thresholds.items()),
        "",
    ]
    for index, row in enumerate(review_groups, start=1):
        verdict = relation_verdict_of(row) or {}
        lines.extend(
            [
                f"## {index}. `{row.get('group_id')}`",
                "",
                f"- dataset: `{row.get('dataset')}` | MR: `{row.get('mr_id')}` "
                f"({row.get('property_family')}, {row.get('relation_type')})",
                f"- source label: `{row.get('source_label')}` | expected follow-up label: "
                f"`{row.get('label')}`",
                f"- gate mode: `{verdict.get('gate_mode')}` | validators: "
                f"`{', '.join(sorted((verdict.get('validators') or {}).keys()))}`",
                "",
                "**Source**",
                "",
                "> " + str(row.get("source_text")).replace("\n", " "),
                "",
                "**Follow-up**",
                "",
                "> " + str(row.get("text")).replace("\n", " "),
                "",
                "Verdicts:",
                "",
                "| annotator question | value |",
                "|---|---|",
                "| transformation valid? | |",
                "| source label correct? | |",
                "| follow-up label correct? | |",
                "| relation valid? | |",
                "| notes | |",
                "",
            ]
        )
    (out_root / "samples_for_review.md").write_text("\n".join(lines), encoding="utf-8")

    with (out_root / "review_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in review_groups:
            writer.writerow(
                {
                    "group_id": row.get("group_id"),
                    "dataset": row.get("dataset"),
                    "mr_id": row.get("mr_id"),
                    "property_family": row.get("property_family"),
                    "relation_type": row.get("relation_type"),
                    "source_label": row.get("source_label"),
                    "expected_followup_label": row.get("label"),
                    "transformation_valid": "",
                    "source_label_correct": "",
                    "followup_label_correct": "",
                    "relation_valid": "",
                    "notes": "",
                }
            )

    codebook = [
        "# SA MR pilot codebook",
        "",
        "Two annotators fill in `review_sheet.csv` independently; a third adjudicates",
        "disagreements. Record raw judgements, not only the final labels.",
        "",
        "## Questions",
        "",
        "- `transformation_valid`: does the follow-up realise the MR named for it?",
        "- `source_label_correct`: is the gold source label the right sentiment for the source text?",
        "- `followup_label_correct`: is the expected follow-up label the right sentiment for the follow-up text?",
        "- `relation_valid`: does the follow-up actually stand in the declared relation to the source?",
        "",
        "## Pre-registered thresholds",
        "",
    ]
    codebook.extend(f"- {name}: >= {value:.2f}" for name, value in thresholds.items())
    codebook.extend(
        [
            "",
            "A MR failing any threshold is revised, re-sampled or dropped; it is never",
            "passed through a weighted average. Thresholds are frozen before the",
            "training results are visible.",
        ]
    )
    (out_root / "codebook.md").write_text("\n".join(codebook) + "\n", encoding="utf-8")

    return {
        "samples_for_review": str(out_root / "samples_for_review.md"),
        "review_sheet": str(out_root / "review_sheet.csv"),
        "codebook": str(out_root / "codebook.md"),
    }


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> dict[str, Any]:
    """Cohen's kappa for two annotators over categorical judgements.

    Implemented directly so the report has no scikit-learn dependency.
    """
    if len(first) != len(second):
        raise PilotReportError("annotator sheets have different lengths")
    if not first:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}
    labels = sorted(set(first) | set(second))
    total = len(first)
    observed = sum(1 for a, b in zip(first, second) if a == b) / total
    counts_first = Counter(first)
    counts_second = Counter(second)
    expected = sum(
        (counts_first[label] / total) * (counts_second[label] / total) for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "n": total,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
    }


def agreement_from_sheets(path_a: Path, path_b: Path, column: str) -> dict[str, Any]:
    rows_a = {row["group_id"]: row for row in _read_csv(path_a)}
    rows_b = {row["group_id"]: row for row in _read_csv(path_b)}
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        raise PilotReportError("the two sheets share no group_id values")
    filled = [
        key
        for key in shared
        if (rows_a[key].get(column) or "").strip() and (rows_b[key].get(column) or "").strip()
    ]
    if not filled:
        raise PilotReportError(f"no completed '{column}' judgements found in both sheets")
    result = cohen_kappa(
        [(rows_a[key][column] or "").strip() for key in filled],
        [(rows_b[key][column] or "").strip() for key in filled],
    )
    return {"column": column, "shared_groups": len(shared), **result}


def _read_csv(path: Path) -> list[dict[str, str]]:
    # utf-8-sig transparently strips a BOM, which spreadsheet exports add and
    # which would otherwise turn the first header into "﻿group_id".
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "group_id" not in rows[0]:
        raise PilotReportError(
            f"{path} has no 'group_id' column (found: {sorted(rows[0])[:4]}...). "
            "Expected a review sheet produced by sa_pilot_report.py."
        )
    return rows


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #

def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SA MR pilot quality report",
        "",
        f"- Method: `{report['method']}`",
        f"- Catalog: `{report['catalog_version']}` sha256 `{report['catalog_sha256'][:16]}…`",
        f"- Domains: {', '.join(sorted(report['domains']))}",
        f"- Validators: {', '.join(report['validator_ids'].values())}",
        f"- Pilot gate: preserve=`{report['preserve_gate']}` flip=`{report['flip_gate']}`",
        "",
        "## Gate risk (headline)",
        "",
        "How often the independent classifier disagrees with an acceptance the",
        "generator-only gate would make. This is the number that decides whether the",
        "final generator-only pipeline can be trusted.",
        "",
        "| scope | generator acceptances | classifier disagreements | false-accept estimate |",
        "|---|---:|---:|---:|",
    ]
    overall = report["overall_gate_risk"]
    lines.append(
        f"| overall | {overall['generator_gate_acceptances']} | "
        f"{overall['classifier_disagreements']} | "
        f"{_fmt_rate(overall['generator_only_false_accept_estimate'])} |"
    )
    for dataset, payload in sorted(report["domains"].items()):
        accepted_total = sum(
            entry["gate_risk"]["generator_gate_acceptances"]
            for entry in payload["per_mr"].values()
        )
        disagreement_total = sum(
            entry["gate_risk"]["classifier_disagreements"]
            for entry in payload["per_mr"].values()
        )
        rate = (disagreement_total / accepted_total) if accepted_total else None
        lines.append(
            f"| {dataset} | {accepted_total} | {disagreement_total} | {_fmt_rate(rate)} |"
        )

    for dataset, payload in sorted(report["domains"].items()):
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                f"Pool: {payload['pool_size']} sources {payload['pool_label_counts']}; "
                f"{payload['accepted_groups']} accepted groups.",
                "",
            ]
        )
        classifier = payload.get("classifier_vs_gold")
        if classifier:
            lines.extend(
                [
                    "Classifier quality on this domain's sources (the measuring instrument): "
                    f"{classifier['correct']}/{classifier['decided']} decided rows correct, "
                    f"{classifier['undecided']} undecided "
                    f"(agreement {_fmt_rate(classifier['agreement_on_decided'])}).",
                    "",
                ]
            )
        lines.extend(
            [
                "| MR | relation | applicable | accepted | acceptance | classifier-vs-gold-free rate |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for mr_id, entry in payload["per_mr"].items():
            lines.append(
                f"| `{mr_id}` | {entry['relation_type']} | {entry['n_applicable']}/{entry['n_pool']} "
                f"({_fmt_rate(entry['applicability_rate'])}) | "
                f"{entry['n_accepted']} | {_fmt_rate(entry['acceptance_rate'])} | "
                f"{_fmt_rate(entry['gate_risk']['generator_only_false_accept_estimate'])} |"
            )
        lines.append("")
        lines.append("Reject reasons per MR:")
        lines.append("")
        for mr_id, entry in payload["per_mr"].items():
            if entry["reject_reasons"]:
                reasons = ", ".join(
                    f"{name}×{count}" for name, count in list(entry["reject_reasons"].items())[:4]
                )
                lines.append(f"- `{mr_id}`: {reasons}")
    lines.append("")
    return "\n".join(lines)


def _fmt_rate(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


# --------------------------------------------------------------------------- #
# CLI: offline recompute
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute the SA MR pilot quality report from frozen artifacts"
    )
    parser.add_argument(
        "--pilot-root",
        default=None,
        help="Pilot output root. Required unless --agreement is used, which reads two sheets directly.",
    )
    parser.add_argument("--datasets", default="imdb,sst2")
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--review-per-mr", type=int, default=0, help="Write review artifacts if > 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agreement",
        nargs=2,
        metavar=("SHEET_A", "SHEET_B"),
        default=None,
        help="Compute Cohen's kappa between two completed review sheets instead",
    )
    parser.add_argument("--agreement-column", default="relation_valid")
    return parser


def load_domain_artifacts(pilot_root: Path, dataset: str) -> dict[str, Any]:
    directory = pilot_root / dataset
    accepted_path = directory / "accepted.jsonl"
    pool_path = pilot_root / "_pilot_sources" / f"{dataset}_pilot_sources.jsonl"
    if not accepted_path.is_file():
        raise PilotReportError(f"missing accepted groups for {dataset}: {accepted_path}")
    if not pool_path.is_file():
        raise PilotReportError(f"missing frozen pilot pool for {dataset}: {pool_path}")
    classifier_gold_path = directory / "classifier_gold.json"
    return {
        "accepted": read_jsonl(accepted_path),
        "rejects": read_jsonl(directory / "accepted.jsonl.rejects.jsonl"),
        "pool": read_jsonl(pool_path),
        "classifier_gold": (
            json.loads(classifier_gold_path.read_text(encoding="utf-8"))
            if classifier_gold_path.is_file()
            else None
        ),
    }


def build_report(
    pilot_root: Path,
    datasets: Sequence[str],
    *,
    run_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    review_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        artifacts = load_domain_artifacts(pilot_root, dataset)
        domains[dataset] = summarize_domain(
            dataset,
            accepted_rows=artifacts["accepted"],
            reject_rows=artifacts["rejects"],
            pool_rows=artifacts["pool"],
            classifier_gold=artifacts["classifier_gold"],
        )
        review_rows.extend(artifacts["accepted"])

    manifest = dict(run_manifest or {})
    thresholds = dict(catalog.get_mr(catalog.MR_ORDER[0]).quality_gate_thresholds)
    return {
        "method": METHOD_NAME,
        "catalog_version": catalog.MR_CATALOG_VERSION,
        "catalog_sha256": catalog.catalog_sha256(),
        "catalog_orientation": catalog.matrix_orientation(),
        "domains": domains,
        "validator_ids": manifest.get("validator_ids", {}),
        "preserve_gate": manifest.get("preserve_gate"),
        "flip_gate": manifest.get("flip_gate"),
        "seed": manifest.get("seed"),
        "overall_gate_risk": overall_gate_risk(domains),
        "acceptance_macro": {
            dataset: macro_over(payload, "acceptance_rate")
            for dataset, payload in domains.items()
        },
        "applicability_macro": {
            dataset: macro_over(payload, "applicability_rate")
            for dataset, payload in domains.items()
        },
        "quality_gate_thresholds": thresholds,
        "_review_rows": review_rows,
    }


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    try:
        args = build_parser().parse_args(argv)
        if args.agreement:
            result = agreement_from_sheets(
                Path(args.agreement[0]), Path(args.agreement[1]), args.agreement_column
            )
            print(json.dumps(result, indent=2))
            return 0

        if not args.pilot_root:
            raise PilotReportError("--pilot-root is required unless --agreement is used")
        pilot_root = Path(args.pilot_root).expanduser().resolve()
        datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
        report = build_report(pilot_root, datasets)
        review_rows = report.pop("_review_rows", [])

        report_path = (
            Path(args.report_output).expanduser().resolve()
            if args.report_output
            else pilot_root / "pilot_report.json"
        )
        write_json(report_path, report)
        (pilot_root / "pilot_report.md").write_text(render_markdown(report), encoding="utf-8")

        if args.review_per_mr > 0:
            groups = select_review_groups(review_rows, args.review_per_mr, seed=args.seed)
            written = write_review_artifacts(
                pilot_root, groups, report["quality_gate_thresholds"]
            )
            print(f"Review artifacts ({len(groups)} groups): {written['review_sheet']}")
        print(f"Report: {report_path}")
        print(f"Markdown: {pilot_root / 'pilot_report.md'}")
        return 0
    except (FileNotFoundError, PilotReportError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
