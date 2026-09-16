#!/usr/bin/env python3
"""Run the SA MR quality pilot on IMDb and SST-2.

The design point is the **shared source pool**: every MR is run over the same
label-balanced set of sources in the same order, so acceptance and applicability
rates have one common denominator and are directly comparable between MRs.  That
is what turns the pilot into an applicability matrix rather than eight unrelated
generation runs.

The pilot runs two validators at once -- the generator's blind self-verification
(the production gate) and an independent local classifier (the measuring
instrument).  The report quantifies how far they disagree, which is what decides
whether the final generator-only pipeline is trustworthy.

Nothing here is a frozen artifact: the pilot's job is to find out which MRs clear
the quality gate before anything is frozen.

Examples:
  # Plan the pools and prompts; loads no model
  python scripts/run_sa_mr_pilot.py --datasets imdb,sst2 --sources-per-domain 4 --dry-run

  # Full pilot
  python scripts/run_sa_mr_pilot.py --datasets imdb,sst2 --sources-per-domain 200 \\
    --offline --device cuda:0 --validator-mode generator,classifier --flip-gate both \\
    --pilot-root data/sa/MR_testing/_pilot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from project_runtime import PROJECT_ROOT, configure_console_encoding
from generate_generic_llm_aug import HFTextGenerator

import generate_sa_mr as engine
import sa_mr_catalog as catalog
import sa_pilot_report as report_module
from sa_validators import SAValidator


METHOD_NAME = "SA-MR-Pilot"
RUNNER_VERSION = "sa_pilot_v1"
DEFAULT_PILOT_ROOT = PROJECT_ROOT / "data" / "sa" / "MR_testing" / "_pilot"
DEFAULT_SOURCES_PER_DOMAIN = 200
SA_LABELS = ("negative", "positive")
DEFAULT_CLASSIFIER_LOCAL = (
    PROJECT_ROOT / "models" / "siebert" / "sentiment-roberta-large-english"
)


def default_classifier_reference() -> str:
    """Prefer a local classifier checkout so the pilot works without network access."""
    if DEFAULT_CLASSIFIER_LOCAL.is_dir():
        return str(DEFAULT_CLASSIFIER_LOCAL)
    return "siebert/sentiment-roberta-large-english"


class PilotError(RuntimeError):
    """Raised when the pilot cannot assemble its inputs."""


# --------------------------------------------------------------------------- #
# Source pools
# --------------------------------------------------------------------------- #

def load_pool(dataset: str, out_root: Path) -> list[dict[str, Any]]:
    path = out_root / "original_dataset" / dataset / "mr_test_source_pool.jsonl"
    if not path.is_file():
        raise PilotError(
            f"missing MR source pool for {dataset}: {path}. "
            "Run scripts/download_sa_datasets.py first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_pilot_pool(
    rows: Sequence[Mapping[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    """Frozen, label-balanced pilot pool in stable ID order."""
    if size <= 0:
        raise PilotError("--sources-per-domain must be positive")
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label"))].append(row)

    per_label = size // len(SA_LABELS)
    if per_label == 0:
        raise PilotError("--sources-per-domain must be at least 2 to stay label-balanced")

    rng = random.Random(seed)
    chosen: list[Mapping[str, Any]] = []
    for label in SA_LABELS:
        bucket = list(by_label.get(label, []))
        if len(bucket) < per_label:
            raise PilotError(
                f"pool has only {len(bucket)} {label} rows but {per_label} are required"
            )
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_label])
    return [dict(row) for row in sorted(chosen, key=lambda row: str(row["source_id"]))]


def write_pool(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Classifier-vs-gold measurement
# --------------------------------------------------------------------------- #

def classifier_gold_agreement(
    validators: Mapping[str, SAValidator], pool_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """How well the measuring instrument does on this domain's own sources.

    Without this number a gate-risk estimate is uninterpretable: a classifier
    that disagrees with the gold labels cannot be used to judge the generator.
    """
    classifier = validators.get("classifier")
    if classifier is None:
        return None
    texts = [str(row["text"]) for row in pool_rows]
    if hasattr(classifier, "predict_many"):
        verdicts = classifier.predict_many(texts, seed=0)
    else:
        verdicts = [classifier.predict(text, 0) for text in texts]

    decided = correct = undecided = 0
    for row, verdict in zip(pool_rows, verdicts):
        if not verdict.usable:
            undecided += 1
            continue
        decided += 1
        correct += int(verdict.predicted_label == str(row["label"]))
    return {
        "total": len(pool_rows),
        "decided": decided,
        "undecided": undecided,
        "correct": correct,
        "agreement_on_decided": (correct / decided) if decided else None,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def make_cached_generator_factory() -> Any:
    """Load the model at most once across every domain in the run."""
    cache: dict[tuple, Any] = {}

    def factory(**kwargs):
        key = (kwargs.get("model_reference"), kwargs.get("device"), kwargs.get("dtype"))
        if key not in cache:
            cache[key] = HFTextGenerator(**kwargs)
        return cache[key]

    return factory


def run_domain(args: argparse.Namespace, dataset: str, pool_path: Path) -> dict[str, Any]:
    pilot_dir = Path(args.pilot_root).expanduser().resolve() / dataset
    pilot_dir.mkdir(parents=True, exist_ok=True)
    accepted = pilot_dir / "accepted.jsonl"

    captured: dict[str, Any] = {}

    def validator_factory(args_ns, generator, model_reference):
        built = engine.build_validators(args_ns, generator, model_reference)
        captured.clear()
        captured.update(built)
        return built

    argv = [
        "--input", str(pool_path),
        "--dataset", dataset,
        "--output", str(accepted),
        "--mrs", "all",
        "--mr-assignment", "all",
        "--num-samples", str(args.sources_per_domain),
        "--label-strategy", "source",
        "--seed", str(args.seed),
        "--model", args.model,
        "--device", args.device,
        "--dtype", args.dtype,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top-p", str(args.top_p),
        "--max-retries", str(args.max_retries),
        "--batch-size", str(args.batch_size),
        "--flip-max-words", str(args.flip_max_words),
        "--validator-mode", args.validator_mode,
        "--classifier-model", args.classifier_model,
        "--classifier-device", args.classifier_device,
        "--preserve-gate", args.preserve_gate,
        "--flip-gate", args.flip_gate,
        "--resume",
    ]
    if args.model_path:
        argv += ["--model-path", args.model_path]
    if args.offline:
        argv.append("--offline")

    print(f"\n=== {dataset}: {args.sources_per_domain} sources × {len(catalog.MR_ORDER)} MRs ===")
    exit_code = engine.main(
        argv,
        generator_factory=make_cached_generator_factory(),
        validator_factory=validator_factory,
    )

    agreement = None
    if captured:
        pool_rows = report_module.read_jsonl(pool_path)
        agreement = classifier_gold_agreement(captured, pool_rows)
        if agreement is not None:
            report_module.write_json(
                Path(args.pilot_root).expanduser().resolve() / dataset / "classifier_gold.json",
                agreement,
            )
            print(
                f"  classifier vs gold on sources: {agreement['correct']}/"
                f"{agreement['decided']} decided correct"
            )

    return {
        "exit_code": exit_code,
        "accepted_path": str(accepted),
        "validator_ids": {
            name: getattr(validator, "validator_id", name)
            for name, validator in captured.items()
        },
        "classifier_vs_gold": agreement,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SA MR quality pilot over shared label-balanced source pools"
    )
    parser.add_argument("--datasets", default="imdb,sst2")
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "sa"))
    parser.add_argument("--pilot-root", default=str(DEFAULT_PILOT_ROOT))
    parser.add_argument("--sources-per-domain", type=int, default=DEFAULT_SOURCES_PER_DOMAIN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mrs", default="all", help="Accepted for symmetry; the pilot runs all MRs")
    parser.add_argument("--model", default=engine.DEFAULT_MODEL)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sequences per forward pass; 1 disables batching",
    )
    parser.add_argument("--flip-max-words", type=int, default=catalog.DEFAULT_FLIP_MAX_WORDS)
    parser.add_argument("--validator-mode", default="generator,classifier")
    parser.add_argument(
        "--classifier-model",
        default=None,
        help="Local classifier path or hub id (default: the local checkout when present)",
    )
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument("--preserve-gate", choices=engine.GATE_MODES, default="both")
    parser.add_argument("--flip-gate", choices=engine.GATE_MODES, default="both")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-count", type=int, default=1)
    parser.add_argument("--review-per-mr", type=int, default=50)
    parser.add_argument("--no-report", action="store_true", help="Skip the report pass")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    try:
        args = build_parser().parse_args(argv)
        if not args.classifier_model:
            args.classifier_model = default_classifier_reference()
        out_root = Path(args.out_root).expanduser().resolve()
        pilot_root = Path(args.pilot_root).expanduser().resolve()
        datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

        pools: dict[str, list[dict[str, Any]]] = {}
        for dataset in datasets:
            rows = load_pool(dataset, out_root)
            pools[dataset] = build_pilot_pool(rows, args.sources_per_domain, args.seed)

        pool_paths: dict[str, Path] = {}
        for dataset, rows in pools.items():
            path = pilot_root / "_pilot_sources" / f"{dataset}_pilot_sources.jsonl"
            write_pool(path, rows)
            pool_paths[dataset] = path
            print(
                f"{dataset}: pilot pool of {len(rows)} sources "
                f"(labels: {dict(_label_counts(rows))}) -> {path}"
            )

        if args.dry_run:
            for dataset in datasets:
                print_dry_run(args, dataset, pool_paths[dataset])
            return 0

        results: dict[str, Any] = {}
        for dataset in datasets:
            results[dataset] = run_domain(args, dataset, pool_paths[dataset])

        if args.no_report:
            return 0 if all(item["exit_code"] == 0 for item in results.values()) else 2

        manifest = {
            "runner": RUNNER_VERSION,
            "seed": args.seed,
            "sources_per_domain": args.sources_per_domain,
            "validator_mode": args.validator_mode,
            "preserve_gate": args.preserve_gate,
            "flip_gate": args.flip_gate,
            # Validators are identical across domains; record them once.
            "validator_ids": results[datasets[0]]["validator_ids"] if datasets else {},
            "datasets": datasets,
            "exit_codes": {name: item["exit_code"] for name, item in results.items()},
        }
        report = report_module.build_report(pilot_root, datasets, run_manifest=manifest)
        report["run"] = manifest
        review_rows = report.pop("_review_rows", [])

        report_module.write_json(pilot_root / "pilot_report.json", report)
        (pilot_root / "pilot_report.md").write_text(
            report_module.render_markdown(report), encoding="utf-8"
        )
        if args.review_per_mr > 0:
            groups = report_module.select_review_groups(
                review_rows, args.review_per_mr, seed=args.seed
            )
            written = report_module.write_review_artifacts(
                pilot_root, groups, report["quality_gate_thresholds"]
            )
            print(f"\nReview packet: {written['samples_for_review']} ({len(groups)} groups)")
            print(f"Review sheet:  {written['review_sheet']}")

        print(f"\nReport: {pilot_root / 'pilot_report.json'}")
        print(f"Summary: {pilot_root / 'pilot_report.md'}")
        print(report_module.render_markdown(report).split("## Gate risk")[1].split("\n## ")[0])
        return 0 if all(item["exit_code"] == 0 for item in results.values()) else 2
    except (FileNotFoundError, PilotError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("label"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def print_dry_run(args: argparse.Namespace, dataset: str, pool_path: Path) -> None:
    print(f"\n=== {dataset} dry run ===")
    engine.main(
        [
            "--input", str(pool_path),
            "--dataset", dataset,
            "--mrs", "all",
            "--num-samples", str(args.sources_per_domain),
            "--label-strategy", "source",
            "--seed", str(args.seed),
            "--flip-max-words", str(args.flip_max_words),
            "--dry-run",
            "--preview-count", str(args.preview_count),
        ],
        generator_factory=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not load a model")
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
