#!/usr/bin/env python3
"""Import the five RQ1 sentiment-analysis domains into frozen canonical splits.

Downloads each upstream file once into ``<out-root>/original_dataset/_raw/<dataset>/``,
parses it through ``scripts/sa_dataset_registry.py``, writes canonical JSONL per
split role, and emits a hash manifest that later phases can verify against.

This script performs no training and no MR generation; it only establishes the
frozen splits that Gate P1 of the RQ1 SA route document requires.

The Hugging Face mirror is used because ``huggingface.co`` is not resolvable in
this environment; GitHub raw is reachable directly.

Examples:
  # Show the resolved URLs, split sizes and expected row counts without downloading
  python scripts/download_sa_datasets.py --datasets imdb,sst2 --dry-run

  # Import every domain and write the registry manifest
  python scripts/download_sa_datasets.py --datasets imdb,sst2,amazon,yelp,twitter \\
    --report data/sa/registry_manifest.json

  # Re-parse from cached raw bytes with no network access
  python scripts/download_sa_datasets.py --offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from project_runtime import PROJECT_ROOT, configure_console_encoding

import sa_dataset_registry as registry
from sa_lexicon import lexicon_agreement


DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "sa"
TWITTER_SANITY_SAMPLE = 200
TWITTER_SANITY_FLOOR = 0.70
DOWNLOAD_TIMEOUT_SECONDS = 300


class DownloadError(RuntimeError):
    """Raised when an upstream artifact cannot be retrieved or verified."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def _download_with_requests(url: str, timeout: int) -> bytes:
    import requests

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    chunks = []
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _download_with_curl(url: str, destination: Path, timeout: int) -> bytes:
    result = subprocess.run(
        [
            "curl", "-L", "--fail", "--silent", "--show-error",
            "--retry", "3", "--max-time", str(timeout),
            "-o", str(destination), url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DownloadError(
            f"curl failed for {url}: {result.stderr.strip() or result.returncode}"
        )
    return destination.read_bytes()


def fetch_source(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    offline: bool = False,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Retrieve ``url`` into ``destination`` unless a cached copy already exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        payload = destination.read_bytes()
        return {
            "retrieved_at": None,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "cached": True,
        }
    if offline:
        raise DownloadError(
            f"--offline requested but {destination} is not cached; run once with network access"
        )
    try:
        payload = _download_with_requests(url, timeout)
        if not payload:
            raise DownloadError(f"empty response body from {url}")
        destination.write_bytes(payload)
    except Exception as exc:  # noqa: BLE001 - fall back to curl for any transport failure
        if isinstance(exc, DownloadError):
            raise
        print(f"  requests failed ({exc}); retrying with curl", file=sys.stderr)
        payload = _download_with_curl(url, destination, timeout)
    return {
        "retrieved_at": utc_now(),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "cached": False,
    }


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #

def import_domain(
    spec: registry.SADatasetSpec,
    *,
    out_root: Path,
    raw_root: Path,
    seed: int,
    mr_pool_size: int | None,
    force: bool,
    offline: bool,
    previous: Mapping[str, Any] | None,
    verify_hashes: bool,
) -> dict[str, Any]:
    dataset_dir = raw_root / spec.dataset
    parsed_by_role: dict[str, registry.ParseResult] = {}
    parsed_pairs: dict[str, registry.ParseResult] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    source_records: list[dict[str, Any]] = []

    previous_sources = {}
    if previous:
        previous_sources = {item["role"]: item for item in previous.get("sources", [])}

    for source_spec in spec.sources:
        destination = dataset_dir / source_spec.filename
        fetch_meta = fetch_source(
            source_spec.url, destination, force=force, offline=offline
        )
        if verify_hashes and not force:
            recorded = previous_sources.get(source_spec.role, {}).get("sha256")
            if recorded and recorded != fetch_meta["sha256"]:
                raise DownloadError(
                    f"{spec.dataset}/{source_spec.filename}: upstream bytes changed "
                    f"(recorded {recorded[:12]}…, got {fetch_meta['sha256'][:12]}…). "
                    "Re-run with --force-download to accept the new revision."
                )
        parsed = registry.parse_source(source_spec.parser, destination.read_bytes())
        if source_spec.role == registry.ROLE_HUMAN_CAD_PAIRS:
            parsed_pairs[source_spec.role] = parsed
        else:
            parsed_by_role[source_spec.role] = parsed

        expected = source_spec.expected_rows
        source_records.append(
            {
                "role": source_spec.role,
                "filename": source_spec.filename,
                "url": source_spec.url,
                "parser": source_spec.parser,
                "sha256": fetch_meta["sha256"],
                "bytes": fetch_meta["bytes"],
                "retrieved_at": fetch_meta["retrieved_at"],
                "cached": fetch_meta["cached"],
                "parsed_rows": len(parsed.records),
                "expected_rows": expected,
                "expected_rows_match": (expected is None or expected == len(parsed.records)),
                "parse_meta": dict(parsed.meta),
                "note": source_spec.note,
            }
        )
        source_meta[source_spec.role] = {
            "filename": source_spec.filename,
            "sha256": fetch_meta["sha256"],
        }

    splits, assembly = registry.build_domain_splits(
        spec, parsed_by_role, source_meta, seed=seed, mr_pool_size=mr_pool_size
    )

    dataset_out = out_root / "original_dataset" / spec.dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, Any] = {}
    for role, rows in splits.items():
        path = dataset_out / f"{role}.jsonl"
        write_jsonl(path, rows)
        split_records[role] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "rows": len(rows),
            "sha256": file_sha256(path),
            "label_counts": registry.label_counts(rows),
            "shared_role": sorted({tuple(row.get("shared_role") or ()) for row in rows})[0]
            if rows and any(row.get("shared_role") for row in rows)
            else [],
        }

    checks = build_dataset_checks(spec, splits)
    auxiliary: dict[str, Any] = {}

    for role, parsed in parsed_pairs.items():
        pairs, pair_meta = registry.build_human_cad_pairs(parsed, source_meta[role])
        pairs_path = dataset_out / f"{role}.jsonl"
        write_jsonl(pairs_path, pairs)
        auxiliary[role] = {
            "path": str(pairs_path.relative_to(PROJECT_ROOT)),
            "pairs": len(pairs),
            "sha256": file_sha256(pairs_path),
            "role": (
                "Human-written counterfactuals for the same cohort. Quality reference for "
                "the flip MRs only; never used for training or as a test split."
            ),
            **pair_meta,
        }
        checks["human_cad_reference"] = registry.check_human_pairs_alignment(
            pairs, splits.get(registry.ROLE_TRAIN_SOURCE_POOL, [])
        )

    return {
        "display_name": spec.display_name,
        "version": spec.version,
        "license_note": spec.license_note,
        "notes": spec.notes,
        "sources": source_records,
        "splits": split_records,
        "assembly": assembly,
        "checks": checks,
        "auxiliary": auxiliary,
        "deviations": registry.build_deviations(spec),
    }


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def build_dataset_checks(
    spec: registry.SADatasetSpec,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    standard = splits.get(registry.ROLE_STANDARD_TEST, [])
    mr_pool = splits.get(registry.ROLE_MR_TEST_SOURCE_POOL, [])

    checks["standard_test"] = registry.duplicate_report(standard)
    checks["mr_test_source_pool"] = registry.duplicate_report(mr_pool)

    if spec.standard_test_is_mr_pool:
        checks["standard_vs_mr_pool"] = {
            "disjoint": False,
            "reason": "declared deviation: standard test and MR source pool are the same rows",
        }
    else:
        standard_ids = {row["source_id"] for row in standard}
        pool_ids = {row["source_id"] for row in mr_pool}
        standard_keys = {registry.normalized_text_key(row["text"]) for row in standard}
        pool_keys = {registry.normalized_text_key(row["text"]) for row in mr_pool}
        overlap_ids = standard_ids & pool_ids
        overlap_keys = standard_keys & pool_keys
        checks["standard_vs_mr_pool"] = {
            "disjoint": not overlap_ids and not overlap_keys,
            "overlapping_source_ids": len(overlap_ids),
            "overlapping_text_keys": len(overlap_keys),
        }
        if overlap_ids or overlap_keys:
            raise registry.SADatasetError(
                f"{spec.dataset}: standard_test and mr_test_source_pool overlap "
                f"({len(overlap_ids)} ids, {len(overlap_keys)} text keys)"
            )

    if spec.dataset == "twitter":
        checks["label_sanity"] = twitter_label_sanity(mr_pool or standard)

    return checks


def twitter_label_sanity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Guard against a silently inverted 0/1 mapping.

    The upstream file uses 0/1; Sentiment140's native encoding is 0/4.  A wrong
    mapping would invert a large share of the evaluation set, so the import
    hard-stops instead of proceeding on a suspect encoding.
    """
    if not rows:
        raise registry.SADatasetError("twitter: no rows available for the label sanity check")
    sample_size = min(TWITTER_SANITY_SAMPLE, len(rows))
    rng = random.Random(registry.DEFAULT_SEED)
    sample = rng.sample(list(rows), sample_size)
    agreement = lexicon_agreement((row["text"], row["label"]) for row in sample)
    rate = agreement["agreement_on_decided"]
    if rate is None or rate < TWITTER_SANITY_FLOOR:
        raise registry.SADatasetError(
            "twitter: lexicon agreement with the parsed labels is "
            f"{rate} on {sample_size} sampled rows, below the {TWITTER_SANITY_FLOOR} floor. "
            "Verify the 0/1 label mapping before importing."
        )
    return {"sample_size": sample_size, "floor": TWITTER_SANITY_FLOOR, **agreement}


def cross_domain_leakage(
    imported: Mapping[str, Mapping[str, Any]],
    all_splits: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Scan every evaluation domain's pools against the IMDb training cohort."""
    imdb_splits = all_splits.get("imdb", {})
    reference_rows = list(imdb_splits.get(registry.ROLE_TRAIN_SOURCE_POOL, [])) + list(
        imdb_splits.get(registry.ROLE_COMMON_DEV, [])
    )
    if not reference_rows:
        return {"status": "skipped", "reason": "imdb training cohort not imported in this run"}
    reference_keys = {registry.normalized_text_key(row["text"]) for row in reference_rows}

    report: dict[str, Any] = {
        "reference": "imdb train_source_pool + common_dev",
        "reference_rows": len(reference_rows),
        "domains": {},
    }
    for dataset, splits in all_splits.items():
        for role in (registry.ROLE_STANDARD_TEST, registry.ROLE_MR_TEST_SOURCE_POOL):
            rows = splits.get(role, [])
            if not rows:
                continue
            overlapping = sum(
                1
                for row in rows
                if registry.normalized_text_key(row["text"]) in reference_keys
            )
            report["domains"].setdefault(dataset, {})[role] = {
                "rows": len(rows),
                "overlapping_text_keys": overlapping,
                "overlap_rate": overlapping / len(rows),
            }
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import the five SA evaluation domains into frozen canonical splits"
    )
    parser.add_argument(
        "--datasets",
        default=",".join(registry.DEFAULT_DATASETS),
        help=f"Comma-separated domain list (default: {','.join(registry.DEFAULT_DATASETS)})",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--raw-root", default=None, help="Raw cache (default: <out-root>/original_dataset/_raw)")
    parser.add_argument("--report", default=None, help="Manifest path (default: <out-root>/registry_manifest.json)")
    parser.add_argument("--seed", type=int, default=registry.DEFAULT_SEED)
    parser.add_argument("--mr-pool-size", type=int, default=None)
    parser.add_argument("--force-download", action="store_true", help="Ignore cached raw files")
    parser.add_argument("--verify-hashes", action="store_true", help="Fail if upstream bytes differ from the manifest")
    parser.add_argument("--offline", action="store_true", help="Parse cached raw files only")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; download nothing")
    return parser


def parse_dataset_list(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in registry.SA_DATASETS]
    if unknown:
        raise ValueError(
            f"unknown dataset(s): {', '.join(unknown)}; "
            f"choose from {', '.join(registry.SA_DATASETS)}"
        )
    if not names:
        raise ValueError("--datasets cannot be empty")
    return names


def print_dry_run(names: Sequence[str], out_root: Path, raw_root: Path) -> None:
    print(f"Registry: {registry.REGISTRY_VERSION}")
    print(f"Output root: {out_root}")
    print(f"Raw cache: {raw_root}")
    for name in names:
        spec = registry.SA_DATASETS[name]
        print(f"\n=== {name} ({spec.display_name}) — {spec.version} ===")
        for source in spec.sources:
            expected = source.expected_rows if source.expected_rows is not None else "?"
            print(f"  [{source.role}] {source.filename}  expected_rows={expected}")
            print(f"      {source.url}")
        if spec.standard_test_is_mr_pool:
            print("  :: standard_test and mr_test_source_pool are the SAME rows (recorded deviation)")
        elif spec.mr_pool_size:
            print(f"  :: mr_test_source_pool = {spec.mr_pool_size} label-balanced rows carved from standard_test")


def main(argv: list[str] | None = None) -> int:
    configure_console_encoding()
    try:
        args = build_parser().parse_args(argv)
        names = parse_dataset_list(args.datasets)
        out_root = Path(args.out_root).expanduser().resolve()
        raw_root = (
            Path(args.raw_root).expanduser().resolve()
            if args.raw_root
            else out_root / "original_dataset" / "_raw"
        )
        report_path = (
            Path(args.report).expanduser().resolve()
            if args.report
            else out_root / "registry_manifest.json"
        )

        if args.dry_run:
            print_dry_run(names, out_root, raw_root)
            return 0

        previous: dict[str, Any] = {}
        if report_path.is_file():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_datasets = previous.get("datasets", {})

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "task": registry.TASK_NAME,
            "registry_version": registry.REGISTRY_VERSION,
            "generated_at": utc_now(),
            "seed": args.seed,
            "text_normalization": registry.TEXT_NORMALIZATION,
            "datasets": {},
        }

        all_splits: dict[str, dict[str, list[dict[str, Any]]]] = {}
        deviations: list[dict[str, Any]] = []

        for name in names:
            spec = registry.SA_DATASETS[name]
            print(f"Importing {name} ({spec.display_name})…")
            entry = import_domain(
                spec,
                out_root=out_root,
                raw_root=raw_root,
                seed=args.seed,
                mr_pool_size=args.mr_pool_size,
                force=args.force_download,
                offline=args.offline,
                previous=previous_datasets.get(name),
                verify_hashes=args.verify_hashes,
            )
            manifest["datasets"][name] = entry
            deviations.extend(entry["deviations"])

            splits = {
                role: load_jsonl(PROJECT_ROOT / record["path"])
                for role, record in entry["splits"].items()
            }
            all_splits[name] = splits
            for source in entry["sources"]:
                if not source["expected_rows_match"]:
                    print(
                        f"  WARNING: {name}/{source['filename']} expected "
                        f"{source['expected_rows']} rows, parsed {source['parsed_rows']}",
                        file=sys.stderr,
                    )
            for role, record in entry["splits"].items():
                print(
                    f"  {role}: {record['rows']} rows {record['label_counts']} "
                    f"-> {record['path']}"
                )

        manifest["deviations"] = deviations
        manifest["leakage"] = cross_domain_leakage(manifest["datasets"], all_splits)
        manifest["totals"] = {
            dataset: sum(record["rows"] for record in entry["splits"].values())
            for dataset, entry in manifest["datasets"].items()
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nManifest: {report_path}")
        if deviations:
            print(
                f"NOTE: {len(deviations)} recorded deviation(s) from the route document; "
                "downstream reports must surface these."
            )
        return 0
    except (FileNotFoundError, registry.SADatasetError, DownloadError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
