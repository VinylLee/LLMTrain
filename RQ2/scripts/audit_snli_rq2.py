#!/usr/bin/env python3
"""Preflight audit for the RQ2 SNLI cohort and merged test set."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from convert_nli_to_ft import KNOWN_MR_IDS, normalize_mr_id  # noqa: E402


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line"] = line_no
            rows.append(row)
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_key(row):
    return (str(row.get("premise", "")).strip(), str(row.get("hypothesis", "")).strip())


def group_report(rows, source_field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("pair_id"))].append(row)
    source_counts = Counter()
    invalid = []
    for pair_id, members in groups.items():
        count = sum(bool(row.get(source_field)) for row in members)
        source_counts[count] += 1
        if count != 1:
            invalid.append({"pair_id": pair_id, "source_count": count, "rows": len(members)})
    return groups, source_counts, invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train = load_jsonl(args.train)
    test = load_jsonl(args.test)
    if not train or not test:
        raise ValueError("训练源或 merged test 为空")

    train_groups = defaultdict(list)
    for row in train:
        train_groups[str(row.get("pair_id"))].append(row)
    test_groups, test_source_counts, test_invalid = group_report(test, "is_source")

    train_sources = [row for members in train_groups.values() for row in members if normalize_mr_id(row.get("mr_id")) == "none"]
    train_source_keys = {text_key(row) for row in train_sources}
    test_source_keys = {text_key(row) for row in test if row.get("is_source")}
    overlap = sorted(train_source_keys & test_source_keys)

    unknown_mr = Counter(
        normalize_mr_id(row.get("mr_id"))
        for row in train
        if normalize_mr_id(row.get("mr_id")) not in KNOWN_MR_IDS
    )
    train_source_count_by_group = Counter(
        sum(normalize_mr_id(row.get("mr_id")) == "none" for row in members)
        for members in train_groups.values()
    )
    labels = Counter(str(row.get("label")) for row in train)
    mr_ids = Counter(normalize_mr_id(row.get("mr_id")) for row in train)

    checks = {
        "nonempty_train": bool(train),
        "nonempty_test": bool(test),
        "train_groups_have_unique_source": train_source_count_by_group.get(1, 0) == len(train_groups),
        "test_groups_have_unique_source": not test_invalid,
        "source_disjoint": not overlap,
        "known_training_mr_ids": not unknown_mr,
        "test_has_source_and_followup": any(row.get("is_source") for row in test) and any(not row.get("is_source") for row in test),
    }
    report = {
        "schema_version": 1,
        "train_path": str(Path(args.train)),
        "test_path": str(Path(args.test)),
        "train_sha256": sha256(args.train),
        "test_sha256": sha256(args.test),
        "train_rows": len(train),
        "train_groups": len(train_groups),
        "train_sources": len(train_sources),
        "test_rows": len(test),
        "test_groups": len(test_groups),
        "test_source_counts": dict(test_source_counts),
        "train_label_distribution": dict(labels),
        "train_mr_distribution": dict(mr_ids),
        "unknown_mr_ids": dict(unknown_mr),
        "train_source_count_by_group": dict(train_source_count_by_group),
        "test_invalid_groups": test_invalid[:20],
        "source_overlap_count": len(overlap),
        "source_overlap_examples": [list(pair) for pair in overlap[:10]],
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "train_rows": len(train), "test_rows": len(test), "source_overlap_count": len(overlap)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
