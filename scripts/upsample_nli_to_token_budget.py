#!/usr/bin/env python3
"""Deterministically repeat post-split Alpaca NLI rows to a reference token budget."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from convert_nli_to_ft import count_row_token_length, register_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def row_sha256(row):
    return hashlib.sha256(canonical_json_bytes(row)).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"instruction", "input", "output"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段: {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"空数据集: {path}")
    return rows


def save_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_distribution(rows):
    return dict(sorted(Counter(row["output"] for row in rows).items()))


def max_label_drift_pp(base_rows, output_rows):
    base = Counter(row["output"] for row in base_rows)
    output = Counter(row["output"] for row in output_rows)
    labels = set(base) | set(output)
    return max(
        abs(base[label] / len(base_rows) - output[label] / len(output_rows)) * 100
        for label in labels
    )


def _stratified_candidate_indices(labels, max_repeat, seed):
    """Build deterministic random cycles whose every prefix stays approximately stratified."""
    label_to_indices = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)
    proportions = {
        label: len(indices) / len(labels)
        for label, indices in label_to_indices.items()
    }
    result = []
    for cycle in range(max_repeat - 1):
        rng = random.Random(seed + cycle * 1_000_003)
        buckets = {}
        for label, indices in sorted(label_to_indices.items()):
            bucket = list(indices)
            rng.shuffle(bucket)
            buckets[label] = bucket
        emitted = Counter()
        emitted_total = 0
        while any(buckets.values()):
            available = [label for label, bucket in buckets.items() if bucket]
            chosen = max(
                available,
                key=lambda label: (
                    proportions[label] * (emitted_total + 1) - emitted[label],
                    label,
                ),
            )
            result.append(buckets[chosen].pop())
            emitted[chosen] += 1
            emitted_total += 1
    return result


def _repair_subset(candidate_indices, lengths, target, tolerance):
    """0/1 subset sum over a small final window, returning candidate positions."""
    upper = target + tolerance
    reachable = bytearray(upper + 1)
    reachable[0] = 1
    previous_sum = [-1] * (upper + 1)
    previous_candidate = [-1] * (upper + 1)

    last_position = -1
    for position, row_index in enumerate(candidate_indices):
        length = lengths[row_index]
        if length > upper:
            continue
        for total in range(upper, length - 1, -1):
            if not reachable[total] and reachable[total - length]:
                reachable[total] = 1
                previous_sum[total] = total - length
                previous_candidate[total] = position
        last_position = position
        if target <= upper and reachable[target]:
            break

    possible = [total for total, value in enumerate(reachable) if value]
    best = min(possible, key=lambda total: (abs(total - target), total > target, total))
    if abs(best - target) > tolerance:
        raise ValueError(
            f"无法在容差内修复 token budget: target={target}, best={best}, "
            f"error={best-target}, candidates={last_position + 1}"
        )

    selected_positions = []
    current = best
    while current:
        position = previous_candidate[current]
        if position < 0:
            raise RuntimeError("subset-sum reconstruction failed")
        selected_positions.append(position)
        current = previous_sum[current]
    selected_positions.reverse()
    return selected_positions, best


def select_repetitions(
    rows,
    lengths,
    target_tokens,
    seed,
    tolerance=32,
    max_repeat=4,
    max_label_drift=1.0,
    repair_window=5000,
    attempts=20,
):
    if len(rows) != len(lengths) or not rows:
        raise ValueError("rows/lengths 必须非空且等长")
    if any(length <= 0 for length in lengths):
        raise ValueError("token length 必须为正")
    base_tokens = sum(lengths)
    if target_tokens < base_tokens:
        raise ValueError(f"目标 token 小于基础数据: {target_tokens} < {base_tokens}")
    if target_tokens > base_tokens * max_repeat:
        raise ValueError(
            f"max_repeat={max_repeat} 容量不足: {target_tokens} > {base_tokens * max_repeat}"
        )

    labels = [row["output"] for row in rows]
    last_error = None
    for attempt in range(attempts):
        attempt_seed = seed + attempt * 10_000_019
        candidates = _stratified_candidate_indices(labels, max_repeat, attempt_seed)
        remaining = target_tokens - base_tokens
        selected = []
        position = 0
        while remaining > repair_window and position < len(candidates):
            row_index = candidates[position]
            selected.append(row_index)
            remaining -= lengths[row_index]
            position += 1
        if remaining < 0:
            last_error = ValueError("greedy token selection overshot unexpectedly")
            continue
        try:
            tail_positions, tail_tokens = _repair_subset(
                candidates[position:], lengths, remaining, tolerance
            )
        except ValueError as exc:
            last_error = exc
            continue
        selected.extend(candidates[position + pos] for pos in tail_positions)

        indexed_rows = [(index, 0) for index in range(len(rows))]
        occurrence = Counter()
        for index in selected:
            occurrence[index] += 1
            indexed_rows.append((index, occurrence[index]))
        random.Random(attempt_seed + 7_919).shuffle(indexed_rows)
        output_rows = [rows[index] for index, _ in indexed_rows]

        repeats = Counter(index for index, _ in indexed_rows)
        observed_max_repeat = max(repeats.values())
        drift = max_label_drift_pp(rows, output_rows)
        final_tokens = base_tokens + sum(lengths[index] for index in selected)
        if observed_max_repeat > max_repeat:
            last_error = ValueError("重复次数超过上限")
            continue
        if abs(final_tokens - target_tokens) > tolerance:
            last_error = ValueError("token 误差超过容差")
            continue
        if drift > max_label_drift:
            last_error = ValueError(
                f"标签比例漂移 {drift:.4f}pp > {max_label_drift:.4f}pp"
            )
            continue
        return {
            "rows": output_rows,
            "repeats": repeats,
            "base_tokens": base_tokens,
            "final_tokens": final_tokens,
            "token_error": final_tokens - target_tokens,
            "max_label_drift_pp": drift,
            "attempt": attempt,
            "tail_tokens": tail_tokens,
        }

    raise ValueError(f"经过 {attempts} 次确定性尝试仍无法满足约束: {last_error}")


def count_lengths(tokenizer, rows):
    lengths = []
    methods = Counter()
    for row in rows:
        length, method = count_row_token_length(tokenizer, row)
        lengths.append(length)
        methods[method] += 1
    return lengths, dict(sorted(methods.items()))


def split_report(
    split_name,
    base_path,
    reference_path,
    output_path,
    base_rows,
    reference_rows,
    base_lengths,
    reference_lengths,
    selection,
):
    output_rows = selection["rows"]
    repeats = selection["repeats"]
    hashes = [row_sha256(row) for row in base_rows]
    repeat_manifest = [
        {
            "row_sha256": hashes[index],
            "label": base_rows[index]["output"],
            "repeat_count": repeats[index],
        }
        for index in range(len(base_rows))
    ]
    repeat_histogram = dict(
        sorted(Counter(repeats.values()).items(), key=lambda item: item[0])
    )
    target_tokens = sum(reference_lengths)
    return {
        "split": split_name,
        "base": {
            "path": str(Path(base_path).resolve()),
            "sha256": file_sha256(base_path),
            "rows": len(base_rows),
            "tokens": sum(base_lengths),
            "label_distribution": label_distribution(base_rows),
        },
        "reference": {
            "path": str(Path(reference_path).resolve()),
            "sha256": file_sha256(reference_path),
            "rows": len(reference_rows),
            "tokens": target_tokens,
        },
        "output": {
            "path": str(Path(output_path).resolve()),
            "sha256": file_sha256(output_path),
            "rows": len(output_rows),
            "tokens": selection["final_tokens"],
            "target_tokens": target_tokens,
            "token_error": selection["token_error"],
            "label_distribution": label_distribution(output_rows),
            "max_label_drift_pp": selection["max_label_drift_pp"],
            "unique_base_rows_covered": len(repeats),
            "max_repeat": max(repeats.values()),
            "repeat_histogram": {str(k): v for k, v in repeat_histogram.items()},
            "selection_attempt": selection["attempt"],
        },
        "repeat_manifest": repeat_manifest,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-validation", required=True)
    parser.add_argument("--reference-train", required=True)
    parser.add_argument("--reference-validation", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tolerance", type=int, default=32)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--max-label-drift-pp", type=float, default=1.0)
    parser.add_argument("--expected-train-tokens", type=int)
    parser.add_argument("--expected-validation-tokens", type=int)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-validation", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer

    paths = {
        "train": (Path(args.base_train), Path(args.reference_train), Path(args.output_train)),
        "validation": (
            Path(args.base_validation),
            Path(args.reference_validation),
            Path(args.output_validation),
        ),
    }
    for trio in paths.values():
        for path in trio[:2]:
            if not path.is_file():
                raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    reports = {}
    base_hash_sets = {}
    method_counts = Counter()
    expected = {
        "train": args.expected_train_tokens,
        "validation": args.expected_validation_tokens,
    }

    prepared = {}
    for offset, (split_name, (base_path, reference_path, output_path)) in enumerate(paths.items()):
        base_rows = load_rows(base_path)
        reference_rows = load_rows(reference_path)
        base_lengths, base_methods = count_lengths(tokenizer, base_rows)
        reference_lengths, reference_methods = count_lengths(tokenizer, reference_rows)
        method_counts.update(base_methods)
        method_counts.update(reference_methods)
        target_tokens = sum(reference_lengths)
        if expected[split_name] is not None and target_tokens != expected[split_name]:
            raise ValueError(
                f"{split_name} 参考 token 漂移: {target_tokens} != {expected[split_name]}"
            )
        hashes = [row_sha256(row) for row in base_rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"{split_name} 基础数据存在重复 converted row，无法唯一审计")
        base_hash_sets[split_name] = set(hashes)
        selection = select_repetitions(
            base_rows,
            base_lengths,
            target_tokens,
            seed=args.seed + offset * 100_003,
            tolerance=args.tolerance,
            max_repeat=args.max_repeat,
            max_label_drift=args.max_label_drift_pp,
        )
        prepared[split_name] = (
            base_path, reference_path, output_path, base_rows, reference_rows,
            base_lengths, reference_lengths, selection,
        )

    overlap = base_hash_sets["train"] & base_hash_sets["validation"]
    if overlap:
        raise ValueError(f"train/validation converted rows 有交集: {len(overlap)}")

    for split_name, values in prepared.items():
        output_path = values[2]
        selection = values[-1]
        save_rows(selection["rows"], output_path)
        reports[split_name] = split_report(split_name, *values)

    register_dataset(
        args.experiment,
        paths["train"][2],
        paths["validation"][2],
        task_type="nli",
    )
    report = {
        "schema_version": 1,
        "status": "pass",
        "experiment": args.experiment,
        "seed": args.seed,
        "strategy": "post_split_deterministic_stratified_random_repeat",
        "token_counting_method_counts": dict(sorted(method_counts.items())),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "tolerance": args.tolerance,
        "max_repeat_allowed": args.max_repeat,
        "max_label_drift_pp_allowed": args.max_label_drift_pp,
        "base_train_validation_overlap": 0,
        "validation_usage_note": (
            "Repeated validation matches exposure budget only; it is not an independent "
            "validation estimate. The unique full_val.json remains preserved."
        ),
        "splits": reports,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"✅ token matching 完成: {args.experiment}")
    for split_name, split in reports.items():
        output = split["output"]
        print(
            f"  {split_name}: {output['rows']} rows, {output['tokens']} tokens "
            f"(target={output['target_tokens']}, error={output['token_error']})"
        )
    print(f"  report: {report_path}")


if __name__ == "__main__":
    main()
