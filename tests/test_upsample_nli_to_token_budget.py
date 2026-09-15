"""Tests for deterministic post-split NLI token-budget matching."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from upsample_nli_to_token_budget import select_repetitions


def make_rows(count=60):
    labels = ("entailment", "neutral", "contradiction")
    return [
        {
            "instruction": f"instruction-{index}",
            "input": f"input-{index}",
            "output": labels[index % len(labels)],
        }
        for index in range(count)
    ]


def test_selection_is_deterministic_and_hits_exact_budget():
    rows = make_rows()
    lengths = [40 + (index % 13) for index in range(len(rows))]
    base = sum(lengths)
    target = base * 2 + sum(lengths[:17])
    first = select_repetitions(
        rows, lengths, target, seed=42, tolerance=0, max_repeat=4
    )
    second = select_repetitions(
        rows, lengths, target, seed=42, tolerance=0, max_repeat=4
    )
    assert first["final_tokens"] == target
    assert first["token_error"] == 0
    assert first["rows"] == second["rows"]
    assert first["repeats"] == second["repeats"]


def test_selection_covers_every_base_row_and_respects_controls():
    rows = make_rows(90)
    lengths = [35 + (index % 19) for index in range(len(rows))]
    target = int(sum(lengths) * 2.65)
    result = select_repetitions(
        rows,
        lengths,
        target,
        seed=44,
        tolerance=32,
        max_repeat=4,
        max_label_drift=1.0,
    )
    assert set(result["repeats"]) == set(range(len(rows)))
    assert max(result["repeats"].values()) <= 4
    assert result["max_label_drift_pp"] <= 1.0
    assert abs(result["token_error"]) <= 32


def test_selection_rejects_impossible_capacity():
    rows = make_rows(9)
    lengths = [50] * len(rows)
    with pytest.raises(ValueError, match="容量不足"):
        select_repetitions(
            rows,
            lengths,
            target_tokens=sum(lengths) * 5,
            seed=42,
            max_repeat=4,
        )
