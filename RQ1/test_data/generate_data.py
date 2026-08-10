#!/usr/bin/env python3
"""Generate minimal NLI test datasets for RQ1 smoke-testing."""
import json
from pathlib import Path

TEST_DATA = Path(__file__).resolve().parent

# ══════════════════════════════════════════════════════════════════════
# 原始 NLI 样本（premise, hypothesis, label）
# label: 0=entailment, 1=neutral, 2=contradiction
# ══════════════════════════════════════════════════════════════════════

ENTAILMENT_PAIRS = [
    ("A man is playing guitar on stage.", "A person is performing music."),
    ("Two dogs are running in the park.", "Animals are outside."),
    ("The woman is reading a book.", "Someone is engaged in reading."),
    ("A child is eating an apple.", "A kid is consuming food."),
    ("The car is driving down the road.", "A vehicle is in motion."),
    ("She poured water into the glass.", "Liquid was transferred to a container."),
    ("The cat is sleeping on the couch.", "A pet is resting on furniture."),
    ("He finished the marathon in three hours.", "Someone completed a race."),
]

NEUTRAL_PAIRS = [
    ("A man is playing guitar on stage.", "The concert is sold out."),
    ("Two dogs are running in the park.", "It is raining heavily."),
    ("The woman is reading a book.", "The library closed early today."),
    ("A child is eating an apple.", "Apples are a popular fruit."),
    ("The car is driving down the road.", "The driver forgot to buy gas."),
    ("She poured water into the glass.", "The glass was made in Italy."),
    ("The cat is sleeping on the couch.", "Cats are independent animals."),
    ("He finished the marathon in three hours.", "The weather was sunny all day."),
]

CONTRADICTION_PAIRS = [
    ("A man is playing guitar on stage.", "Nobody is performing any music."),
    ("Two dogs are running in the park.", "There are no animals outside."),
    ("The woman is reading a book.", "The woman is not engaged in any reading."),
    ("A child is eating an apple.", "The child is refusing to eat anything."),
    ("The car is driving down the road.", "The vehicle is parked and stationary."),
    ("She poured water into the glass.", "The glass remains completely empty."),
    ("The cat is sleeping on the couch.", "No pet is anywhere near the furniture."),
    ("He finished the marathon in three hours.", "He did not complete the race at all."),
]

# ══════════════════════════════════════════════════════════════════════
# MR transformation variants (同一 premise 的不同 hypothesis 变换)
# ══════════════════════════════════════════════════════════════════════

# Source: premise + hypothesis pairs
SOURCES = [
    ("A man is playing guitar on stage.", "A person is performing music.", 0),   # entail
    ("Two dogs are running in the park.", "Animals are outside.", 0),            # entail
    ("The woman is reading a book.", "The library closed early today.", 1),      # neutral
    ("A child is eating an apple.", "The child is refusing to eat anything.", 2), # contra
    ("She poured water into the glass.", "Liquid was transferred to a container.", 0), # entail
]

# MR variants per source
# Format: (mr_id, mr_type, hypothesis, label)
MR_VARIANTS = {
    "synonym_replacement": [
        # (mr_type, hypothesis, label)
        ("inv", "An individual is presenting music.", 0),
        ("inv", "Creatures are outdoors.", 0),
        ("inv", "The female is engaged in reading activity.", 1),
        ("inv", "The kid refuses to consume any food.", 2),
        ("inv", "Fluid was moved into a receptacle.", 0),
    ],
    "voice_switch": [
        ("inv", "Music is being performed by a person.", 0),
        ("inv", "The park has dogs running in it.", 0),
        ("inv", "A book is being read by the woman.", 1),
        ("inv", "Any food is being refused by the child.", 2),
        ("inv", "A container received liquid from her.", 0),
    ],
    "adding_contradiction": [
        ("flip", "A person is performing music but the stage is completely empty.", 2),
        ("flip", "Animals are outside yet there are no animals in sight.", 2),
        ("flip", "The library closed early today but it actually stayed open all night.", 1),
        ("flip", "The child is refusing to eat anything although the child just ate a full meal.", 0),
        ("flip", "Liquid was transferred to a container but the container has a large leak.", 2),
    ],
    "antonym_substitution": [
        ("flip", "Nobody is performing any music.", 2),
        ("flip", "No creatures are anywhere.", 2),
        ("flip", "The woman is actively avoiding all reading.", 2),
        ("flip", "The child eagerly accepts all food offered.", 0),
        ("flip", "No liquid was moved anywhere.", 2),
    ],
}


def generate_original_train():
    """Generate a small original training set (~24 examples)."""
    samples = []
    idx = 0
    for prem, hyp in ENTAILMENT_PAIRS[:8]:
        samples.append({"premise": prem, "hypothesis": hyp, "label": 0, "idx": idx}); idx += 1
    for prem, hyp in NEUTRAL_PAIRS[:8]:
        samples.append({"premise": prem, "hypothesis": hyp, "label": 1, "idx": idx}); idx += 1
    for prem, hyp in CONTRADICTION_PAIRS[:8]:
        samples.append({"premise": prem, "hypothesis": hyp, "label": 2, "idx": idx}); idx += 1
    return samples


def generate_original_test(n=5):
    """Generate a small test set."""
    samples = []
    idx = 0
    pairs_with_labels = [
        (ENTAILMENT_PAIRS[0], 0), (ENTAILMENT_PAIRS[1], 0),
        (NEUTRAL_PAIRS[0], 1), (NEUTRAL_PAIRS[1], 1),
        (CONTRADICTION_PAIRS[0], 2),
    ]
    for (prem, hyp), label in pairs_with_labels:
        samples.append({"premise": prem, "hypothesis": hyp, "label": label, "idx": idx})
        idx += 1
    return samples[:n]


def generate_mr_train():
    """Generate MR-augmented training data with pair_id groups."""
    samples = []
    pair_id = 0
    for prem, hyp, label in SOURCES:
        # Original (mr_id=none)
        samples.append({
            "premise": prem, "hypothesis": hyp, "label": label,
            "pair_id": pair_id, "is_syn": False, "weight": 1.0,
            "mr_id": "none", "mr_type": "inv",
        })
        # MR variants
        for mr_id, variants in MR_VARIANTS.items():
            mr_type, var_hyp, var_label = variants[pair_id]
            samples.append({
                "premise": prem, "hypothesis": var_hyp, "label": var_label,
                "pair_id": pair_id, "is_syn": True, "weight": 1.0,
                "mr_id": mr_id, "mr_type": mr_type,
            })
        pair_id += 1
    return samples


def generate_mr_test():
    """Generate MR test data for each MR type."""
    mr_samples = {}
    for mr_id in ["synonym_replacement", "voice_switch"]:
        samples = []
        for i, (prem, hyp, label) in enumerate(SOURCES):
            if mr_id in MR_VARIANTS:
                mr_type, var_hyp, var_label = MR_VARIANTS[mr_id][i]
                samples.append({
                    "premise": prem, "hypothesis": var_hyp, "label": var_label,
                    "mr_type": mr_id, "mr_category": mr_type,
                })
        mr_samples[mr_id] = samples
    return mr_samples


def write_jsonl(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✅ {path} ({len(samples)} lines)")


def write_summary(path, dataset_name, mr_stats):
    """Write an MR summary file (mimics real summary format)."""
    summary = {
        "dataset": dataset_name,
        "split": "test",
        "timestamp": "20260101_000000",
        "total_mrs": len(mr_stats),
        "save_format": "transformed_only",
        "mr_statistics": {
            mr: {"count": cnt, "category": cat} for mr, (cnt, cat) in mr_stats.items()
        },
        "total_test_cases": sum(cnt for cnt, _ in mr_stats.values()),
        "format_note": "0=entailment, 1=neutral, 2=contradiction",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    print("Generating RQ1 smoke-test datasets...\n")

    # ── Original train ───────────────────────────────────────────────
    orig_train = generate_original_train()
    write_jsonl(TEST_DATA / "original" / "sick" / "train.json", orig_train)

    # ── Original test (4 datasets, same small data) ──────────────────
    orig_test = generate_original_test(n=5)
    for ds in ["snli", "mnlim", "mnlimm", "sick"]:
        write_jsonl(TEST_DATA / "original" / ds / "test.json", orig_test)

    # ── MR train (sick only) ─────────────────────────────────────────
    mr_train = generate_mr_train()
    write_jsonl(TEST_DATA / "mr" / "sick_train.json", mr_train)

    # ── MR test (4 datasets, same small data per MR) ──────────────────
    mr_test_data = generate_mr_test()
    for ds in ["snli", "mnlim", "mnlimm", "sick"]:
        ds_dir = TEST_DATA / "mr" / f"{ds}_test_MR"
        for mr_id, samples in mr_test_data.items():
            write_jsonl(ds_dir / f"{ds}_test_{mr_id}_20260101_000000.json", samples)
        # Summary file
        mr_stats = {
            "synonym_replacement": (len(mr_test_data["synonym_replacement"]), "inv"),
            "voice_switch": (len(mr_test_data["voice_switch"]), "inv"),
        }
        write_summary(ds_dir / f"{ds}_test_summary_20260101_000000.json", ds, mr_stats)

    print("\n✅ All test data generated.")


if __name__ == "__main__":
    main()
