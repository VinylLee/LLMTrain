import json
import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_generic_llm_aug as generic_aug


SOURCE_ROWS = [
    {
        "premise": "A cyclist in a yellow jacket waits beside a traffic light.",
        "hypothesis": "A person is waiting near a road.",
        "label": 0,
    },
    {
        "premise": "Several chefs prepare vegetables in a restaurant kitchen.",
        "hypothesis": "The restaurant serves seafood tonight.",
        "label": "neutral",
    },
    {
        "premise": "A black dog is running through fresh snow.",
        "hypothesis": "No animal is moving outdoors.",
        "label": 2,
    },
]


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_load_jsonl_even_with_json_suffix(tmp_path):
    source = tmp_path / "train.json"
    write_jsonl(source, SOURCE_ROWS)
    assert generic_aug.load_records(source) == SOURCE_ROWS


def test_balanced_tasks_are_deterministic_and_balanced():
    rows = SOURCE_ROWS * 3
    first = generic_aug.select_tasks(rows, 6, seed=17, label_strategy="balanced")
    second = generic_aug.select_tasks(rows, 6, seed=17, label_strategy="balanced")
    assert [(x.source_index, x.target_label) for x in first] == [
        (x.source_index, x.target_label) for x in second
    ]
    assert sorted(task.target_label for task in first) == [
        "contradiction",
        "contradiction",
        "entailment",
        "entailment",
        "neutral",
        "neutral",
    ]


def test_prompt_is_english_generic_augmentation_without_mr_metadata():
    task = generic_aug.select_tasks(SOURCE_ROWS, 1, seed=2, label_strategy="source")[0]
    prompt = "\n".join(message["content"] for message in generic_aug.build_prompt(task))
    assert "Target label:" in prompt
    assert "entailment" in prompt and "neutral" in prompt and "contradiction" in prompt
    lowered = prompt.lower()
    for forbidden in ("metamorphic", "mr_id", "source-follow-up", "operation description"):
        assert forbidden not in lowered


def test_prompt_uses_deterministic_scenario_hint():
    task = generic_aug.select_tasks(SOURCE_ROWS, 1, seed=2, label_strategy="source")[0]
    prompt = "\n".join(message["content"] for message in generic_aug.build_prompt(task))
    assert "Required scenario family:" in prompt
    assert "Use this only as a topic constraint" in prompt



def test_blind_semantic_verification_requires_matching_unambiguous_label():
    accepted = generic_aug.validate_verification(
        json.dumps(
            {
                "predicted_label": "entailment",
                "unambiguous": True,
                "reason": "The hypothesis restates a fact in the premise.",
            }
        ),
        "entailment",
    )
    assert accepted["unambiguous"] is True

    with pytest.raises(generic_aug.GenerationError, match="expected"):
        generic_aug.validate_verification(
            json.dumps({"predicted_label": "neutral", "unambiguous": True, "reason": "unknown"}),
            "entailment",
        )
    with pytest.raises(generic_aug.GenerationError, match="ambiguous"):
        generic_aug.validate_verification(
            json.dumps({"predicted_label": "entailment", "unambiguous": False, "reason": "both"}),
            "entailment",
        )

def test_validation_accepts_fenced_json_and_rejects_copy_or_wrong_label():
    task = generic_aug.select_tasks(SOURCE_ROWS, 1, seed=1, label_strategy="source")[0]
    valid = json.dumps(
        {
            "premise": "Two musicians rehearse a new song inside an empty theater.",
            "hypothesis": "Performers are practicing music in a theater.",
            "label": task.target_label,
        }
    )
    parsed = generic_aug.validate_generation(
        f"```json\n{valid}\n```", task, max_source_similarity=0.85, max_text_chars=1000
    )
    assert parsed["label"] == task.target_label

    copied = json.dumps(
        {
            "premise": task.source["premise"],
            "hypothesis": "This sentence is unrelated to the copied premise.",
            "label": task.target_label,
        }
    )
    with pytest.raises(generic_aug.GenerationError, match="too similar"):
        generic_aug.validate_generation(
            copied, task, max_source_similarity=0.85, max_text_chars=1000
        )

    wrong_label = json.dumps(
        {
            "premise": "A librarian places returned books onto a wooden cart.",
            "hypothesis": "A library worker is handling returned books.",
            "label": next(label for label in generic_aug.LABEL_TO_ID if label != task.target_label),
        }
    )
    with pytest.raises(generic_aug.GenerationError, match="expected"):
        generic_aug.validate_generation(
            wrong_label, task, max_source_similarity=0.85, max_text_chars=1000
        )


def test_main_writes_converter_compatible_rows_and_report(tmp_path):
    source = tmp_path / "train.json"
    output = tmp_path / "generated.jsonl"
    write_jsonl(source, SOURCE_ROWS)

    class FakeGenerator:
        calls = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, messages, seed):
            FakeGenerator.calls += 1
            prompt = messages[-1]["content"]
            target = re.search(r"Target label: (\w+)", prompt).group(1)
            index = FakeGenerator.calls
            examples = {
                "entailment": (
                    f"A painter displays canvas number {index} in a quiet gallery.",
                    f"Canvas number {index} is displayed by an artist.",
                ),
                "neutral": (
                    f"A student opens notebook number {index} before class.",
                    f"The student will receive the highest grade in class number {index}.",
                ),
                "contradiction": (
                    f"A shop closes door number {index} before midnight.",
                    f"Door number {index} remains open throughout the night.",
                ),
            }
            premise, hypothesis = examples[target]
            return json.dumps(
                {"premise": premise, "hypothesis": hypothesis, "label": target}
            )

    exit_code = generic_aug.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--num-samples",
            "3",
            "--seed",
            "42",
            "--label-strategy",
            "balanced",
            "--no-semantic-check",
        ],
        generator_factory=FakeGenerator,
    )
    assert exit_code == 0
    rows = generic_aug.load_records(output)
    assert len(rows) == 3
    assert all(row["mr_id"] == "none" for row in rows)
    assert all(row["type"] == "generic_llm_generated" for row in rows)
    assert sorted(row["label"] for row in rows) == [0, 1, 2]
    assert len({row["pair_id"] for row in rows}) == 3

    report = json.loads(
        output.with_suffix(output.suffix + ".report.json").read_text(encoding="utf-8")
    )
    assert report["method"] == "generic-LLM-Aug"
    assert report["complete"] is True
    assert report["generated_samples"] == 3


def test_dry_run_does_not_require_output_or_construct_model(tmp_path):
    source = tmp_path / "train.json"
    write_jsonl(source, SOURCE_ROWS)

    def forbidden_factory(**kwargs):
        raise AssertionError("dry-run must not construct a model")

    assert generic_aug.main(
        ["--input", str(source), "--num-samples", "1", "--dry-run"],
        generator_factory=forbidden_factory,
    ) == 0
