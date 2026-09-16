"""Offline tests for the pluggable SA sentiment validators."""

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sa_validators as validators
from sa_validators import (
    FlippingScriptedValidator,
    GeneratorSelfValidator,
    LexiconValidator,
    ScriptedValidator,
    ValidationVerdict,
    build_validator,
    normalize_sentiment_label,
    validate_sentiment_verification,
)
from generate_generic_llm_aug import GenerationError


# --------------------------------------------------------------------------- #
# Label normalization
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("positive", "positive"),
        ("POSITIVE", "positive"),
        ("Positive", "positive"),
        ("negative", "negative"),
        ("NEGATIVE", "negative"),
        ("pos", "positive"),
        ("neg", "negative"),
        ("LABEL_0", "negative"),
        ("LABEL_1", "positive"),
        ("label_1", "positive"),
    ],
)
def test_normalize_sentiment_label_handles_upstream_spellings(raw, expected):
    assert normalize_sentiment_label(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "neutral", "mixed", "3"])
def test_normalize_sentiment_label_rejects_non_binary_values(raw):
    assert normalize_sentiment_label(raw) is None


# --------------------------------------------------------------------------- #
# Blind verification parsing
# --------------------------------------------------------------------------- #

def test_validate_sentiment_verification_accepts_a_matching_label():
    response = json.dumps(
        {"predicted_label": "negative", "unambiguous": True, "reason": "clearly negative"}
    )
    result = validate_sentiment_verification(response, "negative")
    assert result["predicted_label"] == "negative"


def test_validate_sentiment_verification_rejects_a_mismatch():
    response = json.dumps(
        {"predicted_label": "positive", "unambiguous": True, "reason": "x"}
    )
    with pytest.raises(GenerationError, match="expected 'negative'"):
        validate_sentiment_verification(response, "negative")


def test_validate_sentiment_verification_rejects_ambiguity():
    response = json.dumps(
        {"predicted_label": "negative", "unambiguous": False, "reason": "mixed"}
    )
    with pytest.raises(GenerationError, match="ambiguous"):
        validate_sentiment_verification(response, "negative")


# --------------------------------------------------------------------------- #
# Generator self validator
# --------------------------------------------------------------------------- #

class _FakeGenerator:
    """Records the prompts it is given and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.seeds = []

    def generate(self, messages, seed):
        self.prompts.append(messages)
        self.seeds.append(seed)
        return self.responses.pop(0)


def test_generator_self_validator_uses_the_blind_prompt():
    generator = _FakeGenerator(
        [json.dumps({"predicted_label": "negative", "unambiguous": True, "reason": "r"})]
    )
    validator = GeneratorSelfValidator(generator)
    verdict = validator.predict("The meal was dreadful.", seed=7)

    assert verdict.predicted_label == "negative"
    assert verdict.unambiguous is True
    prompt = generator.prompts[0][-1]["content"]
    assert "The meal was dreadful." in prompt
    # The blind probe must not be told the gold label, the source, or the MR.
    assert "gold" not in prompt.lower()
    assert "expected" not in prompt.lower()


def test_generator_self_validator_reports_an_error_instead_of_raising():
    generator = _FakeGenerator(["not json at all"])
    validator = GeneratorSelfValidator(generator)
    verdict = validator.predict("Some text.", seed=1)
    assert verdict.error is not None
    assert verdict.usable is False


def test_generator_self_validator_survives_a_generator_exception():
    class Boom:
        def generate(self, messages, seed):
            raise RuntimeError("cuda exploded")

    verdict = GeneratorSelfValidator(Boom()).predict("Some text.", seed=1)
    assert verdict.error is not None
    assert "cuda exploded" in verdict.error


# --------------------------------------------------------------------------- #
# Scripted and lexicon validators
# --------------------------------------------------------------------------- #

def test_scripted_validator_returns_the_configured_label():
    validator = ScriptedValidator({"The film was great.": "positive"}, default="negative")
    assert validator.predict("The film was great.", 0).predicted_label == "positive"
    assert validator.predict("Anything else.", 0).predicted_label == "negative"


def test_scripted_validator_can_force_an_error():
    validator = ScriptedValidator(default="positive", error="forced failure")
    verdict = validator.predict("Some text.", 0)
    assert verdict.usable is False
    assert verdict.error == "forced failure"


def test_scripted_validator_can_report_ambiguity():
    validator = ScriptedValidator(default="positive", unambiguous=False)
    assert validator.predict("Some text.", 0).unambiguous is False


def test_flipping_validator_inverts_the_underlying_opinion():
    base = ScriptedValidator({"A good film.": "positive"})
    flipped = FlippingScriptedValidator(base)
    assert flipped.predict("A good film.", 0).predicted_label == "negative"


def test_lexicon_validator_decides_on_clear_polarity():
    validator = LexiconValidator()
    assert validator.predict("An absolutely wonderful and brilliant film.", 0).predicted_label == "positive"
    assert validator.predict("A dreadful and boring waste of time.", 0).predicted_label == "negative"


def test_lexicon_validator_refuses_when_undecided():
    verdict = LexiconValidator().predict("The film ran for two hours.", 0)
    assert verdict.usable is False
    assert verdict.error == "lexicon_undecided"


def test_lexicon_agreement_reports_undecided_separately():
    validator = LexiconValidator()
    result = validator.agreement(
        [
            ("A wonderful film.", "positive"),
            ("A dreadful film.", "negative"),
            ("The film ran for two hours.", "positive"),
        ]
    )
    assert result["total"] == 3
    assert result["decided"] == 2
    assert result["undecided"] == 1
    assert result["correct"] == 2


# --------------------------------------------------------------------------- #
# Verdict helpers and factory
# --------------------------------------------------------------------------- #

def test_verdict_usable_requires_a_binary_label_and_no_error():
    assert ValidationVerdict("v", predicted_label="positive").usable is True
    assert ValidationVerdict("v", predicted_label=None).usable is False
    assert ValidationVerdict("v", predicted_label="positive", error="x").usable is False
    assert ValidationVerdict("v", predicted_label="neutral").usable is False


def test_verdict_serialises_for_the_audit_trail():
    payload = ValidationVerdict("v", predicted_label="positive", confidence=0.9).to_dict()
    assert payload["validator_id"] == "v"
    assert payload["predicted_label"] == "positive"


def test_build_validator_supports_the_lexicon_backend():
    assert isinstance(build_validator("lexicon"), LexiconValidator)


def test_build_validator_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown validator"):
        build_validator("nope")


def test_build_validator_requires_a_generator_instance():
    fake = _FakeGenerator([])
    built = build_validator("generator", generator=fake)
    assert isinstance(built, GeneratorSelfValidator)
    assert "model_id" not in built.validator_id or built.validator_id
