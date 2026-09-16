"""Offline tests for the eight-MR SA catalog: shape, bridges, prompts and applicability."""

import re
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sa_mr_catalog as catalog


EXPECTED_MR_IDS = [
    "sa_uninformative_context",
    "sa_pronoun_substitution",
    "sa_voice_switch",
    "sa_synonym_nonpolar",
    "sa_tense_shift",
    "sa_case_reversal",
    "sa_controlled_negation_flip",
    "sa_sentiment_antonym_flip",
]

EXPECTED_FAMILIES = {
    "sa_case_reversal": "surface_form_invariance",
    "sa_synonym_nonpolar": "lexical_semantic_equivalence",
    "sa_pronoun_substitution": "reference_preserving_invariance",
    "sa_voice_switch": "structural_invariance",
    "sa_tense_shift": "structural_invariance",
    "sa_uninformative_context": "irrelevant_context_invariance",
    "sa_controlled_negation_flip": "task_specific_label_transition",
    "sa_sentiment_antonym_flip": "task_specific_label_transition",
}


# --------------------------------------------------------------------------- #
# Catalog shape
# --------------------------------------------------------------------------- #

def test_catalog_has_exactly_the_eight_documented_mrs():
    assert list(catalog.MR_ORDER) == EXPECTED_MR_IDS
    assert len(catalog.MR_CATALOG) == 8


def test_relation_split_is_six_preserve_and_two_flip():
    orientation = catalog.matrix_orientation()
    assert orientation["by_relation"] == {"preserve": 6, "flip": 2}


def test_mechanism_split_is_seven_llm_and_one_deterministic():
    orientation = catalog.matrix_orientation()
    assert orientation["by_mechanism"] == {"llm": 7, "deterministic": 1}


@pytest.mark.parametrize("mr_id,family", sorted(EXPECTED_FAMILIES.items()))
def test_property_families_match_the_route_document(mr_id, family):
    assert catalog.get_mr(mr_id).property_family == family


def test_template_ids_are_unique():
    template_ids = [definition.template_id for definition in catalog.MR_CATALOG.values()]
    assert len(template_ids) == len(set(template_ids))


def test_unknown_mr_id_raises_with_the_known_ids():
    with pytest.raises(KeyError, match="sa_tense_shift"):
        catalog.get_mr("sa_does_not_exist")


# --------------------------------------------------------------------------- #
# Bridge onto the shared metric convention
# --------------------------------------------------------------------------- #

def test_mr_type_bridge_never_emits_neutral():
    for definition in catalog.MR_CATALOG.values():
        assert definition.mr_type in {"inv", "flip"}
        assert definition.mr_type != "neutral"


def test_preserve_maps_to_inv_and_flip_maps_to_flip():
    assert catalog.get_mr("sa_tense_shift").mr_type == "inv"
    assert catalog.get_mr("sa_controlled_negation_flip").mr_type == "flip"


@pytest.mark.parametrize(
    "mr_id,source_label,expected",
    [
        ("sa_tense_shift", "positive", "positive"),
        ("sa_tense_shift", "negative", "negative"),
        ("sa_controlled_negation_flip", "positive", "negative"),
        ("sa_sentiment_antonym_flip", "negative", "positive"),
    ],
)
def test_expected_followup_label_covers_all_relation_label_combinations(
    mr_id, source_label, expected
):
    assert catalog.expected_followup_label(mr_id, source_label) == expected


# --------------------------------------------------------------------------- #
# Deterministic case reversal
# --------------------------------------------------------------------------- #

def test_case_reversal_mode_is_deterministic_in_source_id():
    assert catalog.case_reversal_mode("imdb-test-000001") == catalog.case_reversal_mode(
        "imdb-test-000001"
    )
    assert catalog.case_reversal_mode("imdb-test-000001") in catalog.CASE_MODES


def test_case_reversal_produces_the_expected_transform_for_a_known_source():
    source_id = "imdb-test-000001"
    mode = catalog.case_reversal_mode(source_id)
    text = "A Quiet Film."
    result = catalog.apply_case_reversal(text, source_id)
    expected = {"lower": text.lower(), "upper": text.upper(), "swap": text.swapcase()}[mode]
    assert result == expected


def test_case_reversal_preserves_word_order_and_characters():
    text = "The ending was truly dreadful to sit through."
    result = catalog.apply_case_reversal(text, "sst2-test-000042")
    assert result.lower() == text.lower()
    assert len(result) == len(text)


def test_case_reversal_rejects_acronyms_and_emphatic_caps():
    assert catalog.mr_applicable("sa_case_reversal", "I watched this on DVD last night.")[0] is False
    assert catalog.mr_applicable("sa_case_reversal", "This was NEVER any good.")[0] is False


def test_case_reversal_rejects_urls_mentions_and_emoji():
    assert catalog.mr_applicable("sa_case_reversal", "See www.example.com for more.")[0] is False
    assert catalog.mr_applicable("sa_case_reversal", "Ask @someone about this film.")[0] is False
    assert catalog.mr_applicable("sa_case_reversal", "An absolute delight 😀")[0] is False


def test_case_reversal_accepts_plain_text():
    applicable, reason = catalog.mr_applicable(
        "sa_case_reversal", "A quiet little film that never finds its footing."
    )
    assert applicable, reason


def test_case_reversal_is_the_only_deterministic_mr():
    determined = [d.mr_id for d in catalog.MR_CATALOG.values() if d.is_deterministic]
    assert determined == ["sa_case_reversal"]


def test_deterministic_mr_has_no_prompt():
    with pytest.raises(ValueError, match="deterministic"):
        catalog.build_mr_prompt("sa_case_reversal", "Some text.", "positive")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def _prompt_for(mr_id, label="positive"):
    return catalog.build_mr_prompt(mr_id, "A perfectly ordinary sentence about a film.", label)


@pytest.mark.parametrize("mr_id", [m for m in EXPECTED_MR_IDS if m != "sa_case_reversal"])
def test_every_llm_prompt_declares_an_output_contract(mr_id):
    messages = _prompt_for(mr_id)
    user = messages[-1]["content"]
    assert "Return JSON only" in user
    assert '"text"' in user


@pytest.mark.parametrize("mr_id", ["sa_controlled_negation_flip", "sa_sentiment_antonym_flip"])
def test_flip_prompts_state_the_expected_target_label(mr_id):
    user = _prompt_for(mr_id, "positive")[-1]["content"]
    assert "negative" in user
    assert '"edited_label"' in user


def test_flip_prompt_forbids_appending_a_new_sentence():
    user = _prompt_for("sa_controlled_negation_flip")[-1]["content"]
    assert "append" in user.lower()


@pytest.mark.parametrize("mr_id", [m for m in EXPECTED_MR_IDS if m != "sa_case_reversal"])
def test_prompts_never_name_the_mr_or_the_relation(mr_id):
    user = _prompt_for(mr_id)[-1]["content"].lower()
    for forbidden in ("metamorphic", "mr_id", "follow-up", "followup", "source text",
                      "relation", "preserve", "flip"):
        assert forbidden not in user, f"{mr_id} leaks {forbidden!r}"


def test_retry_feedback_is_appended_to_the_prompt():
    messages = catalog.build_mr_prompt(
        "sa_tense_shift", "Some text.", "positive", retry_feedback="too similar"
    )
    assert "too similar" in messages[-1]["content"]


# --------------------------------------------------------------------------- #
# Blind verification prompt
# --------------------------------------------------------------------------- #

def test_verifier_prompt_is_blind():
    """It must not reveal the label, the source text, the MR, or the target."""
    messages = catalog.build_verification_prompt("The meal was outstanding.")
    user = messages[-1]["content"]
    assert "The meal was outstanding." in user
    for forbidden in ("metamorphic", "mr_id", "sa_", "negation", "antonym", "tense",
                      "voice", "synonym", "source", "follow-up", "expected"):
        assert forbidden not in user.lower(), f"verifier leaks {forbidden!r}"


def test_verifier_prompt_does_not_assert_any_gold_label():
    user = catalog.build_verification_prompt("A middling effort.")[-1]["content"]
    assert "the original is" not in user.lower()
    assert "must still be" not in user.lower()
    assert "No label or metadata is provided" in user


# --------------------------------------------------------------------------- #
# Applicability
# --------------------------------------------------------------------------- #

def test_flip_mrs_reject_clearly_mixed_sources():
    mixed = (
        "The lead performance was wonderful and genuinely moving, but the pacing was "
        "dreadful and the script was terrible throughout."
    )
    applicable, reason = catalog.mr_applicable("sa_controlled_negation_flip", mixed)
    assert applicable is False
    assert reason == "flip_mr_source_is_clearly_mixed"


def test_flip_mrs_reject_sources_with_stacked_negation():
    text = "I did not enjoy it and I would never watch it again."
    applicable, reason = catalog.mr_applicable("sa_controlled_negation_flip", text)
    assert applicable is False
    assert reason == "flip_mr_source_has_complex_negation"


def test_flip_mrs_respect_the_configured_word_cap():
    long_text = "This is a wonderful film. " * 60
    applicable, reason = catalog.mr_applicable(
        "sa_controlled_negation_flip", long_text, {"flip_max_words": 50}
    )
    assert applicable is False
    assert reason == "flip_mr_requires_at_most_50_words"


def test_flip_mrs_accept_a_short_single_polarity_source():
    applicable, reason = catalog.mr_applicable(
        "sa_controlled_negation_flip", "An absolutely wonderful and moving film."
    )
    assert applicable, reason


def test_tense_shift_requires_a_finite_verb():
    assert catalog.mr_applicable("sa_tense_shift", "Quite a film indeed, all told.")[0] is False
    long_enough = "The director was absolutely brilliant throughout the whole of this film."
    assert catalog.mr_applicable("sa_tense_shift", long_enough)[0] is True


def test_pronoun_substitution_requires_a_reference_to_replace():
    text = "The film was good. The film was also funny. The film was short."
    assert catalog.mr_applicable("sa_pronoun_substitution", text)[0] is True
    assert catalog.mr_applicable("sa_pronoun_substitution", "Loved it")[0] is False


def test_applicability_returns_a_reason_on_rejection():
    applicable, reason = catalog.mr_applicable("sa_tense_shift", "Short.")
    assert applicable is False
    assert reason


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def test_catalog_sha256_is_stable_across_calls():
    assert catalog.catalog_sha256() == catalog.catalog_sha256()


def test_catalog_sha256_changes_when_a_prompt_changes(monkeypatch):
    before = catalog.catalog_sha256()
    monkeypatch.setattr(catalog, "PROMPT_VERSION", "sa_mr_test_bump")
    assert catalog.catalog_sha256() != before


def test_catalog_sha256_changes_when_the_system_prompt_changes(monkeypatch):
    before = catalog.catalog_sha256()
    monkeypatch.setattr(catalog, "SYSTEM_PROMPT", catalog.SYSTEM_PROMPT + " Be terse.")
    assert catalog.catalog_sha256() != before


def test_catalog_payload_covers_every_mr():
    payload = catalog.catalog_payload()
    assert [entry["mr_id"] for entry in payload["mrs"]] == EXPECTED_MR_IDS


def test_every_mr_carries_its_quality_gate_thresholds():
    for definition in catalog.MR_CATALOG.values():
        thresholds = dict(definition.quality_gate_thresholds)
        assert thresholds["transformation_validity"] == 0.95
        assert thresholds["relation_validity"] == 0.95


def test_flip_templates_do_not_leak_the_mr_name_into_the_operation():
    for mr_id in ("sa_controlled_negation_flip", "sa_sentiment_antonym_flip"):
        operation = catalog.get_mr(mr_id).operation.lower()
        assert not re.search(r"\bmr\b|metamorphic", operation)
