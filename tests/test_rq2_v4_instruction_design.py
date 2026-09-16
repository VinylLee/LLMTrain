"""Unit tests for the RQ2 v4 design (ordered provenance Operation + constraint Relation).

v4 keeps the v3 experimental matrix (grounding-aware 2x2x2, same block order, same
P/O/R/L semantics).  What it changes is the *content model* of two channels:

  Operation : an ordered transformation trace, expanded from ``component_mrs``.
  Relation  : a constraint over the (source label -> follow-up label) mapping.

The properties asserted here are exactly the ones the v3 wording could not
guarantee: order preservation, no sorting/dedup, one Operation renderer for
atomic and composite rows, and no relation-class leakage into the Operation text.

No GPU, no model, no tokenizer.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import convert_nli_to_ft as converter  # noqa: E402
from convert_nli_to_ft import (  # noqa: E402
    INSTRUCTION_NLI,
    build_matched_control_audit,
    build_matched_control_rows_meta,
    build_original_map,
    build_pair_groups,
    convert_to_alpaca,
    select_matched_shuffled_operation_donors_for,
    select_shuffled_relation_kinds_v4,
    sort_samples_by_stable_key,
)
from mr_instruction_design import (  # noqa: E402
    COMPOSITE_OPERATION_FALLBACK,
    CONTROL_MODES,
    CORE_MODES,
    DIAGNOSTIC_MODES,
    INSTRUCTION_TEMPLATES_V4,
    MR_V4_RELATION_KINDS,
    V4_ATOMIC_OPERATION_STEPS,
    V4_COMPOSITE_OPERATION_FALLBACK_STEP,
    V4_OPERATION_TRACE_INTRO,
    V4_RELATION_KIND_DESCRIPTIONS,
    mode_design_meta,
    mode_spec,
    relation_kind_v4,
    render_operation_v4,
    render_relation_v4,
    resolve_operation_trace_v4,
)

V4 = 4


# ============================================================
# Fixtures
# ============================================================
SOURCE = {
    "premise": "A tree service employee is wearing safety equipment while up in a lift.",
    "hypothesis": "The employee works on trimming a tree.",
    "label": 0, "pair_id": 11, "mr_id": "none", "mr_type": "inv",
}
COMPOSITE_FLIP = {
    "premise": SOURCE["premise"],
    "hypothesis": "The person is not trimming a tree.",
    "label": 2, "pair_id": 11, "mr_id": "composite_flip",
    "mr_type": "flip", "component_mrs": ["pronoun_substitution", "negation_flip"],
}
ATOMIC_INV = {
    "premise": SOURCE["premise"], "hypothesis": "The worker trims a tree.",
    "label": 0, "pair_id": 12, "mr_id": "synonym_replacement",
    "mr_type": "inv", "component_mrs": None,
}
ATOMIC_EC = {
    "premise": "A dog is running.", "hypothesis": "No dog is running.",
    "label": 2, "pair_id": 13, "mr_id": "adding_contradiction",
    "mr_type": "flip", "component_mrs": None,
}
ATOMIC_EN = {
    "premise": "A cat sleeps.", "hypothesis": "A cat sleeps outdoors.",
    "label": 1, "pair_id": 14, "mr_id": "uninformative",
    "mr_type": "neutral", "component_mrs": None,
}


def render_v4(sample, mode, original=SOURCE, original_label="", op=None, rel=None):
    mr_id = sample["mr_id"]
    if op is None:
        trace, provenance = resolve_operation_trace_v4(mr_id, sample.get("component_mrs"))
        op = render_operation_v4(trace, provenance)
    if rel is None:
        rel = render_relation_v4(mr_id)
    return converter.build_instruction_for_sample(
        sample=sample, original_sample=original, mode=mode,
        operation_description=op, relation_description=rel,
        original_label=original_label, is_original=(mr_id == "none"),
        template_version=V4,
    )["instruction"]


def make_dataset():
    """A well-formed cohort: one source per pair plus one follow-up."""
    rows = []
    for i, follow in enumerate([ATOMIC_INV, ATOMIC_EC, ATOMIC_EN, COMPOSITE_FLIP]):
        src = dict(SOURCE)
        src["pair_id"] = 100 + i
        src["premise"] = f"Source premise {i}."
        src["hypothesis"] = f"Source hypothesis {i}."
        follow = dict(follow)
        follow["pair_id"] = 100 + i
        follow["premise"] = f"Source premise {i}."
        rows.append(src)
        rows.append(follow)
    rows.append(dict(COMPOSITE_FLIP, pair_id=200, mr_id="composite_neutral",
                     mr_type="neutral", label=1,
                     component_mrs=["pronoun_substitution", "uninformative"]))
    src = dict(SOURCE, pair_id=200, premise="Source premise 4.", hypothesis="Source hypothesis 4.")
    rows.append(src)
    return rows


def dataset():
    rows = sort_samples_by_stable_key(make_dataset())
    groups = build_pair_groups(rows)
    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]
    original_map, missing = build_original_map(groups)
    assert not missing, missing
    return rows, groups, original_map


def converted(mode, require_provenance=False, seed=1041):
    rows, _, original_map = dataset()
    donors, meta = select_matched_shuffled_operation_donors_for(rows, seed)
    rels = select_shuffled_relation_kinds_v4(rows, seed)
    return convert_to_alpaca(
        rows, mode=mode, original_map=original_map,
        shuffled_operation_donors=donors, shuffled_relation_types=rels,
        require_composite_provenance=require_provenance,
        template_version=V4,
    )


# ============================================================
# Ordered trace semantics
# ============================================================
def test_component_order_is_respected():
    a = dict(COMPOSITE_FLIP, component_mrs=["pronoun_substitution", "negation_flip"])
    b = dict(COMPOSITE_FLIP, component_mrs=["negation_flip", "pronoun_substitution"])
    ia = render_v4(a, "operation_only")
    ib = render_v4(b, "operation_only")
    assert ia != ib

    assert ia.index("coreferential") < ia.index("negation marker")
    assert ib.index("negation marker") < ib.index("coreferential")


def test_trace_is_not_sorted_or_deduplicated():
    trace, provenance = resolve_operation_trace_v4(
        "composite_flip", ["voice_switch", "adding_contradiction"])
    assert trace == ("voice_switch", "adding_contradiction"), "must keep stored order"
    assert provenance == "composite_components"

    trace, _ = resolve_operation_trace_v4(
        "composite_flip", ["synonym_replacement", "synonym_replacement"])
    assert trace == ("synonym_replacement", "synonym_replacement"), "must not deduplicate"

    text = render_operation_v4(trace, "composite_components")
    assert text.count("1.") == 1 and text.count("2.") == 1


def test_repeated_component_renders_two_steps():
    text = render_operation_v4(("voice_switch", "voice_switch"), "composite_components")
    assert text.count("A sentence was rewritten between active and passive voice.") == 2


def test_atomic_and_composite_share_one_renderer():
    atomic = render_v4(ATOMIC_INV, "operation_only")
    composite = render_v4(COMPOSITE_FLIP, "operation_only")
    for text in (atomic, composite):
        assert "Input transformation:" in text
        assert V4_OPERATION_TRACE_INTRO in text
        assert "\n1. " in text
    # atomic is simply an arity-1 trace of the same shape
    assert atomic.count("\n1. ") == 1 and "\n2. " not in atomic


def test_same_component_sequence_gives_identical_operation_across_composite_types():
    """Prevents re-introducing relation-class leakage through Operation wording."""
    flip = dict(COMPOSITE_FLIP, mr_id="composite_flip", mr_type="flip")
    inv = dict(COMPOSITE_FLIP, mr_id="composite_inv", mr_type="inv", label=0)
    neutral = dict(COMPOSITE_FLIP, mr_id="composite_neutral", mr_type="neutral", label=1)

    ops = {render_v4(row, "operation_only") for row in (flip, inv, neutral)}
    assert len(ops) == 1, ops

    rels = {render_v4(row, "relation_only") for row in (flip, inv, neutral)}
    assert len(rels) == 3, "relation kind must still differ"


def test_operation_text_carries_no_output_relation_hint():
    for kind, mr_id in MR_V4_RELATION_KINDS.items():
        if kind.startswith("composite"):
            continue
    forbidden = ("label", "entailment", "contradiction", "neutral", "output", "predict")
    texts = list(V4_ATOMIC_OPERATION_STEPS.values()) + [V4_COMPOSITE_OPERATION_FALLBACK_STEP]
    for text in texts:
        low = text.lower()
        for token in forbidden:
            assert token not in low, f"{token!r} leaks into an Operation step: {text}"


# ============================================================
# Composite provenance gate
# ============================================================
def test_composite_without_provenance_falls_back_and_is_counted():
    row = dict(COMPOSITE_FLIP)
    row.pop("component_mrs")
    text = render_v4(row, "operation_only")
    assert V4_COMPOSITE_OPERATION_FALLBACK_STEP in text

    rows, _, original_map = dataset()
    rows = [dict(r) for r in rows]
    for r in rows:
        if r["mr_id"].startswith("composite"):
            r.pop("component_mrs", None)
    _, report = convert_to_alpaca(rows, mode="operation_only", template_version=V4)
    assert report["composite_operation_fallback_count"] == 2


def test_strict_gate_rejects_missing_composite_provenance():
    rows, _, original_map = dataset()
    rows = [dict(r) for r in rows]
    for r in rows:
        if r["mr_id"].startswith("composite"):
            r.pop("component_mrs", None)
    with pytest.raises(ValueError, match="component_mrs provenance"):
        convert_to_alpaca(
            rows, mode="operation_only",
            require_composite_provenance=True, template_version=V4,
        )


def test_strict_gate_passes_when_provenance_present():
    _, report = converted("operation_only", require_provenance=True)
    assert report["composite_operation_fallback_count"] == 0


def test_unknown_composite_component_is_not_silently_dropped():
    trace, provenance = resolve_operation_trace_v4(
        "composite_flip", ["pronoun_substitution", "mystery_mr"])
    assert trace == ()
    assert provenance == "unsupported"


# ============================================================
# Block presence / ordering / leakage (v4 wording)
# ============================================================
@pytest.mark.parametrize("mode", CORE_MODES)
def test_core_mode_block_presence_v4(mode):
    spec = mode_spec(mode)
    instr = render_v4(COMPOSITE_FLIP, mode)
    assert ("Paired source input:" in instr) == bool(spec["pair"])
    assert ("Input transformation:" in instr) == bool(spec["operation"])
    assert ("Output relation:" in instr) == bool(spec["relation"])
    assert "Source label:" not in instr
    assert instr.endswith(INSTRUCTION_NLI)


def test_block_order_is_fixed_v4():
    instr = render_v4(COMPOSITE_FLIP, "full_oracle", original_label="entailment")
    positions = [
        instr.index("Paired source input:"),
        instr.index("Source label:"),
        instr.index("Input transformation:"),
        instr.index("Output relation:"),
        instr.index(INSTRUCTION_NLI),
    ]
    assert positions == sorted(positions)


def test_only_full_oracle_reveals_the_source_label():
    for mode in list(CORE_MODES) + list(CONTROL_MODES):
        assert "Source label:" not in render_v4(COMPOSITE_FLIP, mode, original_label="entailment")
    assert "Source label:" in render_v4(COMPOSITE_FLIP, "full_oracle", original_label="entailment")


@pytest.mark.parametrize("mode", list(CORE_MODES) + list(CONTROL_MODES) + list(DIAGNOSTIC_MODES))
def test_output_is_current_sample_label(mode):
    row = converter.build_instruction_for_sample(
        sample=COMPOSITE_FLIP, original_sample=SOURCE, mode=mode,
        operation_description="x", relation_description="y",
        original_label="entailment", template_version=V4,
    )
    assert row["output"] == "contradiction"
    assert COMPOSITE_FLIP["hypothesis"] in row["input"]


@pytest.mark.parametrize("mode", list(CORE_MODES) + list(CONTROL_MODES) + list(DIAGNOSTIC_MODES))
def test_source_rows_stay_plain_nli_v4(mode):
    assert render_v4(SOURCE, mode, original=SOURCE, original_label="entailment") == INSTRUCTION_NLI


def test_nli_task_block_is_unchanged_from_v3():
    for mode, template in INSTRUCTION_TEMPLATES_V4.items():
        assert template.endswith(INSTRUCTION_NLI), mode


# ============================================================
# Grounding invariance
# ============================================================
@pytest.mark.parametrize("ungrounded, grounded", [
    ("operation_only", "pair_operation"),
    ("operation_relation", "full_specification"),
])
def test_operation_block_is_grounding_invariant(ungrounded, grounded):
    a = render_v4(COMPOSITE_FLIP, ungrounded).split("Input transformation:")[1].split("\n\n")[0]
    b = render_v4(COMPOSITE_FLIP, grounded).split("Input transformation:")[1].split("\n\n")[0]
    assert a == b


def test_relation_block_is_grounding_invariant():
    a = render_v4(COMPOSITE_FLIP, "relation_only").split("Output relation:")[1].split("\n\n")[0]
    b = render_v4(COMPOSITE_FLIP, "pair_relation").split("Output relation:")[1].split("\n\n")[0]
    assert a == b


# ============================================================
# Relation constraint wording
# ============================================================
def test_relation_kinds_are_constraints_not_hedges():
    for text in V4_RELATION_KIND_DESCRIPTIONS.values():
        low = text.lower()
        assert "must" in low
        for hedge in ("likely", "generally", "may ", "tends to", "unlikely"):
            assert hedge not in low, f"hedging {hedge!r} in {text!r}"


def test_relation_text_never_states_the_current_sample_label():
    for text in V4_RELATION_KIND_DESCRIPTIONS.values():
        low = text.lower()
        assert "this sample" not in low
        assert "the correct label" not in low


def test_relation_kind_mapping_matches_the_generator_contract():
    assert relation_kind_v4("synonym_replacement") == "invariance"
    assert relation_kind_v4("composite_inv") == "invariance"
    assert relation_kind_v4("adding_contradiction") == "entailment_to_contradiction"
    assert relation_kind_v4("composite_flip") == "entailment_to_contradiction"
    assert relation_kind_v4("uninformative") == "entailment_to_neutral"
    assert relation_kind_v4("composite_neutral") == "entailment_to_neutral"
    assert relation_kind_v4("tense_shift") is None


def test_ec_and_en_are_conditional_not_totalised():
    """v4 must not invent the C->E / N->N branches the generator never instantiates."""
    ec = V4_RELATION_KIND_DESCRIPTIONS["entailment_to_contradiction"].lower()
    assert "if the source label is entailment" in ec
    assert "contradiction maps to entailment" not in ec
    assert "neutral maps to neutral" not in ec


# ============================================================
# Matched controls
# ============================================================
def test_matched_operation_shuffle_changes_trace_and_text_everywhere():
    rows, _, _ = dataset()
    donors, meta = select_matched_shuffled_operation_donors_for(rows, 1041)
    audit = build_matched_control_audit(meta, donors, None, 1041)
    assert audit["operation_trace_identity_collision_count"] == 0
    assert audit["operation_text_unchanged_count"] == 0
    assert audit["shuffled_operation_no_valid_candidate_count"] == 0
    assert audit["shuffled_operation_same_arity_rate"] == 1.0


def test_matched_operation_shuffle_is_seed_deterministic():
    rows, _, _ = dataset()
    a, _ = select_matched_shuffled_operation_donors_for(rows, 1041)
    b, _ = select_matched_shuffled_operation_donors_for(rows, 1041)
    c, _ = select_matched_shuffled_operation_donors_for(rows, 1042)
    assert a == b
    assert a != c


def test_shuffled_operation_control_preserves_targets_and_pair():
    plain, _ = converted("pair_operation")
    shuffled, report = converted("pair_shuffled_operation")
    assert [r["output"] for r in plain] == [r["output"] for r in shuffled]
    assert [r["input"] for r in plain] == [r["input"] for r in shuffled]
    assert any(a["instruction"] != b["instruction"] for a, b in zip(plain, shuffled))
    assert report["shuffled_operation_identity_collision_count"] == 0


def test_shuffled_relation_control_preserves_targets():
    plain, _ = converted("pair_relation")
    shuffled, _ = converted("pair_shuffled_relation")
    assert [r["output"] for r in plain] == [r["output"] for r in shuffled]
    assert any(a["instruction"] != b["instruction"] for a, b in zip(plain, shuffled))


# ============================================================
# Report metadata
# ============================================================
@pytest.mark.parametrize("mode", CORE_MODES)
def test_v4_report_records_trace_metadata(mode):
    _, report = converted(mode)
    spec = mode_spec(mode)
    assert report["template_version"] == V4
    assert report["pair_present"] is bool(spec["pair"])
    assert report["operation_present"] is bool(spec["operation"])
    assert report["relation_present"] is bool(spec["relation"])
    assert report["label_anchor_present"] is False
    for field in (
        "operation_trace_count",
        "unique_operation_trace_count",
        "operation_arity_distribution",
        "relation_kind_distribution_v4",
        "relation_shortcut_risk_count",
        "composite_operation_fallback_count",
        "operation_unsupported_provenance_count",
    ):
        assert field in report, field


def test_v4_trace_distribution_counts_only_operation_modes():
    _, report = converted("pair_operation")
    assert report["operation_trace_count"] == 5  # 5 augmented rows in the fixture
    assert report["operation_arity_distribution"] == {1: 3, 2: 2}


def test_v4_relation_kind_distribution_and_shortcut_risk():
    _, report = converted("relation_only")
    kinds = report["relation_kind_distribution_v4"]
    assert kinds.get("invariance") == 1
    assert kinds.get("entailment_to_contradiction") == 2
    assert kinds.get("entailment_to_neutral") == 2
    assert report["relation_shortcut_risk_count"] == 4


def test_v4_design_meta_is_unchanged_matrix():
    for mode in CORE_MODES:
        assert mode_design_meta(mode)["core_factorial"] is True
    assert mode_design_meta("full_oracle")["label_anchor"] is True


# ============================================================
# v3 / v2 regression
# ============================================================
def test_v4_does_not_change_v3_output():
    """Same rows, version 3 vs 4: only the added block wording differs."""
    rows, _, original_map = dataset()
    v3, _ = convert_to_alpaca(rows, mode="full_specification",
                              original_map=original_map, template_version=3)
    v4, _ = convert_to_alpaca(rows, mode="full_specification",
                              original_map=original_map, template_version=V4)
    assert [r["output"] for r in v3] == [r["output"] for r in v4]
    # source rows are plain NLI in both versions; only augmented rows carry blocks
    v3_aug = [r for r, s in zip(v3, rows) if s["mr_id"] != "none"]
    v4_aug = [r for r, s in zip(v4, rows) if s["mr_id"] != "none"]
    assert all("Paired source sample:" in r["instruction"] for r in v3_aug)
    assert all("Paired source input:" in r["instruction"] for r in v4_aug)
    assert not any("Paired source" in r["instruction"] for r in v3 if r["instruction"] == INSTRUCTION_NLI)


def test_v2_path_still_renders_legacy_text():
    rows, _, original_map = dataset()
    v2, _ = convert_to_alpaca(rows, mode="pair_operation",
                              original_map=original_map, template_version=2)
    v2_aug = [r for r, s in zip(v2, rows) if s["mr_id"] != "none"]
    assert all("Reference sample:" in r["instruction"] for r in v2_aug)
