"""Unit tests for the RQ2 grounding-aware 2x2x2 MR-information design.

Covers, for the eight core conditions plus controls and the diagnostic mode:
  * which blocks are present (P/O/R/L) and in which fixed order,
  * that the only thing varying across modes is block presence,
  * that no core cell ever leaks the source/reference label,
  * that Operation/Relation wording is identical in grounded and ungrounded
    cells,
  * source rows stay plain NLI in every mode,
  * shuffled controls differ from the correct ones and are seed-deterministic,
  * the conversion report flags.

No GPU, no model, no tokenizer.

NOTE: these assertions use a purpose-built fixture.  The *real* cohort is
dominated by composite MRs whose component provenance is absent, which is why
the Operation text in this fixture is a plain single-transformation sentence.
"""
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import convert_nli_to_ft as converter  # noqa: E402
from convert_nli_to_ft import (  # noqa: E402
    INSTRUCTION_NLI,
    build_grounding_control_audit,
    build_instruction_for_sample,
    build_original_map,
    build_pair_groups,
    convert_to_alpaca,
    select_shuffled_operation_mr_ids,
    select_shuffled_relation_types,
    sort_samples_by_stable_key,
)
from mr_instruction_design import (  # noqa: E402
    COMPOSITE_OPERATION_FALLBACK,
    CONTROL_MODES,
    CORE_MODES,
    DIAGNOSTIC_MODES,
    MR_OPERATION_DESCRIPTIONS,
    MR_RELATION_DESCRIPTIONS,
    RELATION_TYPE_DESCRIPTIONS,
    grounding_status,
    mode_design_meta,
    mode_spec,
    resolve_operation_description,
)

# ============================================================
# Fixture: one pair with a label-preserving and a label-changing follow-up
# ============================================================
SOURCE = {
    "premise": "A cat sat on the mat.",
    "hypothesis": "A cat is on a mat.",
    "label": 0,
    "pair_id": 7,
    "mr_id": "none",
    "mr_type": "inv",
}
# label-preserving (inv)
INV_FOLLOWUP = {
    "premise": "A cat sat on the mat.",
    "hypothesis": "A feline rested on the rug.",
    "label": 0,
    "pair_id": 7,
    "mr_id": "synonym_replacement",
    "mr_type": "inv",
}
# label-changing (flip, E -> C)
FLIP_FOLLOWUP = {
    "premise": "A cat sat on the mat.",
    "hypothesis": "No cat is anywhere near a mat.",
    "label": 2,
    "pair_id": 7,
    "mr_id": "adding_contradiction",
    "mr_type": "flip",
}


def render(sample, mode, original=SOURCE, original_label="", op=None, rel=None):
    mr_id = sample["mr_id"]
    if op is None:
        op, _ = resolve_operation_description(mr_id, sample.get("component_mrs"))
    if rel is None:
        rel = MR_RELATION_DESCRIPTIONS.get(mr_id, "")
    return build_instruction_for_sample(
        sample=sample,
        original_sample=original,
        mode=mode,
        operation_description=op,
        relation_description=rel,
        original_label=original_label,
        is_original=(mr_id == "none"),
    )["instruction"]


# ============================================================
# Mode registry
# ============================================================
def test_eight_core_cells_match_the_declared_2x2x2():
    expected = {
        "none": (0, 0, 0),
        "operation_only": (0, 1, 0),
        "relation_only": (0, 0, 1),
        "operation_relation": (0, 1, 1),
        "pair_only": (1, 0, 0),
        "pair_operation": (1, 1, 0),
        "pair_relation": (1, 0, 1),
        "full_specification": (1, 1, 1),
    }
    assert set(CORE_MODES) == set(expected)
    for mode, (pair, operation, relation) in expected.items():
        spec = mode_spec(mode)
        assert (int(spec["pair"]), int(spec["operation"]), int(spec["relation"])) == (
            pair, operation, relation
        ), mode
        assert not spec["label_anchor"], f"{mode} must not anchor the source label"


def test_only_full_oracle_sets_label_anchor():
    anchored = [m for m, s in converter.MODE_SPECS.items() if s["label_anchor"]]
    assert anchored == ["full_oracle"]
    assert "full_oracle" in DIAGNOSTIC_MODES
    assert "full_oracle" not in CORE_MODES


def test_grounding_status_follows_pair_presence():
    for mode, spec in converter.MODE_SPECS.items():
        expected = "source_grounded" if spec["pair"] else "ungrounded"
        assert grounding_status(spec) == expected


def test_deprecated_aliases_resolve_to_canonical_modes():
    assert converter.resolve_mode_name("relation_aware") == "full_specification"
    assert converter.resolve_mode_name("shuffled_operation") == "pair_shuffled_operation"
    assert converter.resolve_mode_name("pair_operation") == "pair_operation"


# ============================================================
# Per-mode block presence
# ============================================================
@pytest.mark.parametrize("mode", CORE_MODES)
def test_core_mode_block_presence(mode):
    spec = mode_spec(mode)
    instr = render(FLIP_FOLLOWUP, mode)

    assert ("Paired source sample:" in instr) == bool(spec["pair"])
    assert ("Input-transformation specification:" in instr) == bool(spec["operation"])
    assert ("Output-relation specification:" in instr) == bool(spec["relation"])
    assert ("Source label:" in instr) is False

    # the pair block renders the source text iff P=1
    if spec["pair"]:
        assert SOURCE["premise"] in instr and SOURCE["hypothesis"] in instr
    else:
        assert SOURCE["premise"] not in instr and SOURCE["hypothesis"] not in instr

    # every mode ends with the identical NLI task block
    assert instr.endswith(INSTRUCTION_NLI)


def test_full_oracle_shows_source_label_and_all_blocks():
    instr = render(FLIP_FOLLOWUP, "full_oracle", original_label="entailment")
    assert "Paired source sample:" in instr
    assert "Source label:\nentailment" in instr
    assert "Input-transformation specification:" in instr
    assert "Output-relation specification:" in instr


def test_block_order_is_fixed():
    """PAIR -> LABEL_ANCHOR -> OPERATION -> RELATION -> NLI_TASK."""
    instr = render(FLIP_FOLLOWUP, "full_oracle", original_label="entailment")
    positions = [
        instr.index("Paired source sample:"),
        instr.index("Source label:"),
        instr.index("Input-transformation specification:"),
        instr.index("Output-relation specification:"),
        instr.index(INSTRUCTION_NLI),
    ]
    assert positions == sorted(positions)


def test_operation_and_relation_wording_is_grounding_invariant():
    """P=0 and P=1 must render byte-identical O/R payloads."""
    pairs = [
        ("operation_only", "pair_operation", "Input-transformation specification:"),
        ("relation_only", "pair_relation", "Output-relation specification:"),
        ("operation_relation", "full_specification", "Input-transformation specification:"),
    ]
    for ungrounded, grounded, header in pairs:
        a = render(FLIP_FOLLOWUP, ungrounded).split(header)[1].split("\n\n")[0]
        b = render(FLIP_FOLLOWUP, grounded).split(header)[1].split("\n\n")[0]
        assert a == b, (ungrounded, grounded)


def test_only_presence_differs_between_none_and_full_specification():
    """Stripping the added blocks from full_specification yields `none`."""
    full = render(FLIP_FOLLOWUP, "full_specification")
    stripped = full
    for header in (
        "Paired source sample:",
        "Input-transformation specification:",
        "Output-relation specification:",
    ):
        start = stripped.index(header)
        end = stripped.index("\n\n", start) + 2
        stripped = stripped[:start] + stripped[end:]
    assert stripped.strip() == render(FLIP_FOLLOWUP, "none").strip()


# ============================================================
# Leakage
# ============================================================
@pytest.mark.parametrize("mode", list(CORE_MODES) + list(CONTROL_MODES))
def test_no_source_label_leak_in_core_and_control_modes(mode):
    instr = render(FLIP_FOLLOWUP, mode, original_label="entailment")
    assert "Source label:" not in instr
    assert "Reference label" not in instr


@pytest.mark.parametrize("mode", list(CORE_MODES) + list(CONTROL_MODES) + list(DIAGNOSTIC_MODES))
def test_output_is_always_the_current_sample_label(mode):
    row = build_instruction_for_sample(
        sample=FLIP_FOLLOWUP,
        original_sample=SOURCE,
        mode=mode,
        operation_description=MR_OPERATION_DESCRIPTIONS["adding_contradiction"],
        relation_description=MR_RELATION_DESCRIPTIONS["adding_contradiction"],
        original_label="entailment",
    )
    assert row["output"] == "contradiction"
    assert FLIP_FOLLOWUP["premise"] in row["input"]


def test_relation_text_never_names_the_current_target_label():
    """Relation specs state a mapping, never "the label is X for this sample"."""
    for text in RELATION_TYPE_DESCRIPTIONS.values():
        assert "this sample" not in text.lower()
        assert "the correct label" not in text.lower()
        assert "is likely" not in text.lower()
        assert "tends to" not in text.lower()


def test_operation_text_has_no_output_relation_hint():
    forbidden = (
        "label", "relation is preserved", "prediction", "entailment",
        "contradiction", "neutral", "output",
    )
    for mr_id, text in MR_OPERATION_DESCRIPTIONS.items():
        if not text:
            continue
        low = text.lower()
        for token in forbidden:
            assert token not in low, f"{mr_id}: operation text leaks {token!r}"


# ============================================================
# Operation / relation resolvers
# ============================================================
def test_composite_operation_uses_component_provenance_when_available():
    text, provenance = resolve_operation_description(
        "composite_flip", ["pronoun_substitution", "negation_flip"]
    )
    assert provenance == "composite_components"
    assert "1." in text and "2." in text
    assert "coreferential" in text and "negation marker" in text
    # must not mention the composite's own relation class
    assert "contradiction" not in text.lower()


def test_composite_operation_falls_back_to_one_generic_text():
    texts = set()
    for mr_id in ("composite_inv", "composite_flip", "composite_neutral"):
        text, provenance = resolve_operation_description(mr_id, None)
        assert provenance == "composite_fallback"
        texts.add(text)
    assert texts == {COMPOSITE_OPERATION_FALLBACK}, (
        "the three composites must share one fallback wording so that the "
        "operation text cannot reveal the relation class"
    )


def test_relation_description_is_shared_within_a_relation_type():
    assert (
        MR_RELATION_DESCRIPTIONS["synonym_replacement"]
        == MR_RELATION_DESCRIPTIONS["composite_inv"]
    )
    assert (
        MR_RELATION_DESCRIPTIONS["adding_contradiction"]
        == MR_RELATION_DESCRIPTIONS["composite_flip"]
    )
    assert (
        MR_RELATION_DESCRIPTIONS["uninformative"]
        == MR_RELATION_DESCRIPTIONS["composite_neutral"]
    )


# ============================================================
# Source rows
# ============================================================
@pytest.mark.parametrize("mode", list(CORE_MODES) + list(CONTROL_MODES) + list(DIAGNOSTIC_MODES))
def test_source_rows_stay_plain_nli_in_every_mode(mode):
    instr = render(SOURCE, mode, original=SOURCE, original_label="entailment")
    assert instr == INSTRUCTION_NLI
    assert "Paired source sample:" not in instr
    assert "Input-transformation specification:" not in instr
    assert "Output-relation specification:" not in instr


# ============================================================
# Shuffled controls
# ============================================================
MR_CYCLE = [
    "synonym_replacement",       # inv
    "adding_contradiction",      # flip
    "uninformative",             # neutral
    "composite_inv",             # inv
    "composite_flip",            # flip
    "voice_switch",              # inv
]


def make_rows(group_count=12):
    """``group_count`` groups, each = one source row + one augmented follow-up."""
    rows = []
    for i in range(group_count):
        rows.append({
            "premise": f"A cat sat on the mat. ({i})",
            "hypothesis": f"A cat is on a mat. ({i})",
            "label": 0,
            "pair_id": 100 + i,
            "mr_id": "none",
            "mr_type": "inv",
        })
        mr_id = MR_CYCLE[i % len(MR_CYCLE)]
        rows.append({
            "premise": f"A cat sat on the mat. ({i})",
            "hypothesis": f"Modified hypothesis {i}",
            "label": 2 if mr_id.endswith("flip") or mr_id == "adding_contradiction" else i % 3,
            "pair_id": 100 + i,
            "mr_id": mr_id,
            "mr_type": "inv",
        })
    return rows


def tagged(rows):
    """Group ``rows`` and tag every row with its ``_group_key``."""
    groups = build_pair_groups(rows)
    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]
    return groups


def dataset(rows=None):
    """Return ``(rows, groups, original_map)`` for a well-formed fixture."""
    rows = make_rows() if rows is None else rows
    rows = sort_samples_by_stable_key(rows)
    groups = tagged(rows)
    original_map, missing = build_original_map(groups)
    assert not missing, missing
    return rows, groups, original_map


def test_shuffled_operation_is_a_derangement_and_seed_deterministic():
    rows = make_rows()
    a = select_shuffled_operation_mr_ids(rows, 1041)
    b = select_shuffled_operation_mr_ids(rows, 1041)
    c = select_shuffled_operation_mr_ids(rows, 1042)
    assert a == b, "must be deterministic for a fixed seed"
    assert a != c, "different seeds should produce different assignments"

    for row, assigned in zip(rows, a):
        if row["mr_id"] == "none":
            assert assigned is None
        else:
            assert assigned is not None
            assert assigned != row["mr_id"], "identity collision"


def test_shuffled_relation_is_a_derangement_and_seed_deterministic():
    rows = make_rows()
    a = select_shuffled_relation_types(rows, 1041)
    b = select_shuffled_relation_types(rows, 1041)
    assert a == b
    for row, assigned in zip(rows, a):
        if row["mr_id"] == "none":
            assert assigned is None
        else:
            assert assigned is not None
            assert assigned != converter.relation_type_for(row["mr_id"])


@pytest.mark.parametrize(
    "correct_mode, shuffled_mode, payload_key",
    [
        ("pair_operation", "pair_shuffled_operation", "shuffled_descriptions"),
        ("pair_relation", "pair_shuffled_relation", "shuffled_relation_types"),
    ],
)
def test_shuffled_controls_change_specs_but_not_targets(
    correct_mode, shuffled_mode, payload_key
):
    rows, _, original_map = dataset()
    ops = select_shuffled_operation_mr_ids(rows, 1041)
    rels = select_shuffled_relation_types(rows, 1041)
    payload = {"shuffled_descriptions": ops, "shuffled_relation_types": rels}
    kwargs = dict(
        original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
        relation_effects=MR_RELATION_DESCRIPTIONS,
    )

    plain, _ = convert_to_alpaca(rows, mode=correct_mode, **kwargs)
    shuffled, _ = convert_to_alpaca(
        rows, mode=shuffled_mode, **{**kwargs, payload_key: payload[payload_key]}
    )
    # identical training targets and identical current samples
    assert [r["output"] for r in plain] == [r["output"] for r in shuffled]
    assert [r["input"] for r in plain] == [r["input"] for r in shuffled]
    # identical pair block, different specification block
    assert any(
        a["instruction"] != b["instruction"] for a, b in zip(plain, shuffled)
    ), "the shuffled control must actually change the rendered specification"


def test_shuffled_payload_is_ignored_by_correct_modes():
    """A mode whose spec says `correct` must not honour a shuffled payload."""
    rows, _, original_map = dataset()
    ops = select_shuffled_operation_mr_ids(rows, 1041)
    plain, _ = convert_to_alpaca(
        rows, mode="pair_operation", original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
    )
    with_payload, _ = convert_to_alpaca(
        rows, mode="pair_operation", original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
        shuffled_descriptions=ops,
    )
    assert [r["instruction"] for r in plain] == [r["instruction"] for r in with_payload]


def test_shuffled_control_audit_reports_collisions():
    rows = make_rows()
    ops = select_shuffled_operation_mr_ids(rows, 1041)
    rels = select_shuffled_relation_types(rows, 1041)
    audit = build_grounding_control_audit(rows, ops, rels, 1041)
    assert audit["operation_identity_collision_count"] == 0
    assert audit["relation_identity_collision_count"] == 0
    assert audit["relation_text_unchanged_count"] == 0


# ============================================================
# convert_to_alpaca: report + strict mode
# ============================================================
@pytest.mark.parametrize("mode", CORE_MODES)
def test_report_records_design_metadata(mode):
    rows, _, original_map = dataset()
    _, report = convert_to_alpaca(
        rows, mode=mode, original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
        relation_effects=MR_RELATION_DESCRIPTIONS,
    )
    spec = mode_spec(mode)
    assert report["canonical_mode"] == mode
    assert report["pair_present"] is bool(spec["pair"])
    assert report["operation_present"] is bool(spec["operation"])
    assert report["relation_present"] is bool(spec["relation"])
    assert report["label_anchor_present"] is False
    assert report["grounding_status"] == ("source_grounded" if spec["pair"] else "ungrounded")
    for field in (
        "missing_operation_count",
        "missing_relation_count",
        "pair_fallback_count",
        "composite_operation_fallback_count",
        "shuffled_operation_identity_collision_count",
        "shuffled_relation_identity_collision_count",
    ):
        assert field in report


def test_composite_operation_fallback_is_counted():
    rows, _, original_map = dataset()
    _, report = convert_to_alpaca(
        rows, mode="pair_operation", original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
    )
    composites = sum(
        1 for r in rows
        if r["mr_id"] in ("composite_inv", "composite_flip", "composite_neutral")
    )
    assert report["composite_operation_fallback_count"] == composites > 0
    assert report["operation_provenance_distribution"]["composite_fallback"] == composites


def test_strict_mode_fails_when_pair_source_is_missing():
    rows, _, _ = dataset()
    with pytest.raises(ValueError, match="缺少 source sample"):
        convert_to_alpaca(
            rows, mode="pair_operation", original_map={},
            operation_descriptions=MR_OPERATION_DESCRIPTIONS, strict=True,
        )


def test_strict_mode_fails_when_required_operation_is_missing():
    """A row whose mr_id has no Operation definition must not slip through."""
    rows, _, _ = dataset()
    rows = [dict(r) for r in rows]
    rows[-1]["mr_id"] = "mystery_mr"
    with pytest.raises(ValueError, match="缺少操作描述"):
        convert_to_alpaca(rows, mode="operation_only", strict=True)


def test_strict_mode_fails_when_required_relation_is_missing():
    rows, _, _ = dataset()
    rows = [dict(r) for r in rows]
    rows[-1]["mr_id"] = "mystery_mr"
    with pytest.raises(ValueError, match="缺少 relation 描述"):
        convert_to_alpaca(rows, mode="relation_only", strict=True)


def test_relation_type_mismatch_is_counted():
    rows, _, _ = dataset()
    for row in rows:
        if row["mr_id"] == "adding_contradiction":
            row["mr_type"] = "inv"  # wrong on purpose
    _, report = convert_to_alpaca(rows, mode="none")
    assert report["relation_type_mismatch_count"] >= 1


# ============================================================
# Legacy aliases
# ============================================================
def test_relation_aware_alias_equals_full_specification():
    rows, _, original_map = dataset()
    kwargs = dict(
        original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
        relation_effects=MR_RELATION_DESCRIPTIONS,
    )
    a, ra = convert_to_alpaca(rows, mode="relation_aware", **kwargs)
    b, rb = convert_to_alpaca(rows, mode="full_specification", **kwargs)
    assert [r["instruction"] for r in a] == [r["instruction"] for r in b]
    assert ra["canonical_mode"] == rb["canonical_mode"] == "full_specification"


def test_shuffled_operation_alias_equals_pair_shuffled_operation():
    rows, _, original_map = dataset()
    ops = select_shuffled_operation_mr_ids(rows, 1041)
    kwargs = dict(
        original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
        shuffled_descriptions=ops,
    )
    a, ra = convert_to_alpaca(rows, mode="shuffled_operation", **kwargs)
    b, rb = convert_to_alpaca(rows, mode="pair_shuffled_operation", **kwargs)
    assert [r["instruction"] for r in a] == [r["instruction"] for r in b]
    assert ra["canonical_mode"] == rb["canonical_mode"] == "pair_shuffled_operation"


def test_mode_design_meta_matches_registry():
    meta = mode_design_meta("pair_relation")
    assert meta == {
        "canonical_mode": "pair_relation",
        "requested_mode": "pair_relation",
        "pair": True,
        "operation": False,
        "relation": True,
        "label_anchor": False,
        "grounding": "source_grounded",
        "pair_source": "correct",
        "operation_source": "none",
        "relation_source": "correct",
        "role": "core",
        "core_factorial": True,
    }


def test_mismatched_pair_keeps_shape_but_changes_source():
    rows, groups, original_map = dataset()
    mismatched = converter.build_mismatched_pair_map(groups)

    plain, _ = convert_to_alpaca(rows, mode="pair_only", original_map=original_map)
    other, _ = convert_to_alpaca(
        rows, mode="mismatched_pair", original_map=original_map,
        mismatched_pair_map=mismatched,
    )

    assert len(plain) == len(other)
    assert [r["input"] for r in plain] == [r["input"] for r in other]
    assert any(
        a["instruction"] != b["instruction"] for a, b in zip(plain, other)
    ), "mismatched control must actually change the pair"
