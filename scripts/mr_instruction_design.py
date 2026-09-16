#!/usr/bin/env python3
"""RQ2 MR-information design: instruction blocks, mode registry, resolvers.

This module is the single source of truth for the *grounding-aware 2x2x2*
MR-information design used by the RQ2 experiments.  It is deliberately
dependency-free (standard library only) so that the converter, the RQ2 driver
and the inspection tool can all import it without pulling in heavy deps.

Design summary
--------------
Three information dimensions are manipulated independently:

* ``P`` -- Pair / source-follow-up grounding.  When ``P=1`` the paired source
  sample (premise + hypothesis, **never** its label) is rendered.
* ``O`` -- Operation / input-transformation specification.
* ``R`` -- Relation / output-relation specification.

``P`` is a *grounding* factor, not a symmetric peer of ``O``/``R``:

* ``P=0`` renders ``O``/``R`` as ungrounded **metadata / provenance**;
* ``P=1`` renders ``O``/``R`` as specifications **bound to the shown
  source-follow-up pair**.

``L`` (source-label anchoring) is a fourth, *diagnostic-only* dimension.  It is
never part of the core 2x2x2; only ``full_oracle`` sets ``L=1``.

Block order is fixed for every mode::

    [PAIR_BLOCK, if P=1]
    [LABEL_ANCHOR_BLOCK, only if L=1]
    [OPERATION_BLOCK, if O=1]
    [RELATION_BLOCK, if R=1]
    [NLI_TASK_BLOCK]

Only the *presence* of a block may vary across modes -- never its wording, its
position, or the NLI task text.
"""

from typing import Dict, List, Optional, Sequence, Tuple

# ============================================================
# Instruction blocks (canonical wording)
# ============================================================
#: Rendered only when P=1.  Provides ``x_source`` and nothing else: no source
#: label, no operation, no relation, no current target label.
PAIR_BLOCK = (
    "Paired source sample:\n"
    "<source_premise>\n"
    "{original_premise}\n"
    "</source_premise>\n"
    "<source_hypothesis>\n"
    "{original_hypothesis}\n"
    "</source_hypothesis>"
)

#: Diagnostic only (L=1).  Must never appear in a core 2x2x2 cell.
LABEL_ANCHOR_BLOCK = "Source label:\n{original_label}"

#: Rendered whenever O=1 -- identical wording in grounded and ungrounded cells.
OPERATION_BLOCK = "Input-transformation specification:\n{operation_description}"

#: Rendered whenever R=1 -- identical wording in grounded and ungrounded cells.
RELATION_BLOCK = "Output-relation specification:\n{relation_description}"

#: Fixed NLI task block.  Shared verbatim by every mode.
NLI_TASK_BLOCK = (
    "Determine the natural language inference relation between the premise and hypothesis. "
    "Answer with exactly one label: entailment, neutral, or contradiction."
)
NLI_TASK_BLOCK_BINARY = (
    "Determine whether the premise entails the hypothesis. "
    "Answer with exactly one label: entailment or not_entailment."
)

BLOCK_SEPARATOR = "\n\n"

#: Schema version of the block-composition design.  v2 is the frozen legacy
#: per-mode template set retained in ``scripts/convert_nli_to_ft.py``.
INSTRUCTION_DESIGN_VERSION = 3
LEGACY_INSTRUCTION_DESIGN_VERSIONS = (2,)


# ============================================================
# Operation specifications (input-side, declaration not command)
# ============================================================
# Every description is phrased as provenance ("was derived ... by ..."), never
# as an imperative, and never states the expected output relation.
#
# ``source_of_truth`` records where the transformation definition comes from.
# See ``RQ2/MR_RELATION_AUDIT.md`` for the full audit table.
MR_OPERATION_EDITS: Dict[str, str] = {
    "adding_contradiction":
        "inserting additional content that conflicts with the surrounding text",
    "antonym_substitution":
        "replacing one or more content words with context-appropriate antonyms",
    "conditional_clause":
        "adding a conditional clause to one component",
    "negation_flip":
        "adding, removing, or reversing a negation marker",
    "pronoun_substitution":
        "replacing a noun phrase or pronoun with a coreferential form",
    "synonym_replacement":
        "replacing one or more words with context-appropriate synonyms",
    "uninformative":
        "inserting additional content that is not informative about the rest of the text",
    "voice_switch":
        "rewriting a sentence between active and passive voice",
    # Unsupported ids: present only as a defensive whitelist.  They appear in no
    # active dataset and have no generator/oracle definition in either repo.
    "race_sensitive_transformation":
        "replacing or modifying demographic or identity-related terms in the text",
    "same_type_named_entity_substitution":
        "replacing named entities with others of the same category",
    "tense_shift":
        "shifting the tense of one or more verbs",
}

_OPERATION_PREFIX = "The current sample was derived from its source sample by "

#: Applied to composite rows whose component transformations are not recorded.
COMPOSITE_OPERATION_FALLBACK = (
    _OPERATION_PREFIX + "applying multiple input-side transformations."
)
COMPOSITE_OPERATION_HEADER = (
    _OPERATION_PREFIX + "applying multiple input-side transformations:"
)

COMPOSITE_MR_IDS = ("composite_inv", "composite_flip", "composite_neutral")

#: mr_id -> canonical ``Input-transformation specification`` text.
MR_OPERATION_DESCRIPTIONS: Dict[str, str] = {
    mr_id: f"{_OPERATION_PREFIX}{edit}."
    for mr_id, edit in MR_OPERATION_EDITS.items()
}
for _composite_id in COMPOSITE_MR_IDS:
    MR_OPERATION_DESCRIPTIONS[_composite_id] = COMPOSITE_OPERATION_FALLBACK
MR_OPERATION_DESCRIPTIONS["none"] = ""

#: Transformation definitions and their provenance.  ``supported`` is False for
#: ids that no generator/oracle in either repository defines.
MR_OPERATION_SPECS: Dict[str, Dict[str, object]] = {
    "adding_contradiction": {"family": "flip", "supported": True},
    "antonym_substitution": {"family": "flip", "supported": True},
    "conditional_clause": {"family": "neutral", "supported": True},
    "negation_flip": {"family": "flip", "supported": True},
    "pronoun_substitution": {"family": "inv", "supported": True},
    "synonym_replacement": {"family": "inv", "supported": True},
    "uninformative": {"family": "neutral", "supported": True},
    "voice_switch": {"family": "inv", "supported": True},
    "composite_inv": {"family": "inv", "supported": True},
    "composite_flip": {"family": "flip", "supported": True},
    "composite_neutral": {"family": "neutral", "supported": True},
    "race_sensitive_transformation": {"family": None, "supported": False},
    "same_type_named_entity_substitution": {"family": None, "supported": False},
    "tense_shift": {"family": None, "supported": False},
    "none": {"family": None, "supported": True},
}

# provenance tags returned by :func:`resolve_operation_description`
OPERATION_PROVENANCE_EXACT = "exact"
OPERATION_PROVENANCE_COMPONENTS = "composite_components"
OPERATION_PROVENANCE_FALLBACK = "composite_fallback"
OPERATION_PROVENANCE_NONE = "none"


def component_operation_sentence(mr_id: str) -> str:
    """Return the capitalised clause describing one component transformation."""
    edit = MR_OPERATION_EDITS[mr_id]
    return edit[0].upper() + edit[1:] + "."


def resolve_operation_description(
    mr_id: str,
    component_mrs: Optional[Sequence[str]] = None,
) -> Tuple[str, str]:
    """Return ``(operation_description, provenance)`` for ``mr_id``.

    Composite rows use their recorded component transformations when the
    dataset carries them; otherwise they fall back to a single generic
    description.  The fallback is deliberately identical for all three
    composites so that the *wording* cannot reveal the relation class.
    """
    if not mr_id or mr_id == "none":
        return "", OPERATION_PROVENANCE_NONE

    if mr_id in COMPOSITE_MR_IDS:
        components = [str(c).strip().lower().replace("-", "_") for c in (component_mrs or [])]
        components = [c for c in components if c in MR_OPERATION_EDITS]
        if components:
            lines = [f"{i}. {component_operation_sentence(c)}" for i, c in enumerate(components, 1)]
            return COMPOSITE_OPERATION_HEADER + "\n" + "\n".join(lines), OPERATION_PROVENANCE_COMPONENTS
        return COMPOSITE_OPERATION_FALLBACK, OPERATION_PROVENANCE_FALLBACK

    description = MR_OPERATION_DESCRIPTIONS.get(mr_id)
    if not description:
        return "", OPERATION_PROVENANCE_NONE
    return description, OPERATION_PROVENANCE_EXACT


# ============================================================
# Relation specifications (source -> follow-up mapping)
# ============================================================
# Source of truth: MTrain ``augment_snli_all_label_mrs.py::OPERATION_SPECS``
# (``mr_type`` + ``target``) and the MetTrain paper's R_inv / R_E->C / R_E->N
# partition; the flip map is operationalised in
# ``scripts/metamorphic_metrics.py::RELATION_TYPES``.
#
# NOTE: these texts never name the *current* sample's label.  They state the
# abstract source->follow-up mapping only.
RELATION_TYPES: Tuple[str, ...] = ("inv", "flip", "neutral")

RELATION_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "inv": (
        "For valid source-follow-up pairs generated by this metamorphic relation, "
        "the NLI label is preserved: the follow-up label is the same as the source label."
    ),
    "flip": (
        "For valid source-follow-up pairs generated by this metamorphic relation, "
        "the follow-up label is determined from the source label by a fixed label mapping: "
        "entailment maps to contradiction, contradiction maps to entailment, and neutral maps to neutral."
    ),
    "neutral": (
        "For valid source-follow-up pairs generated by this metamorphic relation, "
        "the follow-up label is determined from the source label by a fixed label mapping: "
        "the follow-up label is neutral."
    ),
}

#: mr_id -> official relation type.
MR_RELATION_SPECS: Dict[str, Optional[str]] = {
    "adding_contradiction": "flip",
    "antonym_substitution": "flip",
    "conditional_clause": "neutral",
    "negation_flip": "flip",
    "pronoun_substitution": "inv",
    "synonym_replacement": "inv",
    "uninformative": "neutral",
    "voice_switch": "inv",
    "composite_inv": "inv",
    "composite_flip": "flip",
    "composite_neutral": "neutral",
    "race_sensitive_transformation": None,
    "same_type_named_entity_substitution": None,
    "tense_shift": None,
    "none": None,
}

#: Relation types whose text *directly names* the follow-up label, and therefore
#: short-circuits the task for the corresponding MRs.  Recorded here so the
#: analysis can treat ungrounded ``R`` as a metadata diagnostic for these ids.
SHORTCUT_RELATION_TYPES = ("flip", "neutral")

#: mr_id -> canonical ``Output-relation specification`` text.
MR_RELATION_DESCRIPTIONS: Dict[str, str] = {
    mr_id: (RELATION_TYPE_DESCRIPTIONS[relation_type] if relation_type else "")
    for mr_id, relation_type in MR_RELATION_SPECS.items()
}

# Backwards-compatible alias: the old name of the (now declaration-based) dict.
MR_RELATION_EFFECTS = MR_RELATION_DESCRIPTIONS


def relation_type_for(mr_id: str) -> Optional[str]:
    return MR_RELATION_SPECS.get(mr_id)


def resolve_relation_description(mr_id: str) -> str:
    """Return the ``Output-relation specification`` text for ``mr_id``."""
    return MR_RELATION_DESCRIPTIONS.get(mr_id, "")


def relation_type_is_shortcut(relation_type: Optional[str]) -> bool:
    return relation_type in SHORTCUT_RELATION_TYPES


# ============================================================
# Mode registry: the experimental design as data
# ============================================================
# role:        "core" | "control" | "diagnostic"
# pair:        render PAIR_BLOCK
# label_anchor: render LABEL_ANCHOR_BLOCK (L=1)
# operation:   render OPERATION_BLOCK
# relation:    render RELATION_BLOCK
# *_source:    "correct" | "shuffled" | "mismatched" | "none"
MODE_SPECS: Dict[str, Dict[str, object]] = {
    # ---- core 2x2x2 -------------------------------------------------------
    "none": {
        "pair": False, "label_anchor": False, "operation": False, "relation": False,
        "pair_source": "none", "operation_source": "none", "relation_source": "none",
        "role": "core",
    },
    "operation_only": {
        "pair": False, "label_anchor": False, "operation": True, "relation": False,
        "pair_source": "none", "operation_source": "correct", "relation_source": "none",
        "role": "core",
    },
    "relation_only": {
        "pair": False, "label_anchor": False, "operation": False, "relation": True,
        "pair_source": "none", "operation_source": "none", "relation_source": "correct",
        "role": "core",
    },
    "operation_relation": {
        "pair": False, "label_anchor": False, "operation": True, "relation": True,
        "pair_source": "none", "operation_source": "correct", "relation_source": "correct",
        "role": "core",
    },
    "pair_only": {
        "pair": True, "label_anchor": False, "operation": False, "relation": False,
        "pair_source": "correct", "operation_source": "none", "relation_source": "none",
        "role": "core",
    },
    "pair_operation": {
        "pair": True, "label_anchor": False, "operation": True, "relation": False,
        "pair_source": "correct", "operation_source": "correct", "relation_source": "none",
        "role": "core",
    },
    "pair_relation": {
        "pair": True, "label_anchor": False, "operation": False, "relation": True,
        "pair_source": "correct", "operation_source": "none", "relation_source": "correct",
        "role": "core",
    },
    "full_specification": {
        "pair": True, "label_anchor": False, "operation": True, "relation": True,
        "pair_source": "correct", "operation_source": "correct", "relation_source": "correct",
        "role": "core",
    },
    # ---- semantic / grounding controls ------------------------------------
    "pair_shuffled_operation": {
        "pair": True, "label_anchor": False, "operation": True, "relation": False,
        "pair_source": "correct", "operation_source": "shuffled", "relation_source": "none",
        "role": "control",
    },
    "pair_shuffled_relation": {
        "pair": True, "label_anchor": False, "operation": False, "relation": True,
        "pair_source": "correct", "operation_source": "none", "relation_source": "shuffled",
        "role": "control",
    },
    "mismatched_pair": {
        "pair": True, "label_anchor": False, "operation": False, "relation": False,
        "pair_source": "mismatched", "operation_source": "none", "relation_source": "none",
        "role": "control",
    },
    # ---- diagnostic (L=1) -------------------------------------------------
    "full_oracle": {
        "pair": True, "label_anchor": True, "operation": True, "relation": True,
        "pair_source": "correct", "operation_source": "correct", "relation_source": "correct",
        "role": "diagnostic",
    },
}

CORE_MODES: Tuple[str, ...] = tuple(
    mode for mode, spec in MODE_SPECS.items() if spec["role"] == "core"
)
CONTROL_MODES: Tuple[str, ...] = tuple(
    mode for mode, spec in MODE_SPECS.items() if spec["role"] == "control"
)
DIAGNOSTIC_MODES: Tuple[str, ...] = tuple(
    mode for mode, spec in MODE_SPECS.items() if spec["role"] == "diagnostic"
)
CANONICAL_MODES: Tuple[str, ...] = tuple(MODE_SPECS)

#: Deprecated names kept so historical configs stay loadable.
#:
#: * ``relation_aware`` was in fact ``P1 O1 R1 L0`` -> ``full_specification``.
#: * ``shuffled_operation`` was in fact ``P1 + shuffled O`` -> ``pair_shuffled_operation``.
MODE_ALIASES: Dict[str, str] = {
    "relation_aware": "full_specification",
    "shuffled_operation": "pair_shuffled_operation",
}

ALL_MODE_NAMES: Tuple[str, ...] = CANONICAL_MODES + tuple(MODE_ALIASES)


def is_known_mode(name: str) -> bool:
    return name in MODE_SPECS or name in MODE_ALIASES


def resolve_mode_name(name: str) -> str:
    """Map a (possibly deprecated) mode name onto its canonical name."""
    if name in MODE_SPECS:
        return name
    if name in MODE_ALIASES:
        return MODE_ALIASES[name]
    raise ValueError(
        f"未知 mr_instruction_mode: {name!r}. "
        f"可用: {list(CANONICAL_MODES)}; 别名: {list(MODE_ALIASES)}"
    )


def mode_spec(name: str) -> Dict[str, object]:
    """Return the mode spec for ``name`` (aliases resolved)."""
    return MODE_SPECS[resolve_mode_name(name)]


def grounding_status(spec: Dict[str, object]) -> str:
    """``source_grounded`` when the pair is shown, else ``ungrounded``."""
    return "source_grounded" if spec["pair"] else "ungrounded"


def mode_design_meta(name: str) -> Dict[str, object]:
    """The P/O/R/L block written into experiment metadata (never re-derived)."""
    spec = mode_spec(name)
    return {
        "canonical_mode": resolve_mode_name(name),
        "requested_mode": name,
        "pair": bool(spec["pair"]),
        "operation": bool(spec["operation"]),
        "relation": bool(spec["relation"]),
        "label_anchor": bool(spec["label_anchor"]),
        "grounding": grounding_status(spec),
        "pair_source": spec["pair_source"],
        "operation_source": spec["operation_source"],
        "relation_source": spec["relation_source"],
        "role": spec["role"],
        "core_factorial": spec["role"] == "core",
    }


# ============================================================
# Instruction composition
# ============================================================
def active_block_names(spec: Dict[str, object]) -> List[str]:
    """Block names in canonical order, for one mode spec."""
    names = []
    if spec["pair"]:
        names.append("pair")
    if spec["label_anchor"]:
        names.append("label_anchor")
    if spec["operation"]:
        names.append("operation")
    if spec["relation"]:
        names.append("relation")
    names.append("nli_task")
    return names


_BLOCK_TEMPLATES = {
    "pair": PAIR_BLOCK,
    "label_anchor": LABEL_ANCHOR_BLOCK,
    "operation": OPERATION_BLOCK,
    "relation": RELATION_BLOCK,
}


def build_mode_template(spec: Dict[str, object], binary: bool = False) -> str:
    """Assemble the reference template for a mode spec (blocks in fixed order)."""
    parts = [_BLOCK_TEMPLATES[name] for name in active_block_names(spec) if name != "nli_task"]
    parts.append(NLI_TASK_BLOCK_BINARY if binary else NLI_TASK_BLOCK)
    return BLOCK_SEPARATOR.join(parts)


#: Derived (never hand-written) per-mode template, keyed by canonical mode name.
INSTRUCTION_TEMPLATES: Dict[str, str] = {
    mode: build_mode_template(spec) for mode, spec in MODE_SPECS.items()
}


def compose_instruction(
    mode: str,
    nli_instruction: str,
    original_premise: str = "",
    original_hypothesis: str = "",
    original_label: str = "",
    operation_description: str = "",
    relation_description: str = "",
) -> str:
    """Render the instruction for ``mode`` from pre-resolved block payloads.

    Only block *presence* varies with the mode; wording, order and separators
    are fixed by the registry.
    """
    spec = mode_spec(mode)
    template = build_mode_template(spec)
    return template.format(
        nli_instruction=nli_instruction,
        original_premise=original_premise,
        original_hypothesis=original_hypothesis,
        original_label=original_label,
        operation_description=operation_description,
        relation_description=relation_description,
    )


# ============================================================
# Deterministic block derangement for the shuffled controls
# ============================================================
def _stable_permutation_key(seed: int, value: str) -> str:
    """Deterministic pseudo-random sort key; independent of ``random`` internals."""
    import hashlib

    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def block_derangement(
    buckets: Sequence[Optional[str]],
    keys: Sequence[str],
    seed: int,
) -> List[Optional[str]]:
    """Reassign ``buckets`` so no element keeps its own bucket.

    ``buckets[i]`` is the bucket label of row ``i`` (``None`` for rows that are
    excluded, e.g. ``mr_id == "none"``); ``keys[i]`` is a stable per-row key.

    Algorithm: group rows by bucket (each group contiguous, sorted by ``key``),
    order the groups by ``seed``, then cyclically shift the whole list by the
    size of the largest bucket.  For a row inside a group of size ``s``, the
    shift lands at least ``max_size >= s`` positions later, i.e. strictly past
    its own group's end -- and when the shift wraps, it lands strictly before
    its own group's start.  Either way it cannot land back in its own bucket.
    A cyclic shift is a permutation, so the assigned bucket multiset equals the
    original one (matched marginals).

    Requires ``max_size * 2 <= total`` (Hall's condition); otherwise no full
    derangement exists and ``ValueError`` is raised.

    The result is deterministic given ``(buckets, keys, seed)``, and different
    seeds produce different assignments.
    """
    indexed = [(bucket, keys[i], i) for i, bucket in enumerate(buckets) if bucket is not None]
    result: List[Optional[str]] = [None] * len(buckets)
    if not indexed:
        return result

    grouped: Dict[str, List[Tuple[str, str, int]]] = {}
    for item in indexed:
        grouped.setdefault(item[0], []).append(item)

    total = len(indexed)
    max_size = max(len(members) for members in grouped.values())
    if max_size * 2 > total:
        # Hall's condition: the largest bucket needs more partners than exist.
        raise ValueError(
            "无法生成零 identity collision 的 derangement: "
            f"最大 bucket={max_size} 超过总数的一半 (total={total})"
        )

    # Seed determines the *block order*.  Any order that keeps each bucket's rows
    # contiguous is valid (see the docstring), so this is seed-sensitive without
    # ever placing a row back into its own bucket.  A cyclic rotation would be a
    # no-op whenever all buckets have equal size, hence the hash-based shuffle.
    block_names = sorted(
        grouped,
        key=lambda bucket: _stable_permutation_key(seed, bucket),
    )

    ordered: List[Tuple[str, str, int]] = []
    for bucket in block_names:
        ordered.extend(sorted(grouped[bucket], key=lambda item: item[1]))

    for position, (_, _, index) in enumerate(ordered):
        result[index] = ordered[(position + max_size) % total][0]
    return result


def derangement_collision_count(
    buckets: Sequence[Optional[str]],
    assigned: Sequence[Optional[str]],
) -> int:
    """Rows whose assigned bucket equals their own bucket."""
    return sum(
        1
        for bucket, new_bucket in zip(buckets, assigned)
        if bucket is not None and bucket == new_bucket
    )


def derangement_frequency_mismatch(
    buckets: Sequence[Optional[str]],
    assigned: Sequence[Optional[str]],
) -> bool:
    """True when the assigned bucket multiset differs from the original."""
    from collections import Counter

    return Counter(b for b in buckets if b is not None) != Counter(
        b for b in assigned if b is not None
    )
