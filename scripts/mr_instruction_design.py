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
    version: int = 3,
) -> str:
    """Render the instruction for ``mode`` from pre-resolved block payloads.

    Only block *presence* varies with the mode; wording, order and separators
    are fixed by the registry.  ``version`` selects the block wording: 3 = the
    frozen v3 design, 4 = ordered-provenance Operation + constraint Relation.
    Block order is identical in both.
    """
    if version >= INSTRUCTION_DESIGN_VERSION_V4:
        template = INSTRUCTION_TEMPLATES_V4[resolve_mode_name(mode)]
    else:
        template = build_mode_template(mode_spec(mode))
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
    strict_hall: bool = True,
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
        # Hall's condition: the largest bucket needs more partners than exist, so
        # a complete derangement does not exist.  Strict callers must not silently
        # accept a weaker control; best-effort callers get the rotation, which
        # attains the minimum possible number of collisions (2*max - total).
        if strict_hall:
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


# ============================================================================
# v4: ordered provenance-explicit Operation + constraint-form Relation
# ============================================================================
# v4 keeps the SAME experimental matrix (MODE_SPECS, block order, P/O/R/L
# semantics) as v3.  Only two things change:
#
#   Operation : a generic single-sentence provenance blurb becomes an *ordered*
#               transformation trace T1 -> T2 -> ... -> Tk, expanded from the
#               dataset's ``component_mrs`` sequence.
#   Relation  : hedging prose becomes a *constraint* statement over the
#               (source label -> follow-up label) mapping, restricted to the
#               applicability branch the generator actually instantiates.
#
# Nothing in this section is read by the v3 code path, so v3 output stays
# byte-identical.  See RQ2/RQ2_INSTRUCTION_DESIGN_V4.md and
# RQ2/MR_RELATION_AUDIT_V4.md.

INSTRUCTION_DESIGN_VERSION_V4 = 4

#: v4 pair wording: "sample" reads like a few-shot exemplar; "source input"
#: states the grounding role of the shown text explicitly.
PAIR_BLOCK_V4 = (
    "Paired source input:\n"
    "<source_premise>\n"
    "{original_premise}\n"
    "</source_premise>\n"
    "<source_hypothesis>\n"
    "{original_hypothesis}\n"
    "</source_hypothesis>"
)

OPERATION_BLOCK_V4 = "Input transformation:\n{operation_description}"
RELATION_BLOCK_V4 = "Output relation:\n{relation_description}"

#: LABEL_ANCHOR_BLOCK and NLI_TASK_BLOCK are reused verbatim from v3: v4 changes
#: the MR-information representation, not the base NLI task.

V4_OPERATION_TRACE_INTRO = (
    "The follow-up input was derived from its source input through the following "
    "ordered transformation sequence:"
)

#: Canonical factual step sentences, one per atomic transformation.  Declarative
#: past tense: they state what happened to the input, never what the output must
#: be.  Semantics follow the generator contract (MTrain OPERATION_SPECS), see
#: RQ2/MR_RELATION_AUDIT_V4.md.
V4_ATOMIC_OPERATION_STEPS: Dict[str, str] = {
    "synonym_replacement":
        "One or more words were replaced with context-appropriate synonyms.",
    "pronoun_substitution":
        "A noun phrase or pronoun was replaced with a coreferential expression.",
    "voice_switch":
        "A sentence was rewritten between active and passive voice.",
    "conditional_clause":
        "A conditional clause was added to one component of the input.",
    "negation_flip":
        "A negation marker was added, removed, or reversed.",
    "antonym_substitution":
        "One or more content words were replaced with context-appropriate antonyms.",
    "uninformative":
        "Additional content that is semantically irrelevant to the affected component was inserted.",
    "adding_contradiction":
        "Additional content that conflicts with a proposition in the affected component was inserted.",
}

#: Used only when a composite row carries no component provenance.  Formal RQ2
#: v4 conversions must not contain this: the strict gate counts it and fails.
V4_COMPOSITE_OPERATION_FALLBACK_STEP = (
    "Multiple input-side transformations were applied in an unspecified order."
)

#: Provenance tag for an mr_id / component with no v4 step definition.  Treated
#: as "missing" by the converter so it can never silently enter a formal run.
OPERATION_PROVENANCE_UNSUPPORTED = "unsupported"

# ============================================================
# v4 relation kinds
# ============================================================
# The v3 `flip` text totalised the mapping to entailment<->contradiction with
# neutral->neutral.  That 3-class map is a *metric* operationalisation
# (scripts/metamorphic_metrics.py) used to build a total function; the MR formal
# definition is the applicability-restricted branch the generator instantiates
# (MTrain OPERATION_SPECS: target="contradiction", eligible only when the source
# label is entailment; paper: R_E->C = {r | O_r(E)=C}).
#
# v4 therefore states the constraint on the branch that actually occurs and does
# not invent the C->E / N->N branches.
V4_RELATION_KINDS: Tuple[str, ...] = (
    "invariance",
    "entailment_to_contradiction",
    "entailment_to_neutral",
)

V4_RELATION_KIND_DESCRIPTIONS: Dict[str, str] = {
    "invariance": (
        "For a valid source-follow-up pair, the output labels must satisfy the "
        "following constraint:\n"
        "the follow-up label must be the same as the source label."
    ),
    "entailment_to_contradiction": (
        "For a valid source-follow-up pair to which this relation applies, the "
        "output labels must satisfy the following constraint:\n"
        "if the source label is entailment, the follow-up label must be contradiction."
    ),
    "entailment_to_neutral": (
        "For a valid source-follow-up pair to which this relation applies, the "
        "output labels must satisfy the following constraint:\n"
        "if the source label is entailment, the follow-up label must be neutral."
    ),
}

#: mr_id -> v4 relation kind.  Derived from the same source of truth as v3's
#: MR_RELATION_SPECS; kept explicit so the mapping is auditable in one place.
MR_V4_RELATION_KINDS: Dict[str, Optional[str]] = {
    "synonym_replacement": "invariance",
    "pronoun_substitution": "invariance",
    "voice_switch": "invariance",
    "composite_inv": "invariance",
    "adding_contradiction": "entailment_to_contradiction",
    "antonym_substitution": "entailment_to_contradiction",
    "negation_flip": "entailment_to_contradiction",
    "composite_flip": "entailment_to_contradiction",
    "uninformative": "entailment_to_neutral",
    "conditional_clause": "entailment_to_neutral",
    "composite_neutral": "entailment_to_neutral",
    "race_sensitive_transformation": None,
    "same_type_named_entity_substitution": None,
    "tense_shift": None,
    "none": None,
}

#: v4 kinds whose constraint names the follow-up label conditionally.  Ungrounded
#: R remains a shortcut risk for these (see the design doc); that is a property
#: of the design, not a wording bug, and must not be "fixed" by hedging.
V4_SHORTCUT_RELATION_KINDS = ("entailment_to_contradiction", "entailment_to_neutral")


def relation_kind_v4(mr_id: str) -> Optional[str]:
    return MR_V4_RELATION_KINDS.get(mr_id)


def relation_kind_requires_entailment_source(kind: Optional[str]) -> bool:
    return kind in ("entailment_to_contradiction", "entailment_to_neutral")


def render_relation_v4(mr_id: str) -> str:
    """Return the v4 ``Output relation`` payload for ``mr_id``."""
    kind = relation_kind_v4(mr_id)
    if kind is None:
        return ""
    return V4_RELATION_KIND_DESCRIPTIONS[kind]


def _norm_mr_id(value) -> str:
    if value is None:
        return "none"
    return str(value).strip().lower().replace("-", "_")


def resolve_operation_trace_v4(
    mr_id: str,
    component_mrs: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[str, ...], str]:
    """Return ``(trace_id, provenance)`` for one row.

    ``trace_id`` is the **ordered** sequence of atomic transformations that
    produced the follow-up input.  ``component_mrs`` is authoritative and is
    used exactly as stored: never sorted, never deduplicated, never filtered --
    the array order is the real execution order, and a repeated entry means the
    transformation ran twice.

    Composite rows with no usable provenance yield an empty trace plus the
    ``composite_fallback`` tag so callers can count and (in strict mode) reject
    them.
    """
    if not mr_id or mr_id == "none":
        return (), OPERATION_PROVENANCE_NONE

    if mr_id in COMPOSITE_MR_IDS:
        components = [_norm_mr_id(c) for c in (component_mrs or [])]
        if not components:
            return (), OPERATION_PROVENANCE_FALLBACK
        if all(c in V4_ATOMIC_OPERATION_STEPS for c in components):
            return tuple(components), OPERATION_PROVENANCE_COMPONENTS
        # Provenance is present but names a transformation we cannot describe.
        # Never silently drop the unknown component: report it as unsupported.
        return (), OPERATION_PROVENANCE_UNSUPPORTED

    if mr_id not in V4_ATOMIC_OPERATION_STEPS:
        return (), OPERATION_PROVENANCE_UNSUPPORTED
    return (mr_id,), OPERATION_PROVENANCE_EXACT


def render_operation_v4(trace_id: Sequence[str], provenance: str) -> str:
    """Render the v4 ``Input transformation`` payload.

    Atomic and composite rows share one format: a numbered ordered list.  An
    atomic row is simply an arity-1 trace.
    """
    if provenance in (OPERATION_PROVENANCE_NONE, OPERATION_PROVENANCE_UNSUPPORTED):
        return ""
    if provenance == OPERATION_PROVENANCE_FALLBACK:
        steps = [V4_COMPOSITE_OPERATION_FALLBACK_STEP]
    else:
        steps = [V4_ATOMIC_OPERATION_STEPS[c] for c in trace_id]
    body = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
    return f"{V4_OPERATION_TRACE_INTRO}\n{body}"


def operation_arity(trace_id: Sequence[str]) -> int:
    return len(trace_id)


# ============================================================
# v4 block composition
# ============================================================
_V4_BLOCK_TEMPLATES = {
    "pair": PAIR_BLOCK_V4,
    "label_anchor": LABEL_ANCHOR_BLOCK,
    "operation": OPERATION_BLOCK_V4,
    "relation": RELATION_BLOCK_V4,
}


def build_mode_template_v4(spec: Dict[str, object], binary: bool = False) -> str:
    """Same block order as v3, with the v4 block wording."""
    parts = [_V4_BLOCK_TEMPLATES[name]
             for name in active_block_names(spec) if name != "nli_task"]
    parts.append(NLI_TASK_BLOCK_BINARY if binary else NLI_TASK_BLOCK)
    return BLOCK_SEPARATOR.join(parts)


INSTRUCTION_TEMPLATES_V4: Dict[str, str] = {
    mode: build_mode_template_v4(spec) for mode, spec in MODE_SPECS.items()
}


# ============================================================
# v4 matched derangement for the shuffled-operation control
# ============================================================
def select_matched_shuffled_operation_donors(rows_meta: Sequence[Dict[str, object]],
                                             seed: int) -> List[Optional[int]]:
    """Choose a donor row for every augmented row (matched derangement).

    ``rows_meta[i]`` must provide ``trace_id`` (tuple), ``arity`` (int),
    ``relation_kind`` (str or None) and ``key`` (stable string).  Source rows
    must pass ``trace_id = None`` and get ``None`` back.

    Matching preferences, in order:

    1. same transformation arity (1-step <-> 1-step, 2-step <-> 2-step, ...);
    2. same relation kind (invariance <-> invariance, E->C <-> E->C, ...);
    3. the assigned trace must differ from the row's own trace;
    4. rows are visited in a seed-dependent order so the control is not one
       fixed permutation shared by every seed.

    Returns a donor-index list aligned to ``rows_meta``.  When a stratum cannot
    be deranged internally (Hall's condition fails), the residual fixed points
    are repaired by swapping donors with another stratum; the number of such
    rows is reported by the caller via
    :func:`build_matched_control_audit`.
    """
    n = len(rows_meta)
    donors: List[Optional[int]] = [None] * n
    active = [i for i, meta in enumerate(rows_meta) if meta.get("trace_id") is not None]
    if not active:
        return donors

    # ---- phase 1: derange within each arity stratum -------------------------
    strata: Dict[int, List[int]] = {}
    for i in active:
        strata.setdefault(int(rows_meta[i]["arity"]), []).append(i)

    for arity in sorted(strata):
        members = strata[arity]
        grouped: Dict[Tuple[str, ...], List[int]] = {}
        for i in members:
            grouped.setdefault(tuple(rows_meta[i]["trace_id"]), []).append(i)

        # Block order is seed-dependent so different seeds give different
        # assignments, while the cyclic shift below still guarantees a
        # different trace (contiguous blocks + shift >= own block size).
        block_order = sorted(
            grouped,
            key=lambda trace: _stable_permutation_key(seed, "\x1f".join(trace)),
        )
        ordered: List[int] = []
        for trace in block_order:
            ordered.extend(sorted(grouped[trace], key=lambda i: str(rows_meta[i]["key"])))

        offset = max(len(v) for v in grouped.values())
        total = len(ordered)
        for position, i in enumerate(ordered):
            donors[i] = ordered[(position + offset) % total]

    # ---- phase 2: repair residual fixed points across strata ---------------
    fixed = [i for i in active if rows_meta[donors[i]]["trace_id"] == rows_meta[i]["trace_id"]]
    repair_order = sorted(
        fixed,
        key=lambda i: _stable_permutation_key(seed, str(rows_meta[i]["key"])),
    )
    for i in repair_order:
        if rows_meta[donors[i]]["trace_id"] != rows_meta[i]["trace_id"]:
            continue  # already repaired by an earlier swap
        candidates = sorted(
            (j for j in active
             if j != i
             and rows_meta[j]["arity"] != rows_meta[i]["arity"]
             and rows_meta[donors[j]]["trace_id"] != rows_meta[i]["trace_id"]
             and rows_meta[donors[i]]["trace_id"] != rows_meta[j]["trace_id"]),
            key=lambda j: _stable_permutation_key(seed, str(rows_meta[j]["key"])),
        )
        if not candidates:
            continue
        j = candidates[0]
        donors[i], donors[j] = donors[j], donors[i]

    return donors


def build_matched_control_audit(
    rows_meta: Sequence[Dict[str, object]],
    donors: Sequence[Optional[int]],
    assigned_relation_kinds: Optional[Sequence[Optional[str]]] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Quality report for the v4 shuffled-operation / shuffled-relation controls."""
    total = 0
    trace_collisions = 0
    text_collisions = 0
    same_arity = 0
    same_kind = 0
    cross_arity_fallback = 0
    cross_kind_fallback = 0
    no_candidate = 0

    for i, meta in enumerate(rows_meta):
        if meta.get("trace_id") is None:
            continue
        total += 1
        donor = donors[i]
        if donor is None:
            no_candidate += 1
            continue

        own_trace = tuple(meta["trace_id"])
        donor_trace = tuple(rows_meta[donor]["trace_id"])
        if own_trace == donor_trace:
            trace_collisions += 1
        if meta.get("operation_text") is not None and \
                meta["operation_text"] == rows_meta[donor].get("operation_text"):
            text_collisions += 1

        if int(rows_meta[donor]["arity"]) == int(meta["arity"]):
            same_arity += 1
        else:
            cross_arity_fallback += 1

        own_kind = meta.get("relation_kind")
        donor_kind = rows_meta[donor].get("relation_kind")
        if own_kind is not None and own_kind == donor_kind:
            same_kind += 1
        else:
            cross_kind_fallback += 1

    report = {
        "seed": seed,
        "total_augmented": total,
        "operation_trace_identity_collision_count": trace_collisions,
        "operation_text_unchanged_count": text_collisions,
        "shuffled_operation_same_arity_count": same_arity,
        "shuffled_operation_same_arity_rate": (same_arity / total) if total else 0.0,
        "shuffled_operation_same_relation_kind_count": same_kind,
        "shuffled_operation_same_relation_kind_rate": (same_kind / total) if total else 0.0,
        "shuffled_operation_cross_arity_fallback_count": cross_arity_fallback,
        "shuffled_operation_cross_relation_fallback_count": cross_kind_fallback,
        "shuffled_operation_no_valid_candidate_count": no_candidate,
    }

    if assigned_relation_kinds is not None:
        rel_total = 0
        rel_collisions = 0
        for i, meta in enumerate(rows_meta):
            if meta.get("relation_kind") is None:
                continue
            rel_total += 1
            if assigned_relation_kinds[i] == meta["relation_kind"]:
                rel_collisions += 1
        report["total_relation_eligible"] = rel_total
        report["shuffled_relation_identity_collision_count"] = rel_collisions

    return report
