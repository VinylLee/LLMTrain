"""The eight core sentiment-analysis metamorphic relations (SA-RQ1).

Each entry is a declarative ``MRDefinition``: what the relation means, how it is
realised (LLM rewrite or deterministic transform), when it applies, and how its
output is checked.  ``scripts/generate_sa_mr.py`` is the engine that consumes
this catalog; it holds no MR knowledge of its own.

The catalog mirrors section 6 of the RQ1 SA route document: six
``preserve`` relations (the text's overall polarity must survive the edit) and
two ``flip`` relations (a strictly validated positive <-> negative reversal).
``sa_case_reversal`` is deliberately deterministic -- surface-form invariance
needs no generator noise.

Freezing a catalog version means bumping ``PROMPT_VERSION``: any prompt edit
invalidates previously generated data, and ``catalog_sha256()`` makes that
detectable in the run report.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sa_lexicon import (
    has_contrast_marker,
    count_negation_cues,
    polarity_scores,
)

MR_CATALOG_VERSION = "sa_mr_catalog_v1"
PROMPT_VERSION = "sa_mr_v1"

RELATION_PRESERVE = "preserve"
RELATION_FLIP = "flip"

MECHANISM_LLM = "llm"
MECHANISM_DETERMINISTIC = "deterministic"

FAMILY_IRRELEVANT_CONTEXT = "irrelevant_context_invariance"
FAMILY_REFERENCE_PRESERVING = "reference_preserving_invariance"
FAMILY_STRUCTURAL = "structural_invariance"
FAMILY_LEXICAL_EQUIVALENCE = "lexical_semantic_equivalence"
FAMILY_SURFACE_FORM = "surface_form_invariance"
FAMILY_LABEL_TRANSITION = "task_specific_label_transition"

#: Bridge onto the shared metric convention used by ``metamorphic_metrics``.
#: ``neutral`` is never emitted by SA.
RELATION_TO_MR_TYPE = {RELATION_PRESERVE: "inv", RELATION_FLIP: "flip"}

OPPOSITE_LABEL = {"positive": "negative", "negative": "positive"}

#: Flip MRs pre-screen on length, but the real gate is the validator check.
#: The cap is deliberately generous: the human CAD counterfactuals show that the
#: IMDb reviews in this project's source pool were ALL successfully flipped by
#: annotators, so a low cap would encode a generator limitation as an
#: applicability verdict.  The pilot measures relation validity by length bucket
#: to find where quality actually degrades.
DEFAULT_FLIP_MAX_WORDS = 200

SYSTEM_PROMPT = (
    "You are a careful dataset editor for binary sentiment analysis. "
    "You edit exactly one text at a time and you reply with JSON only."
)

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict sentiment annotator. Read only the text given to you "
    "and reply with JSON only."
)

VERIFIER_USER_TEMPLATE = """Read the text below and decide its overall sentiment.

Label definitions:
- positive: the text expresses an overall favourable opinion about its subject.
- negative: the text expresses an overall unfavourable opinion about its subject.

Rules:
- Judge the text as a whole, not individual words.
- For a mixed text, use the dominant overall attitude.
- If you cannot decide, set "unambiguous" to false.

Text:
{text}

Return exactly one JSON object containing:
- "predicted_label": exactly positive or negative
- "unambiguous": true only if no other label is reasonably defensible
- "reason": one short explanation grounded only in the text above
No label or metadata is provided alongside the text. Do not infer one from the wording.
Return JSON only:"""


class NotApplicable(ValueError):
    """Raised when an MR's applicability condition is not met for a source text."""


# --------------------------------------------------------------------------- #
# Shared text predicates
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|@\w+", re.IGNORECASE)
_EMOJI_RE = re.compile(
    "[\U0001f300-\U0001faff☀-➿⬀-⯿️]"
)
_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "was", "were", "are",
        "but", "not", "you", "his", "her", "she", "him", "they", "them",
        "their", "have", "has", "had", "been", "being", "all", "any",
        "can", "will", "would", "could", "should", "there", "here", "what",
        "when", "which", "who", "whom", "how", "why", "its", "it's", "very",
        "too", "also", "just", "only", "even", "still", "more", "most",
        "much", "many", "some", "such", "than", "then", "them", "these",
        "those", "into", "onto", "from", "about", "over", "under", "after",
        "before", "because", "while", "where", "does", "did", "doing", "done",
    }
)

_COMMON_VERBS = frozenset(
    {
        "is", "are", "was", "were", "be", "been", "being", "am",
        "has", "have", "had", "do", "does", "did", "done", "doing",
        "make", "makes", "made", "making", "get", "gets", "got", "getting",
        "go", "goes", "went", "going", "come", "comes", "came", "coming",
        "see", "sees", "saw", "seen", "seeing", "watch", "watches", "watched",
        "like", "likes", "liked", "love", "loves", "loved", "hate", "hates",
        "hated", "enjoy", "enjoys", "enjoyed", "want", "wants", "wanted",
        "need", "needs", "needed", "think", "thinks", "thought", "know",
        "knows", "knew", "known", "find", "finds", "found", "give", "gives",
        "gave", "given", "take", "takes", "took", "taken", "play", "plays",
        "played", "work", "works", "worked", "act", "acts", "acted", "tell",
        "tells", "told", "feel", "feels", "felt", "look", "looks", "looked",
        "seem", "seems", "seemed", "keep", "keeps", "kept", "let", "lets",
        "put", "puts", "say", "says", "said", "show", "shows", "showed",
        "leave", "leaves", "left", "call", "calls", "called", "read", "reads",
        "write", "writes", "wrote", "written", "buy", "buys", "bought",
        "spend", "spends", "spent", "try", "tries", "tried", "use", "uses",
        "used", "recommend", "recommends", "recommended", "waste", "wastes",
        "wasted", "deserve", "deserves", "deserved", "remind", "reminds",
        "reminded", "deliver", "delivers", "delivered", "offer", "offers",
        "offered", "produce", "produces", "produced", "create", "creates",
        "created", "turn", "turns", "turned", "start", "starts", "started",
        "stop", "stops", "stopped", "wait", "waits", "waited", "belong",
        "belongs", "belonged", "matter", "matters", "mattered", "cost",
        "costs", "cost", "serve", "serves", "served", "arrive", "arrives",
        "arrived", "return", "returns", "returned", "expect", "expects",
        "expected", "believe", "believes", "believed", "realize", "realizes",
        "realized", "understand", "understands", "understood", "remember",
        "remembers", "remembered", "forget", "forgets", "forgot", "mean",
        "means", "meant", "help", "helps", "helped", "helping", "bring",
        "brings", "brought", "hold", "holds", "held", "follow", "follows",
        "followed", "lose", "loses", "lost", "win", "wins", "won", "fail",
        "fails", "failed", "succeed", "succeeds", "succeeded", "die", "dies",
        "died", "live", "lives", "lived", "grow", "grows", "grew", "grown",
        "sell", "sells", "sold", "pay", "pays", "paid", "order", "orders",
        "ordered", "ship", "ships", "shipped", "fit", "fits", "fitted",
        "last", "lasts", "lasted", "sound", "sounds", "sounded", "allow",
        "allows", "allowed", "end", "ends", "ended", "begin", "begins",
        "began", "begun", "run", "runs", "ran", "kills", "kill", "killed",
    }
)


def word_count(text: str) -> int:
    return len(text.split())


def has_common_verb(text: str) -> bool:
    return any(token.casefold() in _COMMON_VERBS for token in _WORD_RE.findall(text))


def has_named_entity(text: str) -> bool:
    """A capitalised multi-letter token that is not sentence-initial."""
    words = text.split()
    for position, word in enumerate(words):
        stripped = word.strip(".,!?;:\"'()")
        if position == 0:
            continue
        if len(stripped) > 2 and stripped[:1].isupper() and stripped[1:].islower():
            return True
    return False


def has_repeated_content_token(text: str) -> bool:
    tokens = [token.casefold() for token in _WORD_RE.findall(text)]
    counts = Counter(
        token for token in tokens if token not in _STOPWORDS and len(token) > 3
    )
    return any(count >= 2 for count in counts.values())


def count_nonpolar_content_tokens(text: str) -> int:
    positive_terms, negative_terms = _polar_terms()
    total = 0
    for token in (token.casefold() for token in _WORD_RE.findall(text)):
        if token in _STOPWORDS or token in positive_terms or token in negative_terms:
            continue
        if len(token) > 3:
            total += 1
    return total


def _polar_terms() -> tuple[frozenset[str], frozenset[str]]:
    from sa_lexicon import NEGATIVE_TERMS, POSITIVE_TERMS

    return POSITIVE_TERMS, NEGATIVE_TERMS


def is_clearly_mixed(text: str) -> bool:
    """Whether both polarities are attested enough to make the text genuinely mixed.

    A single incidental word on the minority side is not enough evidence: the
    requirement is at least two minority hits *and* a near-balanced ratio.  This
    is a coarse pre-screen only -- the authoritative whole-text polarity judgement
    belongs to the validators in the acceptance gate.
    """
    positive, negative = polarity_scores(text)
    if positive == 0 or negative == 0:
        return False
    minority = min(positive, negative)
    return minority >= 2 and minority / max(positive, negative) >= 0.5


def contains_url_or_emoji(text: str) -> bool:
    return bool(_URL_RE.search(text) or _EMOJI_RE.search(text))


def count_allcaps_tokens(text: str) -> int:
    return sum(1 for word in text.split() if len(word) > 1 and word.isupper())


def is_flip_eligible(text: str, max_words: int) -> tuple[bool, str]:
    """Shared applicability gate for both polarity-flip MRs."""
    words = word_count(text)
    if words > max_words:
        return False, f"flip_mr_requires_at_most_{max_words}_words"
    if count_negation_cues(text) >= 2:
        return False, "flip_mr_source_has_complex_negation"
    if is_clearly_mixed(text):
        return False, "flip_mr_source_is_clearly_mixed"
    return True, ""


# --------------------------------------------------------------------------- #
# Applicability rules
# --------------------------------------------------------------------------- #

def _applicable_uninformative_context(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    words = word_count(text)
    if words < 8:
        return False, "text_too_short_for_added_context"
    if words > 200:
        return False, "text_too_long_for_added_context"
    return True, ""


def _applicable_pronoun_substitution(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    if word_count(text) < 12:
        return False, "text_too_short_for_reference_rewrite"
    if not (has_repeated_content_token(text) or has_named_entity(text)):
        return False, "no_repeated_or_named_reference_to_replace"
    return True, ""


def _applicable_voice_switch(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    if word_count(text) < 8:
        return False, "text_too_short_for_voice_rewrite"
    if not has_common_verb(text):
        return False, "no_finite_verb_detected"
    return True, ""


def _applicable_synonym_nonpolar(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    if word_count(text) < 8:
        return False, "text_too_short_for_synonym_rewrite"
    if count_nonpolar_content_tokens(text) < 2:
        return False, "fewer_than_two_nonpolar_content_words"
    return True, ""


def _applicable_tense_shift(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    if word_count(text) < 8:
        return False, "text_too_short_for_tense_rewrite"
    if not has_common_verb(text):
        return False, "no_finite_verb_detected"
    return True, ""


def _applicable_case_reversal(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    if contains_url_or_emoji(text):
        return False, "contains_url_mention_or_emoji"
    if count_allcaps_tokens(text) >= 1:
        return False, "contains_acronym_or_emphatic_caps"
    return True, ""


def _applicable_negation_flip(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    return is_flip_eligible(text, int(context.get("flip_max_words", DEFAULT_FLIP_MAX_WORDS)))


def _applicable_antonym_flip(text: str, context: Mapping[str, Any]) -> tuple[bool, str]:
    return is_flip_eligible(text, int(context.get("flip_max_words", DEFAULT_FLIP_MAX_WORDS)))


# --------------------------------------------------------------------------- #
# Deterministic case reversal
# --------------------------------------------------------------------------- #

CASE_MODES = ("lower", "upper", "swap")


def case_reversal_mode(source_id: str) -> str:
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return CASE_MODES[int.from_bytes(digest[:8], "big") % len(CASE_MODES)]


def apply_case_reversal(text: str, source_id: str) -> str:
    """Deterministic surface-form change; word order and characters are preserved."""
    mode = case_reversal_mode(source_id)
    if mode == "lower":
        return text.lower()
    if mode == "upper":
        return text.upper()
    return text.swapcase()


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

_PRESERVE_TEMPLATE = """{operation}

Rules - all are mandatory:
1. The edited text must express the SAME overall sentiment as the original. \
The original is {source_label}; the result must still be {source_label}.
2. Keep every evaluated object, fact, opinion and its strength unchanged.
3. Do not add or remove any opinion, degree word or negation.
4. Do not add headings, explanations, notes, quotes or commentary.
5. Return the complete edited text, not a summary and not only part of it.
6. If this edit is not possible without changing the meaning, return {{"text": null}}.

Original sentiment: {source_label}
Original text:
{source_text}

Return exactly one JSON object with a single string field "text" holding the \
complete edited text, or {{"text": null}} if the edit is not possible. \
Return JSON only:"""

_FLIP_TEMPLATE = """{operation}

Rules - all are mandatory:
1. The edited text must express the OPPOSITE overall sentiment. \
The original is {source_label}; the result must be {expected_label}.
2. Reverse the text's central evaluation itself. Do NOT simply append a sentence \
such as "I did not like it." at the end.
3. Change only what the reversal needs. Keep the topic, the evaluated object, and \
every other detail unchanged.
4. Touch exactly one predicate. Do not stack negations and do not rely on double \
negation.
5. The result must read as a natural, complete text with the same length and \
register as the original.
6. Do not add headings, explanations, notes, quotes or commentary.
7. If the text has mixed sentiment, or no single expression controls its overall \
sentiment, return {{"text": null}}.

Original sentiment: {source_label}
Original text:
{source_text}

Return exactly one JSON object with these string fields:
- "text": the complete edited text, or null if the reversal is not possible
- "edited_label": the overall sentiment of your edited text, exactly "{expected_label}"
Return JSON only:"""


OPERATIONS = {
    "sa_uninformative_context": (
        "Add one short piece of background information that is emotionally neutral "
        "and unrelated to the main subject being evaluated. Put it either before the "
        "first sentence or after the last sentence of the text."
    ),
    "sa_pronoun_substitution": (
        "Rewrite the text so that repeated mentions of the same person, object or "
        "organisation are replaced by an equivalent pronoun or a shorter referring "
        "expression. The text must still refer to exactly the same things."
    ),
    "sa_voice_switch": (
        "Change the grammatical voice of the main clause: if it is in the active "
        "voice, rewrite it in the passive voice; if it is in the passive voice, "
        "rewrite it in the active voice."
    ),
    "sa_synonym_nonpolar": (
        "Replace a few ordinary words of the text with suitable synonyms. Replace "
        "only words that do NOT carry the text's main sentiment - for example plain "
        "nouns, neutral verbs, or ordinary descriptive words."
    ),
    "sa_tense_shift": (
        "Change the tense of the main verbs: if the text is mainly in the present "
        "tense, rewrite it in the past tense; if it is mainly in the past tense, "
        "rewrite it in the present tense."
    ),
    "sa_controlled_negation_flip": (
        "Reverse the overall sentiment of the text by adding a negation to, or "
        "removing a negation from, the single predicate that carries the text's "
        "main evaluation."
    ),
    "sa_sentiment_antonym_flip": (
        "Reverse the overall sentiment of the text by replacing the central "
        "sentiment expression with an expression of the opposite polarity."
    ),
}


# --------------------------------------------------------------------------- #
# Definition and catalog
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MRDefinition:
    mr_id: str
    property_family: str
    relation_type: str
    mechanism: str
    template_id: str
    prompt_version: str
    operation: str
    applicability: Callable[[str, Mapping[str, Any]], tuple[bool, str]]
    min_words: int
    min_chars: int
    max_chars: int
    min_source_similarity: float
    max_source_similarity: float
    output_fields: tuple[str, ...]
    requires_flip_cue: bool = False
    deterministic_fn: Callable[[str, str], str] | None = None
    #: Bounds on len(follow-up) / len(source).  ``None`` skips the check, which is
    #: required for MRs that add content or that preserve length exactly.
    length_ratio_bounds: tuple[float, float] | None = (0.6, 1.7)
    notes: str = ""
    #: Quality-gate thresholds carried with the MR so the pilot can compare.
    quality_gate_thresholds: tuple[tuple[str, float], ...] = (
        ("transformation_validity", 0.95),
        ("relation_validity", 0.95),
    )

    @property
    def is_deterministic(self) -> bool:
        return self.mechanism == MECHANISM_DETERMINISTIC

    @property
    def mr_type(self) -> str:
        return RELATION_TO_MR_TYPE[self.relation_type]


def _preserve(
    mr_id: str,
    family: str,
    applicability: Callable[[str, Mapping[str, Any]], tuple[bool, str]],
    *,
    min_words: int = 8,
    max_chars: int = 6000,
    length_ratio_bounds: tuple[float, float] | None = (0.6, 1.7),
    notes: str = "",
) -> MRDefinition:
    return MRDefinition(
        mr_id=mr_id,
        property_family=family,
        relation_type=RELATION_PRESERVE,
        mechanism=MECHANISM_LLM,
        template_id=f"sa_mr/{mr_id.removeprefix('sa_')}/v1",
        prompt_version=PROMPT_VERSION,
        operation=OPERATIONS[mr_id],
        applicability=applicability,
        min_words=min_words,
        min_chars=16,
        max_chars=max_chars,
        min_source_similarity=0.55,
        max_source_similarity=0.985,
        output_fields=("text",),
        length_ratio_bounds=length_ratio_bounds,
        notes=notes,
    )


def _flip(
    mr_id: str,
    applicability: Callable[[str, Mapping[str, Any]], tuple[bool, str]],
    *,
    notes: str = "",
) -> MRDefinition:
    return MRDefinition(
        mr_id=mr_id,
        property_family=FAMILY_LABEL_TRANSITION,
        relation_type=RELATION_FLIP,
        mechanism=MECHANISM_LLM,
        template_id=f"sa_mr/{mr_id.removeprefix('sa_')}/v1",
        prompt_version=PROMPT_VERSION,
        operation=OPERATIONS[mr_id],
        applicability=applicability,
        min_words=4,
        min_chars=12,
        max_chars=1500,
        min_source_similarity=0.45,
        max_source_similarity=0.985,
        output_fields=("text", "followup_label"),
        requires_flip_cue=True,
        notes=notes,
    )


MR_CATALOG: dict[str, MRDefinition] = {
    definition.mr_id: definition
    for definition in (
        _preserve(
            "sa_uninformative_context",
            FAMILY_IRRELEVANT_CONTEXT,
            _applicable_uninformative_context,
            length_ratio_bounds=(0.6, 2.5),
            notes="Added span must stay neutral and must not introduce a new main subject.",
        ),
        _preserve(
            "sa_pronoun_substitution",
            FAMILY_REFERENCE_PRESERVING,
            _applicable_pronoun_substitution,
            min_words=12,
            notes="Only applies when the referenced entity is unambiguous.",
        ),
        _preserve(
            "sa_voice_switch",
            FAMILY_STRUCTURAL,
            _applicable_voice_switch,
            notes="Only applies to clauses that convert without changing focus.",
        ),
        _preserve(
            "sa_synonym_nonpolar",
            FAMILY_LEXICAL_EQUIVALENCE,
            _applicable_synonym_nonpolar,
            notes="Sentiment-bearing expressions are out of scope for this MR.",
        ),
        _preserve(
            "sa_tense_shift",
            FAMILY_STRUCTURAL,
            _applicable_tense_shift,
            notes="Tense change must not alter event reality or temporal ordering.",
        ),
        MRDefinition(
            mr_id="sa_case_reversal",
            property_family=FAMILY_SURFACE_FORM,
            relation_type=RELATION_PRESERVE,
            mechanism=MECHANISM_DETERMINISTIC,
            template_id="sa_mr/case_reversal/deterministic_v1",
            prompt_version=PROMPT_VERSION,
            operation=(
                "Change only the letter case of the text, cycling deterministically "
                "between lower, upper and swapped case."
            ),
            applicability=_applicable_case_reversal,
            min_words=2,
            min_chars=8,
            max_chars=6000,
            min_source_similarity=0.0,
            max_source_similarity=1.0,
            output_fields=("text",),
            deterministic_fn=apply_case_reversal,
            length_ratio_bounds=None,
            notes=(
                "Deterministic: no prompt is sent. Normalised text is identical to the "
                "source by construction, so the engine must skip the usual diff and "
                "similarity gates for this MR only."
            ),
        ),
        _flip(
            "sa_controlled_negation_flip",
            _applicable_negation_flip,
            notes="Acceptance requires a validated whole-text polarity reversal.",
        ),
        _flip(
            "sa_sentiment_antonym_flip",
            _applicable_antonym_flip,
            notes="Acceptance requires a validated whole-text polarity reversal.",
        ),
    )
}

MR_ORDER: tuple[str, ...] = tuple(MR_CATALOG)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def get_mr(mr_id: str) -> MRDefinition:
    try:
        return MR_CATALOG[mr_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown mr_id {mr_id!r}; known ids: {', '.join(MR_ORDER)}"
        ) from exc


def expected_followup_label(mr_id: str, source_label: str) -> str:
    definition = get_mr(mr_id)
    if definition.relation_type == RELATION_FLIP:
        return OPPOSITE_LABEL[source_label]
    return source_label


def mr_applicable(
    mr_id: str, text: str, context: Mapping[str, Any] | None = None
) -> tuple[bool, str]:
    definition = get_mr(mr_id)
    return definition.applicability(text, dict(context or {}))


def relation_to_mr_type(relation_type: str) -> str:
    try:
        return RELATION_TO_MR_TYPE[relation_type]
    except KeyError as exc:
        raise ValueError(f"unknown relation type {relation_type!r}") from exc


def build_mr_prompt(
    mr_id: str,
    source_text: str,
    source_label: str,
    retry_feedback: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for one LLM-backed MR rewrite."""
    definition = get_mr(mr_id)
    if definition.is_deterministic:
        raise ValueError(f"{mr_id} is deterministic and has no prompt")
    expected = expected_followup_label(mr_id, source_label)
    template = (
        _FLIP_TEMPLATE if definition.relation_type == RELATION_FLIP else _PRESERVE_TEMPLATE
    )
    user = template.format(
        operation=definition.operation,
        source_label=source_label,
        expected_label=expected,
        source_text=source_text,
    )
    if retry_feedback:
        user += (
            "\n\nYour previous response was rejected for this reason: "
            f"{retry_feedback}\nProduce a corrected edit and return JSON only."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_verification_prompt(text: str) -> list[dict[str, str]]:
    """Blind sentiment probe: no label, no source text, no MR name."""
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": VERIFIER_USER_TEMPLATE.format(text=text)},
    ]


def catalog_payload() -> dict[str, Any]:
    return {
        "catalog_version": MR_CATALOG_VERSION,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "verifier_system_prompt": VERIFIER_SYSTEM_PROMPT,
        "verifier_user_template": VERIFIER_USER_TEMPLATE,
        "preserve_template": _PRESERVE_TEMPLATE,
        "flip_template": _FLIP_TEMPLATE,
        "mrs": [
            {
                "mr_id": definition.mr_id,
                "property_family": definition.property_family,
                "relation_type": definition.relation_type,
                "mechanism": definition.mechanism,
                "template_id": definition.template_id,
                "operation": definition.operation,
                "min_words": definition.min_words,
                "max_chars": definition.max_chars,
                "min_source_similarity": definition.min_source_similarity,
                "max_source_similarity": definition.max_source_similarity,
                "output_fields": list(definition.output_fields),
                "requires_flip_cue": definition.requires_flip_cue,
                "quality_gate_thresholds": dict(definition.quality_gate_thresholds),
            }
            for definition in MR_CATALOG.values()
        ],
    }


def catalog_sha256() -> str:
    payload = json.dumps(catalog_payload(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def matrix_orientation() -> dict[str, Any]:
    """Per-MR summary used by the pilot report header."""
    counts = Counter(definition.relation_type for definition in MR_CATALOG.values())
    families = Counter(definition.property_family for definition in MR_CATALOG.values())
    mechanisms = Counter(definition.mechanism for definition in MR_CATALOG.values())
    return {
        "mr_count": len(MR_CATALOG),
        "by_relation": dict(counts),
        "by_family": dict(families),
        "by_mechanism": dict(mechanisms),
    }
