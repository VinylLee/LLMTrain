"""Minimal transparent polarity lexicon shared by SA import checks and validators.

This exists for three narrow jobs and nothing else:

* sanity-checking an upstream label mapping during dataset import
  (``scripts/download_sa_datasets.py``) -- a silently inverted 0/1 mapping is the
  single most dangerous failure mode in the import pipeline;
* driving the applicability rules and flip-cue checks in
  ``scripts/sa_mr_catalog.py``;
* giving the test suite and ``LexiconValidator`` an offline opinion about a
  text's polarity without downloading a classifier.

It is deliberately NOT the pilot's measurement instrument.  That role belongs to
the local sentiment classifier in ``scripts/sa_validators.py``; this lexicon is
only a coarse, fully auditable backstop.
"""

from __future__ import annotations

import re
from typing import Iterable

POSITIVE_TERMS = frozenset(
    {
        "amazing", "awesome", "beautiful", "best", "brilliant", "charming",
        "classic", "clever", "compelling", "delightful", "enjoyable",
        "enjoyed", "excellent", "exciting", "fabulous", "fantastic",
        "favorite", "favourite", "fun", "funny", "genius", "gorgeous",
        "great", "gripping", "heartwarming", "hilarious", "impressive",
        "incredible", "insightful", "inspiring", "loved", "lovely",
        "masterpiece", "memorable", "moving", "outstanding", "perfect",
        "phenomenal", "pleasant", "pleased", "powerful", "recommend",
        "refreshing", "remarkable", "riveting", "satisfying", "solid",
        "stunning", "superb", "sweet", "talented", "terrific", "touching",
        "wonderful", "worth", "worthy",
    }
)

NEGATIVE_TERMS = frozenset(
    {
        "appalling", "awful", "bad", "bland", "boring", "cheesy", "cliche",
        "cliched", "confusing", "cringe", "dull", "disappointing",
        "disappointed", "disaster", "dreadful", "dull", "embarrassing",
        "forgettable", "garbage", "hate", "hated", "horrible", "horrendous",
        "insipid", "lame", "lousy", "mess", "mediocre", "messy", "miserable",
        "nasty", "nonsense", "obnoxious", "offensive", "painful", "pathetic",
        "poor", "predictable", "pretentious", "ridiculous", "rubbish",
        "shallow", "sloppy", "stupid", "tedious", "terrible", "ugly",
        "unbearable", "unfunny", "uninteresting", "unwatchable", "weak",
        "worse", "worst", "worthless",
    }
)

NEGATION_CUES = frozenset(
    {"not", "no", "never", "cannot", "without", "nothing", "none", "nor",
     "neither", "hardly", "barely", "isn", "wasn", "aren", "weren", "don",
     "doesn", "didn", "won", "can", "couldn", "shouldn", "wouldn", "'t"}
)

CONTRAST_CUES = frozenset(
    {"but", "although", "though", "however", "yet", "nevertheless",
     "nonetheless", "despite", "whereas", "while", "unfortunately",
     "regrettably"}
)

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic word tokens, keeping simple contractions."""
    return _WORD_RE.findall(text.casefold())


def polarity_scores(text: str) -> tuple[int, int]:
    """Return ``(positive_hits, negative_hits)``."""
    tokens = tokenize(text)
    positive = sum(1 for token in tokens if token in POSITIVE_TERMS)
    negative = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    return positive, negative


def polarity_opinion(text: str) -> str | None:
    """Coarse polarity, or ``None`` when the lexicon cannot decide."""
    positive, negative = polarity_scores(text)
    if positive == negative:
        return None
    return "positive" if positive > negative else "negative"


def count_negation_cues(text: str) -> int:
    return sum(1 for token in tokenize(text) if token in NEGATION_CUES)


def has_contrast_marker(text: str, within_tokens: int | None = None) -> bool:
    """Whether a contrastive marker appears, optionally only in the opening span."""
    tokens = tokenize(text)
    if within_tokens is not None:
        tokens = tokens[:within_tokens]
    return any(token in CONTRAST_CUES for token in tokens)


def lexicon_agreement(pairs: Iterable[tuple[str, str]]) -> dict[str, object]:
    """Agreement between the lexicon's opinion and gold labels.

    ``pairs`` is an iterable of ``(text, gold_label)``.  Rows the lexicon cannot
    decide are reported separately rather than counted as errors, so a low
    ``decided`` share is visible instead of silently deflating agreement.
    """
    decided = correct = undecided = 0
    for text, gold in pairs:
        opinion = polarity_opinion(text)
        if opinion is None:
            undecided += 1
            continue
        decided += 1
        correct += int(opinion == gold)
    total = decided + undecided
    return {
        "total": total,
        "decided": decided,
        "undecided": undecided,
        "correct": correct,
        "agreement_on_decided": (correct / decided) if decided else None,
    }
