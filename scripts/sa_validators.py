"""Pluggable sentiment validators for SA metamorphic-relation groups.

A validator answers one question: *what is the overall polarity of this text?*
The engine asks it twice per flip group -- once about the follow-up and once
about the source -- so a polarity reversal can be told apart from "the model got
the source wrong too".

Two families exist for two different jobs:

``GeneratorSelfValidator``
    The production path.  A blind second pass through the same generator, at
    temperature 0, seeing only the text -- no label, no source text, no MR name.
    This is what makes the final pipeline generator-only.

``LocalClassifierValidator``
    The pilot's measuring instrument.  A separate sentence-classification model,
    used to quantify how far the generator-only gate can be trusted.  It is
    deliberately NOT part of the final pipeline.

``LexiconValidator`` and ``ScriptedValidator`` need neither torch nor a network
and exist for offline tests and import-time sanity checks.

Every validator returns a ``ValidationVerdict``; failures are reported as a
verdict with ``error`` set rather than raised, so one broken call cannot abort a
long pilot run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from generate_generic_llm_aug import GenerationError, extract_json_object

from sa_lexicon import lexicon_agreement, polarity_opinion
from sa_mr_catalog import build_verification_prompt

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
SA_LABELS = (LABEL_NEGATIVE, LABEL_POSITIVE)
OPPOSITE_LABEL = {LABEL_POSITIVE: LABEL_NEGATIVE, LABEL_NEGATIVE: LABEL_POSITIVE}

DEFAULT_CLASSIFIER_ID = "siebert/sentiment-roberta-large-english"
DEFAULT_CLASSIFIER_CONFIDENCE = 0.70

_LABEL_ALIASES = {
    "positive": LABEL_POSITIVE,
    "pos": LABEL_POSITIVE,
    "label_1": LABEL_POSITIVE,
    "negative": LABEL_NEGATIVE,
    "neg": LABEL_NEGATIVE,
    "label_0": LABEL_NEGATIVE,
}

#: Applied only when a model exposes no usable ``id2label``.
DEFAULT_INDEX_LABELS = {0: LABEL_NEGATIVE, 1: LABEL_POSITIVE}


def normalize_sentiment_label(value: Any) -> str | None:
    """Map an upstream label spelling onto ``positive``/``negative``."""
    if value is None:
        return None
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if text in _LABEL_ALIASES:
        return _LABEL_ALIASES[text]
    if text.startswith("positive"):
        return LABEL_POSITIVE
    if text.startswith("negative"):
        return LABEL_NEGATIVE
    return None


@dataclass(frozen=True)
class ValidationVerdict:
    validator_id: str
    predicted_label: str | None = None
    confidence: float | None = None
    unambiguous: bool | None = None
    reason: str = ""
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.predicted_label in SA_LABELS

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "unambiguous": self.unambiguous,
            "reason": self.reason,
            "error": self.error,
        }


@runtime_checkable
class SAValidator(Protocol):
    validator_id: str

    def predict(self, text: str, seed: int) -> ValidationVerdict: ...


class ValidatorError(RuntimeError):
    """Raised when a validator cannot be constructed."""


# --------------------------------------------------------------------------- #
# Blind response parsing
# --------------------------------------------------------------------------- #

def validate_sentiment_verification(response: str, expected_label: str) -> dict[str, Any]:
    """Parse a blind verification response and require the expected polarity."""
    value = extract_json_object(response)
    predicted = normalize_sentiment_label(value.get("predicted_label"))
    if predicted is None:
        raise GenerationError(
            f"verification returned an invalid label: {value.get('predicted_label')!r}"
        )
    unambiguous = value.get("unambiguous")
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    if unambiguous is not True:
        raise GenerationError(f"verification marked the text as ambiguous: {reason[:300]}")
    if predicted != expected_label:
        raise GenerationError(
            f"verification predicted {predicted!r}, expected {expected_label!r}: "
            f"{reason[:300]}"
        )
    return {"predicted_label": predicted, "unambiguous": True, "reason": reason[:500]}


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #

class GeneratorSelfValidator:
    """Blind second pass through the generator itself (the production gate)."""

    def __init__(
        self,
        generator: Any,
        *,
        validator_id: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.generator = generator
        self.temperature = temperature
        self.validator_id = validator_id or f"generator:{getattr(generator, 'model_id', 'self')}"

    def predict(self, text: str, seed: int) -> ValidationVerdict:
        try:
            response = self.generator.generate(build_verification_prompt(text), seed)
        except Exception as exc:  # noqa: BLE001 - one failure must not abort a run
            return ValidationVerdict(
                validator_id=self.validator_id, error=f"{type(exc).__name__}: {exc}"[:300]
            )
        return self._parse(response)

    def predict_many(self, texts: Sequence[str], seed: int) -> list[ValidationVerdict]:
        """Blind-audit several texts, in one generator batch where supported."""
        prompts = [build_verification_prompt(text) for text in texts]
        batch_fn = getattr(self.generator, "generate_batch", None)
        if batch_fn is not None:
            try:
                responses = list(batch_fn(prompts, [seed] * len(prompts)))
            except Exception as exc:  # noqa: BLE001 - report, never abort the run
                message = f"{type(exc).__name__}: {exc}"[:300]
                return [
                    ValidationVerdict(validator_id=self.validator_id, error=message)
                    for _ in prompts
                ]
        else:
            verdicts: list[ValidationVerdict] = []
            for prompt in prompts:
                try:
                    verdicts.append(self._parse(self.generator.generate(prompt, seed)))
                except Exception as exc:  # noqa: BLE001
                    message = f"{type(exc).__name__}: {exc}"[:300]
                    verdicts.append(
                        ValidationVerdict(validator_id=self.validator_id, error=message)
                    )
            return verdicts
        return [self._parse(response) for response in responses]

    def _parse(self, response: str) -> ValidationVerdict:
        try:
            value = extract_json_object(response)
        except GenerationError as exc:
            return ValidationVerdict(validator_id=self.validator_id, error=str(exc)[:300])
        label = normalize_sentiment_label(value.get("predicted_label"))
        unambiguous = value.get("unambiguous")
        reason = value.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        return ValidationVerdict(
            validator_id=self.validator_id,
            predicted_label=label,
            unambiguous=unambiguous if isinstance(unambiguous, bool) else None,
            reason=reason[:300],
            error=None if label else "unparseable predicted_label",
        )


class LocalClassifierValidator:
    """Local sentence-classification model used as the pilot's measuring instrument."""

    def __init__(
        self,
        model_reference: str = DEFAULT_CLASSIFIER_ID,
        *,
        device: str = "auto",
        offline: bool = False,
        confidence_threshold: float = DEFAULT_CLASSIFIER_CONFIDENCE,
        batch_size: int = 16,
        max_tokens: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ValidatorError(
                "LocalClassifierValidator requires PyTorch and Transformers"
            ) from exc

        from project_runtime import apply_offline_mode

        if offline:
            apply_offline_mode(True)
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.model_id = model_reference
        self.validator_id = f"classifier:{model_reference}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_reference, trust_remote_code=trust_remote_code, local_files_only=offline
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_reference, trust_remote_code=trust_remote_code, local_files_only=offline
        ).to(self.device)
        self.model.eval()
        self.index_labels = self._resolve_index_labels()

    def _resolve_index_labels(self) -> dict[int, str]:
        config = getattr(self.model, "config", None)
        raw = getattr(config, "id2label", None) or {}
        resolved: dict[int, str] = {}
        for index, name in raw.items():
            label = normalize_sentiment_label(name)
            if label:
                resolved[int(index)] = label
        if len(resolved) != 2:
            # Models without a usable id2label fall back to the SST-2 convention.
            return dict(DEFAULT_INDEX_LABELS)
        return resolved

    def predict(self, text: str, seed: int) -> ValidationVerdict:
        return self.predict_many([text], seed)[0]

    def predict_many(self, texts: Sequence[str], seed: int) -> list[ValidationVerdict]:
        verdicts: list[ValidationVerdict] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            try:
                verdicts.extend(self._predict_batch(batch))
            except Exception as exc:  # noqa: BLE001 - report, never abort the run
                message = f"{type(exc).__name__}: {exc}"[:300]
                verdicts.extend(
                    ValidationVerdict(validator_id=self.validator_id, error=message)
                    for _ in batch
                )
        return verdicts

    def _predict_batch(self, batch: Sequence[str]) -> list[ValidationVerdict]:
        encoded = self.tokenizer(
            list(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        ).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**encoded).logits
        probabilities = self.torch.softmax(logits.float(), dim=-1).cpu()
        results: list[ValidationVerdict] = []
        for row in probabilities:
            best = int(self.torch.argmax(row).item())
            confidence = float(row[best].item())
            label = self.index_labels.get(best)
            results.append(
                ValidationVerdict(
                    validator_id=self.validator_id,
                    predicted_label=label,
                    confidence=confidence,
                    unambiguous=confidence >= self.confidence_threshold,
                    reason="",
                    error=None if label else f"no label mapping for class index {best}",
                )
            )
        return results


class LexiconValidator:
    """Offline polarity opinion from the shared lexicon; refuses when undecided."""

    def __init__(self, validator_id: str = "lexicon:v1") -> None:
        self.validator_id = validator_id

    def predict(self, text: str, seed: int) -> ValidationVerdict:
        opinion = polarity_opinion(text)
        return ValidationVerdict(
            validator_id=self.validator_id,
            predicted_label=opinion,
            confidence=None,
            unambiguous=opinion is not None,
            reason="" if opinion else "lexicon could not decide",
            error=None if opinion else "lexicon_undecided",
        )

    def agreement(self, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
        return lexicon_agreement(pairs)


class ScriptedValidator:
    """Deterministic validator driven by a lookup table; used by the test suite."""

    def __init__(
        self,
        labels_by_text: Mapping[str, str] | None = None,
        *,
        default: str | None = None,
        validator_id: str = "scripted",
        unambiguous: bool = True,
        error: str | None = None,
    ) -> None:
        self.labels_by_text = dict(labels_by_text or {})
        self.default = default
        self.validator_id = validator_id
        self.unambiguous = unambiguous
        self.forced_error = error

    def predict(self, text: str, seed: int) -> ValidationVerdict:
        if self.forced_error:
            return ValidationVerdict(validator_id=self.validator_id, error=self.forced_error)
        label = self.labels_by_text.get(text, self.default)
        return ValidationVerdict(
            validator_id=self.validator_id,
            predicted_label=label,
            confidence=1.0,
            unambiguous=self.unambiguous,
            reason="scripted",
            error=None if label else "scripted validator has no label for this text",
        )


class FlippingScriptedValidator:
    """Test double that inverts whatever polarity it is shown.

    Lets the suite assert that the source-prior check works without hand-labelling
    every fixture string.
    """

    def __init__(self, base: SAValidator, validator_id: str = "flipping") -> None:
        self.base = base
        self.validator_id = validator_id

    def predict(self, text: str, seed: int) -> ValidationVerdict:
        verdict = self.base.predict(text, seed)
        if not verdict.usable:
            return verdict
        return ValidationVerdict(
            validator_id=self.validator_id,
            predicted_label=OPPOSITE_LABEL[verdict.predicted_label],
            confidence=verdict.confidence,
            unambiguous=verdict.unambiguous,
            reason=verdict.reason,
        )


VALIDATOR_FACTORIES: dict[str, Callable[..., SAValidator]] = {
    "lexicon": LexiconValidator,
}


def build_validator(name: str, **kwargs: Any) -> SAValidator:
    """Construct a validator by short name; generator/classifier need caller context."""
    if name == "generator":
        return GeneratorSelfValidator(kwargs["generator"])
    if name == "classifier":
        return LocalClassifierValidator(
            kwargs.get("model_reference", DEFAULT_CLASSIFIER_ID),
            device=kwargs.get("device", "auto"),
            offline=kwargs.get("offline", False),
            confidence_threshold=kwargs.get(
                "confidence_threshold", DEFAULT_CLASSIFIER_CONFIDENCE
            ),
        )
    if name == "lexicon":
        return LexiconValidator()
    raise ValueError(f"unknown validator {name!r}; choose from generator, classifier, lexicon")
