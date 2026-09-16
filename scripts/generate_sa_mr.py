#!/usr/bin/env python3
"""Generate implicit MR augmentation groups for binary sentiment analysis.

A *group* is one source text plus one accepted MR follow-up.  The follow-up is
produced by the MR named in ``scripts/sa_mr_catalog.py``; seven MRs are LLM
rewrites and ``sa_case_reversal`` is a deterministic case transform.

The generator is never told that it is producing augmentation data: the prompts
describe a single editing task in ordinary language and never mention MRs,
source/follow-up structure, or preserve/flip relations.  The MR identity lives in
the row metadata only, which is what makes the downstream training condition
*implicit* rather than instruction-conditioned.

Acceptance is a two-stage gate:

1. **Structural** -- JSON contract, length bounds, the text must actually have
   changed, similarity and length-ratio bands, and no label-word or meta
   commentary leaking into the text.
2. **Relation** -- at least one configured validator must agree that the
   follow-up has the expected polarity.  For ``flip`` MRs the same validator must
   *also* have judged the source correctly, otherwise "both texts read negative"
   would be accepted as a successful reversal of a positive source.

Validators are pluggable (``scripts/sa_validators.py``): ``generator`` is the
blind self-verification pass that the final pipeline uses, and ``classifier`` is
the pilot-only diagnostic used to measure how far that self-check can be trusted.

Examples:
  # Inspect sampling and the exact prompts; loads no model and writes nothing
  python scripts/generate_sa_mr.py --dataset imdb \\
    --input data/sa/original_dataset/imdb/mr_test_source_pool.jsonl \\
    --mrs all --num-samples 2 --dry-run

  # Real smoke run on one MR
  python scripts/generate_sa_mr.py --dataset imdb \\
    --input data/sa/original_dataset/imdb/mr_test_source_pool.jsonl \\
    --mr sa_tense_shift --num-samples 6 --seed 42 --offline \\
    --output data/sa/MR_testing/_smoke/imdb/tense_shift.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from project_runtime import PROJECT_ROOT, configure_console_encoding
from generate_generic_llm_aug import (
    GenerationError,
    HFTextGenerator,
    canonical_sha256,
    extract_json_object,
    file_sha256,
    load_records,
    normalized_text,
    similarity,
    utc_now,
)

import sa_mr_catalog as catalog
import sa_validators as validators_module
from sa_validators import SAValidator, ValidationVerdict


METHOD_NAME = "SA-MR-Gen"
DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "models" / "google" / "gemma-3-4b-it"
RESEARCH_ROLE = (
    "Implicit MR augmentation for sentiment analysis: source and follow-up enter "
    "training as ordinary classification samples with no MR, operation, pairing or "
    "relation information (mr_instruction_mode=none)."
)
SCHEMA_VERSION = 1

TASK_NAME = "sentiment_analysis"
GROUP_SUFFIX = "v1"

SA_LABELS = ("negative", "positive")
OPPOSITE_LABEL = {"positive": "negative", "negative": "positive"}

#: Terms the model must not inject into the edited text.  Only flagged when the
#: source did not already contain them, so ordinary review wording is not punished.
_LABEL_LEAK_RE = re.compile(
    r"\b(positive|negative|sentiment|polarity|label|classification)\b", re.IGNORECASE
)
_META_LEAK_RE = re.compile(
    r"\b(metamorphic|transformation|transformed|rewritten|rewrite|paraphrase[d]?|"
    r"source text|original text|follow-?up|augmentation|as an ai|i cannot|"
    r"i'm sorry|here is the|here's the)\b",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```")


class SAMRGenerationError(GenerationError):
    """A candidate follow-up failed a structural or format check."""


class MRDeclined(GenerationError):
    """The generator reported that the edit is not possible for this source."""


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MRGenTask:
    output_index: int
    source_index: int
    mr_id: str
    dataset: str
    source: Mapping[str, Any]

    @property
    def source_id(self) -> str:
        return str(self.source["source_id"])

    @property
    def group_id(self) -> str:
        return f"{self.source_id}::{self.mr_id}::{GROUP_SUFFIX}"

    @property
    def source_text(self) -> str:
        return str(self.source["text"])

    @property
    def source_label(self) -> str:
        return str(self.source["label"])


def validate_source(row: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    missing = [field for field in ("source_id", "text", "label") if field not in row]
    if missing:
        raise ValueError(f"source row {source_index} is missing {', '.join(missing)}")
    text = str(row["text"]).strip()
    if not text:
        raise ValueError(f"source row {source_index} has empty text")
    label = str(row["label"]).strip().casefold()
    if label not in SA_LABELS:
        raise ValueError(f"source row {source_index} has label outside {SA_LABELS}: {label!r}")
    return {**row, "text": text, "label": label}


def sample_sources(
    records: Sequence[Mapping[str, Any]], num_samples: int | None, seed: int, label_strategy: str
) -> list[dict[str, Any]]:
    """Deterministically sample sources, optionally exactly label-balanced."""
    if num_samples is None:
        num_samples = len(records)
    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")
    if num_samples > len(records):
        raise ValueError(
            f"requested {num_samples} sources but the input has only {len(records)}; "
            "sampling is intentionally without replacement"
        )

    rng = random.Random(seed)
    if label_strategy == "source":
        indices = sorted(rng.sample(range(len(records)), num_samples))
    elif label_strategy == "balanced":
        by_label: dict[str, list[int]] = {label: [] for label in SA_LABELS}
        for index, row in enumerate(records):
            label = str(row.get("label", "")).strip().casefold()
            if label not in by_label:
                raise ValueError(f"source row {index} has label outside {SA_LABELS}: {label!r}")
            by_label[label].append(index)
        for bucket in by_label.values():
            rng.shuffle(bucket)
        quota = num_samples // len(SA_LABELS)
        remainder = num_samples - quota * len(SA_LABELS)
        chosen: list[int] = []
        for position, label in enumerate(SA_LABELS):
            wanted = quota + (1 if position < remainder else 0)
            if wanted > len(by_label[label]):
                raise ValueError(
                    f"cannot sample {wanted} {label} rows; only {len(by_label[label])} available"
                )
            chosen.extend(by_label[label][:wanted])
        indices = sorted(chosen)
    else:
        raise ValueError(f"unknown label strategy: {label_strategy}")

    originals = [validate_source(records[index], index) for index in indices]
    # Keep the pool order stable so every MR sees the same sources in the same order.
    return [
        {**row, "_source_index": index} for row, index in zip(originals, indices)
    ]


def select_tasks(
    sources: Sequence[Mapping[str, Any]],
    mr_ids: Sequence[str],
    dataset: str,
    assignment: str,
) -> list[MRGenTask]:
    """Build the (source, MR) task list.

    ``all`` gives every source every MR -- the pilot's design, so acceptance rates
    share one denominator.  ``round_robin`` spreads MRs across sources, which
    matches the eventual training-data shape.
    """
    if not mr_ids:
        raise ValueError("at least one MR is required")
    for mr_id in mr_ids:
        catalog.get_mr(mr_id)
    if assignment not in ("all", "round_robin"):
        raise ValueError(f"unknown MR assignment: {assignment}")

    tasks: list[MRGenTask] = []
    output_index = 0
    if assignment == "all":
        for mr_id in mr_ids:
            for source in sources:
                tasks.append(
                    MRGenTask(
                        output_index=output_index,
                        source_index=int(source["_source_index"]),
                        mr_id=mr_id,
                        dataset=dataset,
                        source=source,
                    )
                )
                output_index += 1
    else:
        for position, source in enumerate(sources):
            mr_id = mr_ids[position % len(mr_ids)]
            tasks.append(
                MRGenTask(
                    output_index=output_index,
                    source_index=int(source["_source_index"]),
                    mr_id=mr_id,
                    dataset=dataset,
                    source=source,
                )
            )
            output_index += 1
    return tasks


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #

def _seed_from(*parts: Any) -> int:
    digest = hashlib.sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def mr_task_seed(seed: int, task: MRGenTask, attempt: int) -> int:
    return _seed_from("gen", seed, task.source_id, task.mr_id, attempt)


def verification_seed(seed: int, task: MRGenTask, attempt: int, validator_id: str) -> int:
    return _seed_from("verify", validator_id, seed, task.group_id, attempt)


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #

def build_followup_text(raw_text: str) -> str:
    return " ".join(raw_text.split())


def run_structural_checks(
    source_text: str,
    followup_text: str,
    definition: catalog.MRDefinition,
    *,
    max_chars: int,
) -> dict[str, Any]:
    """Validate one candidate follow-up; raise ``SAMRGenerationError`` on failure."""
    if not followup_text:
        raise SAMRGenerationError("text_empty")
    if len(followup_text) < definition.min_chars:
        raise SAMRGenerationError("text_too_short")
    if len(followup_text) > max_chars:
        raise SAMRGenerationError("text_too_long")
    if _CODE_FENCE_RE.search(followup_text):
        raise SAMRGenerationError("meta_leak_code_fence")

    source_key = normalized_text(source_text)
    followup_key = normalized_text(followup_text)

    if definition.is_deterministic:
        # Case reversal is length- and word-preserving by construction, so the
        # usual "must have changed" gate is inverted: the raw text must differ
        # while the normalised text must be identical.
        if definition.deterministic_fn is not None and followup_text == source_text:
            raise SAMRGenerationError("unchanged_text")
        if followup_key != source_key:
            raise SAMRGenerationError("case_transform_changed_words")
    else:
        if followup_key == source_key:
            raise SAMRGenerationError("unchanged_text")
        score = similarity(source_text, followup_text)
        if score < definition.min_source_similarity:
            raise SAMRGenerationError(
                f"similarity_below_min:{score:.3f}<{definition.min_source_similarity:.3f}"
            )
        if score > definition.max_source_similarity:
            raise SAMRGenerationError(
                f"similarity_above_max:{score:.3f}>{definition.max_source_similarity:.3f}"
            )

    if definition.length_ratio_bounds is not None:
        low, high = definition.length_ratio_bounds
        ratio = len(followup_text) / max(len(source_text), 1)
        if not low <= ratio <= high:
            raise SAMRGenerationError(f"length_ratio_out_of_bounds:{ratio:.2f}")

    source_leaks = {match.group(0).casefold() for match in _LABEL_LEAK_RE.finditer(source_text)}
    followup_leaks = {
        match.group(0).casefold() for match in _LABEL_LEAK_RE.finditer(followup_text)
    }
    introduced = followup_leaks - source_leaks
    if introduced:
        raise SAMRGenerationError(f"label_word_leak:{sorted(introduced)[0]}")

    meta_match = _META_LEAK_RE.search(followup_text)
    if meta_match:
        raise SAMRGenerationError(f"meta_leak:{meta_match.group(0).casefold()}")

    return {
        "text_nonempty": True,
        "within_char_bounds": True,
        "diff_nonempty": True,
        "similarity": None if definition.is_deterministic else round(similarity(source_text, followup_text), 4),
        "length_ratio": round(len(followup_text) / max(len(source_text), 1), 4),
        "label_word_leak": False,
        "meta_leak": False,
    }


def validate_mr_output(
    response: str, task: MRGenTask, definition: catalog.MRDefinition
) -> dict[str, Any]:
    """Parse a generator response into ``{text, followup_label}``."""
    value = extract_json_object(response)
    if "text" not in value:
        raise SAMRGenerationError("missing_field:text")
    raw_text = value["text"]
    if raw_text is None:
        raise MRDeclined("generator_declined")
    if not isinstance(raw_text, str):
        raise SAMRGenerationError("text_not_string")

    followup_label = None
    if definition.relation_type == catalog.RELATION_FLIP:
        if "edited_label" not in value:
            raise SAMRGenerationError("missing_field:edited_label")
        declared = str(value["edited_label"]).strip().casefold()
        if declared not in SA_LABELS:
            raise SAMRGenerationError("edited_label_invalid")
        expected = catalog.expected_followup_label(task.mr_id, task.source_label)
        if declared != expected:
            raise SAMRGenerationError(f"edited_label_mismatch:{declared}!={expected}")
        followup_label = declared

    return {"text": build_followup_text(raw_text), "followup_label": followup_label}


# --------------------------------------------------------------------------- #
# Relation gate
# --------------------------------------------------------------------------- #

GATE_MODES = ("generator", "classifier", "both", "any")


def gate_validator_names(mode: str) -> tuple[list[str], bool]:
    """Return ``(validator names, require_all)`` for a gate mode."""
    if mode == "generator":
        return ["generator"], True
    if mode == "classifier":
        return ["classifier"], True
    if mode == "both":
        return ["generator", "classifier"], True
    if mode == "any":
        return ["generator", "classifier"], False
    raise ValueError(f"unknown gate mode {mode!r}; choose from {', '.join(GATE_MODES)}")


def _validator_passes(
    name: str,
    source_verdict: ValidationVerdict | None,
    followup_verdict: ValidationVerdict | None,
    definition: catalog.MRDefinition,
    source_label: str,
    expected_label: str,
) -> tuple[bool, str]:
    if followup_verdict is None:
        return False, f"{name}_not_run"
    if followup_verdict.error is not None or not followup_verdict.usable:
        return False, f"{name}_followup_unusable"
    if followup_verdict.unambiguous is False:
        return False, f"{name}_followup_ambiguous"

    if definition.relation_type == catalog.RELATION_FLIP:
        # The load-bearing source-prior check: without it, "source and follow-up
        # both read negative" would count as a successful flip of a positive source.
        if source_verdict is None or not source_verdict.usable:
            return False, f"{name}_source_unusable"
        if source_verdict.predicted_label != source_label:
            return False, f"{name}_source_prior_mismatch"
        if followup_verdict.predicted_label != expected_label:
            return False, f"{name}_flip_not_achieved"
    else:
        if followup_verdict.predicted_label != source_label:
            return False, f"{name}_polarity_not_preserved"
    return True, ""


def evaluate_relation(
    *,
    definition: catalog.MRDefinition,
    source_label: str,
    verdicts: Mapping[str, Mapping[str, ValidationVerdict]],
    gate_mode: str,
) -> dict[str, Any]:
    expected_label = catalog.expected_followup_label(definition.mr_id, source_label)
    names, require_all = gate_validator_names(gate_mode)
    results: dict[str, Any] = {}
    outcomes: list[bool] = []
    reasons: list[str] = []

    for name in names:
        pair = verdicts.get(name)
        if pair is None:
            results[name] = {"ran": False, "pass": False, "reason": f"{name}_not_configured"}
            outcomes.append(False)
            reasons.append(f"{name}_not_configured")
            continue
        passed, reason = _validator_passes(
            name,
            pair.get("source"),
            pair.get("followup"),
            definition,
            source_label,
            expected_label,
        )
        outcomes.append(passed)
        if not passed:
            reasons.append(reason)
        results[name] = {
            "ran": True,
            "pass": passed,
            "reason": reason,
            "source_predicted": (pair.get("source").predicted_label if pair.get("source") else None),
            "followup_predicted": (
                pair.get("followup").predicted_label if pair.get("followup") else None
            ),
            "followup_unambiguous": (
                pair.get("followup").unambiguous if pair.get("followup") else None
            ),
            "followup_confidence": (
                pair.get("followup").confidence if pair.get("followup") else None
            ),
        }

    passed = all(outcomes) if require_all else any(outcomes)
    return {
        "gate_mode": gate_mode,
        "relation_type": definition.relation_type,
        "expected_followup_label": expected_label,
        "source_label": source_label,
        "passed": passed,
        "validators": results,
        "validators_agree": len(set(outcomes)) == 1 if len(outcomes) > 1 else None,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

def _make_row(
    *,
    task: MRGenTask,
    definition: catalog.MRDefinition,
    sample_suffix: str,
    is_source: bool,
    text: str,
    label: str,
    generator_id: str | None,
    generation_seed: int | None,
    automatic_checks: Mapping[str, Any],
) -> dict[str, Any]:
    group_id = task.group_id
    row = {
        "task": TASK_NAME,
        "dataset": task.dataset,
        "split": "mr_test",
        "group_id": group_id,
        "sample_id": f"{group_id}::{sample_suffix}",
        "source_id": task.source_id,
        "is_source": is_source,
        "text": text,
        "label": label,
        "source_text": task.source_text,
        "source_label": task.source_label,
        "mr_id": None if is_source else task.mr_id,
        "property_family": definition.property_family,
        "relation_type": definition.relation_type,
        "template_id": None if is_source else definition.template_id,
        "generator_id": generator_id,
        "prompt_version": definition.prompt_version,
        "generation_seed": generation_seed,
        "automatic_checks": dict(automatic_checks),
        "accepted": True,
        # Bridge keys so scripts/metamorphic_metrics.py works unchanged.
        "pair_id": group_id,
        "mr_type": definition.mr_type,
        "pair": f"{task.dataset}:{task.source_id}",
    }
    row["provenance_hash"] = canonical_sha256(
        {
            "group_id": group_id,
            "sample_id": row["sample_id"],
            "text": text,
            "label": label,
            "prompt_version": definition.prompt_version,
            "template_id": row["template_id"],
        }
    )
    return row


def make_group_rows(
    task: MRGenTask,
    definition: catalog.MRDefinition,
    followup_text: str,
    structural: Mapping[str, Any],
    relation: Mapping[str, Any],
    *,
    generator_id: str | None,
    generation_seed: int | None,
    attempts: int,
) -> list[dict[str, Any]]:
    checks = {
        "structural": dict(structural),
        "relation_verdict": dict(relation),
        "attempts": attempts,
    }
    source_row = _make_row(
        task=task,
        definition=definition,
        sample_suffix="source",
        is_source=True,
        text=task.source_text,
        label=task.source_label,
        generator_id=None,
        generation_seed=None,
        automatic_checks={},
    )
    followup_row = _make_row(
        task=task,
        definition=definition,
        sample_suffix="followup",
        is_source=False,
        text=followup_text,
        label=catalog.expected_followup_label(task.mr_id, task.source_label),
        generator_id=generator_id,
        generation_seed=generation_seed,
        automatic_checks=checks,
    )
    return [source_row, followup_row]


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #

def read_existing_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, row in enumerate(load_records(path), start=1):
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"cannot resume: output row {line_number} has no group_id")
        by_group[group_id].append(row)
    for group_id, rows in by_group.items():
        sources = sum(1 for row in rows if row.get("is_source"))
        followups = len(rows) - sources
        if sources != 1 or followups != 1:
            raise ValueError(
                f"cannot resume: group {group_id!r} has {sources} source and "
                f"{followups} follow-up rows; expected exactly one of each"
            )
    return dict(by_group)


def validate_resume_groups(
    existing: Mapping[str, Sequence[Mapping[str, Any]]], tasks: Iterable[MRGenTask]
) -> None:
    task_map = {task.group_id: task for task in tasks}
    unexpected = sorted(set(existing) - set(task_map))
    if unexpected:
        raise ValueError(f"cannot resume: output has unexpected groups, e.g. {unexpected[:3]}")
    for group_id, rows in existing.items():
        task = task_map[group_id]
        for row in rows:
            if row.get("mr_id") not in (None, task.mr_id):
                raise ValueError(f"cannot resume: mr_id mismatch in group {group_id}")
            if str(row.get("source_id")) != task.source_id:
                raise ValueError(f"cannot resume: source_id mismatch in group {group_id}")


# --------------------------------------------------------------------------- #
# Model references
# --------------------------------------------------------------------------- #

def resolve_model_references(args: argparse.Namespace) -> tuple[str, str]:
    if args.model_path:
        model_reference = str(Path(args.model_path).expanduser().resolve())
    elif args.model == DEFAULT_MODEL and DEFAULT_LOCAL_MODEL.is_dir():
        model_reference = str(DEFAULT_LOCAL_MODEL.resolve())
    else:
        model_reference = args.model
    tokenizer_reference = (
        str(Path(args.tokenizer_path).expanduser().resolve())
        if args.tokenizer_path
        else model_reference
    )
    return model_reference, tokenizer_reference


# --------------------------------------------------------------------------- #
# Batched generation and verification
# --------------------------------------------------------------------------- #

class BatchedGenerator:
    """Generate several prompts in one forward pass.

    ``HFTextGenerator.generate`` runs one sequence at a time, which leaves the GPU
    idle between calls and dominates pilot wall-clock.  This wrapper reuses the
    already-loaded model and tokenizer and pads a whole batch of chat prompts so a
    single decode loop serves many groups.

    A batched sample uses the first task's seed for the entire batch.  This does
    not affect verification, which runs greedily, so only the MR rewrite itself
    loses a little per-task seed independence; the run report records that
    batching was on.
    """

    def __init__(self, inner: Any, batch_size: int = 8) -> None:
        self.inner = inner
        self.batch_size = max(1, int(batch_size))

    # Forward the sampling knobs so TemperatureOverrideGenerator can drive these.
    @property
    def temperature(self) -> float:
        return getattr(self.inner, "temperature", 0.0)

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.inner.temperature = value

    @property
    def top_p(self) -> float:
        return getattr(self.inner, "top_p", 1.0)

    @top_p.setter
    def top_p(self, value: float) -> None:
        self.inner.top_p = value

    def generate(self, messages: list[dict[str, str]], seed: int) -> str:
        return self.inner.generate(messages, seed)

    def generate_batch(
        self, messages_list: Sequence[list[dict[str, str]]], seeds: Sequence[int]
    ) -> list[str]:
        results: list[str] = []
        for start in range(0, len(messages_list), self.batch_size):
            chunk = list(messages_list[start : start + self.batch_size])
            chunk_seeds = list(seeds[start : start + self.batch_size])
            if not chunk:
                continue
            results.extend(self._forward(chunk, chunk_seeds))
        return results

    def _forward(
        self, chunk: Sequence[list[dict[str, str]]], seeds: Sequence[int]
    ) -> list[str]:
        tokenizer = self.inner.tokenizer
        model = self.inner.model
        torch = self.inner.torch
        device = self.inner.device

        # Left padding is required so every row's generation starts at the same
        # position; right padding would make the model continue the pad tokens.
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer.apply_chat_template(
                chunk,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )
        finally:
            tokenizer.padding_side = previous_side
        encoded = encoded.to(device)

        torch.manual_seed(seeds[0])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seeds[0])

        temperature = getattr(self.inner, "temperature", 0.0)
        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.inner.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=temperature, top_p=self.inner.top_p)
        with torch.inference_mode():
            outputs = model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[-1]
        return [
            tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip()
            for row in outputs
        ]


def generate_batch(
    generator: Any,
    prompts: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
) -> list[str]:
    """Batch when the backend supports it, otherwise fall back to one at a time."""
    batch_fn = getattr(generator, "generate_batch", None)
    if batch_fn is not None:
        return list(batch_fn(list(prompts), list(seeds)))
    return [generator.generate(prompt, seed) for prompt, seed in zip(prompts, seeds)]


def predict_many(validator: Any, texts: Sequence[str], seed: int) -> list[ValidationVerdict]:
    """Batch a validator when it can, otherwise fall back to one at a time."""
    many = getattr(validator, "predict_many", None)
    if many is not None:
        return list(many(list(texts), seed))
    return [validator.predict(text, seed) for text in texts]


def collect_verdicts_batched(
    validators_map: Mapping[str, SAValidator],
    requests: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, dict[str, ValidationVerdict]]]:
    """Resolve every validator probe for a round of candidate follow-ups.

    Each request is ``{"group_id", "definition", "source_text", "followup_text"}``.
    Probes are de-duplicated by ``(validator, text)`` before dispatch, which matters
    because every flip group re-checks its source and those sources recur across
    MRs.  For flip relations each validator judges both texts, so a polarity
    reversal can be told apart from "the source was misread too".
    """
    slots: list[tuple[str, str]] = []
    slot_index: dict[tuple[str, str], int] = {}
    pending_probes: list[tuple[str, str, str, int]] = []

    for request in requests:
        definition = request["definition"]
        roles = [("followup", str(request["followup_text"]))]
        if definition.relation_type == catalog.RELATION_FLIP:
            roles.append(("source", str(request["source_text"])))
        for name in validators_map:
            for role, text in roles:
                key = (name, text)
                if key not in slot_index:
                    slot_index[key] = len(slots)
                    slots.append(key)
                pending_probes.append((str(request["group_id"]), name, role, slot_index[key]))

    resolved: dict[int, ValidationVerdict] = {}
    for name, validator in validators_map.items():
        indices = [index for index, (owner, _) in enumerate(slots) if owner == name]
        if not indices:
            continue
        texts = [slots[index][1] for index in indices]
        verdicts = predict_many(validator, texts, seed)
        for index, verdict in zip(indices, verdicts):
            resolved[index] = verdict

    grouped: dict[str, dict[str, dict[str, ValidationVerdict]]] = {}
    for group_id, name, role, index in pending_probes:
        grouped.setdefault(group_id, {}).setdefault(name, {})[role] = resolved[index]
    return grouped


class TemperatureOverrideGenerator:
    """Reuse one loaded model while forcing a sampling temperature per call.

    The blind self-verification pass must run greedily.  Building a second model
    instance just to change one generation parameter would double GPU memory, so
    this wrapper flips the temperature around each call instead.
    """

    def __init__(self, inner: Any, *, temperature: float, top_p: float = 1.0) -> None:
        self.inner = inner
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, messages: list[dict[str, str]], seed: int) -> str:
        sentinel = object()
        previous_temperature = getattr(self.inner, "temperature", sentinel)
        previous_top_p = getattr(self.inner, "top_p", sentinel)
        self.inner.temperature = self.temperature
        self.inner.top_p = self.top_p
        try:
            return self.inner.generate(messages, seed)
        finally:
            if previous_temperature is not sentinel:
                self.inner.temperature = previous_temperature
            if previous_top_p is not sentinel:
                self.inner.top_p = previous_top_p


def build_validators(
    args: argparse.Namespace,
    generator: Any,
    model_reference: str,
) -> dict[str, SAValidator]:
    names = [name.strip() for name in args.validator_mode.split(",") if name.strip()]
    if not names:
        raise ValueError("--validator-mode cannot be empty")
    built: dict[str, SAValidator] = {}
    for name in names:
        if name == "generator":
            built["generator"] = validators_module.GeneratorSelfValidator(
                TemperatureOverrideGenerator(
                    generator, temperature=args.verifier_temperature, top_p=1.0
                )
            )
        elif name == "classifier":
            built["classifier"] = validators_module.LocalClassifierValidator(
                args.classifier_model,
                device=args.classifier_device,
                offline=args.offline,
                confidence_threshold=args.classifier_confidence,
            )
        elif name == "lexicon":
            built["lexicon"] = validators_module.LexiconValidator()
        else:
            raise ValueError(f"unknown validator {name!r}")
    for mode in (args.preserve_gate, args.flip_gate):
        required, _ = gate_validator_names(mode)
        missing = [name for name in required if name not in built]
        if missing:
            raise ValueError(
                f"gate mode {mode!r} needs validator(s) {missing} "
                f"but --validator-mode is {args.validator_mode!r}"
            )
    return built


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_mr_report(
    path: Path,
    *,
    args: argparse.Namespace,
    tasks: Sequence[MRGenTask],
    accepted_group_ids: set[str],
    rejected_reasons: Counter,
    skipped_reasons: Counter,
    failures: Sequence[Mapping[str, Any]],
    started_at: str,
    model_reference: str,
    input_path: Path,
    output_path: Path,
    validator_ids: Mapping[str, str],
) -> None:
    per_mr: dict[str, Any] = {}
    for mr_id in dict.fromkeys(task.mr_id for task in tasks):
        definition = catalog.get_mr(mr_id)
        mr_tasks = [task for task in tasks if task.mr_id == mr_id]
        accepted = sum(1 for task in mr_tasks if task.group_id in accepted_group_ids)
        per_mr[mr_id] = {
            "property_family": definition.property_family,
            "relation_type": definition.relation_type,
            "mechanism": definition.mechanism,
            "template_id": definition.template_id,
            "requested_groups": len(mr_tasks),
            "accepted_groups": accepted,
            "accepted_rate": (accepted / len(mr_tasks)) if mr_tasks else None,
        }
    accepted_groups = len(accepted_group_ids)

    report = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD_NAME,
        "research_role": RESEARCH_ROLE,
        "catalog_version": catalog.MR_CATALOG_VERSION,
        "catalog_sha256": catalog.catalog_sha256(),
        "prompt_version": catalog.PROMPT_VERSION,
        "catalog_orientation": catalog.matrix_orientation(),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path) if output_path.is_file() else None,
        "requested_model": args.model,
        "resolved_model_reference": model_reference,
        "generator_id": model_reference,
        "validator_ids": dict(validator_ids),
        "validator_mode": args.validator_mode,
        "preserve_gate": args.preserve_gate,
        "flip_gate": args.flip_gate,
        "seed": args.seed,
        "label_strategy": args.label_strategy,
        "mr_assignment": args.mr_assignment,
        "requested_groups": len(tasks),
        "accepted_groups": accepted_groups,
        "complete": accepted_groups == len(tasks),
        "per_mr": per_mr,
        "generation_parameters": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_retries": args.max_retries,
            "offline": args.offline,
            "flip_max_words": args.flip_max_words,
            "max_chars": args.max_chars,
            "batch_size": args.batch_size,
            "verifier_temperature": args.verifier_temperature,
        },
        "automatic_checks": [
            "valid JSON object with the declared fields",
            "text is non-empty and inside the character bounds",
            "the follow-up actually differs from the source",
            "similarity and length-ratio bands",
            "no label-word or meta commentary introduced",
            "blind polarity agreement by the configured validators",
            "flip MRs additionally require the source prior to be recovered",
        ],
        "rejected_attempts_by_reason": dict(rejected_reasons.most_common()),
        "skipped_by_reason": dict(skipped_reasons.most_common()),
        "failed_groups": list(failures),
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate implicit MR augmentation groups for sentiment analysis"
    )
    parser.add_argument("--input", required=True, help="Canonical SA source JSONL")
    parser.add_argument("--dataset", required=True, help="Domain name, e.g. imdb or sst2")
    parser.add_argument("--output", help="Accepted groups JSONL (required unless --dry-run)")
    parser.add_argument("--report-output", help="Report path (default: <output>.report.json)")
    parser.add_argument("--mr", help="A single MR id")
    parser.add_argument(
        "--mrs", nargs="+", default=None, help="MR ids, or 'all' (default: all)"
    )
    parser.add_argument(
        "--mr-assignment",
        choices=("all", "round_robin"),
        default="all",
        help="Give every source every MR (default), or spread MRs across sources",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Sources to use (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-strategy", choices=("source", "balanced"), default="balanced"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--verifier-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the blind self-verification pass (greedy by default)",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Sequences per forward pass. 1 disables batching. Batching typically "
            "gives several times the throughput of one-at-a-time generation."
        ),
    )
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--flip-max-words", type=int, default=catalog.DEFAULT_FLIP_MAX_WORDS)
    parser.add_argument(
        "--validator-mode",
        default="generator",
        help="Comma-separated validators: generator, classifier, lexicon",
    )
    parser.add_argument("--classifier-model", default=validators_module.DEFAULT_CLASSIFIER_ID)
    parser.add_argument("--classifier-device", default="auto")
    parser.add_argument(
        "--classifier-confidence", type=float, default=validators_module.DEFAULT_CLASSIFIER_CONFIDENCE
    )
    parser.add_argument("--preserve-gate", choices=GATE_MODES, default="generator")
    parser.add_argument("--flip-gate", choices=GATE_MODES, default="generator")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preview tasks/prompts; write nothing")
    parser.add_argument("--preview-count", type=int, default=2)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.output:
        raise ValueError("--output is required unless --dry-run is used")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.preview_count < 0:
        raise ValueError("--preview-count cannot be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    return args


def resolve_mr_ids(args: argparse.Namespace) -> list[str]:
    if args.mr and args.mrs:
        raise ValueError("--mr and --mrs are mutually exclusive")
    if args.mr:
        catalog.get_mr(args.mr)
        return [args.mr]
    if not args.mrs or args.mrs == ["all"]:
        return list(catalog.MR_ORDER)
    for mr_id in args.mrs:
        catalog.get_mr(mr_id)
    return list(dict.fromkeys(args.mrs))


def print_dry_run(tasks: Sequence[MRGenTask], args: argparse.Namespace) -> None:
    print(f"Method: {METHOD_NAME}")
    print(f"Catalog: {catalog.MR_CATALOG_VERSION} sha256={catalog.catalog_sha256()[:16]}…")
    print(f"Requested groups: {len(tasks)}")
    print(f"Seed: {args.seed}  label strategy: {args.label_strategy}")
    print(f"MR assignment: {args.mr_assignment}")
    print(f"Distribution: {dict(Counter(task.mr_id for task in tasks))}")
    previewed = 0
    for task in tasks:
        if previewed >= args.preview_count:
            break
        definition = catalog.get_mr(task.mr_id)
        ok, reason = catalog.mr_applicable(
            task.mr_id, task.source_text, {"flip_max_words": args.flip_max_words}
        )
        print(
            f"\n--- {task.group_id} ({definition.mechanism}, {definition.relation_type}) "
            f"applicable={ok}{'' if ok else f' reason={reason}'} ---"
        )
        if definition.is_deterministic:
            print(f"[deterministic] {definition.operation}")
            continue
        for message in catalog.build_mr_prompt(task.mr_id, task.source_text, task.source_label):
            print(f"[{message['role']}]\n{message['content']}")
        previewed += 1


def main(
    argv: list[str] | None = None,
    generator_factory: Callable[..., Any] | None = None,
    validator_factory: Callable[..., Mapping[str, SAValidator]] | None = None,
) -> int:
    configure_console_encoding()
    try:
        args = parse_args(argv)
        input_path = Path(args.input).expanduser().resolve()
        records = load_records(input_path)
        mr_ids = resolve_mr_ids(args)
        sources = sample_sources(records, args.num_samples, args.seed, args.label_strategy)
        tasks = select_tasks(sources, mr_ids, args.dataset, args.mr_assignment)

        if args.dry_run:
            print_dry_run(tasks, args)
            return 0

        output_path = Path(args.output).expanduser().resolve()
        report_path = (
            Path(args.report_output).expanduser().resolve()
            if args.report_output
            else output_path.with_suffix(output_path.suffix + ".report.json")
        )
        rejects_path = output_path.with_suffix(output_path.suffix + ".rejects.jsonl")

        if output_path.exists() and not args.resume and not args.overwrite:
            raise FileExistsError(
                f"output already exists: {output_path}. Use --resume or --overwrite."
            )
        if args.overwrite:
            for path in (output_path, rejects_path):
                if path.exists():
                    path.unlink()

        existing = read_existing_groups(output_path) if args.resume else {}
        validate_resume_groups(existing, tasks)

        model_reference, tokenizer_reference = resolve_model_references(args)
        factory = generator_factory or HFTextGenerator
        generator = factory(
            model_reference=model_reference,
            tokenizer_reference=tokenizer_reference,
            device=args.device,
            dtype=args.dtype,
            offline=args.offline,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # Wrap only real Transformers generators: injected test doubles have no
        # tokenizer/model, and the batch helpers fall back to one-at-a-time for them.
        batching_enabled = args.batch_size > 1 and hasattr(generator, "tokenizer") and hasattr(
            generator, "model"
        )
        if batching_enabled:
            generator = BatchedGenerator(generator, batch_size=args.batch_size)
        if validator_factory is not None:
            builders = validator_factory(args, generator, model_reference)
        else:
            builders = build_validators(args, generator, model_reference)
        validator_ids = {name: getattr(v, "validator_id", name) for name, v in builders.items()}

        started_at = utc_now()
        rejected_reasons: Counter = Counter()
        skipped_reasons: Counter = Counter()
        failures: list[dict[str, Any]] = []
        pending = [task for task in tasks if task.group_id not in existing]
        accepted_group_ids: set[str] = set(existing)
        print(
            f"{METHOD_NAME}: {len(tasks)} groups requested, {len(accepted_group_ids)} existing, "
            f"{len(pending)} pending; model={args.model}; validators={sorted(builders)}"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8", newline="\n") as output_handle, \
                rejects_path.open("a" if args.resume else "w", encoding="utf-8", newline="\n") as reject_handle:

            def emit_reject(payload: dict[str, Any]) -> None:
                reject_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                reject_handle.flush()

            def emit_rows(rows: Sequence[Mapping[str, Any]]) -> None:
                for row in rows:
                    output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()

            def report_accept(task: MRGenTask, definition: catalog.MRDefinition) -> None:
                accepted_group_ids.add(task.group_id)
                print(
                    f"  [{len(accepted_group_ids)}/{len(tasks)}] {task.group_id} "
                    f"{task.source_label} -> "
                    f"{catalog.expected_followup_label(task.mr_id, task.source_label)}"
                )

            def finalize_unresolved(pairs: Sequence[tuple[MRGenTask, catalog.MRDefinition]]) -> None:
                """Record groups that never produced an accepted follow-up.

                A group whose every attempt was a decline is a statement about
                applicability, not a generation failure, so it is skipped rather
                than failed.
                """
                for task, _definition in pairs:
                    history = outcomes.get(task.group_id, [])
                    reason = feedback.get(task.group_id)
                    verdict_payload = dict(last_checks.get(task.group_id) or {})
                    if history and all(outcome == "declined" for outcome in history):
                        skipped_reasons["generator_declined"] += 1
                        emit_reject(
                            {
                                "group_id": task.group_id,
                                "dataset": task.dataset,
                                "mr_id": task.mr_id,
                                "source_id": task.source_id,
                                "source_label": task.source_label,
                                "stage": "generator_declined",
                                "reason": reason,
                                **verdict_payload,
                            }
                        )
                        continue
                    failures.append(
                        {
                            "group_id": task.group_id,
                            "dataset": task.dataset,
                            "mr_id": task.mr_id,
                            "reason": reason,
                        }
                    )
                    emit_reject(
                        {
                            "group_id": task.group_id,
                            "dataset": task.dataset,
                            "mr_id": task.mr_id,
                            "source_id": task.source_id,
                            "source_label": task.source_label,
                            "stage": "validation",
                            "reason": reason,
                            "attempts": len(history),
                            **verdict_payload,
                        }
                    )
                    print(f"  FAILED {task.group_id}: {reason}", file=sys.stderr)
                    if args.fail_fast:
                        return

            def resolve_candidates(entries: Mapping[str, dict[str, Any]]) -> list[tuple[MRGenTask, catalog.MRDefinition]]:
                """Batch-verify and gate one round's structurally valid candidates.

                Every validator probe for the whole round goes out in one batched
                call per validator, which is where most of the wall-clock saving
                comes from: previously each group made its own generator call.
                Returns the pairs that failed the gate and may be retried.
                """
                if not entries:
                    return []
                verdicts_map = collect_verdicts_batched(
                    builders,
                    [
                        {
                            "group_id": group_id,
                            "definition": entry["definition"],
                            "source_text": entry["task"].source_text,
                            "followup_text": entry["text"],
                        }
                        for group_id, entry in entries.items()
                    ],
                    seed=args.seed,
                )
                retry: list[tuple[MRGenTask, catalog.MRDefinition]] = []
                for group_id, entry in entries.items():
                    task = entry["task"]
                    definition = entry["definition"]
                    gate_mode = (
                        args.flip_gate
                        if definition.relation_type == catalog.RELATION_FLIP
                        else args.preserve_gate
                    )
                    relation = evaluate_relation(
                        definition=definition,
                        source_label=task.source_label,
                        verdicts=verdicts_map.get(group_id, {}),
                        gate_mode=gate_mode,
                    )
                    last_checks.setdefault(group_id, {})["relation_verdict"] = relation
                    if not relation["passed"]:
                        outcomes[group_id].append("gate")
                        feedback[group_id] = (
                            "the relation check failed: " + ",".join(relation["reasons"])
                        )
                        rejected_reasons[feedback[group_id]] += 1
                        retry.append((task, definition))
                        continue
                    emit_rows(
                        make_group_rows(
                            task,
                            definition,
                            entry["text"],
                            entry["structural"],
                            relation,
                            generator_id=None if definition.is_deterministic else model_reference,
                            generation_seed=(
                                None
                                if definition.is_deterministic
                                else mr_task_seed(args.seed, task, entry["attempt"])
                            ),
                            attempts=entry["attempt"] + 1,
                        )
                    )
                    report_accept(task, definition)
                return retry

            # Applicability is a pure text predicate, so settle it before the model
            # is consulted at all.
            deterministic_pairs: list[tuple[MRGenTask, catalog.MRDefinition]] = []
            open_pairs: list[tuple[MRGenTask, catalog.MRDefinition]] = []
            feedback: dict[str, str | None] = {}
            outcomes: dict[str, list[str]] = defaultdict(list)
            last_checks: dict[str, dict[str, Any]] = {}

            for task in pending:
                definition = catalog.get_mr(task.mr_id)
                applicable, reason = catalog.mr_applicable(
                    task.mr_id, task.source_text, {"flip_max_words": args.flip_max_words}
                )
                if not applicable:
                    skipped_reasons[f"not_applicable:{reason}"] += 1
                    emit_reject(
                        {
                            "group_id": task.group_id,
                            "dataset": task.dataset,
                            "mr_id": task.mr_id,
                            "source_id": task.source_id,
                            "source_label": task.source_label,
                            "stage": "applicability",
                            "reason": reason,
                        }
                    )
                    continue
                feedback[task.group_id] = None
                if definition.is_deterministic:
                    deterministic_pairs.append((task, definition))
                else:
                    open_pairs.append((task, definition))

            # Deterministic MRs need no generator and are never retried.
            deterministic_entries: dict[str, dict[str, Any]] = {}
            for task, definition in deterministic_pairs:
                try:
                    assert definition.deterministic_fn is not None
                    text = definition.deterministic_fn(task.source_text, task.source_id)
                    structural = run_structural_checks(
                        task.source_text, text, definition, max_chars=args.max_chars
                    )
                except SAMRGenerationError as exc:
                    outcomes[task.group_id].append("structural")
                    failures.append(
                        {
                            "group_id": task.group_id,
                            "dataset": task.dataset,
                            "mr_id": task.mr_id,
                            "reason": str(exc),
                        }
                    )
                    emit_reject(
                        {
                            "group_id": task.group_id,
                            "dataset": task.dataset,
                            "mr_id": task.mr_id,
                            "source_id": task.source_id,
                            "source_label": task.source_label,
                            "stage": "validation",
                            "reason": str(exc),
                            "attempts": 1,
                        }
                    )
                    continue
                deterministic_entries[task.group_id] = {
                    "text": text,
                    "structural": structural,
                    "task": task,
                    "definition": definition,
                    "attempt": 0,
                }
            # Deterministic MRs have no prompt, so a gate failure is terminal
            # rather than retryable -- they must not enter the retry loop below.
            finalize_unresolved(resolve_candidates(deterministic_entries))

            # LLM-backed MRs: one batched generation round per attempt.
            for attempt in range(args.max_retries + 1):
                if not open_pairs:
                    break
                prompts = [
                    catalog.build_mr_prompt(
                        task.mr_id, task.source_text, task.source_label, feedback.get(task.group_id)
                    )
                    for task, _ in open_pairs
                ]
                seeds = [mr_task_seed(args.seed, task, attempt) for task, _ in open_pairs]
                responses = generate_batch(generator, prompts, seeds)
                response_by_group = {
                    task.group_id: response for (task, _), response in zip(open_pairs, responses)
                }

                entries: dict[str, dict[str, Any]] = {}
                next_open: list[tuple[MRGenTask, catalog.MRDefinition]] = []
                for task, definition in open_pairs:
                    try:
                        parsed = validate_mr_output(
                            response_by_group[task.group_id], task, definition
                        )
                        structural = run_structural_checks(
                            task.source_text, parsed["text"], definition, max_chars=args.max_chars
                        )
                    except MRDeclined as exc:
                        # A truthful "this edit is impossible" is a judgement about
                        # applicability, not a generation failure -- but retry once
                        # in case the model was merely lazy.
                        outcomes[task.group_id].append("declined")
                        feedback[task.group_id] = str(exc)
                        rejected_reasons[str(exc)] += 1
                        next_open.append((task, definition))
                        continue
                    except SAMRGenerationError as exc:
                        outcomes[task.group_id].append("structural")
                        feedback[task.group_id] = str(exc)
                        rejected_reasons[f"structural:{exc}"] += 1
                        last_checks.setdefault(task.group_id, {})["structural"] = None
                        next_open.append((task, definition))
                        continue
                    last_checks.setdefault(task.group_id, {})["structural"] = structural
                    entries[task.group_id] = {
                        "text": parsed["text"],
                        "structural": structural,
                        "task": task,
                        "definition": definition,
                        "attempt": attempt,
                    }

                next_open.extend(resolve_candidates(entries))
                open_pairs = next_open

            # Whatever is still unresolved exhausted its retries.
            finalize_unresolved(open_pairs)

        write_mr_report(
            report_path,
            args=args,
            tasks=tasks,
            accepted_group_ids=accepted_group_ids,
            rejected_reasons=rejected_reasons,
            skipped_reasons=skipped_reasons,
            failures=failures,
            started_at=started_at,
            model_reference=model_reference,
            input_path=input_path,
            output_path=output_path,
            validator_ids=validator_ids,
        )
        print(f"Output: {output_path}")
        print(f"Rejects: {rejects_path}")
        print(f"Report: {report_path}")
        if failures:
            print(
                f"Incomplete: {len(failures)} group(s) failed; rerun with --resume.",
                file=sys.stderr,
            )
            return 2
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
