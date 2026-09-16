#!/usr/bin/env python3
"""Generate a generic LLM augmentation baseline for three-way NLI.

The generator sees one ordinary NLI example and a requested target label.  It
is deliberately not given MR names, operation descriptions, relation effects,
or source/follow-up structure.  This makes the output suitable for the RQ1
control that asks whether high-quality generic synthetic data alone explains
changes in standard accuracy or MR compliance.

The input may be a JSON array or JSONL (including JSONL files named ``.json``).
The output is JSONL compatible with ``scripts/convert_nli_to_ft.py``.

Examples:
  # Inspect deterministic sampling and the exact English prompts (no model load)
  python scripts/generate_generic_llm_aug.py \
    --input data/nli/original_dataset/snli/train.json \
    --num-samples 3 --seed 42 --dry-run

  # Generate a small real sample with the default Gemma-3-4B model
  python scripts/generate_generic_llm_aug.py \
    --input data/nli/original_dataset/snli/train.json \
    --output data/nli/generic_llm_aug/snli/gemma3_4b_seed42.jsonl \
    --num-samples 3 --seed 42 --offline

  # Resume an interrupted full generation run
  python scripts/generate_generic_llm_aug.py \
    --input data/nli/original_dataset/snli/train.json \
    --output data/nli/generic_llm_aug/snli/gemma3_4b_seed42.jsonl \
    --num-samples 5340 --seed 42 --offline --resume
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from project_runtime import PROJECT_ROOT, apply_offline_mode, configure_console_encoding


METHOD_NAME = "generic-LLM-Aug"
PROMPT_VERSION = "generic_nli_v6"
DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "models" / "google" / "gemma-3-4b-it"
LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
LABEL_ALIASES = {
    "0": "entailment",
    "1": "neutral",
    "2": "contradiction",
    "entailment": "entailment",
    "entails": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradictory": "contradiction",
}
LABEL_QUALITY_GUIDANCE = {
    "entailment": (
        "Make the hypothesis a direct consequence of facts explicitly stated in the premise: "
        "use a safe paraphrase, hypernym, or deletion of detail. Every factual claim in the "
        "hypothesis must be supported by the premise. Do not infer a purpose, emotion, cause, "
        "future action, location, duration, or likely activity."
    ),
    "neutral": (
        "Make the hypothesis compatible with the premise while adding exactly one independent, "
        "clearly unstated fact. That fact must be possible but neither implied nor ruled out by "
        "the premise; do not make it a disguised contradiction or a likely-purpose inference."
    ),
    "contradiction": (
        "State one explicitly mutually exclusive fact about the same named entity at the same "
        "explicitly named time. Put the time and entity in both sentences, use matching tense, "
        "and make the opposite attribute unmistakable (for example open versus closed). Never "
        "rely on different times, possible future events, different unnamed entities, or world "
        "knowledge to create an apparent conflict."
    ),
}
SCENARIO_HINTS = (
    "public transportation or travel",
    "outdoor recreation",
    "school or classroom",
    "music performance or rehearsal",
    "library or archive",
    "gardening or plants",
    "weather observation",
    "repair workshop",
    "animal care",
    "retail store",
    "office work",
    "museum exhibit",
)



SYSTEM_PROMPT = (
    "You are a meticulous dataset writer for three-way natural language inference (NLI). "
    "Think through the logic privately, then return exactly one high-quality training instance as JSON only."
)

USER_PROMPT_TEMPLATE = """Create one original NLI training instance. Treat the example as a discardable style reference only; do not reuse its people, objects, setting, event, or distinctive wording. Choose a fresh everyday situation and varied vocabulary.

Required scenario family: {scenario_hint}
Use this only as a topic constraint; invent the people, objects, and event yourself. Do not copy content from the example.

Target label: {target_label}

Label definitions:
- entailment: the hypothesis must be true if the premise is true.
- neutral: the hypothesis may be true, but its truth cannot be determined from the premise.
- contradiction: the hypothesis must be false if the premise is true.

Construction guidance for this target:
{label_guidance}


Requirements:
1. Write a substantially new premise and hypothesis about a different situation. Do not copy, paraphrase, or minimally edit either example sentence, and do not preserve its main nouns or named entities.
2. Express one simple, verifiable relationship; avoid long lists of facts, rhetorical language, stereotypes, and details that require outside knowledge.
3. Make the target label unambiguous and logically correct. Both sentences must be natural, self-contained, plausible, and similar in difficulty to ordinary NLI data.
4. Keep entity identity, quantifiers, scope, tense, and time aligned whenever they matter. Do not change a subject or silently introduce a second event.
5. Do not use intentions, occupations, typical behavior, causal assumptions, or likely events as evidence. A possibility is not entailment, and an unmentioned fact is not contradiction.
6. Before answering, silently run this check: (i) write down only what the premise explicitly states; (ii) test whether the hypothesis is forced, merely possible, or impossible; (iii) revise until only the requested label is defensible.
7. For entailment, make every hypothesis claim licensed by the premise. For neutral, ensure the added fact could be true or false without changing the premise. For contradiction, make the opposing facts simultaneous and explicit.
8. Do not mention these instructions, the example, or the label inside the premise or hypothesis.
9. Return exactly one JSON object with string fields "premise", "hypothesis", and "label". Set "label" to "{target_label}".

Example:
Premise: {source_premise}
Hypothesis: {source_hypothesis}
Label: {source_label}

Return JSON only:"""

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict three-way NLI data auditor. Classify only from the candidate "
    "premise and hypothesis, and return JSON only."
)
VERIFIER_USER_PROMPT_TEMPLATE = """Independently audit this candidate NLI pair.

Label definitions:
- entailment: the hypothesis must be true if the premise is true.
- neutral: the hypothesis may be true, but its truth cannot be determined from the premise.
- contradiction: the hypothesis must be false if the premise is true.
Audit rules:
- Treat an unstated fact as unknown, not false.
- Call a pair entailment only when every hypothesis claim is forced by the premise.
- Call a pair contradiction only when the two claims cannot both hold in the same event and time.
- If two labels remain defensible, set "unambiguous" to false.


Candidate:
Premise: {premise}
Hypothesis: {hypothesis}

Return one JSON object containing:
- "predicted_label": exactly entailment, neutral, or contradiction
- "unambiguous": true only if no other label is reasonably defensible
- "reason": one short explanation grounded only in the two sentences
Do not infer a desired label from wording or metadata; none is provided.
Return JSON only:"""



class GenerationError(ValueError):
    """Raised when a model response cannot become a valid augmentation row."""


@dataclass(frozen=True)
class GenerationTask:
    output_index: int
    source_index: int
    target_label: str
    source: dict[str, Any]

    @property
    def augmentation_id(self) -> str:
        return f"generic_llm_aug_{self.output_index:08d}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or line-delimited JSON without trusting suffixes."""
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Input dataset is empty: {path}")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} of {path} is not a JSON object")
            records.append(item)
        return records

    if isinstance(parsed, dict):
        for key in ("data", "examples", "records"):
            candidate = parsed.get(key)
            if isinstance(candidate, list):
                parsed = candidate
                break
        else:
            # A JSONL file holding exactly one record parses as a bare object.
            # Under this function's contract that is still valid input, so treat
            # it as a single-record file rather than rejecting it.
            parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise ValueError("Input must be a JSON array of objects or JSONL objects")
    return parsed


def normalize_label(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid NLI label: {value!r}")
    if isinstance(value, int):
        if value in ID_TO_LABEL:
            return ID_TO_LABEL[value]
        raise ValueError(f"Unknown numeric NLI label: {value!r}")
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    raise ValueError(f"Unknown three-way NLI label: {value!r}")


def validate_source(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    try:
        premise = row["premise"].strip()
        hypothesis = row["hypothesis"].strip()
        label = normalize_label(row["label"])
    except KeyError as exc:
        raise ValueError(f"Source row {source_index} is missing field {exc.args[0]!r}") from exc
    except AttributeError as exc:
        raise ValueError(f"Source row {source_index} premise/hypothesis must be strings") from exc
    if not premise or not hypothesis:
        raise ValueError(f"Source row {source_index} has an empty premise or hypothesis")
    return {**row, "premise": premise, "hypothesis": hypothesis, "_normalized_label": label}


def select_tasks(
    records: list[dict[str, Any]],
    num_samples: int | None,
    seed: int,
    label_strategy: str,
) -> list[GenerationTask]:
    if num_samples is None:
        num_samples = len(records)
    if num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero")
    if num_samples > len(records):
        raise ValueError(
            f"Requested {num_samples} samples, but the input has only {len(records)} rows; "
            "sampling is intentionally without replacement"
        )

    rng = random.Random(seed)
    source_indices = rng.sample(range(len(records)), num_samples)
    sources = [validate_source(records[index], index) for index in source_indices]

    if label_strategy == "source":
        targets = [source["_normalized_label"] for source in sources]
    elif label_strategy == "balanced":
        targets = [ID_TO_LABEL[index % len(ID_TO_LABEL)] for index in range(num_samples)]
        rng.shuffle(targets)
    else:
        raise ValueError(f"Unknown label strategy: {label_strategy}")

    return [
        GenerationTask(
            output_index=output_index,
            source_index=source_index,
            target_label=target,
            source=source,
        )
        for output_index, (source_index, source, target) in enumerate(
            zip(source_indices, sources, targets)
        )
    ]


def build_prompt(task: GenerationTask, retry_feedback: str | None = None) -> list[dict[str, str]]:
    prompt = USER_PROMPT_TEMPLATE.format(
        scenario_hint=SCENARIO_HINTS[task.output_index % len(SCENARIO_HINTS)],
        target_label=task.target_label,
        label_guidance=LABEL_QUALITY_GUIDANCE[task.target_label],
        source_premise=task.source["premise"],
        source_hypothesis=task.source["hypothesis"],
        source_label=task.source["_normalized_label"],
    )
    if retry_feedback:
        prompt += (
            "\n\nYour previous response was rejected for this reason: "
            f"{retry_feedback}\nGenerate a corrected, fully new instance and return JSON only."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_verification_prompt(generated: dict[str, str]) -> list[dict[str, str]]:
    prompt = VERIFIER_USER_PROMPT_TEMPLATE.format(
        premise=generated["premise"],
        hypothesis=generated["hypothesis"],
    )
    return [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def validate_verification(response: str, target_label: str) -> dict[str, Any]:
    value = extract_json_object(response)
    try:
        predicted_label = normalize_label(value.get("predicted_label"))
    except ValueError as exc:
        raise GenerationError(f"semantic audit returned an invalid label: {exc}") from exc

    unambiguous = value.get("unambiguous")
    reason = value.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    if unambiguous is not True:
        raise GenerationError(f"semantic audit marked the pair ambiguous: {reason[:300]}")
    if predicted_label != target_label:
        raise GenerationError(
            f"semantic audit predicted {predicted_label!r}, expected {target_label!r}: "
            f"{reason[:300]}"
        )
    return {
        "predicted_label": predicted_label,
        "unambiguous": True,
        "reason": reason[:500],
    }


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise GenerationError("response does not contain a valid JSON object")


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, normalized_text(left), normalized_text(right)).ratio()


def validate_generation(
    response: str,
    task: GenerationTask,
    max_source_similarity: float,
    max_text_chars: int,
) -> dict[str, str]:
    value = extract_json_object(response)
    missing = [field for field in ("premise", "hypothesis", "label") if field not in value]
    if missing:
        raise GenerationError(f"missing JSON fields: {', '.join(missing)}")
    if not isinstance(value["premise"], str) or not isinstance(value["hypothesis"], str):
        raise GenerationError("premise and hypothesis must be strings")

    premise = " ".join(value["premise"].split())
    hypothesis = " ".join(value["hypothesis"].split())
    if len(premise) < 8 or len(hypothesis) < 8:
        raise GenerationError("premise and hypothesis must each contain at least 8 characters")
    if len(premise) > max_text_chars or len(hypothesis) > max_text_chars:
        raise GenerationError(f"premise or hypothesis exceeds {max_text_chars} characters")
    if normalized_text(premise) == normalized_text(hypothesis):
        raise GenerationError("premise and hypothesis are identical")

    try:
        label = normalize_label(value["label"])
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc
    if label != task.target_label:
        raise GenerationError(f"model returned label {label!r}, expected {task.target_label!r}")

    source_texts = (task.source["premise"], task.source["hypothesis"])
    generated_texts = (premise, hypothesis)
    highest_similarity = max(
        similarity(generated, source)
        for generated in generated_texts
        for source in source_texts
    )
    if highest_similarity > max_source_similarity:
        raise GenerationError(
            f"generated text is too similar to the source ({highest_similarity:.3f} > "
            f"{max_source_similarity:.3f})"
        )
    return {"premise": premise, "hypothesis": hypothesis, "label": label}


def task_seed(seed: int, task: GenerationTask, attempt: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{task.output_index}:{task.source_index}:{task.target_label}:{attempt}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def semantic_audit_seed(seed: int, task: GenerationTask, attempt: int) -> int:
    digest = hashlib.sha256(
        f"semantic:{seed}:{task.output_index}:{task.source_index}:{attempt}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


class HFTextGenerator:
    """Small lazy Transformers wrapper; importing this module does not import torch."""

    def __init__(
        self,
        model_reference: str,
        tokenizer_reference: str,
        device: str,
        dtype: str,
        offline: bool,
        trust_remote_code: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Real generation requires PyTorch and Transformers in the active environment"
            ) from exc

        if offline:
            apply_offline_mode(True)
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch = torch
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_reference,
            trust_remote_code=trust_remote_code,
            local_files_only=offline,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_reference,
            torch_dtype=dtype_map[dtype],
            trust_remote_code=trust_remote_code,
            local_files_only=offline,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, messages: list[dict[str, str]], seed: int) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        self.torch.manual_seed(seed)
        if self.device.type == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0, prompt_tokens:], skip_special_tokens=True).strip()


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


def resolve_semantic_verifier_references(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve an optional verifier model; absent options reuse the generator model."""
    if not args.semantic_verifier_model and not args.semantic_verifier_model_path:
        return resolve_model_references(args)
    if args.semantic_verifier_model_path:
        model_reference = str(Path(args.semantic_verifier_model_path).expanduser().resolve())
    else:
        model_reference = args.semantic_verifier_model
        if model_reference == DEFAULT_MODEL and DEFAULT_LOCAL_MODEL.is_dir():
            model_reference = str(DEFAULT_LOCAL_MODEL.resolve())
    tokenizer_reference = (
        str(Path(args.semantic_verifier_tokenizer_path).expanduser().resolve())
        if args.semantic_verifier_tokenizer_path
        else model_reference
    )
    return model_reference, tokenizer_reference


def read_existing_output(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = load_records(path)
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        augmentation_id = row.get("augmentation_id")
        if not isinstance(augmentation_id, str) or not augmentation_id:
            raise ValueError(
                f"Cannot resume: output row {line_number} has no valid augmentation_id"
            )
        if augmentation_id in by_id:
            raise ValueError(f"Cannot resume: duplicate augmentation_id {augmentation_id!r}")
        by_id[augmentation_id] = row
    return by_id


def validate_resume_rows(existing: dict[str, dict[str, Any]], tasks: Iterable[GenerationTask]) -> None:
    task_map = {task.augmentation_id: task for task in tasks}
    unexpected = sorted(set(existing) - set(task_map))
    if unexpected:
        raise ValueError(f"Cannot resume: output contains unexpected ids, e.g. {unexpected[:3]}")
    for augmentation_id, row in existing.items():
        task = task_map[augmentation_id]
        if row.get("source_index") != task.source_index:
            raise ValueError(f"Cannot resume: source_index mismatch for {augmentation_id}")
        if normalize_label(row.get("label")) != task.target_label:
            raise ValueError(f"Cannot resume: label mismatch for {augmentation_id}")


def make_output_row(
    generated: dict[str, str],
    task: GenerationTask,
    args: argparse.Namespace,
    successful_attempt: int,
    semantic_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "premise": generated["premise"],
        "hypothesis": generated["hypothesis"],
        "label": LABEL_TO_ID[generated["label"]],
        "pair_id": task.augmentation_id,
        "mr_id": "none",
        "is_syn": True,
        "weight": 1.0,
        "type": "generic_llm_generated",
        "augmentation_method": METHOD_NAME,
        "augmentation_id": task.augmentation_id,
        "source_index": task.source_index,
        "source_label": task.source["_normalized_label"],
        "scenario_hint": SCENARIO_HINTS[task.output_index % len(SCENARIO_HINTS)],
        "target_label": task.target_label,
        "generation_model": args.model,
        "generation_seed": task_seed(args.seed, task, successful_attempt),
        "generation_attempt": successful_attempt,
        "semantic_audit": semantic_audit,
        "prompt_version": PROMPT_VERSION,
    }


def write_report(
    path: Path,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    tasks: list[GenerationTask],
    generated_rows: int,
    rejected_reasons: Counter,
    failures: list[dict[str, Any]],
    started_at: str,
    model_reference: str,
    semantic_verifier_reference: str | None,
) -> None:
    report = {
        "schema_version": 1,
        "method": METHOD_NAME,
        "research_role": (
            "Generic synthetic NLI control without MR specifications or source-follow-up structure"
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": canonical_sha256(
            {
                "system": SYSTEM_PROMPT,
                "scenario_hints": SCENARIO_HINTS,
                "user_template": USER_PROMPT_TEMPLATE,
                "label_quality_guidance": LABEL_QUALITY_GUIDANCE,
            }
        ),
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path) if output_path.is_file() else None,
        "requested_model": args.model,
        "resolved_model_reference": model_reference,
        "semantic_check_enabled": args.semantic_check,
        "semantic_verifier_model": args.semantic_verifier_model or (args.model if args.semantic_check else None),
        "resolved_semantic_verifier_reference": semantic_verifier_reference,
        "semantic_verifier_mode": (
            "separate_model" if args.semantic_verifier_model or args.semantic_verifier_model_path
            else ("same_model_blind_second_pass" if args.semantic_check else "disabled")
        ),
        "seed": args.seed,
        "label_strategy": args.label_strategy,
        "requested_samples": len(tasks),
        "generated_samples": generated_rows,
        "complete": generated_rows == len(tasks) and not failures,
        "target_label_distribution": dict(Counter(task.target_label for task in tasks)),
        "generation_parameters": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_retries": args.max_retries,
            "max_source_similarity": args.max_source_similarity,
            "max_text_chars": args.max_text_chars,
            "offline": args.offline,
        },
        "automatic_checks": [
            "valid JSON object",
            "required fields and string types",
            "target label equality",
            "non-empty and non-identical texts",
            "maximum source-text similarity",
        ],
        "semantic_label_validation": (
            "Blind verifier checked each candidate; human audit is still recommended"
            if args.semantic_check
            else "Disabled; prompted and structurally checked only; semantic audit is still required"
        ),
        "rejected_attempts_by_reason": dict(rejected_reasons),
        "failed_tasks": failures,
        "started_at": started_at,
        "finished_at": utc_now(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_dry_run(tasks: list[GenerationTask], args: argparse.Namespace) -> None:
    print(f"Method: {METHOD_NAME}")
    print(f"Input rows selected: {len(tasks)}")
    print(f"Seed: {args.seed}")
    print(f"Label strategy: {args.label_strategy}")
    print(f"Target labels: {dict(Counter(task.target_label for task in tasks))}")
    for task in tasks[: args.preview_count]:
        print(f"\n--- prompt {task.output_index} (source_index={task.source_index}) ---")
        for message in build_prompt(task):
            print(f"[{message['role']}]\n{message['content']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generic-LLM-Aug NLI data without providing MR specifications to the generator"
        )
    )
    parser.add_argument("--input", required=True, help="Original NLI JSON/JSONL dataset")
    parser.add_argument("--output", help="Output JSONL (required unless --dry-run)")
    parser.add_argument("--report-output", help="Audit report path (default: <output>.report.json)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-path", help="Local model directory; overrides --model for loading")
    parser.add_argument("--tokenizer-path", help="Tokenizer directory/id (default: resolved model)")
    parser.add_argument("--num-samples", type=int, default=None, help="Rows to generate (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic source/generation seed")
    parser.add_argument(
        "--label-strategy",
        choices=("source", "balanced"),
        default="source",
        help="Preserve sampled source labels (default) or assign an exactly balanced target schedule",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-retries", type=int, default=2, help="Retries after the first attempt")
    parser.add_argument("--max-source-similarity", type=float, default=0.85)
    parser.add_argument("--max-text-chars", type=int, default=1000)
    parser.add_argument("--offline", action="store_true", help="Use local model/tokenizer files only")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--semantic-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Blindly audit each candidate with a second semantic pass (default: enabled)",
    )
    parser.add_argument("--semantic-verifier-model", help="Optional separate model id for blind audit")
    parser.add_argument("--semantic-verifier-model-path", help="Optional local verifier model directory")
    parser.add_argument("--semantic-verifier-tokenizer-path", help="Optional verifier tokenizer directory/id")
    parser.add_argument("--resume", action="store_true", help="Append only missing deterministic tasks")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Preview sampling/prompts; write nothing")
    parser.add_argument("--preview-count", type=int, default=3)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.output:
        raise ValueError("--output is required unless --dry-run is used")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if not 0 <= args.temperature:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if not 0 <= args.max_source_similarity <= 1:
        raise ValueError("--max-source-similarity must be in [0, 1]")
    if args.preview_count < 0:
        raise ValueError("--preview-count cannot be negative")
    return args


def main(
    argv: list[str] | None = None,
    generator_factory: Callable[..., Any] | None = None,
) -> int:
    configure_console_encoding()
    try:
        args = parse_args(argv)
        input_path = Path(args.input).expanduser().resolve()
        records = load_records(input_path)
        tasks = select_tasks(records, args.num_samples, args.seed, args.label_strategy)
        if args.dry_run:
            print_dry_run(tasks, args)
            return 0

        output_path = Path(args.output).expanduser().resolve()
        report_path = (
            Path(args.report_output).expanduser().resolve()
            if args.report_output
            else output_path.with_suffix(output_path.suffix + ".report.json")
        )
        if output_path.exists() and not args.resume and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --resume or explicit --overwrite."
            )
        if args.overwrite and output_path.exists():
            output_path.unlink()
        existing = read_existing_output(output_path) if args.resume else {}
        validate_resume_rows(existing, tasks)

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

        semantic_verifier = None
        semantic_verifier_reference = None
        if args.semantic_check:
            semantic_verifier_reference, verifier_tokenizer_reference = (
                resolve_semantic_verifier_references(args)
            )
            if (semantic_verifier_reference, verifier_tokenizer_reference) == (
                model_reference,
                tokenizer_reference,
            ):
                semantic_verifier = generator
            else:
                semantic_verifier = factory(
                    model_reference=semantic_verifier_reference,
                    tokenizer_reference=verifier_tokenizer_reference,
                    device=args.device,
                    dtype=args.dtype,
                    offline=args.offline,
                    trust_remote_code=args.trust_remote_code,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = utc_now()
        rejected_reasons: Counter = Counter()
        failures: list[dict[str, Any]] = []
        completed = len(existing)
        pending = [task for task in tasks if task.augmentation_id not in existing]
        print(
            f"{METHOD_NAME}: {len(tasks)} requested, {completed} existing, "
            f"{len(pending)} pending; model={args.model}"
        )

        with output_path.open("a", encoding="utf-8", newline="\n") as output_handle:
            for pending_index, task in enumerate(pending, start=1):
                feedback = None
                generated = None
                semantic_audit = None
                for attempt in range(args.max_retries + 1):
                    response = generator.generate(
                        build_prompt(task, feedback), task_seed(args.seed, task, attempt)
                    )
                    try:
                        generated = validate_generation(
                            response,
                            task,
                            max_source_similarity=args.max_source_similarity,
                            max_text_chars=args.max_text_chars,
                        )
                        if semantic_verifier is not None:
                            audit_response = semantic_verifier.generate(
                                build_verification_prompt(generated),
                                semantic_audit_seed(args.seed, task, attempt),
                            )
                            semantic_audit = validate_verification(
                                audit_response, task.target_label
                            )
                        break
                    except GenerationError as exc:
                        generated = None
                        semantic_audit = None
                        feedback = str(exc)
                        rejected_reasons[feedback] += 1
                if generated is None:
                    failure = {
                        "augmentation_id": task.augmentation_id,
                        "source_index": task.source_index,
                        "target_label": task.target_label,
                        "reason": feedback,
                    }
                    failures.append(failure)
                    print(f"  FAILED {task.augmentation_id}: {feedback}", file=sys.stderr)
                    if args.fail_fast:
                        break
                    continue

                output_handle.write(
                    json.dumps(
                        make_output_row(generated, task, args, attempt, semantic_audit),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_handle.flush()
                completed += 1
                print(
                    f"  [{pending_index}/{len(pending)}] {task.augmentation_id} "
                    f"label={task.target_label}"
                )

        write_report(
            report_path,
            args,
            input_path,
            output_path,
            tasks,
            completed,
            rejected_reasons,
            failures,
            started_at,
            model_reference,
            semantic_verifier_reference,
        )
        print(f"Output: {output_path}")
        print(f"Report: {report_path}")
        if failures:
            print(
                f"Generation incomplete: {len(failures)} task(s) failed; rerun with --resume.",
                file=sys.stderr,
            )
            return 2
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
