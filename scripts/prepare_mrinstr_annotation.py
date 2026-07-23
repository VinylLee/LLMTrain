#!/usr/bin/env python3
"""Prepare and verify the Stage 2 human-review batches.

The generated JSONL files contain no model predictions and no pre-filled human
decisions.  They are deterministic for a fixed cohort, seed, tokenizer, and
generator version.  Existing annotation files are never overwritten.
"""

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from project_runtime import (
    PROJECT_ROOT,
    apply_offline_mode,
    configure_console_encoding,
    resolve_model_reference,
)
from convert_nli_to_ft import (
    LABEL_NAMES_3CLASS,
    MR_OPERATION_DESCRIPTIONS,
    build_instruction_for_sample,
    build_original_map,
    build_pair_groups,
    canonical_sha256,
    compute_data_signature,
    compute_sample_key,
    count_row_token_length,
    normalize_mr_id,
)


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
ANNOTATOR_IDS = ("annotator_a", "annotator_b")
BLOCKER_MR_IDS = (
    "adding_contradiction",
    "composite_flip",
    "conditional_clause",
    "pronoun_substitution",
)
PILOT_EXPECTED_COUNTS = {
    "adding_contradiction": 667,
    "composite_flip": 327,
    "conditional_clause": 119,
    "pronoun_substitution": 1,
}
REQUIRED_RESPONSE_FIELDS = (
    "reference_pair_valid",
    "transformation_valid",
    "transformation_quality",
    "nli_label",
    "stored_label_agrees",
    "coreference_clear",
)
ALLOWED_VALUES = {
    "reference_pair_valid": ("yes", "no", "uncertain"),
    "transformation_valid": ("yes", "no", "uncertain"),
    "transformation_quality": (
        "clean",
        "awkward_but_interpretable",
        "ambiguous",
        "ungrammatical",
    ),
    "nli_label": ("entailment", "neutral", "contradiction", "uncertain"),
    "stored_label_agrees": ("yes", "no", "uncertain"),
    "coreference_clear": ("yes", "no", "not_applicable", "uncertain"),
}
ALLOWED_REASON_CODES = (
    "not_logically_contradictory",
    "unsupported_extra_information",
    "relation_ambiguous",
    "reference_current_mismatch",
    "invalid_transformation",
    "coreference_ambiguous",
    "conditional_scope_ambiguous",
    "composite_effect_heterogeneous",
    "grammar_blocks_interpretation",
    "stored_label_supported",
)
TARGET_FILENAMES = (
    "annotation_batch_manifest.json",
    "annotator_a.jsonl",
    "annotator_b.jsonl",
)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def workspace_relative(path):
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_name(value):
    if isinstance(value, bool):
        raise ValueError(f"Boolean is not a valid NLI label: {value!r}")
    if isinstance(value, int):
        if value not in LABEL_NAMES_3CLASS:
            raise ValueError(f"Unknown numeric NLI label: {value!r}")
        return LABEL_NAMES_3CLASS[value]
    normalized = str(value).strip().lower()
    if normalized not in LABEL_NAMES_3CLASS.values():
        raise ValueError(f"Unknown NLI label: {value!r}")
    return normalized


def attach_group_keys(sampled_rows, source_file_id):
    groups = build_pair_groups(
        sampled_rows,
        [source_file_id] * len(sampled_rows),
    )
    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]
    return groups


def build_review_contexts(
    sampled_rows,
    *,
    sampled_file,
    tokenizer,
    cutoff_len,
    expected_counts=None,
):
    """Build one immutable review context for every blocker sample."""
    groups = attach_group_keys(sampled_rows, Path(sampled_file).stem)
    original_map, source_issues = build_original_map(groups)
    if source_issues:
        raise ValueError(f"Invalid source pairing in review cohort: {source_issues[:5]}")

    data_signature = compute_data_signature(sampled_rows)
    sampled_sha256 = file_sha256(sampled_file)
    sampled_reference = workspace_relative(sampled_file)
    group_map = {group["group_key"]: group for group in groups}
    blocker_rows = [
        sample for sample in sampled_rows
        if normalize_mr_id(sample.get("mr_id")) in BLOCKER_MR_IDS
    ]
    counts = Counter(normalize_mr_id(sample.get("mr_id")) for sample in blocker_rows)
    if expected_counts is not None and dict(counts) != dict(expected_counts):
        raise ValueError(
            "Blocker counts do not match the approved pilot scope: "
            f"expected={dict(expected_counts)}, actual={dict(counts)}"
        )

    sample_keys = [compute_sample_key(sample) for sample in blocker_rows]
    duplicates = [key for key, count in Counter(sample_keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"Stable sample keys are not unique: {duplicates[:5]}")

    contexts = []
    for sample in blocker_rows:
        mr_id = normalize_mr_id(sample.get("mr_id"))
        original = original_map[sample["_group_key"]]
        review_row = build_instruction_for_sample(
            sample,
            original,
            "pair_operation",
            MR_OPERATION_DESCRIPTIONS[mr_id],
            is_original=False,
        )
        token_length, counting_method = count_row_token_length(tokenizer, review_row)
        current_key = compute_sample_key(sample)
        siblings = []
        for sibling in group_map[sample["_group_key"]]["samples"]:
            sibling_mr_id = normalize_mr_id(sibling.get("mr_id"))
            sibling_key = compute_sample_key(sibling)
            if sibling_mr_id == "none" or sibling_key == current_key:
                continue
            siblings.append({
                "sample_key": sibling_key,
                "mr_id": sibling_mr_id,
                "stored_label": label_name(sibling["label"]),
                "premise": sibling["premise"].strip(),
                "hypothesis": sibling["hypothesis"].strip(),
            })
        siblings.sort(key=lambda item: item["sample_key"])
        contexts.append({
            "sample_key": current_key,
            "pair_id": sample.get("pair_id"),
            "mr_id": mr_id,
            "stored_label": label_name(sample["label"]),
            "source": {
                "premise": original["premise"].strip(),
                "hypothesis": original["hypothesis"].strip(),
                "label": label_name(original["label"]),
            },
            "current": {
                "premise": sample["premise"].strip(),
                "hypothesis": sample["hypothesis"].strip(),
            },
            "operation_description": MR_OPERATION_DESCRIPTIONS[mr_id],
            "sibling_augmented_rows": siblings,
            "tokenization": {
                "context_token_length": token_length,
                "cutoff_len": cutoff_len,
                "over_cutoff": token_length > cutoff_len,
                "counting_method": counting_method,
            },
            "provenance": {
                "sampled_file": sampled_reference,
                "sampled_file_sha256": sampled_sha256,
                "data_signature": data_signature,
            },
        })
    return sorted(contexts, key=lambda item: item["sample_key"]), dict(sorted(counts.items()))


def build_annotator_rows(
    contexts,
    *,
    annotator_id,
    seed,
    unique_per_batch=100,
    repeats_per_batch=5,
):
    """Create independently ordered batches with concealed repeat flags."""
    if annotator_id not in ANNOTATOR_IDS:
        raise ValueError(f"Unknown annotator slot: {annotator_id}")
    if unique_per_batch < 1 or repeats_per_batch < 1:
        raise ValueError("Batch and repeat counts must both be positive")

    unique_rows = copy.deepcopy(contexts)
    random.Random(f"{seed}:{annotator_id}:unique:v{GENERATOR_VERSION}").shuffle(unique_rows)
    output = []
    slot = annotator_id.removeprefix("annotator_").upper()
    for batch_index, start in enumerate(range(0, len(unique_rows), unique_per_batch), 1):
        chunk = unique_rows[start:start + unique_per_batch]
        if len(chunk) < repeats_per_batch:
            raise ValueError(
                f"Final batch has {len(chunk)} unique rows, fewer than "
                f"{repeats_per_batch} required repeats"
            )
        repeat_rng = random.Random(
            f"{seed}:{annotator_id}:batch:{batch_index}:repeat:v{GENERATOR_VERSION}"
        )
        repeated = repeat_rng.sample(chunk, repeats_per_batch)
        presentations = copy.deepcopy(chunk + repeated)
        random.Random(
            f"{seed}:{annotator_id}:batch:{batch_index}:order:v{GENERATOR_VERSION}"
        ).shuffle(presentations)
        batch_id = f"{slot}-B{batch_index:03d}"
        for position, context in enumerate(presentations, 1):
            row = {
                "schema_version": SCHEMA_VERSION,
                "presentation_id": f"{batch_id}-P{position:03d}",
                "review_batch_id": batch_id,
                **context,
                "annotator_id": annotator_id,
                "reference_pair_valid": None,
                "transformation_valid": None,
                "transformation_quality": None,
                "nli_label": None,
                "stored_label_agrees": None,
                "coreference_clear": None,
                "minimal_reason_code": [],
                "free_text_note": "",
                "review_status": "pending",
            }
            output.append(row)
    return output


def annotator_statistics(rows):
    keys = [row["sample_key"] for row in rows]
    batches = {}
    for row in rows:
        batches.setdefault(row["review_batch_id"], []).append(row)
    batch_stats = {}
    for batch_id, batch_rows in sorted(batches.items()):
        counts = Counter(row["sample_key"] for row in batch_rows)
        batch_stats[batch_id] = {
            "unique_count": len(counts),
            "presentation_count": len(batch_rows),
            "repeat_presentation_count": sum(count - 1 for count in counts.values()),
        }
    return {
        "unique_count": len(set(keys)),
        "presentation_count": len(rows),
        "repeat_presentation_count": len(rows) - len(set(keys)),
        "batch_count": len(batches),
        "batch_statistics": batch_stats,
        "order_signature": canonical_sha256(keys),
    }


def validate_pristine_rows(rows, annotator_id, expected_keys, repeats_per_batch):
    errors = []
    if not rows:
        return [f"{annotator_id}: file is empty"]
    if {row.get("sample_key") for row in rows} != set(expected_keys):
        errors.append(f"{annotator_id}: unique sample-key coverage mismatch")
    if any(row.get("annotator_id") != annotator_id for row in rows):
        errors.append(f"{annotator_id}: annotator_id mismatch")
    presentation_ids = [row.get("presentation_id") for row in rows]
    if len(presentation_ids) != len(set(presentation_ids)):
        errors.append(f"{annotator_id}: duplicate presentation_id")
    for index, row in enumerate(rows, 1):
        if row.get("review_status") != "pending":
            errors.append(f"{annotator_id}:{index}: review_status is not pending")
        for field in REQUIRED_RESPONSE_FIELDS:
            if row.get(field) is not None:
                errors.append(f"{annotator_id}:{index}: {field} is pre-filled")
        if row.get("minimal_reason_code") != [] or row.get("free_text_note") != "":
            errors.append(f"{annotator_id}:{index}: reason or note is pre-filled")
        forbidden = {"prediction", "model_prediction", "training_result", "mode"}
        if forbidden.intersection(row):
            errors.append(f"{annotator_id}:{index}: forbidden biasing field present")
    batches = {}
    for row in rows:
        batches.setdefault(row.get("review_batch_id"), []).append(row)
    for batch_id, batch_rows in batches.items():
        counts = Counter(row.get("sample_key") for row in batch_rows)
        repeats = sum(count - 1 for count in counts.values())
        if repeats < repeats_per_batch:
            errors.append(f"{annotator_id}:{batch_id}: only {repeats} repeat rows")
        if any(count > 2 for count in counts.values()):
            errors.append(f"{annotator_id}:{batch_id}: a sample appears more than twice")
    return errors


def build_manifest(
    *,
    config_path,
    cohort_id,
    seed,
    sampled_path,
    contexts,
    blocker_counts,
    annotator_paths,
    annotator_rows,
    tokenizer_reference,
    local_tokenizer_path,
    tokenizer_class,
    cutoff_len,
    unique_per_batch,
    repeats_per_batch,
):
    data_signature = contexts[0]["provenance"]["data_signature"] if contexts else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "purpose": "dual-independent-human-review-of-stage2-blockers",
        "config": workspace_relative(config_path),
        "cohort_id": cohort_id,
        "seed": seed,
        "source": {
            "sampled_file": workspace_relative(sampled_path),
            "sampled_file_sha256": file_sha256(sampled_path),
            "data_signature": data_signature,
        },
        "review_scope": {
            "blocker_mr_ids": list(BLOCKER_MR_IDS),
            "by_mr": blocker_counts,
            "unique_sample_count": len(contexts),
            "stable_sample_key_signature": canonical_sha256(
                [context["sample_key"] for context in contexts]
            ),
        },
        "protocol": {
            "document": ".research/MR_INSTRUCTION_DATA_REVIEW_PROTOCOL.md",
            "version": "1.0-draft",
            "independent_annotators": 2,
            "bias_controls": [
                "no_model_predictions",
                "no_training_results",
                "no_instruction_mode_names",
            ],
        },
        "quality_control": {
            "unique_rows_per_batch": unique_per_batch,
            "minimum_repeat_rows_per_batch": repeats_per_batch,
            "repeat_flags_exposed_to_annotators": False,
        },
        "tokenization": {
            "tokenizer_reference": tokenizer_reference,
            "local_tokenizer_path": local_tokenizer_path,
            "tokenizer_class": tokenizer_class,
            "cutoff_len": cutoff_len,
            "context_definition": "reference text + operation description + current NLI row",
        },
        "response_schema": {
            "required_fields": list(REQUIRED_RESPONSE_FIELDS),
            "allowed_values": {key: list(value) for key, value in ALLOWED_VALUES.items()},
            "allowed_reason_codes": list(ALLOWED_REASON_CODES),
            "completion_status": "complete",
        },
        "annotators": {},
    }
    for annotator_id in ANNOTATOR_IDS:
        path = annotator_paths[annotator_id]
        manifest["annotators"][annotator_id] = {
            "file": workspace_relative(path),
            "initial_template_sha256": file_sha256(path),
            **annotator_statistics(annotator_rows[annotator_id]),
        }
    manifest["sha256"] = canonical_sha256(manifest)
    return manifest


def verify_artifacts(output_dir, *, verify_initial_hashes=True):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "annotation_batch_manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    payload = dict(manifest)
    recorded_hash = payload.pop("sha256", None)
    errors = []
    if recorded_hash != canonical_sha256(payload):
        errors.append("annotation_batch_manifest.json self-hash mismatch")

    source = manifest.get("source") or {}
    sampled_path = resolve_path(source.get("sampled_file", ""))
    if not sampled_path.is_file():
        errors.append(f"sampled source is missing: {sampled_path}")
        expected_keys = set()
    else:
        if file_sha256(sampled_path) != source.get("sampled_file_sha256"):
            errors.append("sampled source SHA-256 mismatch")
        sampled_rows = load_jsonl(sampled_path)
        if compute_data_signature(sampled_rows) != source.get("data_signature"):
            errors.append("sampled source data signature mismatch")
        groups = attach_group_keys(sampled_rows, sampled_path.stem)
        del groups
        expected_keys = {
            compute_sample_key(row) for row in sampled_rows
            if normalize_mr_id(row.get("mr_id")) in BLOCKER_MR_IDS
        }

    all_orders = []
    repeats_per_batch = manifest["quality_control"]["minimum_repeat_rows_per_batch"]
    for annotator_id in ANNOTATOR_IDS:
        item = manifest.get("annotators", {}).get(annotator_id) or {}
        path = resolve_path(item.get("file", ""))
        if not path.is_file():
            errors.append(f"{annotator_id}: annotation file is missing")
            continue
        if verify_initial_hashes and file_sha256(path) != item.get("initial_template_sha256"):
            errors.append(f"{annotator_id}: initial template SHA-256 mismatch")
        rows = load_jsonl(path)
        errors.extend(validate_pristine_rows(
            rows,
            annotator_id,
            expected_keys,
            repeats_per_batch,
        ))
        malformed = [
            index for index, row in enumerate(rows, 1)
            if not row.get("sample_key") or not row.get("review_batch_id")
            or not row.get("presentation_id")
        ]
        if malformed:
            errors.append(
                f"{annotator_id}: rows missing presentation identity: {malformed[:5]}"
            )
            continue
        stats = annotator_statistics(rows)
        for field in (
            "unique_count",
            "presentation_count",
            "repeat_presentation_count",
            "batch_count",
            "order_signature",
        ):
            if stats[field] != item.get(field):
                errors.append(f"{annotator_id}: manifest {field} mismatch")
        all_orders.append([row["sample_key"] for row in rows])
    if len(all_orders) == 2 and all_orders[0] == all_orders[1]:
        errors.append("annotator files have identical presentation order")
    if len(expected_keys) != manifest.get("review_scope", {}).get("unique_sample_count"):
        errors.append("manifest unique-sample count mismatch")
    if errors:
        raise ValueError("Annotation artifact verification failed:\n- " + "\n- ".join(errors))
    return {
        "status": "PASS",
        "unique_sample_count": len(expected_keys),
        "annotator_presentation_counts": {
            annotator_id: manifest["annotators"][annotator_id]["presentation_count"]
            for annotator_id in ANNOTATOR_IDS
        },
        "manifest_sha256": manifest["sha256"],
    }


def prepare(args, tokenizer=None):
    config_path = resolve_path(args.config)
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    defaults = config.get("experiment_defaults") or {}
    cohort_id = defaults.get("cohort_id")
    if not cohort_id:
        raise ValueError("experiment_defaults.cohort_id is required")
    cohort_dir = (
        PROJECT_ROOT / "data" / "ft_datasets" / "_cohorts" /
        cohort_id / f"seed_{args.seed}"
    )
    sampled_path = cohort_dir / "sampled.json"
    output_dir = resolve_path(args.output_dir)
    target_paths = [output_dir / name for name in TARGET_FILENAMES]

    if args.verify_only:
        return verify_artifacts(output_dir)
    existing = [path for path in target_paths if path.exists()]
    if existing:
        if len(existing) == len(target_paths):
            return verify_artifacts(output_dir)
        raise FileExistsError(
            "Annotation directory is only partially populated; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    if not sampled_path.is_file():
        raise FileNotFoundError(sampled_path)
    sampled_rows = load_jsonl(sampled_path)
    tokenizer_reference = config.get("tokenizer", config.get("model"))
    local_tokenizer_path = config.get("local_tokenizer_path", config.get("local_model_path"))
    if tokenizer is None:
        apply_offline_mode(True)
        from transformers import AutoTokenizer
        resolved_tokenizer = resolve_model_reference(
            tokenizer_reference,
            local_tokenizer_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer,
            trust_remote_code=True,
            local_files_only=True,
        )

    cutoff_len = defaults.get("cutoff_len", 512)
    contexts, blocker_counts = build_review_contexts(
        sampled_rows,
        sampled_file=sampled_path,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        expected_counts=PILOT_EXPECTED_COUNTS,
    )
    annotator_rows = {
        annotator_id: build_annotator_rows(
            contexts,
            annotator_id=annotator_id,
            seed=args.seed,
            unique_per_batch=args.unique_per_batch,
            repeats_per_batch=args.repeats_per_batch,
        )
        for annotator_id in ANNOTATOR_IDS
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    annotator_paths = {
        annotator_id: output_dir / f"{annotator_id}.jsonl"
        for annotator_id in ANNOTATOR_IDS
    }
    for annotator_id, path in annotator_paths.items():
        write_jsonl(path, annotator_rows[annotator_id])
    manifest = build_manifest(
        config_path=config_path,
        cohort_id=cohort_id,
        seed=args.seed,
        sampled_path=sampled_path,
        contexts=contexts,
        blocker_counts=blocker_counts,
        annotator_paths=annotator_paths,
        annotator_rows=annotator_rows,
        tokenizer_reference=tokenizer_reference,
        local_tokenizer_path=local_tokenizer_path,
        tokenizer_class=type(tokenizer).__name__,
        cutoff_len=cutoff_len,
        unique_per_batch=args.unique_per_batch,
        repeats_per_batch=args.repeats_per_batch,
    )
    manifest_path = output_dir / "annotation_batch_manifest.json"
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return verify_artifacts(output_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare or verify the dual-independent MR blocker review batch."
    )
    parser.add_argument(
        "--config",
        default="experiments_config_mrinstr_pilot.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="artifacts/mrinstr_annotation",
    )
    parser.add_argument("--unique-per-batch", type=int, default=100)
    parser.add_argument("--repeats-per-batch", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    configure_console_encoding()
    args = parse_args(argv)
    result = prepare(args)
    print("Annotation batch verification: PASS")
    print(f"Unique blocker samples: {result['unique_sample_count']}")
    for annotator_id, count in result["annotator_presentation_counts"].items():
        print(f"{annotator_id}: {count} presentations")
    print(f"Manifest SHA-256: {result['manifest_sha256']}")


if __name__ == "__main__":
    main()
