"""Declarative registry for the sentiment-analysis (SA) RQ1 evaluation domains.

This module owns the *pure* half of SA data import: which upstream files make up
each domain, how their bytes are parsed, how labels map onto the binary SA label
space, how stable sample IDs are formed, and how the frozen split roles are
assembled.  It performs no network or filesystem IO -- ``download_sa_datasets.py``
is the thin CLI that does that.

Roles follow the RQ1 SA route document (section 5.2):

``train_source_pool``    common source cohort for OR / AutoCAD / MR training
``common_dev``           shared checkpoint-selection set
``standard_test``        ordinary in-domain classification evaluation
``mr_test_source_pool``  sources reserved for MR generation

Label space is always the two strings ``positive`` / ``negative``; upstream
0/1 or ``Positive``/``Negative`` encodings are normalized at import time.

See ``DEFAULTS`` and ``SA_DATASETS`` for the frozen upstream versions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

TASK_NAME = "sentiment_analysis"
REGISTRY_VERSION = "sa_registry_v1"
DEFAULT_SEED = 42

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
SA_LABELS: tuple[str, str] = (LABEL_NEGATIVE, LABEL_POSITIVE)
OPPOSITE_LABEL = {LABEL_POSITIVE: LABEL_NEGATIVE, LABEL_NEGATIVE: LABEL_POSITIVE}

ROLE_TRAIN_SOURCE_POOL = "train_source_pool"
ROLE_COMMON_DEV = "common_dev"
ROLE_STANDARD_TEST = "standard_test"
ROLE_MR_TEST_SOURCE_POOL = "mr_test_source_pool"
SA_ROLES: tuple[str, ...] = (
    ROLE_TRAIN_SOURCE_POOL,
    ROLE_COMMON_DEV,
    ROLE_STANDARD_TEST,
    ROLE_MR_TEST_SOURCE_POOL,
)

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

#: Normalization applied to every imported text, recorded in each manifest.
TEXT_NORMALIZATION = (
    "strip HTML <br> separators, replace runs of whitespace with a single space, strip ends"
)

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

#: Label encodings seen upstream.
POS_NEG_WORD_LABELS = {
    "positive": LABEL_POSITIVE,
    "negative": LABEL_NEGATIVE,
    "pos": LABEL_POSITIVE,
    "neg": LABEL_NEGATIVE,
}
BINARY_INT_LABELS = {"0": LABEL_NEGATIVE, "1": LABEL_POSITIVE}


class SADatasetError(ValueError):
    """Raised when an upstream file cannot be parsed into canonical SA rows."""


# --------------------------------------------------------------------------- #
# Records and parse results
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ParsedRecord:
    text: str
    label: str
    source_file_index: int


@dataclass(frozen=True)
class ParseResult:
    records: tuple[ParsedRecord, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SASourceSpec:
    """One upstream file that contributes rows to a dataset."""

    role: str
    filename: str
    url: str
    parser: str
    id_prefix: str
    expected_rows: int | None = None
    note: str = ""


@dataclass(frozen=True)
class SADatasetSpec:
    dataset: str
    display_name: str
    text_field: str
    label_field: str
    version: str
    license_note: str
    sources: tuple[SASourceSpec, ...]
    #: Rows carved out of an over-large split into ``mr_test_source_pool``.
    mr_pool_size: int = 0
    #: Set when ``standard_test`` and ``mr_test_source_pool`` are the same rows.
    standard_test_is_mr_pool: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- #
# Text normalization and stable identifiers
# --------------------------------------------------------------------------- #

def normalize_text(value: str) -> str:
    """Collapse HTML line breaks and whitespace runs; preserve all other content."""
    text = _BR_RE.sub(" ", value.replace("\r\n", "\n").replace("\r", "\n"))
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalized_text_key(text: str) -> str:
    """Key used for duplicate and leakage checks (word tokens, case-folded)."""
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_sample_id(id_prefix: str, source_file_index: int) -> str:
    """Stable, human-readable ID derived from the upstream file order."""
    return f"{id_prefix}-{source_file_index:06d}"


def make_canonical_row(
    *,
    dataset: str,
    split_role: str,
    sample_id: str,
    text: str,
    label: str,
    source_file: str,
    source_file_index: int,
    source_file_sha256: str,
    shared_role: Sequence[str] = (),
) -> dict[str, Any]:
    row = {
        "task": TASK_NAME,
        "dataset": dataset,
        "split_role": split_role,
        "source_id": sample_id,
        "sample_id": sample_id,
        "text": text,
        "label": label,
        "source_file": source_file,
        "source_file_index": source_file_index,
        "source_file_sha256": source_file_sha256,
        "shared_role": list(shared_role),
    }
    row["provenance_hash"] = canonical_row_hash(row)
    return row


def canonical_row_hash(row: Mapping[str, Any]) -> str:
    payload = {
        "dataset": row.get("dataset"),
        "source_id": row.get("source_id"),
        "text": row.get("text"),
        "label": row.get("label"),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# --------------------------------------------------------------------------- #
# Parsers: raw bytes -> ParsedRecord tuple
# --------------------------------------------------------------------------- #

def _decode(raw: bytes) -> str:
    return raw.decode("utf-8-sig")


def _finalize(
    records: list[ParsedRecord], meta: Mapping[str, Any]
) -> ParseResult:
    if not records:
        raise SADatasetError("parser produced no records")
    return ParseResult(records=tuple(records), meta=dict(meta))


def parse_cad_tsv(raw: bytes) -> ParseResult:
    """IMDb CAD split: header ``Sentiment\\tText`` with ``Positive``/``Negative``."""
    reader = csv.DictReader(io.StringIO(_decode(raw)), delimiter="\t")
    fieldnames = {(name or "").strip() for name in (reader.fieldnames or [])}
    if not {"Sentiment", "Text"} <= fieldnames:
        raise SADatasetError(f"unexpected CAD TSV header: {sorted(fieldnames)}")
    records: list[ParsedRecord] = []
    for offset, row in enumerate(reader, start=1):
        label = POS_NEG_WORD_LABELS.get((row.get("Sentiment") or "").strip().casefold())
        if label is None:
            raise SADatasetError(f"CAD TSV row {offset}: unknown label {row.get('Sentiment')!r}")
        text = normalize_text(row.get("Text") or "")
        if not text:
            raise SADatasetError(f"CAD TSV row {offset}: empty text")
        records.append(ParsedRecord(text=text, label=label, source_file_index=offset))
    return _finalize(records, {"format": "cad_tsv", "label_encoding": "Positive/Negative"})


def parse_sst2_tsv(raw: bytes) -> ParseResult:
    """SST-2 TSV, no header.

    Tolerates both ``text<TAB>label`` and ``index<TAB>text<TAB>label`` and both
    the literal ``positive``/``negative`` encoding and the numeric ``0``/``1``
    encoding.  Which variant was seen is recorded in ``meta`` so the mapping
    stays auditable.
    """
    records: list[ParsedRecord] = []
    label_encodings: Counter = Counter()
    index_columns = 0
    for line_number, line in enumerate(_decode(raw).splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise SADatasetError(f"SST-2 line {line_number}: fewer than two tab fields")
        label_raw = parts[-1].strip().casefold()
        body = parts[:-1]
        if len(body) >= 2 and body[0].strip().isdigit():
            body = body[1:]
            index_columns += 1
        label = POS_NEG_WORD_LABELS.get(label_raw) or BINARY_INT_LABELS.get(label_raw)
        if label is None:
            raise SADatasetError(f"SST-2 line {line_number}: unknown label {parts[-1]!r}")
        label_encodings["words" if label_raw in POS_NEG_WORD_LABELS else "ints"] += 1
        text = normalize_text("\t".join(body))
        if not text:
            raise SADatasetError(f"SST-2 line {line_number}: empty text")
        records.append(ParsedRecord(text=text, label=label, source_file_index=len(records) + 1))
    return _finalize(
        records,
        {
            "format": "sst2_tsv",
            "label_encoding": "+".join(sorted(label_encodings)),
            "rows_with_leading_index_column": index_columns,
        },
    )


def parse_twitter_tsv(raw: bytes) -> ParseResult:
    """Twitter corpus: ``label<TAB>text`` with the text kept verbatim."""
    records: list[ParsedRecord] = []
    label_values: Counter = Counter()
    for line_number, line in enumerate(_decode(raw).splitlines(), start=1):
        if not line.strip():
            continue
        label_raw, separator, body = line.partition("\t")
        if not separator:
            raise SADatasetError(f"Twitter line {line_number}: no tab separator")
        label_key = label_raw.strip()
        label_values[label_key] += 1
        label = BINARY_INT_LABELS.get(label_key)
        if label is None:
            raise SADatasetError(
                f"Twitter line {line_number}: label {label_key!r} is not 0/1; "
                "Sentiment140 native 0/4 labels must be remapped before import"
            )
        text = normalize_text(body)
        if not text:
            raise SADatasetError(f"Twitter line {line_number}: empty text")
        records.append(ParsedRecord(text=text, label=label, source_file_index=len(records) + 1))
    return _finalize(
        records,
        {"format": "twitter_tsv", "label_encoding": "0/1", "raw_label_counts": dict(label_values)},
    )


def parse_label_text_csv(raw: bytes) -> ParseResult:
    """Amazon-style CSV: header ``label,context`` with 0/1 labels."""
    reader = csv.DictReader(io.StringIO(_decode(raw)))
    fieldnames = {(name or "").strip() for name in (reader.fieldnames or [])}
    if not {"label", "context"} <= fieldnames:
        raise SADatasetError(f"unexpected label/context CSV header: {sorted(fieldnames)}")
    records: list[ParsedRecord] = []
    for offset, row in enumerate(reader, start=1):
        label = BINARY_INT_LABELS.get((row.get("label") or "").strip())
        if label is None:
            raise SADatasetError(f"CSV row {offset}: unknown label {row.get('label')!r}")
        text = normalize_text(row.get("context") or "")
        if not text:
            raise SADatasetError(f"CSV row {offset}: empty text")
        records.append(ParsedRecord(text=text, label=label, source_file_index=offset))
    return _finalize(records, {"format": "label_text_csv", "label_encoding": "0/1"})


def _parse_parquet_text_label(raw: bytes, text_column: str, label_column: str) -> ParseResult:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(raw))
    columns = set(table.column_names)
    if not {text_column, label_column} <= columns:
        raise SADatasetError(f"unexpected parquet columns: {sorted(columns)}")
    texts = table.column(text_column).to_pylist()
    labels = table.column(label_column).to_pylist()
    records: list[ParsedRecord] = []
    for offset, (text, label_value) in enumerate(zip(texts, labels), start=1):
        label = BINARY_INT_LABELS.get(str(label_value).strip())
        if label is None:
            raise SADatasetError(f"parquet row {offset}: unknown label {label_value!r}")
        normalized = normalize_text(text or "")
        if not normalized:
            raise SADatasetError(f"parquet row {offset}: empty text")
        records.append(ParsedRecord(text=normalized, label=label, source_file_index=offset))
    return _finalize(records, {"format": "parquet", "label_encoding": "0/1"})


def parse_yelp_parquet(raw: bytes) -> ParseResult:
    """Yelp Polarity: parquet with ``text`` / ``label`` (0 = negative, 1 = positive)."""
    return _parse_parquet_text_label(raw, "text", "label")


def parse_imdb_parquet(raw: bytes) -> ParseResult:
    """Full IMDb: parquet with ``text`` / ``label`` (0 = negative, 1 = positive)."""
    return _parse_parquet_text_label(raw, "text", "label")


PARSERS: dict[str, Callable[[bytes], ParseResult]] = {
    "cad_tsv": parse_cad_tsv,
    "sst2_tsv": parse_sst2_tsv,
    "twitter_tsv": parse_twitter_tsv,
    "label_text_csv": parse_label_text_csv,
    "yelp_parquet": parse_yelp_parquet,
    "imdb_parquet": parse_imdb_parquet,
}


def parse_source(parser: str, raw: bytes) -> ParseResult:
    try:
        parser_fn = PARSERS[parser]
    except KeyError as exc:
        raise SADatasetError(f"unknown parser {parser!r}") from exc
    return parser_fn(raw)


# --------------------------------------------------------------------------- #
# Frozen upstream specifications
# --------------------------------------------------------------------------- #

_PAIRCFR_RAW = "https://raw.githubusercontent.com/Siki-cloud/paircfr/main/runimdb/data"
_TWITTER_RAW = (
    "https://raw.githubusercontent.com/cblancac/SentimentAnalysisBert/main/data"
)


def _hf_file(repo_id: str, path: str) -> str:
    return f"{HF_MIRROR_ENDPOINT}/datasets/{repo_id}/resolve/main/{path}"


IMDB = SADatasetSpec(
    dataset="imdb",
    display_name="IMDb",
    text_field="text",
    label_field="label",
    version="paircfr-original17+revised17",
    license_note="Research use; CAD splits from Kaushik et al. 2020 via the PairCFR mirror of acmi-lab/counterfactually-augmented-data",
    sources=(
        SASourceSpec(
            role=ROLE_TRAIN_SOURCE_POOL,
            filename="original17_train.tsv",
            url=f"{_PAIRCFR_RAW}/original17/train.tsv",
            parser="cad_tsv",
            id_prefix="imdb-train",
            expected_rows=1707,
            note="common source cohort for OR / AutoCAD / MR",
        ),
        SASourceSpec(
            role=ROLE_COMMON_DEV,
            filename="original17_dev.tsv",
            url=f"{_PAIRCFR_RAW}/original17/dev.tsv",
            parser="cad_tsv",
            id_prefix="imdb-dev",
            expected_rows=245,
            note="shared checkpoint selection set",
        ),
        SASourceSpec(
            role=ROLE_STANDARD_TEST,
            filename="original17_test.tsv",
            url=f"{_PAIRCFR_RAW}/original17/test.tsv",
            parser="cad_tsv",
            id_prefix="imdb-test",
            expected_rows=488,
            note="in-domain test; also serves as the MR source pool (see deviations)",
        ),
        SASourceSpec(
            role="human_cad_reference",
            filename="revised17_train.tsv",
            url=f"{_PAIRCFR_RAW}/revised17/train.tsv",
            parser="cad_tsv",
            id_prefix="imdb-cadrev-train",
            expected_rows=1707,
            note="human counterfactuals; quality reference for flip MRs, never a test set",
        ),
    ),
    standard_test_is_mr_pool=True,
    notes="Human-CAD revised test is deliberately excluded per route document section 12.5",
)

AMAZON = SADatasetSpec(
    dataset="amazon",
    display_name="Amazon",
    text_field="text",
    label_field="label",
    version="Siki-77/amazon6_5core_polarity@main",
    license_note="Apache-2.0; derived from Amazon Reviews (Ni et al. 2019) six-genre 5-core",
    sources=(
        SASourceSpec(
            role=ROLE_STANDARD_TEST,
            filename="test.csv",
            url=_hf_file("Siki-77/amazon6_5core_polarity", "test.csv"),
            parser="label_text_csv",
            id_prefix="amazon-test",
            expected_rows=106740,
        ),
    ),
    mr_pool_size=400,
)

YELP = SADatasetSpec(
    dataset="yelp",
    display_name="Yelp Polarity",
    text_field="text",
    label_field="label",
    version="fancyzhx/yelp_polarity@main",
    license_note="See upstream dataset card; Yelp Open Dataset derived",
    sources=(
        SASourceSpec(
            role=ROLE_STANDARD_TEST,
            filename="yelp_polarity_test.parquet",
            url=_hf_file("fancyzhx/yelp_polarity", "plain_text/test-00000-of-00001.parquet"),
            parser="yelp_parquet",
            id_prefix="yelp-test",
            expected_rows=38000,
        ),
    ),
    mr_pool_size=400,
)

TWITTER = SADatasetSpec(
    dataset="twitter",
    display_name="Twitter",
    text_field="text",
    label_field="label",
    version="cblancac/SentimentAnalysisBert test_62k",
    license_note="See upstream repository (TSATC corpus)",
    sources=(
        SASourceSpec(
            role=ROLE_STANDARD_TEST,
            filename="twitter_test_62k.txt",
            url=f"{_TWITTER_RAW}/test_62k.txt",
            parser="twitter_tsv",
            id_prefix="twitter-test",
            expected_rows=61998,
        ),
    ),
    mr_pool_size=400,
    notes="Short, noisy social-media text; MR applicability is expected to differ from review domains",
)

SST2 = SADatasetSpec(
    dataset="sst2",
    display_name="SST-2",
    text_field="text",
    label_field="label",
    version="gpt3mix/sst2@main",
    license_note="Third-party mirror of GLUE SST-2; verify upstream licence before publication",
    sources=(
        SASourceSpec(
            role=ROLE_STANDARD_TEST,
            filename="sst2_test.tsv",
            url=_hf_file("gpt3mix/sst2", "data/test.tsv"),
            parser="sst2_tsv",
            id_prefix="sst2-test",
            expected_rows=1820,
        ),
    ),
    mr_pool_size=400,
    notes="Short single sentences; primary short-text contrast against IMDb long reviews",
)

SA_DATASETS: dict[str, SADatasetSpec] = {
    spec.dataset: spec
    for spec in (IMDB, AMAZON, YELP, TWITTER, SST2)
}

DEFAULT_MR_POOL_SIZE = 400
DEFAULT_DATASETS: tuple[str, ...] = tuple(SA_DATASETS)


# --------------------------------------------------------------------------- #
# Split assembly
# --------------------------------------------------------------------------- #

def stratified_carve(
    rows: Sequence[Mapping[str, Any]], size: int, seed: int = DEFAULT_SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split ``rows`` into a label-balanced ``size``-row carve and the remainder.

    Selection is deterministic in ``seed`` and label-stratified, so the carve is
    balanced whenever both labels have enough rows.  Returns ``(carve, remainder)``
    preserving the original row order inside each part.
    """
    if size < 0:
        raise SADatasetError("carve size cannot be negative")
    if size > len(rows):
        raise SADatasetError(f"cannot carve {size} rows from {len(rows)}")

    import random

    by_label: dict[str, list[int]] = {label: [] for label in SA_LABELS}
    for index, row in enumerate(rows):
        label = row["label"]
        if label not in by_label:
            raise SADatasetError(f"row {index} has label outside the SA label space: {label!r}")
        by_label[label].append(index)

    rng = random.Random(seed)
    per_label = size // len(SA_LABELS)
    remainder_size = size - per_label * len(SA_LABELS)

    chosen: list[int] = []
    for position, label in enumerate(SA_LABELS):
        wanted = per_label + (1 if position < remainder_size else 0)
        available = by_label[label]
        if wanted > len(available):
            raise SADatasetError(
                f"cannot carve {wanted} {label} rows; only {len(available)} available"
            )
        chosen.extend(rng.sample(available, wanted))

    chosen_set = set(chosen)
    carve = [dict(rows[index]) for index in sorted(chosen_set)]
    remainder = [dict(row) for index, row in enumerate(rows) if index not in chosen_set]
    return carve, remainder


def assign_shared_role(
    rows: Iterable[dict[str, Any]], roles: Sequence[str]
) -> list[dict[str, Any]]:
    """Tag rows as belonging to more than one split role and refresh the hash."""
    shared = list(roles)
    tagged = []
    for row in rows:
        updated = {**row, "shared_role": shared}
        updated["provenance_hash"] = canonical_row_hash(updated)
        tagged.append(updated)
    return tagged


def build_domain_splits(
    spec: SADatasetSpec,
    parsed_by_role: Mapping[str, ParseResult],
    source_meta: Mapping[str, Mapping[str, Any]],
    seed: int = DEFAULT_SEED,
    mr_pool_size: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Assemble canonical rows per split role for one domain.

    ``source_meta`` maps a role to that source file's ``{filename, sha256}`` record
    so each canonical row can point back at the exact bytes it came from.
    """
    if mr_pool_size is None:
        mr_pool_size = spec.mr_pool_size or DEFAULT_MR_POOL_SIZE

    def rows_for(role: str) -> list[dict[str, Any]]:
        parsed = parsed_by_role[role]
        meta = source_meta[role]
        source_spec = next(item for item in spec.sources if item.role == role)
        return [
            make_canonical_row(
                dataset=spec.dataset,
                split_role=role,
                sample_id=make_sample_id(source_spec.id_prefix, record.source_file_index),
                text=record.text,
                label=record.label,
                source_file=meta["filename"],
                source_file_index=record.source_file_index,
                source_file_sha256=meta["sha256"],
            )
            for record in parsed.records
        ]

    splits: dict[str, list[dict[str, Any]]] = {}
    assembly: dict[str, Any] = {}

    if spec.standard_test_is_mr_pool:
        standard = rows_for(ROLE_STANDARD_TEST)
        splits[ROLE_STANDARD_TEST] = assign_shared_role(
            standard, (ROLE_STANDARD_TEST, ROLE_MR_TEST_SOURCE_POOL)
        )
        splits[ROLE_MR_TEST_SOURCE_POOL] = [dict(row) for row in splits[ROLE_STANDARD_TEST]]
        assembly["standard_test_equals_mr_test_source_pool"] = True
    else:
        standard = rows_for(ROLE_STANDARD_TEST)
        carve, remainder = stratified_carve(standard, mr_pool_size, seed)
        for row in carve:
            row["split_role"] = ROLE_MR_TEST_SOURCE_POOL
        splits[ROLE_STANDARD_TEST] = remainder
        splits[ROLE_MR_TEST_SOURCE_POOL] = carve
        assembly["standard_test_equals_mr_test_source_pool"] = False
        assembly["mr_pool_size_requested"] = mr_pool_size

    for role in (ROLE_TRAIN_SOURCE_POOL, ROLE_COMMON_DEV):
        if role in parsed_by_role:
            splits[role] = rows_for(role)

    for role in ("human_cad_reference",):
        if role in parsed_by_role:
            splits[role] = rows_for(role)

    return splits, assembly


def label_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["label"] for row in rows))


def duplicate_report(
    rows: Sequence[Mapping[str, Any]], *, corpus: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Within-set duplicate keys, plus overlap against an optional reference corpus."""
    keys = [normalized_text_key(row["text"]) for row in rows]
    counter = Counter(keys)
    duplicates = {key: count for key, count in counter.items() if count > 1 and key}
    report: dict[str, Any] = {
        "rows": len(rows),
        "unique_text_keys": len(counter),
        "duplicate_keys": len(duplicates),
        "duplicate_rows": sum(count - 1 for count in duplicates.values()),
    }
    if corpus is not None:
        corpus_keys = {normalized_text_key(row["text"]) for row in corpus}
        overlapping = sum(1 for key in counter if key and key in corpus_keys)
        report["reference_rows"] = len(corpus)
        report["overlapping_text_keys"] = overlapping
        report["overlap_rate"] = (overlapping / len(counter)) if counter else None
    return report


def build_deviations(spec: SADatasetSpec) -> list[dict[str, Any]]:
    """Machine-readable record of where the frozen splits depart from the route doc."""
    if not spec.standard_test_is_mr_pool:
        return []
    return [
        {
            "id": f"{spec.dataset}_standard_test_eq_mr_source_pool",
            "spec_ref": "5.2 / 5.4",
            "decision": "user-approved",
            "rationale": (
                "The CAD test split originals are the natural held-out cohort: they are "
                "guaranteed disjoint from the CAD train and dev splits and share the "
                "training cohort's length distribution, so they are reused as the MR "
                "generation source pool instead of carving a separate subset."
            ),
            "consequence": (
                "IMDb standard-test accuracy and IMDb MR metrics are computed over the same "
                "source strings, so they are NOT independent pieces of evidence. Every "
                "downstream report must print this warning. The full 25k IMDb test split is "
                "the ready fix if an independent standard test is needed later."
            ),
        }
    ]
