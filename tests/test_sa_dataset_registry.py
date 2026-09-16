"""Offline tests for SA dataset import: parsers, label maps, IDs and split assembly."""

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sa_dataset_registry as registry


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #

def test_cad_tsv_maps_positive_negative_words():
    raw = b"Sentiment\tText\nPositive\tA wonderful film.\nNegative\tA dreadful film.\n"
    result = registry.parse_cad_tsv(raw)
    assert [record.label for record in result.records] == ["positive", "negative"]
    assert result.records[0].text == "A wonderful film."


def test_cad_tsv_rejects_unknown_label():
    raw = b"Sentiment\tText\nMixed\tSomething.\n"
    with pytest.raises(registry.SADatasetError):
        registry.parse_cad_tsv(raw)


def test_sst2_tsv_without_index_column():
    raw = b"A good film .\tpositive\nA bad film .\tnegative\n"
    result = registry.parse_sst2_tsv(raw)
    assert [record.text for record in result.records] == ["A good film .", "A bad film ."]
    assert result.meta["label_encoding"] == "words"
    assert result.meta["rows_with_leading_index_column"] == 0


def test_sst2_tsv_with_index_column_is_stripped():
    raw = b"0\tA good film .\tpositive\n1\tA bad film .\tnegative\n"
    result = registry.parse_sst2_tsv(raw)
    assert [record.text for record in result.records] == ["A good film .", "A bad film ."]
    assert result.meta["rows_with_leading_index_column"] == 2


def test_sst2_tsv_accepts_numeric_encoding_and_records_it():
    raw = b"A good film .\t1\nA bad film .\t0\n"
    result = registry.parse_sst2_tsv(raw)
    assert [record.label for record in result.records] == ["positive", "negative"]
    assert result.meta["label_encoding"] == "ints"


def test_sst2_text_containing_a_tab_is_not_treated_as_an_index():
    raw = b"I saw film number 3\tin a row\tpositive\n"
    result = registry.parse_sst2_tsv(raw)
    assert result.records[0].text == "I saw film number 3 in a row"


def test_twitter_parser_preserves_embedded_tabs_and_rejects_zero_four_labels():
    raw = b"1\tA great\tfilm with a tab\n0\tA bad film\n"
    result = registry.parse_twitter_tsv(raw)
    assert result.records[0].text == "A great film with a tab"
    assert [record.label for record in result.records] == ["positive", "negative"]

    with pytest.raises(registry.SADatasetError, match="0/4"):
        registry.parse_twitter_tsv(b"4\tSentiment140 native label\n")


def test_label_text_csv_maps_binary_ints():
    raw = b"label,context\n1,Good product\n0,Bad product\n"
    result = registry.parse_label_text_csv(raw)
    assert [record.label for record in result.records] == ["positive", "negative"]
    assert result.records[0].text == "Good product"


def test_cad_paired_tsv_records_batch_ids():
    raw = (
        b"Sentiment\tText\tbatch_id\n"
        b"Negative\tNot good at all.\t4\n"
        b"Positive\tReally good indeed.\t4\n"
    )
    result = registry.parse_cad_paired_tsv(raw)
    assert result.meta["batch_ids"] == ["4", "4"]
    assert result.meta["pair_key"] == "batch_id"


def test_build_human_cad_pairs_requires_inverted_labels():
    raw = (
        b"Sentiment\tText\tbatch_id\n"
        b"Negative\tNot good at all.\t4\n"
        b"Negative\tAlso not good.\t4\n"
    )
    parsed = registry.parse_cad_paired_tsv(raw)
    meta = {"filename": "paired.tsv", "sha256": "deadbeef"}
    with pytest.raises(registry.SADatasetError, match="matching labels"):
        registry.build_human_cad_pairs(parsed, meta)


def test_build_human_cad_pairs_orders_original_before_counterfactual():
    raw = (
        b"Sentiment\tText\tbatch_id\n"
        b"Negative\tNot good at all.\t4\n"
        b"Positive\tReally good indeed.\t4\n"
    )
    parsed = registry.parse_cad_paired_tsv(raw)
    pairs, meta = registry.build_human_cad_pairs(
        parsed, {"filename": "paired.tsv", "sha256": "deadbeef"}
    )
    assert meta["pair_count"] == 1
    assert pairs[0]["source_label"] == "negative"
    assert pairs[0]["counterfactual_label"] == "positive"
    assert pairs[0]["counterfactual_text"] == "Really good indeed."


# --------------------------------------------------------------------------- #
# Normalization and identifiers
# --------------------------------------------------------------------------- #

def test_normalize_text_collapses_html_breaks_and_whitespace():
    assert registry.normalize_text("Line one<br />Line two") == "Line one Line two"
    assert registry.normalize_text("  spaced   out  ") == "spaced out"
    assert registry.normalize_text("a<br/>b") == "a b"


def test_sample_ids_are_stable_and_zero_padded():
    assert registry.make_sample_id("imdb-test", 7) == "imdb-test-000007"
    assert registry.make_sample_id("imdb-test", 7) == registry.make_sample_id("imdb-test", 7)


def test_canonical_row_hash_ignores_split_role_but_tracks_text():
    base = {
        "dataset": "imdb",
        "source_id": "imdb-test-000001",
        "text": "A film.",
        "label": "positive",
    }
    first = registry.canonical_row_hash({**base, "split_role": "standard_test"})
    second = registry.canonical_row_hash({**base, "split_role": "mr_test_source_pool"})
    assert first == second
    assert first != registry.canonical_row_hash({**base, "text": "Another film."})


# --------------------------------------------------------------------------- #
# Split assembly
# --------------------------------------------------------------------------- #

def _rows(n_positive, n_negative):
    rows = []
    for index in range(n_positive):
        rows.append({"source_id": f"p{index}", "text": f"good {index}", "label": "positive"})
    for index in range(n_negative):
        rows.append({"source_id": f"n{index}", "text": f"bad {index}", "label": "negative"})
    return rows


def test_stratified_carve_is_balanced_and_deterministic():
    rows = _rows(300, 300)
    carve, remainder = registry.stratified_carve(rows, 400, seed=42)
    assert len(carve) == 400
    assert len(remainder) == 200
    counts = registry.label_counts(carve)
    assert counts == {"positive": 200, "negative": 200}

    again, _ = registry.stratified_carve(rows, 400, seed=42)
    assert [row["source_id"] for row in carve] == [row["source_id"] for row in again]


def test_stratified_carve_changes_with_seed():
    rows = _rows(300, 300)
    first, _ = registry.stratified_carve(rows, 100, seed=42)
    second, _ = registry.stratified_carve(rows, 100, seed=43)
    assert [row["source_id"] for row in first] != [row["source_id"] for row in second]


def test_stratified_carve_raises_when_a_label_is_exhausted():
    with pytest.raises(registry.SADatasetError, match="cannot carve"):
        registry.stratified_carve(_rows(10, 300), 300, seed=42)


def _import_style_spec(**overrides):
    base = dict(
        dataset="toy",
        display_name="Toy",
        text_field="text",
        label_field="label",
        version="v1",
        license_note="test",
        sources=(),
    )
    base.update(overrides)
    return registry.SADatasetSpec(**base)


def test_build_domain_splits_carves_disjoint_mr_pool():
    spec = _import_style_spec(mr_pool_size=100)
    parsed = registry.ParseResult(
        records=tuple(
            registry.ParsedRecord(text=row["text"], label=row["label"], source_file_index=index)
            for index, row in enumerate(_rows(300, 300), start=1)
        ),
        meta={},
    )
    source = registry.SASourceSpec(
        role=registry.ROLE_STANDARD_TEST,
        filename="test.tsv",
        url="http://example.invalid/test.tsv",
        parser="cad_tsv",
        id_prefix="toy-test",
    )
    spec = _import_style_spec(sources=(source,), mr_pool_size=100)
    splits, assembly = registry.build_domain_splits(
        spec,
        {registry.ROLE_STANDARD_TEST: parsed},
        {registry.ROLE_STANDARD_TEST: {"filename": "test.tsv", "sha256": "abc"}},
        seed=42,
    )
    standard = splits[registry.ROLE_STANDARD_TEST]
    pool = splits[registry.ROLE_MR_TEST_SOURCE_POOL]
    assert len(pool) == 100
    assert len(standard) == 500
    assert not ({row["source_id"] for row in standard} & {row["source_id"] for row in pool})
    assert assembly["standard_test_equals_mr_test_source_pool"] is False
    assert registry.label_counts(pool) == {"positive": 50, "negative": 50}


def test_build_domain_splits_shares_roles_when_declared():
    spec = _import_style_spec(standard_test_is_mr_pool=True, sources=(
        registry.SASourceSpec(
            role=registry.ROLE_STANDARD_TEST,
            filename="test.tsv",
            url="http://example.invalid/test.tsv",
            parser="cad_tsv",
            id_prefix="toy-test",
        ),
    ))
    parsed = registry.ParseResult(
        records=(
            registry.ParsedRecord(text="good", label="positive", source_file_index=1),
            registry.ParsedRecord(text="bad", label="negative", source_file_index=2),
        ),
        meta={},
    )
    splits, assembly = registry.build_domain_splits(
        spec,
        {registry.ROLE_STANDARD_TEST: parsed},
        {registry.ROLE_STANDARD_TEST: {"filename": "test.tsv", "sha256": "abc"}},
    )
    assert assembly["standard_test_equals_mr_test_source_pool"] is True
    assert [row["source_id"] for row in splits[registry.ROLE_STANDARD_TEST]] == [
        row["source_id"] for row in splits[registry.ROLE_MR_TEST_SOURCE_POOL]
    ]
    assert splits[registry.ROLE_STANDARD_TEST][0]["shared_role"] == [
        registry.ROLE_STANDARD_TEST,
        registry.ROLE_MR_TEST_SOURCE_POOL,
    ]


def test_imdb_declares_the_shared_pool_deviation():
    deviations = registry.build_deviations(registry.IMDB)
    assert len(deviations) == 1
    deviation = deviations[0]
    assert deviation["id"] == "imdb_standard_test_eq_mr_source_pool"
    assert deviation["decision"] == "user-approved"
    assert "NOT independent" in deviation["consequence"]


def test_only_imdb_declares_a_deviation():
    for name in ("amazon", "yelp", "twitter", "sst2"):
        assert registry.build_deviations(registry.SA_DATASETS[name]) == []


# --------------------------------------------------------------------------- #
# Registry shape
# --------------------------------------------------------------------------- #

def test_registry_covers_the_five_route_document_domains():
    assert set(registry.SA_DATASETS) == {"imdb", "amazon", "yelp", "twitter", "sst2"}


def test_every_llm_parser_is_registered():
    for spec in registry.SA_DATASETS.values():
        for source in spec.sources:
            assert source.parser in registry.PARSERS, source.parser


def test_imdb_sources_match_the_route_document_split_sizes():
    roles = {source.role: source for source in registry.IMDB.sources}
    assert roles[registry.ROLE_TRAIN_SOURCE_POOL].expected_rows == 1707
    assert roles[registry.ROLE_COMMON_DEV].expected_rows == 245
    assert roles[registry.ROLE_STANDARD_TEST].expected_rows == 488


def test_duplicate_report_detects_repeats_and_corpus_overlap():
    rows = [
        {"text": "A good film.", "label": "positive"},
        {"text": "A good film.", "label": "positive"},
        {"text": "A bad film.", "label": "negative"},
    ]
    within = registry.duplicate_report(rows)
    assert within["rows"] == 3
    assert within["duplicate_rows"] == 1

    corpus = [{"text": "A bad film.", "label": "negative"}]
    against = registry.duplicate_report(rows, corpus=corpus)
    assert against["overlapping_text_keys"] == 1


def test_normalized_text_key_ignores_case_and_punctuation():
    assert registry.normalized_text_key("A Great Film!") == registry.normalized_text_key(
        "a great film"
    )
