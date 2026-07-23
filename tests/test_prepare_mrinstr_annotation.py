import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import prepare_mrinstr_annotation as annotation


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is False
        text = " ".join(message["content"] for message in messages)
        return list(range(len(text.split())))


def sample(pair_id, mr_id, label, suffix):
    return {
        "pair_id": pair_id,
        "mr_id": mr_id,
        "premise": f"premise {suffix}",
        "hypothesis": f"hypothesis {suffix}",
        "label": label,
        "idx": suffix,
    }


def write_source(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_review_contexts_covers_only_blockers_and_includes_pair_context(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(annotation, "PROJECT_ROOT", tmp_path)
    sampled_path = tmp_path / "sampled.json"
    rows = [
        sample(7, "none", 0, "source"),
        sample(7, "adding_contradiction", 2, "risk"),
        sample(7, "synonym_replacement", 0, "sibling"),
    ]
    write_source(sampled_path, rows)

    contexts, counts = annotation.build_review_contexts(
        rows,
        sampled_file=sampled_path,
        tokenizer=FakeTokenizer(),
        cutoff_len=20,
    )

    assert counts == {"adding_contradiction": 1}
    assert len(contexts) == 1
    context = contexts[0]
    assert context["source"]["label"] == "entailment"
    assert context["stored_label"] == "contradiction"
    assert context["current"]["hypothesis"] == "hypothesis risk"
    assert [row["mr_id"] for row in context["sibling_augmented_rows"]] == [
        "synonym_replacement"
    ]
    assert context["tokenization"]["context_token_length"] > 0
    assert context["provenance"]["sampled_file"] == "sampled.json"


def test_annotator_batches_are_independent_and_have_five_repeats_per_chunk():
    contexts = [{"sample_key": f"key-{index:04d}"} for index in range(205)]
    first = annotation.build_annotator_rows(
        contexts,
        annotator_id="annotator_a",
        seed=42,
    )
    second = annotation.build_annotator_rows(
        contexts,
        annotator_id="annotator_b",
        seed=42,
    )

    assert len(first) == 220
    assert len(second) == 220
    assert {row["sample_key"] for row in first} == {
        context["sample_key"] for context in contexts
    }
    assert [row["sample_key"] for row in first] != [
        row["sample_key"] for row in second
    ]
    for rows in (first, second):
        stats = annotation.annotator_statistics(rows)
        assert stats["unique_count"] == 205
        assert stats["batch_count"] == 3
        assert stats["repeat_presentation_count"] == 15
        assert all(
            item["repeat_presentation_count"] == 5
            for item in stats["batch_statistics"].values()
        )
        assert all(row["nli_label"] is None for row in rows)
        assert all(row["review_status"] == "pending" for row in rows)


def test_prepare_rows_rejects_a_final_batch_smaller_than_repeat_requirement():
    contexts = [{"sample_key": f"key-{index}"} for index in range(102)]
    with pytest.raises(ValueError, match="Final batch"):
        annotation.build_annotator_rows(
            contexts,
            annotator_id="annotator_a",
            seed=42,
            unique_per_batch=100,
            repeats_per_batch=5,
        )


def test_manifest_and_pristine_artifacts_verify(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    sampled_path = tmp_path / "sampled.json"
    source_rows = [
        sample(1, "none", 0, "source-1"),
        sample(1, "adding_contradiction", 2, "risk-1"),
        sample(2, "none", 1, "source-2"),
        sample(2, "conditional_clause", 1, "risk-2"),
    ]
    write_source(sampled_path, source_rows)
    contexts, counts = annotation.build_review_contexts(
        source_rows,
        sampled_file=sampled_path,
        tokenizer=FakeTokenizer(),
        cutoff_len=512,
    )
    output_dir = tmp_path / "artifacts" / "mrinstr_annotation"
    output_dir.mkdir(parents=True)
    rows = {
        annotator_id: annotation.build_annotator_rows(
            contexts,
            annotator_id=annotator_id,
            seed=42,
            repeats_per_batch=1,
        )
        for annotator_id in annotation.ANNOTATOR_IDS
    }
    paths = {
        annotator_id: output_dir / f"{annotator_id}.jsonl"
        for annotator_id in annotation.ANNOTATOR_IDS
    }
    for annotator_id, path in paths.items():
        annotation.write_jsonl(path, rows[annotator_id])
    manifest = annotation.build_manifest(
        config_path=config_path,
        cohort_id="test-cohort",
        seed=42,
        sampled_path=sampled_path,
        contexts=contexts,
        blocker_counts=counts,
        annotator_paths=paths,
        annotator_rows=rows,
        tokenizer_reference="fake/tokenizer",
        local_tokenizer_path="models/fake",
        tokenizer_class="FakeTokenizer",
        cutoff_len=512,
        unique_per_batch=100,
        repeats_per_batch=1,
    )
    manifest_path = output_dir / "annotation_batch_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = annotation.verify_artifacts(output_dir)
    assert result["status"] == "PASS"
    assert result["unique_sample_count"] == 2
    assert result["annotator_presentation_counts"] == {
        "annotator_a": 3,
        "annotator_b": 3,
    }

    with open(paths["annotator_a"], "a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="initial template SHA-256 mismatch"):
        annotation.verify_artifacts(output_dir)
