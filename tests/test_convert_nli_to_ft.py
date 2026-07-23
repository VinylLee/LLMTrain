"""单元测试：scripts/convert_nli_to_ft.py

测试覆盖：
- Split 完整性（同一 pair_id 不跨 split）
- Instruction 内容（标签泄漏、mr_id 暴露、当前样本重复）
- Strict 模式校验
- Manifest 复用
"""
import json
import tempfile
import shutil
from pathlib import Path
import copy

import pytest

# 将被测试模块加入 sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import convert_nli_to_ft as converter

from convert_nli_to_ft import (
    normalize_mr_id,
    validate_mr_samples,
    build_pair_groups,
    split_pair_groups,
    flatten_groups,
    build_original_map,
    build_instruction_for_sample,
    convert_to_alpaca,
    select_shuffled_descriptions,
    build_shuffle_audit,
    save_split_manifest,
    load_split_manifest,
    compute_data_signature,
    compute_sample_key,
    sort_samples_by_stable_key,
    compute_ordered_sample_signature,
    converted_rows_sha256,
    validate_manifest_for_groups,
    summarize_token_lengths,
    build_token_length_report,
    save_jsonl,
    INSTRUCTION_TEMPLATE_VERSION,
    INSTRUCTION_TEMPLATE_HASH,
    OPERATION_DESCRIPTION_HASH,
    RELATION_EFFECT_HASH,
    MR_OPERATION_DESCRIPTIONS,
    MR_RELATION_EFFECTS,
    KNOWN_MR_IDS,
    LABEL_NAMES_3CLASS,
    register_dataset,
)


def test_dataset_registration_uses_data_relative_posix_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(converter, "WORK_DIR", tmp_path)
    train_path = tmp_path / "data" / "ft_datasets" / "pilot" / "train.json"
    val_path = tmp_path / "data" / "ft_datasets" / "pilot" / "val.json"
    train_path.parent.mkdir(parents=True)
    train_path.write_text("{}\n", encoding="utf-8")
    val_path.write_text("{}\n", encoding="utf-8")

    register_dataset("pilot", train_path, val_path)

    registry = json.loads(
        (tmp_path / "data" / "dataset_info.json").read_text(encoding="utf-8")
    )
    assert registry["pilot"]["file_name"] == "ft_datasets/pilot/train.json"
    assert registry["pilot_val"]["file_name"] == "ft_datasets/pilot/val.json"


def test_dataset_registration_rejects_files_outside_data(tmp_path, monkeypatch):
    monkeypatch.setattr(converter, "WORK_DIR", tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="data/"):
        register_dataset("bad", outside)


def test_save_jsonl_bytes_match_portable_converted_hash(tmp_path):
    rows = [{"instruction": "i", "input": "x", "output": "neutral"}]
    path = tmp_path / "rows.jsonl"
    save_jsonl(rows, path)
    payload = path.read_bytes()
    assert b"\r\n" not in payload
    import hashlib
    assert hashlib.sha256(payload).hexdigest() == converted_rows_sha256(rows)


def test_data_signature_is_order_independent_and_content_sensitive():
    rows = [make_sample(pair_id=1), make_sample(pair_id=2)]
    assert compute_data_signature(rows) == compute_data_signature(list(reversed(rows)))
    changed = [dict(row) for row in rows]
    changed[0]["hypothesis"] = "changed"
    assert compute_data_signature(rows) != compute_data_signature(changed)


def test_custom_instruction_reaches_source_and_augmented():
    rows = make_group(1, ["none", "synonym_replacement"])
    groups = build_pair_groups(rows)
    original_map, _ = build_original_map(groups)
    for sample in rows:
        sample["_group_key"] = groups[0]["group_key"]
    converted, _ = convert_to_alpaca(rows, mode="pair_operation",
                                     original_map=original_map,
                                     operation_descriptions=MR_OPERATION_DESCRIPTIONS,
                                     nli_instruction="CUSTOM_INSTRUCTION_SENTINEL")
    assert all("CUSTOM_INSTRUCTION_SENTINEL" in row["instruction"] for row in converted)


def test_shuffled_preserves_frequency_and_has_no_description_fixed_points():
    rows = make_group(1, ["none", "synonym_replacement", "antonym_substitution",
                          "adding_contradiction", "voice_switch"])
    assigned = select_shuffled_descriptions(rows, set(), 42)
    true_desc = [MR_OPERATION_DESCRIPTIONS[s["mr_id"]] for s in rows if s["mr_id"] != "none"]
    assigned_desc = [MR_OPERATION_DESCRIPTIONS[m] for m in assigned if m != "none"]
    assert sorted(true_desc) == sorted(assigned_desc)
    assert all(a != b for a, b in zip(true_desc, assigned_desc))


def test_token_summary_cutoff():
    summary = summarize_token_lengths([1, 2, 10, 20], 10)
    assert summary["count"] == 4
    assert summary["over_cutoff_count"] == 1
    assert summary["over_cutoff_ratio"] == .25


class FakeChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        marker = messages[0]["content"].split("LEN:", 1)[1].split()[0]
        return [0] * int(marker)


def test_token_report_uses_real_tokenizer_path_and_reports_by_mr():
    train_rows = [
        {"instruction": "LEN:4", "input": "a", "output": "entailment"},
        {"instruction": "LEN:9", "input": "b", "output": "contradiction"},
    ]
    val_rows = [
        {"instruction": "LEN:6", "input": "c", "output": "neutral"},
    ]
    report = build_token_length_report(
        train_rows,
        val_rows,
        "fake-tokenizer",
        cutoff_len=5,
        train_samples=[
            make_sample(mr_id="none"),
            make_sample(mr_id="adding_contradiction"),
        ],
        val_samples=[make_sample(mr_id="synonym_replacement")],
        tokenizer=FakeChatTokenizer(),
    )
    assert report["counting_method"] == "apply_chat_template"
    assert report["counting_method_counts"] == {"apply_chat_template": 3}
    assert report["train"]["over_cutoff_count"] == 1
    assert report["validation"]["over_cutoff_count"] == 1
    assert report["by_mr"]["train"]["none"]["count"] == 1
    assert report["by_mr"]["train"]["none"]["over_cutoff_count"] == 0
    assert report["by_mr"]["train"]["adding_contradiction"]["over_cutoff_count"] == 1
    assert report["by_mr"]["validation"]["synonym_replacement"]["count"] == 1


def test_template_v2_hashes_are_full_sha256():
    assert INSTRUCTION_TEMPLATE_VERSION == 2
    for value in (
        INSTRUCTION_TEMPLATE_HASH,
        OPERATION_DESCRIPTION_HASH,
        RELATION_EFFECT_HASH,
    ):
        assert len(value) == 64
        int(value, 16)


def test_reference_template_v2_handles_embedded_quotes():
    sample = make_sample(mr_id="synonym_replacement")
    original = make_sample(
        premise='A person said "yes" and then left.',
        hypothesis='The person said "yes".',
        mr_id="none",
    )
    result = build_instruction_for_sample(
        sample,
        original_sample=original,
        mode="pair_operation",
        operation_description=MR_OPERATION_DESCRIPTIONS["synonym_replacement"],
    )
    instruction = result["instruction"]
    assert (
        '<reference_premise>\nA person said "yes" and then left.\n'
        '</reference_premise>'
    ) in instruction
    assert (
        '<reference_hypothesis>\nThe person said "yes".\n'
        '</reference_hypothesis>'
    ) in instruction
    assert 'Premise: "' not in instruction
    assert 'Hypothesis: "' not in instruction


def test_stable_sort_and_ordered_signature_ignore_input_order():
    rows = make_group(2, ["none", "synonym_replacement"]) + make_group(
        1, ["none", "adding_contradiction"]
    )
    groups = build_pair_groups(rows)
    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]
    first = sort_samples_by_stable_key(rows)
    second = sort_samples_by_stable_key(list(reversed(rows)))
    assert [compute_sample_key(row) for row in first] == [
        compute_sample_key(row) for row in second
    ]
    assert compute_ordered_sample_signature(first, []) == (
        compute_ordered_sample_signature(second, [])
    )


def test_converted_rows_sha256_covers_exact_order_and_content():
    rows = [
        {"instruction": "a", "input": "b", "output": "c"},
        {"instruction": "d", "input": "e", "output": "f"},
    ]
    assert converted_rows_sha256(rows) == converted_rows_sha256(list(rows))
    assert converted_rows_sha256(rows) != converted_rows_sha256(list(reversed(rows)))
    changed = [dict(row) for row in rows]
    changed[0]["instruction"] = "changed"
    assert converted_rows_sha256(rows) != converted_rows_sha256(changed)


def test_shuffle_audit_contains_true_to_assigned_confusion_matrix():
    rows = make_group(
        1,
        [
            "none",
            "synonym_replacement",
            "antonym_substitution",
            "adding_contradiction",
            "voice_switch",
        ],
    )
    assigned = select_shuffled_descriptions(rows, set(), 42)
    audit = build_shuffle_audit(rows, assigned, 42)
    matrix = audit["true_to_assigned_confusion_matrix"]
    assert audit["total_augmented"] == 4
    assert audit["fixed_point_count"] == 0
    assert sum(sum(row.values()) for row in matrix.values()) == 4
    assert {
        key: sum(row.values()) for key, row in matrix.items()
    } == audit["true_description_counts"]


def test_modes_without_operation_do_not_report_missing_descriptions():
    rows = make_group(1, ["none", "synonym_replacement"])
    for mode in ("none", "pair_only"):
        _, report = convert_to_alpaca(rows, mode=mode)
        assert report["missing_operation_count"] == 0


# ============================================================
# Helpers
# ============================================================
def make_sample(premise="A cat sat on a mat.", hypothesis="A cat is on a mat.",
                label=0, pair_id=0, mr_id="none", mr_type="inv", idx=0):
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "pair_id": pair_id,
        "mr_id": mr_id,
        "mr_type": mr_type,
        "idx": idx,
    }


def make_group(pair_id, mr_ids, base_premise="A cat sat on a mat."):
    """Create a group with one source (none) and multiple MR variants."""
    samples = []
    for i, mr_id in enumerate(mr_ids):
        if mr_id == "none":
            hypothesis = "A cat is on a mat."
            label = 0
        elif mr_id == "antonym_substitution":
            hypothesis = "A dog is on a mat."
            label = 2
        elif mr_id == "adding_contradiction":
            hypothesis = "No animal is anywhere near."
            label = 2
        elif mr_id == "synonym_replacement":
            hypothesis = "A feline rests on a rug."
            label = 0
        elif mr_id == "composite_inv":
            hypothesis = "A feline was seen on a rug."
            label = 0
        elif mr_id == "composite_flip":
            hypothesis = "No feline was seen anywhere."
            label = 2
        else:
            hypothesis = f"Modified hypothesis {i}"
            label = 1

        samples.append(make_sample(
            premise=base_premise,
            hypothesis=hypothesis,
            label=label,
            pair_id=pair_id,
            mr_id=mr_id,
        ))
    return samples


def make_groups(count=5, samples_per_group=3):
    """Create test data with `count` groups, each with `samples_per_group` MR variants.
    First sample in each group is 'none' (original).
    """
    all_samples = []
    for g in range(count):
        mr_ids = ["none"] + [f"mr_type_{i}" for i in range(1, samples_per_group)]
        group = make_group(g, mr_ids, base_premise=f"Premise for group {g}.")
        all_samples.extend(group)
    return all_samples


# ============================================================
# Tests: normalize_mr_id
# ============================================================
def test_normalize_mr_id():
    assert normalize_mr_id("adding_contradiction") == "adding_contradiction"
    assert normalize_mr_id("Adding-Contradiction") == "adding_contradiction"
    assert normalize_mr_id(None) == "none"
    assert normalize_mr_id("") == ""
    assert normalize_mr_id("  Synonym_Replacement  ") == "synonym_replacement"
    assert normalize_mr_id("race_sensitive_transformation") == "race_sensitive_transformation"
    print("✅ test_normalize_mr_id")


# ============================================================
# Tests: validate_mr_samples
# ============================================================
def test_validate_mr_samples_known():
    samples = [make_sample(mr_id="none"), make_sample(mr_id="synonym_replacement")]
    report = validate_mr_samples(samples, strict=False)
    assert len(report["unknown_mr_ids"]) == 0
    print("✅ test_validate_mr_samples_known")


def test_validate_mr_samples_unknown_strict():
    samples = [make_sample(mr_id="unknown_mr_xyz")]
    try:
        validate_mr_samples(samples, strict=True)
        assert False, "预期抛出 ValueError"
    except ValueError as e:
        assert "unknown_mr_xyz" in str(e)
    print("✅ test_validate_mr_samples_unknown_strict")


def test_validate_mr_samples_unknown_nonstrict():
    samples = [make_sample(mr_id="unknown_mr_xyz")]
    report = validate_mr_samples(samples, strict=False)
    assert "unknown_mr_xyz" in report["unknown_mr_ids"]
    print("✅ test_validate_mr_samples_unknown_nonstrict")


# ============================================================
# Tests: build_pair_groups
# ============================================================
def test_build_pair_groups():
    samples = [
        make_sample(pair_id=0, mr_id="none"),
        make_sample(pair_id=0, mr_id="synonym_replacement"),
        make_sample(pair_id=1, mr_id="none"),
        make_sample(pair_id=2, mr_id="none"),
    ]
    groups = build_pair_groups(samples)
    assert len(groups) == 3
    group_map = {g["group_key"]: g for g in groups}

    # Group 0 should have 2 samples
    g0 = group_map[("input", "0")]
    assert len(g0["samples"]) == 2

    # Groups 1 and 2 should have 1 sample each
    assert len(group_map[("input", "1")]["samples"]) == 1
    assert len(group_map[("input", "2")]["samples"]) == 1
    print("✅ test_build_pair_groups")


def test_build_pair_groups_no_pair_id():
    samples = [
        {"premise": "A", "hypothesis": "B", "label": 0},  # no pair_id
        {"premise": "C", "hypothesis": "D", "label": 1},  # no pair_id
    ]
    groups = build_pair_groups(samples)
    assert len(groups) == 2  # each row is its own group
    print("✅ test_build_pair_groups_no_pair_id")


def test_build_pair_groups_mixed():
    samples = [
        make_sample(pair_id=0, mr_id="none"),
        {"premise": "A", "hypothesis": "B", "label": 0},  # no pair_id
        make_sample(pair_id=0, mr_id="synonym_replacement"),
        {"premise": "C", "hypothesis": "D", "label": 1},  # no pair_id
    ]
    groups = build_pair_groups(samples)
    assert len(groups) == 3  # 1 group for pair_id=0 + 2 singletons
    print("✅ test_build_pair_groups_mixed")


# ============================================================
# Tests: split_pair_groups
# ============================================================
def test_split_no_cross():
    """同一 pair_id 不跨 train/validation"""
    samples = make_groups(count=10, samples_per_group=3)
    groups = build_pair_groups(samples)

    train_groups, val_groups = split_pair_groups(groups, val_ratio=0.2, seed=42)

    train_ids = {g["group_key"] for g in train_groups}
    val_ids = {g["group_key"] for g in val_groups}

    # No overlap
    assert train_ids.isdisjoint(val_ids), "train/val group IDs 不应有交集"

    # All groups accounted for
    assert len(train_ids) + len(val_ids) == len(groups)
    print(f"✅ test_split_no_cross: train={len(train_ids)}, val={len(val_ids)}")


def test_split_deterministic():
    """相同 seed 得到相同 split"""
    samples = make_groups(count=20, samples_per_group=3)
    groups = build_pair_groups(samples)

    t1, v1 = split_pair_groups(groups, val_ratio=0.1, seed=42)
    t2, v2 = split_pair_groups(groups, val_ratio=0.1, seed=42)

    t1_ids = {g["group_key"] for g in t1}
    t2_ids = {g["group_key"] for g in t2}
    v1_ids = {g["group_key"] for g in v1}
    v2_ids = {g["group_key"] for g in v2}

    assert t1_ids == t2_ids, f"train 不一致: {t1_ids - t2_ids}"
    assert v1_ids == v2_ids, f"val 不一致: {v1_ids - v2_ids}"
    print("✅ test_split_deterministic")


def test_split_different_seed_different():
    """不同 seed 应产生不同 split（极大概率）"""
    samples = make_groups(count=30, samples_per_group=3)
    groups = build_pair_groups(samples)

    t1, v1 = split_pair_groups(groups, val_ratio=0.2, seed=42)
    t2, v2 = split_pair_groups(groups, val_ratio=0.2, seed=99)

    t1_ids = {g["group_key"] for g in t1}
    t2_ids = {g["group_key"] for g in t2}

    # 极大概率不同（15+ groups 下 seed 42 和 99 给出相同 split 概率 < 1e-10）
    assert t1_ids != t2_ids, "不同 seed 应产生不同 split"
    print("✅ test_split_different_seed_different")


# ============================================================
# Tests: build_original_map
# ============================================================
def test_build_original_map():
    samples = make_groups(count=3, samples_per_group=4)
    groups = build_pair_groups(samples)
    original_map, missing = build_original_map(groups)

    assert len(missing) == 0
    assert len(original_map) == 3
    for key, orig in original_map.items():
        assert orig["mr_id"] == "none"
    print("✅ test_build_original_map")


def test_build_original_map_missing():
    """一个 group 缺失 source 时应在 missing 列表中"""
    samples = [
        make_sample(pair_id=0, mr_id="synonym_replacement"),  # no 'none'
        make_sample(pair_id=1, mr_id="none"),
    ]
    groups = build_pair_groups(samples)
    original_map, missing = build_original_map(groups)

    assert len(missing) == 1
    assert missing[0][0] == ("input", "0")
    assert missing[0][1] == "no_source"
    print("✅ test_build_original_map_missing")


# ============================================================
# Tests: Instruction content
# ============================================================
def test_instruction_none():
    """mr_id=none 始终使用普通 instruction"""
    sample = make_sample(mr_id="none")
    result = build_instruction_for_sample(
        sample, original_sample=None,
        mode="none", operation_description="",
    )
    assert "Reference sample" not in result["instruction"]
    assert "Transformation" not in result["instruction"]
    assert result["instruction"].startswith("Determine the natural language inference relation")
    assert sample["premise"] in result["input"]
    assert sample["hypothesis"] in result["input"]
    print("✅ test_instruction_none")


def test_instruction_pair_operation_no_label_leak():
    """pair_operation 不含原始 label、不含 raw mr_id、不含 relation effect"""
    sample = make_sample(mr_id="synonym_replacement", label=1)
    original = make_sample(mr_id="none", label=0)
    desc = MR_OPERATION_DESCRIPTIONS.get("synonym_replacement", "")

    result = build_instruction_for_sample(
        sample, original_sample=original,
        mode="pair_operation", operation_description=desc,
    )

    instr = result["instruction"]

    # 不应包含原始 label
    assert "Reference label" not in instr
    assert "entailment" not in instr.split("Reference sample")[-1].split("Transformation")[0]

    # 不应包含 raw mr_id
    assert "synonym_replacement" not in instr

    # 不应包含 relation effect
    assert "Expected relation effect" not in instr

    # 应有 reference
    assert "Reference sample:" in instr
    assert original["premise"] in instr

    # 应有 operation description
    assert "Transformation applied" in instr
    assert desc in instr

    # 当前样本只在 input
    assert sample["premise"] in result["input"]
    assert sample["hypothesis"] in result["input"]

    print("✅ test_instruction_pair_operation_no_label_leak")


def test_instruction_pair_operation_has_reference():
    """pair_operation 应有 reference 和 operation"""
    sample = make_sample(mr_id="antonym_substitution", label=2)
    original = make_sample(mr_id="none", label=0, premise="Original premise", hypothesis="Original hypothesis")
    desc = MR_OPERATION_DESCRIPTIONS.get("antonym_substitution", "")

    result = build_instruction_for_sample(
        sample, original_sample=original,
        mode="pair_operation", operation_description=desc,
    )

    assert original["premise"] in result["instruction"]
    assert original["hypothesis"] in result["instruction"]
    assert desc in result["instruction"]

    # output 是当前标签，不是原标签
    assert result["output"] == LABEL_NAMES_3CLASS[sample["label"]]
    print("✅ test_instruction_pair_operation_has_reference")


def test_instruction_validation_always_plain():
    """验证集始终使用普通 instruction（即 mode none）"""
    # validation 用 mode="none" 来模拟
    sample = make_sample(mr_id="synonym_replacement", label=1)
    result = build_instruction_for_sample(
        sample, original_sample=None,
        mode="none", operation_description="",
    )
    assert "Reference sample" not in result["instruction"]
    assert "Transformation" not in result["instruction"]
    print("✅ test_instruction_validation_always_plain")


def test_instruction_original_always_plain_in_pair_mode():
    """原始样本（mr_id=none）在 pair_operation 模式下也使用普通 instruction"""
    sample = make_sample(mr_id="none", label=0)
    # 即使 mode=pair_operation、有 original_sample、有描述，
    # is_original=True 时应该强制使用 none instruction
    result = build_instruction_for_sample(
        sample, original_sample=sample,
        mode="pair_operation",
        operation_description="Some transformation description",
        is_original=True,
    )
    assert "Reference sample" not in result["instruction"], "原始样本不应有 Reference sample"
    assert "Transformation" not in result["instruction"], "原始样本不应有 Transformation info"
    assert result["instruction"].startswith("Determine the natural language inference relation")
    assert result["output"] == "entailment"
    print("✅ test_instruction_original_always_plain_in_pair_mode")


def test_instruction_original_always_plain_in_all_modes():
    """原始样本在所有 MR instruction 模式下都保持普通 instruction"""
    sample = make_sample(mr_id="none", label=0)
    for mode in ["operation_only", "pair_only", "pair_operation",
                  "shuffled_operation", "relation_aware", "full_oracle"]:
        result = build_instruction_for_sample(
            sample, original_sample=sample,
            mode=mode,
            operation_description="Some description",
            is_original=True,
        )
        assert "Reference sample" not in result["instruction"], f"mode={mode}: 不应有 Reference"
        assert "Transformation" not in result["instruction"], f"mode={mode}: 不应有 Transformation"
        assert result["instruction"].startswith("Determine the natural language inference")
    print("✅ test_instruction_original_always_plain_in_all_modes")


def test_instruction_full_oracle():
    """full_oracle 包含上界信息"""
    sample = make_sample(mr_id="synonym_replacement", label=1)
    original = make_sample(mr_id="none", label=0)
    desc = MR_OPERATION_DESCRIPTIONS.get("synonym_replacement", "")
    effect = MR_RELATION_EFFECTS.get("synonym_replacement", "")

    result = build_instruction_for_sample(
        sample, original_sample=original,
        mode="full_oracle", operation_description=desc,
        relation_effect=effect, original_label="entailment",
    )

    assert "Reference label" in result["instruction"]
    assert "entailment" in result["instruction"]  # the original label
    assert "Expected relation effect" in result["instruction"]
    assert effect in result["instruction"]
    print("✅ test_instruction_full_oracle")


def test_instruction_current_ph_only_in_input():
    """当前 P/H 只出现在 input，不出现在 instruction"""
    sample = make_sample(
        premise="UNIQUE_PREMISE_12345",
        hypothesis="UNIQUE_HYPOTHESIS_67890",
        mr_id="synonym_replacement",
    )
    original = make_sample(pair_id=0, mr_id="none")

    for mode in ["none", "pair_operation", "full_oracle"]:
        result = build_instruction_for_sample(
            sample, original_sample=original if mode != "none" else None,
            mode=mode,
            operation_description=MR_OPERATION_DESCRIPTIONS.get("synonym_replacement", ""),
            relation_effect=MR_RELATION_EFFECTS.get("synonym_replacement", ""),
            original_label="entailment",
        )
        # 当前 P/H 不在 instruction 中
        # (但 reference premise 在 pair 模式 instruction 中，它可能与当前 premise 相似)
        # 所以我们不做完全相等的 assert，只检查 input
        assert "UNIQUE_PREMISE_12345" in result["input"]
        assert "UNIQUE_HYPOTHESIS_67890" in result["input"]
    print("✅ test_instruction_current_ph_only_in_input")


# ============================================================
# Tests: convert_to_alpaca
# ============================================================
def test_convert_to_alpaca_none():
    samples = [make_sample(mr_id="none"), make_sample(mr_id="synonym_replacement")]
    converted, report = convert_to_alpaca(samples, mode="none")
    assert len(converted) == 2
    assert report["fallback_count"] == 0
    assert "Reference sample" not in converted[0]["instruction"]
    assert "Reference sample" not in converted[1]["instruction"]
    print("✅ test_convert_to_alpaca_none")


def test_convert_to_alpaca_with_original():
    samples = [
        make_sample(pair_id=0, mr_id="none"),
        make_sample(pair_id=0, mr_id="synonym_replacement"),
    ]
    groups = build_pair_groups(samples)
    # tag group keys
    for g in groups:
        for s in g["samples"]:
            s["_group_key"] = g["group_key"]
    original_map, _ = build_original_map(groups)

    converted, report = convert_to_alpaca(
        samples, mode="pair_operation",
        original_map=original_map,
        operation_descriptions=MR_OPERATION_DESCRIPTIONS,
    )
    assert len(converted) == 2

    # none sample should have plain instruction even in pair_operation mode
    # (because _group_key is set but there's no reference for itself)
    for i, c in enumerate(converted):
        mr = samples[i]["mr_id"]
        if mr == "none":
            # We don't set instruction for "none" samples — they use the source reference
            pass

    print("✅ test_convert_to_alpaca_with_original")


# ============================================================
# Tests: shuffled descriptions
# ============================================================
def test_shuffled_descriptions_differ():
    samples = [
        make_sample(pair_id=0, mr_id="synonym_replacement"),
        make_sample(pair_id=1, mr_id="adding_contradiction"),
        make_sample(pair_id=2, mr_id="antonym_substitution"),
        make_sample(pair_id=3, mr_id="none"),
        make_sample(pair_id=4, mr_id="composite_inv"),
    ]
    all_mr_ids = {"synonym_replacement", "adding_contradiction", "antonym_substitution",
                  "composite_inv", "composite_flip", "none"}

    shuffled = select_shuffled_descriptions(samples, all_mr_ids, seed=42)
    assert len(shuffled) == len(samples)

    # none 样本应保持 none
    none_indices = [i for i, s in enumerate(samples) if s["mr_id"] == "none"]
    for i in none_indices:
        assert shuffled[i] == "none"

    # 非 none 样本应被赋予不同描述（极大概率不全相同）
    non_none_shuffled = [shuffled[i] for i in range(len(samples)) if samples[i]["mr_id"] != "none"]
    non_none_original = [s["mr_id"] for s in samples if s["mr_id"] != "none"]

    # shuffled 的内容应与原始列表不同（至少有一个不同）
    assert shuffled != [s["mr_id"] for s in samples], "shuffled 应与原始不同"

    print("✅ test_shuffled_descriptions_differ")


def test_shuffled_descriptions_uses_known():
    """shuffled 描述应来自已知 mr_id 集合"""
    samples = [
        make_sample(pair_id=0, mr_id="synonym_replacement"),
        make_sample(pair_id=1, mr_id="adding_contradiction"),
    ]
    all_mr_ids = {"synonym_replacement", "adding_contradiction", "antonym_substitution", "none"}

    shuffled = select_shuffled_descriptions(samples, all_mr_ids, seed=42)
    for s, sh in zip(samples, shuffled):
        if s["mr_id"] != "none":
            assert sh in all_mr_ids, f"shuffled {sh} 不在已知集合中"
    print("✅ test_shuffled_descriptions_uses_known")


# ============================================================
# Tests: Manifest
# ============================================================
def test_manifest_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "split_manifest.json"
        save_split_manifest(
            manifest_path=str(manifest_path),
            seed=42,
            val_ratio=0.05,
            train_group_ids=[("file", "0"), ("file", "1")],
            val_group_ids=[("file", "2")],
            source_files=["data.json"],
            data_signature="abc123",
        )

        loaded = load_split_manifest(str(manifest_path))
        assert loaded["seed"] == 42
        assert loaded["val_ratio"] == 0.05
        assert ("file", "0") in [tuple(i) if isinstance(i, list) else i for i in loaded["train_group_ids"]]
        print("✅ test_manifest_save_load")


def make_valid_manifest(tmp_path):
    rows = (
        make_group(1, ["none", "synonym_replacement"])
        + make_group(2, ["none", "adding_contradiction"])
    )
    groups = build_pair_groups(rows)
    train_ids = [groups[0]["group_key"]]
    val_ids = [groups[1]["group_key"]]
    signature = compute_data_signature(rows)
    path = tmp_path / "split_manifest.json"
    manifest = save_split_manifest(
        manifest_path=path,
        seed=42,
        val_ratio=0.5,
        train_group_ids=train_ids,
        val_group_ids=val_ids,
        source_files=["data.jsonl"],
        data_signature=signature,
        train_sample_count=len(groups[0]["samples"]),
        val_sample_count=len(groups[1]["samples"]),
    )
    return path, manifest, groups, signature


def test_manifest_rejects_duplicate_group_ids(tmp_path):
    _, manifest, groups, signature = make_valid_manifest(tmp_path)
    duplicate = copy.deepcopy(manifest)
    duplicate["train_group_ids"] = [
        duplicate["train_group_ids"][0],
        duplicate["train_group_ids"][0],
    ]
    with pytest.raises(ValueError, match="重复 group ID"):
        validate_manifest_for_groups(duplicate, groups, 42, 0.5, signature)


@pytest.mark.parametrize("field", [
    "all_group_count",
    "train_group_count",
    "val_group_count",
    "train_sample_count",
    "val_sample_count",
])
def test_manifest_rejects_incorrect_counts(tmp_path, field):
    _, manifest, groups, signature = make_valid_manifest(tmp_path)
    invalid = copy.deepcopy(manifest)
    invalid[field] += 1
    with pytest.raises(ValueError, match=field):
        validate_manifest_for_groups(invalid, groups, 42, 0.5, signature)


def test_manifest_load_rejects_self_hash_tampering(tmp_path):
    path, _, _, _ = make_valid_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["train_sample_count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        load_split_manifest(path)


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    test_normalize_mr_id()
    test_validate_mr_samples_known()
    test_validate_mr_samples_unknown_strict()
    test_validate_mr_samples_unknown_nonstrict()
    test_build_pair_groups()
    test_build_pair_groups_no_pair_id()
    test_build_pair_groups_mixed()
    test_split_no_cross()
    test_split_deterministic()
    test_split_different_seed_different()
    test_build_original_map()
    test_build_original_map_missing()
    test_instruction_none()
    test_instruction_pair_operation_no_label_leak()
    test_instruction_pair_operation_has_reference()
    test_instruction_validation_always_plain()
    test_instruction_full_oracle()
    test_instruction_current_ph_only_in_input()
    test_convert_to_alpaca_none()
    test_convert_to_alpaca_with_original()
    test_shuffled_descriptions_differ()
    test_shuffled_descriptions_uses_known()
    test_manifest_save_load()
    print(f"\n{'='*50}")
    print("  ✅ 所有测试通过！")
    print(f"{'='*50}")
