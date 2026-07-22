#!/usr/bin/env python3
"""
将 NLI 数据集转换为 LLaMA-Factory 微调格式（Alpaca），支持：
- Group-aware train/validation split（按 pair_id 分组）
- 多种 MR-instruction 模式（operation_only, pair_only, pair_operation, 等）
- 共享 split manifest（确保不同 instruction 变体使用相同样本）
- Strict pairing 校验
- Conversion report 输出

用法:
  # 基础转换（与旧版行为一致）
  python scripts/convert_nli_to_ft.py \
    --input data/nli/mettrain/mnlim_lr0.0051_gemma-3-4b-it-qat/augmented_data.json \
    --name mettrain_mnlim_4413_gemma3_4b \
    --split --val-ratio 0.05

  # MR-instruction 模式（生成 manifest）
  python scripts/convert_nli_to_ft.py \
    --input data/nli/mettrain/mnlim_lr0.0051_gemma-3-4b-it-qat/augmented_data.json \
    --name mettrain_mnlim_4413_gemma3_4b_mrinstr \
    --split --val-ratio 0.05 \
    --mr-instruction-mode pair_operation \
    --strict-pairing \
    --write-split-manifest data/ft_datasets/mettrain_mnlim_4413_gemma3_4b/split_manifest.json

  # 复用 split manifest（同一 cohort 的其他变体）
  python scripts/convert_nli_to_ft.py \
    --input data/nli/mettrain/mnlim_lr0.0051_gemma-3-4b-it-qat/augmented_data.json \
    --name mettrain_mnlim_4413_gemma3_4b_mrinstr_shuffled \
    --split --val-ratio 0.05 \
    --mr-instruction-mode shuffled_operation \
    --strict-pairing \
    --split-manifest data/ft_datasets/mettrain_mnlim_4413_gemma3_4b/split_manifest.json
"""
import json
import os
import argparse
import random
import hashlib
from pathlib import Path

from project_runtime import PROJECT_ROOT
from collections import Counter, defaultdict


def serialize_counter_keys(counter):
    """将 Counter 中不可 JSON 序列化的 key（如 tuple）转为字符串 key"""
    result = {}
    for k, v in counter.items():
        if isinstance(k, tuple):
            result[str(k)] = v
        elif isinstance(k, (int, float, bool)):
            result[str(k)] = v
        else:
            result[k] = v
    return result

WORK_DIR = PROJECT_ROOT

# ============================================================
# 标签映射
# ============================================================
LABEL_NAMES_3CLASS = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES_BINARY = {0: "entailment", 1: "not_entailment"}  # 保留，但 RTE 已从主实验移除

# ============================================================
# 已知 MR ID 列表（来自实际数据审计）
# ============================================================
KNOWN_MR_IDS = frozenset({
    "none",
    "adding_contradiction",
    "antonym_substitution",
    "composite_flip",
    "composite_inv",
    "composite_neutral",
    "conditional_clause",
    "negation_flip",
    "pronoun_substitution",
    "race_sensitive_transformation",
    "same_type_named_entity_substitution",
    "synonym_replacement",
    "tense_shift",
    "uninformative",
    "voice_switch",
})

# ============================================================
# MR 操作描述（仅描述文本变换，不含标签暗示）
# 主实验（pair_operation, operation_only）只能使用此字典
# ============================================================
MR_OPERATION_DESCRIPTIONS = {
    "adding_contradiction":
        "Additional information was inserted, and the inserted content is incompatible "
        "with part of the reference text.",

    "antonym_substitution":
        "One or more content words were replaced with context-appropriate antonyms.",

    "composite_flip":
        "Several controlled textual edits were applied.",

    "composite_inv":
        "Multiple transformations were performed.",

    "composite_neutral":
        "A combination of text modifications was used.",

    "conditional_clause":
        "A conditional clause was added to one component.",

    "negation_flip":
        "A negation marker was added, removed, or reversed in one component.",

    "pronoun_substitution":
        "A noun phrase or pronoun was replaced by a coreferential form.",

    "race_sensitive_transformation":
        "Demographic or identity-related terms in the text were replaced or modified.",

    "same_type_named_entity_substitution":
        "Named entities (e.g., person names, locations) were replaced with others of the same category.",

    "synonym_replacement":
        "One or more words were replaced with context-appropriate synonyms.",

    "tense_shift":
        "The tense of one or more verbs was shifted.",

    "uninformative":
        "Additional content with little direct relevance was inserted.",

    "voice_switch":
        "A sentence was rewritten between active and passive voice.",

    "none": "",
}

# ============================================================
# MR 输出关系描述（仅用于 relation_aware / full_oracle）
# 主实验代码路径不得读取此字典
# ============================================================
MR_RELATION_EFFECTS = {
    "adding_contradiction":
        "The inserted content conflicts with the reference, so the original "
        "relation is likely disrupted.",

    "antonym_substitution":
        "Replacing words with opposites may invert or alter the semantic relation.",

    "composite_flip":
        "The combined transformations are likely to change the original relation.",

    "composite_inv":
        "The combined transformations tend to preserve the original relation.",

    "composite_neutral":
        "The combined transformations tend to shift the relation toward neutrality.",

    "conditional_clause":
        "Adding a condition may modify the relation by introducing a hypothetical context.",

    "negation_flip":
        "Adding or removing negation may reverse the truth conditions.",

    "pronoun_substitution":
        "Replacing with coreferential expressions generally preserves the relation.",

    "race_sensitive_transformation":
        "Modifying demographic terms may affect the semantic relation between premise and hypothesis.",

    "same_type_named_entity_substitution":
        "Replacing named entities generally preserves the relational structure.",

    "synonym_replacement":
        "Replacing with synonyms tends to preserve the original relation.",

    "tense_shift":
        "Changing tense may alter the temporal relationship between events.",

    "uninformative":
        "Adding irrelevant content is unlikely to change the core relation.",

    "voice_switch":
        "Switching between active and passive voice generally preserves meaning.",

    "none": "",
}

# ============================================================
# Instruction 模板（详见 MR_AS_INSTRUCTION_PLAN.md Section 5）
# ============================================================
INSTRUCTION_NLI = (
    "Determine the natural language inference relation between the premise and hypothesis. "
    "Answer with exactly one label: entailment, neutral, or contradiction."
)
INSTRUCTION_NLI_BINARY = (
    "Determine whether the premise entails the hypothesis. "
    "Answer with exactly one label: entailment or not_entailment."
)

INSTRUCTION_TEMPLATE_VERSION = 2

INSTRUCTION_TEMPLATES = {
    "none": "{nli_instruction}",

    "operation_only": (
        "Transformation information:\n"
        "{operation_description}\n\n"
        "{nli_instruction}"
    ),

    "pair_only": (
        "Reference sample:\n"
        "<reference_premise>\n"
        "{original_premise}\n"
        "</reference_premise>\n"
        "<reference_hypothesis>\n"
        "{original_hypothesis}\n"
        "</reference_hypothesis>\n\n"
        "{nli_instruction}"
    ),

    "pair_operation": (
        "Reference sample:\n"
        "<reference_premise>\n"
        "{original_premise}\n"
        "</reference_premise>\n"
        "<reference_hypothesis>\n"
        "{original_hypothesis}\n"
        "</reference_hypothesis>\n\n"
        "Transformation applied to create the current sample:\n"
        "{operation_description}\n\n"
        "{nli_instruction}"
    ),

    "shuffled_operation": (
        "Reference sample:\n"
        "<reference_premise>\n"
        "{original_premise}\n"
        "</reference_premise>\n"
        "<reference_hypothesis>\n"
        "{original_hypothesis}\n"
        "</reference_hypothesis>\n\n"
        "Transformation applied to create the current sample:\n"
        "{operation_description}\n\n"
        "{nli_instruction}"
    ),

    "relation_aware": (
        "Reference sample:\n"
        "<reference_premise>\n"
        "{original_premise}\n"
        "</reference_premise>\n"
        "<reference_hypothesis>\n"
        "{original_hypothesis}\n"
        "</reference_hypothesis>\n\n"
        "Transformation applied to create the current sample:\n"
        "{operation_description}\n\n"
        "Expected relation effect:\n"
        "{relation_effect}\n\n"
        "{nli_instruction}"
    ),

    "full_oracle": (
        "Reference sample:\n"
        "<reference_premise>\n"
        "{original_premise}\n"
        "</reference_premise>\n"
        "<reference_hypothesis>\n"
        "{original_hypothesis}\n"
        "</reference_hypothesis>\n"
        "Reference label: {original_label}\n\n"
        "Transformation:\n"
        "{operation_description}\n\n"
        "Expected relation effect:\n"
        "{relation_effect}\n\n"
        "{nli_instruction}"
    ),
}

MR_INSTRUCTION_MODES = frozenset({
    "none", "operation_only", "pair_only", "pair_operation",
    "shuffled_operation", "relation_aware", "full_oracle",
})

# 需要 original label 的模式（用于校验）
MODES_REQUIRING_ORIGINAL_LABEL = frozenset({"full_oracle"})

# 需要 relation effect 的模式
MODES_REQUIRING_RELATION_EFFECT = frozenset({"relation_aware", "full_oracle"})

MODES_REQUIRING_OPERATION = frozenset({
    "operation_only", "pair_operation", "shuffled_operation",
    "relation_aware", "full_oracle",
})


def canonical_sha256(value):
    """Return SHA-256 of canonical UTF-8 JSON."""
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


INSTRUCTION_TEMPLATE_HASH = canonical_sha256({
    "version": INSTRUCTION_TEMPLATE_VERSION,
    "templates": INSTRUCTION_TEMPLATES,
})
OPERATION_DESCRIPTION_HASH = canonical_sha256(MR_OPERATION_DESCRIPTIONS)
RELATION_EFFECT_HASH = canonical_sha256(MR_RELATION_EFFECTS)


# ============================================================
# MR ID 规范化
# ============================================================
def normalize_mr_id(value):
    """规范化 MR ID：strip/lower/replace '-' with '_'，None 或缺失 → 'none'"""
    if value is None:
        return "none"
    return str(value).strip().lower().replace("-", "_")


# ============================================================
# Preflight 校验
# ============================================================
def validate_mr_samples(samples, strict=True, known_mr_ids=KNOWN_MR_IDS):
    """对样本集合做 preflight 校验。

    Returns:
        report dict 包含未知 mr_id 列表、缺失 source 数等。
    当 strict=True 且发现问题时抛出 ValueError。
    """
    report = {
        "total_samples": len(samples),
        "unknown_mr_ids": {},
        "missing_source_groups": [],
        "multi_source_groups": [],
        "fallback_count": 0,
    }

    unknown_mr_ids = Counter()
    for s in samples:
        mr_id = normalize_mr_id(s.get("mr_id"))
        if mr_id not in known_mr_ids:
            unknown_mr_ids[mr_id] += 1

    if unknown_mr_ids:
        report["unknown_mr_ids"] = dict(unknown_mr_ids)
        if strict:
            raise ValueError(
                f"未知 mr_id: {dict(unknown_mr_ids)}。"
                f" 已知: {sorted(known_mr_ids)}"
            )

    return report


# ============================================================
# Group 构建
# ============================================================
def build_pair_groups(samples, source_file_ids=None):
    """按 (source_file_id, pair_id) 分组。

    Args:
        samples: 样本列表（每个 dict 应有 'pair_id' 字段）
        source_file_ids: 每个样本来源文件 ID 列表（与 samples 等长），
                         为 None 时统一使用 "input"

    Returns:
        list of groups, 每个 group 是一个 dict:
          {"group_key": (file_id, pair_id_str), "samples": [...]}
    """
    if source_file_ids is None:
        source_file_ids = ["input"] * len(samples)

    groups_dict = {}
    auto_index = 0

    for sid, s in zip(source_file_ids, samples):
        raw_pid = s.get("pair_id")
        if raw_pid is None:
            # 无 pair_id → 每行独立 group
            group_key = (sid, f"__row_{auto_index}")
            auto_index += 1
        else:
            group_key = (sid, str(raw_pid).strip())

        if group_key not in groups_dict:
            groups_dict[group_key] = {
                "group_key": group_key,
                "samples": [],
            }
        groups_dict[group_key]["samples"].append(s)

    return list(groups_dict.values())


def split_pair_groups(groups, val_ratio, seed, stratify=False):
    """按 group 级别拆分 train/validation。

    同一个 group（相同 pair_id）的所有样本必须全部进入 train 或全部进入 val。
    """
    random.seed(seed)

    # 构建 group_key -> group 列表
    group_keys = [g["group_key"] for g in groups]

    # 打乱 group 顺序
    indices = list(range(len(groups)))
    random.shuffle(indices)

    val_count = max(1, int(len(groups) * val_ratio))
    train_count = len(groups) - val_count

    train_indices = set(indices[:train_count])
    val_indices = set(indices[train_count:])

    train_groups = [groups[i] for i in train_indices]
    val_groups = [groups[i] for i in val_indices]

    # 校验：group ID 集合不相交
    train_ids = {g["group_key"] for g in train_groups}
    val_ids = {g["group_key"] for g in val_groups}
    assert train_ids.isdisjoint(val_ids), "train/val group IDs 不应有交集"

    return train_groups, val_groups


def flatten_groups(groups):
    """展开 group 列表为样本列表"""
    result = []
    for g in groups:
        result.extend(g["samples"])
    return result


# ============================================================
# Source map 构建
# ============================================================
def build_original_map(groups):
    """在 group 内找 mr_id=none 的样本作为 source（原始对照）。

    Returns:
        group_key -> original_sample dict（不含 mr_id 或 label 泄漏）
        对于没有 'none' 样本的 group，返回 None 并在 report 中记录
    """
    original_map = {}
    missing = []

    for g in groups:
        key = g["group_key"]
        originals = [s for s in g["samples"] if normalize_mr_id(s.get("mr_id")) == "none"]
        if len(originals) == 1:
            original_map[key] = originals[0]
        elif len(originals) > 1:
            # 多个 source — strict 模式下应报错，这里记录
            missing.append((key, "multi_source", len(originals)))
        else:
            missing.append((key, "no_source", 0))

    return original_map, missing


# ============================================================
# Instruction 构建（纯函数）
# ============================================================
def build_instruction_for_sample(
    sample,
    original_sample,
    mode,
    operation_description,
    relation_effect="",
    original_label="",
    is_binary=False,
    is_original=False,
    nli_instruction=None,
):
    """根据 mode 构建 instruction 文本。

    Args:
        sample: 当前样本 dict（有 premise, hypothesis, label）
        original_sample: 原始对照样本 dict，或 None
        mode: MR instruction mode
        operation_description: 操作描述文本
        relation_effect: 关系效果描述（仅 relation_aware / full_oracle）
        original_label: 原始标签名（仅 full_oracle）
        is_binary: 是否二分类
        is_original: 当前样本是否为原始 source（mr_id=none）。
                      原始样本始终使用普通 NLI instruction。

    Returns:
        Alpaca 格式 dict: {"instruction": ..., "input": ..., "output": ...}
    """
    nli_instruction = nli_instruction or (INSTRUCTION_NLI_BINARY if is_binary else INSTRUCTION_NLI)

    # input 始终是当前样本
    input_text = (
        f"Premise: {sample['premise'].strip()}\n"
        f"Hypothesis: {sample['hypothesis'].strip()}"
    )

    # output 始终是当前样本的标签
    label = sample["label"]
    label_names = LABEL_NAMES_BINARY if is_binary else LABEL_NAMES_3CLASS
    if isinstance(label, int):
        output_label = label_names.get(label, str(label))
    else:
        output_label = str(label)

    # 原始样本（mr_id=none）始终使用普通 NLI instruction，不添加任何 MR 信息
    if mode != "none" and is_original:
        mode = "none"

    if mode == "none":
        instruction = INSTRUCTION_TEMPLATES["none"].format(
            nli_instruction=nli_instruction
        )

    elif mode in ("operation_only",):
        instruction = INSTRUCTION_TEMPLATES["operation_only"].format(
            operation_description=operation_description,
            nli_instruction=nli_instruction,
        )

    elif mode == "pair_only":
        if original_sample is None:
            # fallback: 无 original 时使用普通 instruction
            instruction = INSTRUCTION_TEMPLATES["none"].format(
                nli_instruction=nli_instruction
            )
        else:
            instruction = INSTRUCTION_TEMPLATES["pair_only"].format(
                original_premise=original_sample["premise"].strip(),
                original_hypothesis=original_sample["hypothesis"].strip(),
                nli_instruction=nli_instruction,
            )

    elif mode in ("pair_operation", "shuffled_operation"):
        if original_sample is None:
            instruction = INSTRUCTION_TEMPLATES["none"].format(
                nli_instruction=nli_instruction
            )
        else:
            instruction = INSTRUCTION_TEMPLATES["pair_operation"].format(
                original_premise=original_sample["premise"].strip(),
                original_hypothesis=original_sample["hypothesis"].strip(),
                operation_description=operation_description,
                nli_instruction=nli_instruction,
            )

    elif mode == "relation_aware":
        if original_sample is None:
            instruction = INSTRUCTION_TEMPLATES["none"].format(
                nli_instruction=nli_instruction
            )
        else:
            instruction = INSTRUCTION_TEMPLATES["relation_aware"].format(
                original_premise=original_sample["premise"].strip(),
                original_hypothesis=original_sample["hypothesis"].strip(),
                operation_description=operation_description,
                relation_effect=relation_effect,
                nli_instruction=nli_instruction,
            )

    elif mode == "full_oracle":
        if original_sample is None:
            instruction = INSTRUCTION_TEMPLATES["none"].format(
                nli_instruction=nli_instruction
            )
        else:
            instruction = INSTRUCTION_TEMPLATES["full_oracle"].format(
                original_premise=original_sample["premise"].strip(),
                original_hypothesis=original_sample["hypothesis"].strip(),
                original_label=original_label,
                operation_description=operation_description,
                relation_effect=relation_effect,
                nli_instruction=nli_instruction,
            )
    else:
        raise ValueError(f"未知 mr_instruction_mode: {mode}")

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_label,
    }


# ============================================================
# Shuffled 描述选择
# ============================================================
def select_shuffled_descriptions(samples, mr_ids, seed):
    """为 shuffled_operation 模式打乱操作描述。

    保持样本和标签不变，只打乱操作描述。
    确保尽可能不分配回原 MR。
    """
    del mr_ids  # retained for API compatibility
    indexed = []
    result = ["none"] * len(samples)
    for index, sample in enumerate(samples):
        mr_id = normalize_mr_id(sample.get("mr_id"))
        if mr_id != "none":
            description = MR_OPERATION_DESCRIPTIONS.get(mr_id, "")
            indexed.append((compute_sample_key(sample), index, mr_id, description))
    if not indexed:
        return result
    indexed.sort(key=lambda item: item[0])
    counts = Counter(item[3] for item in indexed)
    if max(counts.values()) * 2 > len(indexed):
        raise ValueError(f"无法生成零固定点描述重排: {dict(counts)}")
    # A rotation by the largest bucket size is a multiset derangement whenever
    # the feasibility condition above holds. Seed deterministically rotates the
    # stable sample order without changing description frequencies.
    offset = max(counts.values())
    ordered = sorted(indexed, key=lambda item: (item[3], item[0]))
    rotation = seed % len(ordered)
    ordered = ordered[rotation:] + ordered[:rotation]
    assigned = [ordered[(i + offset) % len(ordered)][2] for i in range(len(ordered))]
    for item, assigned_mr in zip(ordered, assigned):
        if MR_OPERATION_DESCRIPTIONS[assigned_mr] == item[3]:
            raise ValueError("描述 derangement 出现固定点")
        result[item[1]] = assigned_mr
    return result


def build_shuffle_audit(samples, assigned_mr_ids, shuffle_seed):
    """Build a deterministic audit for a shuffled-operation assignment."""
    augmented_pairs = []
    confusion = defaultdict(Counter)
    for sample, assigned_mr in zip(samples, assigned_mr_ids):
        true_mr = normalize_mr_id(sample.get("mr_id"))
        if true_mr == "none":
            continue
        true_description = MR_OPERATION_DESCRIPTIONS[true_mr]
        assigned_description = MR_OPERATION_DESCRIPTIONS[assigned_mr]
        augmented_pairs.append({
            "sample_key": compute_sample_key(sample),
            "true_mr_id": true_mr,
            "assigned_mr_id": assigned_mr,
            "true_description": true_description,
            "assigned_description": assigned_description,
        })
        confusion[true_description][assigned_description] += 1

    true_counts = Counter(p["true_description"] for p in augmented_pairs)
    assigned_counts = Counter(p["assigned_description"] for p in augmented_pairs)
    fixed_points = sum(
        p["true_description"] == p["assigned_description"]
        for p in augmented_pairs
    )
    if true_counts != assigned_counts or fixed_points:
        raise ValueError("shuffled audit 失败：频率不一致或存在描述固定点")

    return {
        "shuffle_seed": shuffle_seed,
        "total_augmented": len(augmented_pairs),
        "fixed_point_count": fixed_points,
        "true_description_counts": dict(sorted(true_counts.items())),
        "assigned_description_counts": dict(sorted(assigned_counts.items())),
        "true_to_assigned_confusion_matrix": {
            true_description: dict(sorted(assignments.items()))
            for true_description, assignments in sorted(confusion.items())
        },
        "mapping_signature": canonical_sha256(
            sorted(augmented_pairs, key=lambda pair: pair["sample_key"])
        ),
    }


# ============================================================
# 转换主函数（支持 instruction mode）
# ============================================================
def convert_to_alpaca(
    samples,
    mode="none",
    is_binary=False,
    original_map=None,
    operation_descriptions=None,
    relation_effects=None,
    shuffled_descriptions=None,
    strict=False,
    nli_instruction=None,
):
    """将 NLI 样本转换为 Alpaca 格式，支持 MR instruction 模式。

    Args:
        samples: 样本列表
        mode: MR instruction mode
        is_binary: 是否二分类
        original_map: group_key -> original_sample 的映射
        operation_descriptions: mr_id -> 操作描述文本 的映射
        relation_effects: mr_id -> 关系效果描述 的映射
        shuffled_descriptions: 可选，shuffled mr_id 列表（与 samples 等长）
        strict: 若 True，对数据完整性问题报错

    Returns:
        (converted list, report dict)
    """
    if operation_descriptions is None:
        operation_descriptions = {}

    converted = []
    label_dist = Counter()
    mr_dist = Counter()
    mr_label_dist = Counter()
    fallback_count = 0
    missing_operation = 0

    for i, s in enumerate(samples):
        mr_id = normalize_mr_id(s.get("mr_id"))
        mr_dist[mr_id] += 1
        label_names = LABEL_NAMES_BINARY if is_binary else LABEL_NAMES_3CLASS
        label_val = s["label"]
        if isinstance(label_val, int):
            label_name = label_names.get(label_val, str(label_val))
        else:
            label_name = str(label_val)
        mr_label_dist[(mr_id, label_name)] += 1

        # 确定使用的 mr_id（可能被 shuffled）
        effective_mr_id = mr_id
        if mode == "shuffled_operation" and shuffled_descriptions is not None:
            effective_mr_id = shuffled_descriptions[i]

        # 只在实际使用 operation description 的模式检查描述完整性。
        operation_desc = ""
        if mode in MODES_REQUIRING_OPERATION:
            operation_desc = operation_descriptions.get(effective_mr_id, "")
            if not operation_desc and effective_mr_id != "none":
                missing_operation += 1
                if strict:
                    raise ValueError(f"缺少操作描述: mr_id={effective_mr_id}")

        # 获取关系效果描述（仅某些 mode 需要）
        relation_effect = ""
        if mode in MODES_REQUIRING_RELATION_EFFECT:
            relation_effect = relation_effects.get(effective_mr_id, "") if relation_effects else ""

        # 获取原始样本（用于 pair 相关 mode）
        original_sample = None
        original_label = ""
        if original_map is not None and mode not in ("none", "operation_only"):
            group_key = s.get("_group_key")
            if group_key is not None:
                original_sample = original_map.get(group_key)
            if original_sample is None:
                fallback_count += 1

        # 获取原始标签（仅 full_oracle）
        if mode == "full_oracle" and original_sample is not None:
            ol = original_sample.get("label")
            if isinstance(ol, int):
                original_label = label_names.get(ol, str(ol))
            else:
                original_label = str(ol)

        # 构建 Alpaca 格式
        result = build_instruction_for_sample(
            sample=s,
            original_sample=original_sample,
            mode=mode,
            operation_description=operation_desc,
            relation_effect=relation_effect,
            original_label=original_label,
            is_binary=is_binary,
            is_original=(effective_mr_id == "none"),
            nli_instruction=nli_instruction,
        )

        converted.append(result)
        label_dist[result["output"]] += 1

    report = {
        "total": len(converted),
        "label_distribution": dict(label_dist),
        "mr_distribution": dict(mr_dist),
        "mr_label_cross": serialize_counter_keys(mr_label_dist),
        "fallback_count": fallback_count,
        "missing_operation_count": missing_operation,
    }

    return converted, report


# ============================================================
# Split Manifest
# ============================================================
def compute_data_signature(samples):
    """计算样本数据的签名（用于检测数据变更）"""
    canonical = []
    fields = ("pair_id", "mr_id", "premise", "hypothesis", "label", "idx", "id", "_source")
    for sample in samples:
        canonical.append({key: sample.get(key) for key in fields if key in sample})
    canonical.sort(key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_sample_key(sample):
    group_key = sample.get("_group_key")
    if isinstance(group_key, tuple):
        group_key = list(group_key)
    payload = {
        key: sample.get(key)
        for key in (
            "pair_id", "mr_id", "premise", "hypothesis", "label",
            "idx", "id", "_source",
        )
        if key in sample
    }
    if group_key is not None:
        payload["_group_key"] = group_key
    return canonical_sha256(payload)


def sort_samples_by_stable_key(samples):
    """Return samples in mode-independent stable-key order."""
    return sorted(
        samples,
        key=lambda sample: (
            compute_sample_key(sample),
            json.dumps(sample, sort_keys=True, ensure_ascii=False, default=list),
        ),
    )


def compute_ordered_sample_signature(train_samples, val_samples):
    """Hash the exact train/validation sample-key order used for conversion."""
    return canonical_sha256({
        "version": 1,
        "train": [compute_sample_key(sample) for sample in train_samples],
        "validation": [compute_sample_key(sample) for sample in val_samples],
    })


def converted_rows_sha256(rows):
    """Hash the exact UTF-8 JSONL bytes emitted by save_jsonl."""
    digest = hashlib.sha256()
    for row in rows:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def save_split_manifest(
    manifest_path,
    seed,
    val_ratio,
    train_group_ids,
    val_group_ids,
    source_files,
    data_signature,
    train_sample_count=0,
    val_sample_count=0,
):
    """保存 split manifest 到 JSON 文件"""
    manifest = {
        "schema_version": 2,
        "signature_version": 1,
        "seed": seed,
        "val_ratio": val_ratio,
        "group_key_version": 1,
        "source_files": source_files,
        "train_group_ids": sorted(train_group_ids, key=str),
        "val_group_ids": sorted(val_group_ids, key=str),
        "all_group_count": len(train_group_ids) + len(val_group_ids),
        "train_group_count": len(train_group_ids),
        "val_group_count": len(val_group_ids),
        "train_sample_count": train_sample_count,
        "val_sample_count": val_sample_count,
        "data_signature": data_signature,
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    manifest["sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def load_split_manifest(manifest_path):
    """加载 split manifest"""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest 不存在: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    stored_hash = manifest.get("sha256")
    payload = dict(manifest)
    payload.pop("sha256", None)
    calculated_hash = canonical_sha256(payload)
    if not stored_hash or stored_hash != calculated_hash:
        raise ValueError(
            f"Split manifest self-hash 不匹配: stored={stored_hash} "
            f"calculated={calculated_hash}"
        )
    return manifest


def deserialize_manifest_group_ids(manifest, field):
    values = manifest.get(field)
    if not isinstance(values, list):
        raise ValueError(f"manifest {field} 必须是 list")
    normalized = [tuple(value) if isinstance(value, list) else value for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"manifest {field} 含重复 group ID")
    return set(normalized)


def verify_manifest(manifest, train_group_ids, val_group_ids, data_signature):
    """校验 manifest 与当前数据的 group IDs 一致"""
    expected_train = set(sorted(train_group_ids, key=str))
    expected_val = set(sorted(val_group_ids, key=str))
    actual_train = deserialize_manifest_group_ids(manifest, "train_group_ids")
    actual_val = deserialize_manifest_group_ids(manifest, "val_group_ids")

    if expected_train != actual_train or expected_val != actual_val:
        raise ValueError(
            f"Split manifest 与当前数据不匹配！\n"
            f"  实际 train group IDs: {len(expected_train)} 个\n"
            f"  manifest train group IDs: {len(actual_train)} 个\n"
            f"  实际 val group IDs: {len(expected_val)} 个\n"
            f"  manifest val group IDs: {len(actual_val)} 个\n"
            f"  train 差集: {expected_train - actual_train}\n"
            f"  val 差集: {expected_val - actual_val}"
        )

    if manifest.get("schema_version") != 2:
        raise ValueError(f"不支持的 manifest schema: {manifest.get('schema_version')}")
    if data_signature != manifest.get("data_signature"):
        raise ValueError(f"数据签名不匹配: manifest={manifest.get('data_signature')} current={data_signature}")


def validate_manifest_for_groups(manifest, groups, seed, val_ratio, data_signature):
    all_ids = {g["group_key"] for g in groups}
    train_ids = deserialize_manifest_group_ids(manifest, "train_group_ids")
    val_ids = deserialize_manifest_group_ids(manifest, "val_group_ids")
    if train_ids & val_ids:
        raise ValueError("manifest train/validation groups 有交集")
    if train_ids | val_ids != all_ids:
        raise ValueError("manifest groups 未严格覆盖当前数据")
    if manifest.get("seed") != seed:
        raise ValueError(f"manifest seed 不匹配: {manifest.get('seed')} != {seed}")
    if abs(float(manifest.get("val_ratio")) - float(val_ratio)) > 1e-12:
        raise ValueError("manifest val_ratio 不匹配")
    if manifest.get("all_group_count") != len(all_ids):
        raise ValueError("manifest all_group_count 不匹配")
    if manifest.get("train_group_count") != len(train_ids):
        raise ValueError("manifest train_group_count 不匹配")
    if manifest.get("val_group_count") != len(val_ids):
        raise ValueError("manifest val_group_count 不匹配")
    group_sizes = {group["group_key"]: len(group["samples"]) for group in groups}
    current_train_sample_count = sum(group_sizes[group_id] for group_id in train_ids)
    current_val_sample_count = sum(group_sizes[group_id] for group_id in val_ids)
    if manifest.get("train_sample_count") != current_train_sample_count:
        raise ValueError(
            "manifest train_sample_count 不匹配: "
            f"{manifest.get('train_sample_count')} != {current_train_sample_count}"
        )
    if manifest.get("val_sample_count") != current_val_sample_count:
        raise ValueError(
            "manifest val_sample_count 不匹配: "
            f"{manifest.get('val_sample_count')} != {current_val_sample_count}"
        )
    verify_manifest(manifest, train_ids, val_ids, data_signature)
    return train_ids, val_ids


def percentile(values, q):
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[index]


def summarize_token_lengths(lengths, cutoff_len):
    over = sum(length > cutoff_len for length in lengths)
    return {"count": len(lengths), "p50": percentile(lengths, .50),
            "p90": percentile(lengths, .90), "p95": percentile(lengths, .95),
            "p99": percentile(lengths, .99), "max": max(lengths, default=0),
            "over_cutoff_count": over,
            "over_cutoff_ratio": over / len(lengths) if lengths else 0.0}


def count_row_token_length(tokenizer, row):
    messages = [
        {"role": "user", "content": row["instruction"] + "\n\n" + row["input"]},
        {"role": "assistant", "content": row["output"]},
    ]
    try:
        length = len(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        ))
        return length, "apply_chat_template"
    except (AttributeError, ValueError, NotImplementedError):
        text = "\n\n".join(message["content"] for message in messages)
        return len(tokenizer(text)["input_ids"]), "tokenizer_fallback"


def summarize_token_lengths_by_mr(lengths, samples, cutoff_len):
    if samples is None:
        return {}
    if len(lengths) != len(samples):
        raise ValueError("token lengths 与 samples 数量不一致")
    buckets = {}
    for length, sample in zip(lengths, samples):
        mr_id = normalize_mr_id(sample.get("mr_id"))
        buckets.setdefault(mr_id, []).append(length)
    return {
        mr_id: summarize_token_lengths(values, cutoff_len)
        for mr_id, values in sorted(buckets.items())
    }


def build_token_length_report(
    train_rows,
    val_rows,
    tokenizer_path,
    cutoff_len,
    train_samples=None,
    val_samples=None,
    tokenizer=None,
):
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
    methods = Counter()

    def count_rows(rows):
        lengths = []
        for row in rows:
            length, method = count_row_token_length(tokenizer, row)
            lengths.append(length)
            methods[method] += 1
        return lengths

    train_lengths = count_rows(train_rows)
    val_lengths = count_rows(val_rows)
    counting_method = (
        "tokenizer_fallback" if methods.get("tokenizer_fallback")
        else "apply_chat_template"
    )
    return {
        "counting_method": counting_method,
        "counting_method_counts": dict(sorted(methods.items())),
        "tokenizer_path": tokenizer_path,
        "cutoff_len": cutoff_len,
        "train": summarize_token_lengths(train_lengths, cutoff_len),
        "validation": summarize_token_lengths(val_lengths, cutoff_len),
        "by_mr": {
            "train": summarize_token_lengths_by_mr(
                train_lengths, train_samples, cutoff_len
            ),
            "validation": summarize_token_lengths_by_mr(
                val_lengths, val_samples, cutoff_len
            ),
        },
    }


# ============================================================
# Conversion Report 输出
# ============================================================
def generate_conversion_report(
    mode,
    source_files,
    seed,
    groups_count,
    report_data,
    instruction_lengths=None,
    token_lengths=None,
    cutoff_len=None,
):
    """生成并打印 conversion report"""
    lines = [
        "=" * 60,
        "  Conversion Report",
        "=" * 60,
        f"  Mode: {mode}",
        f"  Source files: {source_files}",
        f"  Seed: {seed}",
        f"  Groups: {groups_count}",
        f"  Total samples: {report_data.get('total', 0)}",
        f"  Fallback count: {report_data.get('fallback_count', 0)}",
        f"  Missing operation desc: {report_data.get('missing_operation_count', 0)}",
        "",
        "  Label distribution:",
    ]
    for label, count in sorted(report_data.get("label_distribution", {}).items()):
        lines.append(f"    {label}: {count}")
    lines.append("")

    mr_dist = report_data.get("mr_distribution", {})
    if mr_dist:
        lines.append("  MR distribution:")
        for mr_id, count in sorted(mr_dist.items()):
            lines.append(f"    {mr_id}: {count}")

    if instruction_lengths:
        lines.append("")
        lines.append("  Instruction length (chars):")
        if instruction_lengths:
            lines.append(f"    min: {min(instruction_lengths)}")
            lines.append(f"    median: {sorted(instruction_lengths)[len(instruction_lengths)//2]}")
            lines.append(f"    max: {max(instruction_lengths)}")

    if token_lengths and cutoff_len:
        lines.append("")
        lines.append(f"  Token length (cutoff={cutoff_len}):")
        if token_lengths:
            lines.append(f"    min: {min(token_lengths)}")
            lines.append(f"    median: {sorted(token_lengths)[len(token_lengths)//2]}")
            lines.append(f"    max: {max(token_lengths)}")
            exceeded = sum(1 for t in token_lengths if t > cutoff_len)
            lines.append(f"    >{cutoff_len}: {exceeded}/{len(token_lengths)} ({exceeded/len(token_lengths)*100:.1f}%)")

    lines.append("=" * 60)
    print("\n".join(lines))


# ============================================================
# JSON 文件操作
# ============================================================
def load_jsonl(filepath):
    """加载 JSONL 文件"""
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "premise" in d and "hypothesis" in d and "label" in d:
                    samples.append(d)
            except json.JSONDecodeError:
                continue
    print(f"  📥 {Path(filepath).name}: {len(samples)} 条有效样本")
    return samples


def save_jsonl(samples, filepath):
    """保存为 JSONL 文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  💾 保存 {len(samples)} 条 → {filepath}")


def register_dataset(dataset_name, train_file, val_file=None, task_type="nli"):
    """注册数据集到 dataset_info.json"""
    info_path = WORK_DIR / "data" / "dataset_info.json"

    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            info = {}
    else:
        info = {}

    entry = {
        "file_name": str(train_file),
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        },
    }
    if task_type != "nli":
        entry["task_type"] = task_type
    info[dataset_name] = entry

    if val_file:
        val_entry = {
            "file_name": str(val_file),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            },
        }
        if task_type != "nli":
            val_entry["task_type"] = task_type
        info[dataset_name + "_val"] = val_entry

    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"  📝 已注册数据集 '{dataset_name}' → {info_path}")


# ============================================================
# 参数解析
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="将 NLI 数据转为 LLaMA-Factory 微调格式（Alpaca）"
    )
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="输入文件（JSONL 格式，每行有 premise, hypothesis, label 字段）")
    parser.add_argument("--name", "-n", default="nli_dataset",
                        help="数据集名称（用于注册到 dataset_info.json）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（默认: data/ft_datasets/<name>/）")
    parser.add_argument("--output-name", default=None,
                        help="输出文件名前缀（默认: full）")
    parser.add_argument("--split", "-s", action="store_true",
                        help="是否拆分为训练集和验证集")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例（默认 0.1）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数（用于测试）")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="是否打乱数据")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--instruction", type=str, default=None,
                        help="instruction 文本（覆盖内置 NLI 指令）")

    # MR-instruction 模式
    parser.add_argument("--mr-instruction", type=str, default=None,
                        help="[DEPRECATED: 请使用 --mr-instruction-mode] "
                             "兼容旧版参数，将映射为 pair_operation")
    parser.add_argument("--mr-instruction-mode",
                        choices=sorted(MR_INSTRUCTION_MODES),
                        default="none",
                        help="MR instruction 模式（默认: none）")
    parser.add_argument("--strict-pairing", action="store_true", default=False,
                        help="严格配对模式：未知 mr_id、缺失 source 等直接报错")
    parser.add_argument("--split-manifest", type=str, default=None,
                        help="读取共享 split manifest 路径")
    parser.add_argument("--write-split-manifest", type=str, default=None,
                        help="写入共享 split manifest 路径")
    parser.add_argument("--report-token-lengths", action="store_true", default=False,
                        help="输出 token 长度统计（需要 transformers）")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--cutoff-len", type=int, default=512)
    parser.add_argument(
        "--instruction-template-version",
        type=int,
        default=INSTRUCTION_TEMPLATE_VERSION,
        help=f"MR instruction template schema version (current: {INSTRUCTION_TEMPLATE_VERSION})",
    )

    # 兼容旧参数
    parser.add_argument("--binary", action="store_true", default=None,
                        help="[DEPRECATED] 强制二分类模式。RTE 已从主实验移除。")

    args = parser.parse_args()

    # 处理 --mr-instruction 兼容别名
    if args.mr_instruction is not None:
        print("  ⚠️  --mr-instruction 已弃用，请改用 --mr-instruction-mode")
        if args.mr_instruction_mode == "none":
            args.mr_instruction_mode = "pair_operation"

    return args


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    random.seed(args.seed)

    if args.instruction_template_version != INSTRUCTION_TEMPLATE_VERSION:
        raise ValueError(
            "不支持的 instruction template version: "
            f"{args.instruction_template_version}; "
            f"当前仅支持 v{INSTRUCTION_TEMPLATE_VERSION}"
        )

    # 兼容旧版 --binary（RTE 已移除，但仍保持功能以防外部调用）
    is_binary = args.binary if args.binary is not None else False
    task_type = "nli-binary" if is_binary else "nli"
    if is_binary:
        print("  ⚠️  二分类模式已启用。注意：RTE 已从主实验中移除。")

    mode = args.mr_instruction_mode
    active_manifest = None

    # 确定使用的 label_names 和 nli instruction
    label_names = LABEL_NAMES_BINARY if is_binary else LABEL_NAMES_3CLASS
    nli_instruction = (
        args.instruction or
        (INSTRUCTION_NLI_BINARY if is_binary else INSTRUCTION_NLI)
    )

    print(f"\n{'='*60}")
    print(f"  🔄 NLI → LLaMA-Factory 格式转换")
    print(f"  数据集: {args.name}")
    print(f"  任务类型: {'二分类' if is_binary else '三分类'}")
    print(f"  MR Instruction Mode: {mode}")
    print(f"  Strict Pairing: {args.strict_pairing}")
    print(f"{'='*60}\n")

    # ============================
    # 1. 加载所有输入文件
    # ============================
    all_samples = []
    source_file_ids = []
    for filepath in args.input:
        path = Path(filepath)
        if not path.exists():
            print(f"  ⚠️ 文件不存在: {path}")
            continue
        samples = load_jsonl(path)
        file_id = path.stem  # 文件名（不含扩展名）
        all_samples.extend(samples)
        source_file_ids.extend([file_id] * len(samples))

    if not all_samples:
        print("  ❌ 没有有效样本")
        return

    print(f"\n  📊 总计: {len(all_samples)} 条")

    # ============================
    # 2. Preflight 校验
    # ============================
    try:
        preflight_report = validate_mr_samples(
            all_samples,
            strict=args.strict_pairing,
        )
    except ValueError as e:
        print(f"  ❌ Preflight 校验失败: {e}")
        raise

    # ============================
    # 4. 构建 pair groups（先于 max_samples，确保 group 完整性）
    # ============================
    groups = build_pair_groups(all_samples, source_file_ids)
    print(f"  📦 Pair groups: {len(groups)}")

    # ============================
    # 3. 可选限制样本数（按 group 级别采样）
    # ============================
    if args.max_samples:
        # 先估算需要多少个 group 能达到目标样本数
        total_available = sum(len(g["samples"]) for g in groups)
        if total_available > args.max_samples:
            # 按 group 随机采样，尽可能接近 max_samples
            random.shuffle(groups)
            sampled_groups = []
            sampled_count = 0
            for g in groups:
                if sampled_count + len(g["samples"]) <= args.max_samples or not sampled_groups:
                    sampled_groups.append(g)
                    sampled_count += len(g["samples"])
                else:
                    break
            groups = sampled_groups
            all_samples = flatten_groups(groups)
            print(f"  ✂️ Group-level 采样至 {len(all_samples)} 条（{len(groups)} groups）")

    # ============================
    # 5. Group-aware split
    # ============================
    train_groups = groups  # default: no split
    val_groups = []

    if args.split and len(groups) > 1:
        if args.split_manifest:
            # 复用已有 split manifest
            manifest = load_split_manifest(args.split_manifest)
            active_manifest = manifest
            print(f"  📋 加载 split manifest: {args.split_manifest}")

            data_signature = compute_data_signature(all_samples)
            manifest_train, manifest_val = validate_manifest_for_groups(
                manifest, groups, args.seed, args.val_ratio, data_signature)
            # 构建 group_key -> group 索引
            group_map = {g["group_key"]: g for g in groups}

            # 从 manifest 读取 group IDs
            train_ids, val_ids = manifest_train, manifest_val

            # 验证所有 ID 都在当前数据中
            all_ids = set(g["group_key"] for g in groups)
            missing_train = train_ids - all_ids
            missing_val = val_ids - all_ids
            if missing_train or missing_val:
                raise ValueError(
                    f"Split manifest 中的 group ID 不在当前数据中: "
                    f"train missing={missing_train}, val missing={missing_val}"
                )

            train_groups = [group_map[i] for i in train_ids if i in group_map]
            val_groups = [group_map[i] for i in val_ids if i in group_map]

            if not train_groups:
                raise ValueError("Split manifest 没有匹配的训练组")
            print(f"  📋 复用 manifest: {len(train_groups)} train / {len(val_groups)} val groups")
        else:
            # 新拆分：使用 +999 偏移 seed，避免与 --max-samples 的 random state 冲突
            # 这样不同 seed 产生不同的 group 选择和不同的拆分，相同 seed 完全可重现
            train_groups, val_groups = split_pair_groups(
                groups,
                val_ratio=args.val_ratio,
                seed=args.seed + 999,
            )
            print(f"  ✂️ 拆分: {len(train_groups)} train / {len(val_groups)} val groups")

            # 保存 manifest
            if args.write_split_manifest:
                train_ids = [g["group_key"] for g in train_groups]
                val_ids = [g["group_key"] for g in val_groups]
                active_manifest = save_split_manifest(
                    manifest_path=args.write_split_manifest,
                    seed=args.seed,
                    val_ratio=args.val_ratio,
                    train_group_ids=train_ids,
                    val_group_ids=val_ids,
                    source_files=args.input,
                    data_signature=compute_data_signature(all_samples),
                    train_sample_count=sum(len(g["samples"]) for g in train_groups),
                    val_sample_count=sum(len(g["samples"]) for g in val_groups),
                )
                print(f"  📋 保存 split manifest: {args.write_split_manifest}")

    # ============================
    # 6. 构建 source map
    # ============================
    original_map, missing_sources = build_original_map(train_groups)
    if missing_sources:
        print(f"  ⚠️  有 {len(missing_sources)} 个 group 缺少 source 样本")
        for key, reason, count in missing_sources[:5]:
            warn = f"    {key}: {reason} (count={count})"
            if args.strict_pairing:
                raise ValueError(f"Strict 模式: 增强样本缺少 source: {warn}")
            print(warn)

    # 给每个样本标记 group_key
    for g in groups:
        for s in g["samples"]:
            s["_group_key"] = g["group_key"]

    # ============================
    # 7. 生成 shuffled 描述（若需要）
    # ============================
    shuffled_descriptions = None
    shuffle_audit = None
    if mode == "shuffled_operation":
        train_samples_flat = sort_samples_by_stable_key(flatten_groups(train_groups))
        all_mr_ids = set(normalize_mr_id(s.get("mr_id")) for s in all_samples)
        shuffle_seed = args.seed + 999
        shuffled_descriptions = select_shuffled_descriptions(
            train_samples_flat, all_mr_ids, seed=shuffle_seed
        )
        shuffle_audit = build_shuffle_audit(
            train_samples_flat,
            shuffled_descriptions,
            shuffle_seed,
        )
        print(f"  🔀 Shuffled operation descriptions 已生成")

    # ============================
    # 8. 转换：训练集（按指定 mode）
    # ============================
    train_samples = sort_samples_by_stable_key(flatten_groups(train_groups))
    print(f"\n  🔄 转换训练集（mode={mode}）: {len(train_samples)} 条")

    # 准备操作描述 map（过滤掉不在 ANY_MR 中的 key）
    op_map = {k: v for k, v in MR_OPERATION_DESCRIPTIONS.items()}
    rel_map = {k: v for k, v in MR_RELATION_EFFECTS.items()}

    train_converted, train_report = convert_to_alpaca(
        train_samples,
        mode=mode,
        is_binary=is_binary,
        original_map=original_map if mode not in ("none", "operation_only") else None,
        operation_descriptions=op_map,
        relation_effects=rel_map if mode in MODES_REQUIRING_RELATION_EFFECT else None,
        shuffled_descriptions=shuffled_descriptions,
        strict=args.strict_pairing,
        nli_instruction=nli_instruction,
    )

    print(f"  📈 训练集标签分布: {dict(train_report['label_distribution'])}")

    # ============================
    # 9. 转换：验证集（始终 mode=none）
    # ============================
    val_converted = []
    val_report = {}
    val_samples = []
    if val_groups:
        val_samples = sort_samples_by_stable_key(flatten_groups(val_groups))
        print(f"\n  🔄 转换验证集（mode=none）: {len(val_samples)} 条")
        val_converted, val_report = convert_to_alpaca(
            val_samples,
            mode="none",
            is_binary=is_binary,
            strict=False,
            nli_instruction=nli_instruction,
        )
        print(f"  📈 验证集标签分布: {dict(val_report.get('label_distribution', {}))}")

    # ============================
    # 10. 输出目录
    # ============================
    if args.output_dir:
        out_dir = WORK_DIR / args.output_dir / args.name
    else:
        out_dir = WORK_DIR / "data" / "ft_datasets" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================
    # 11. Instruction 长度统计
    # ============================
    all_converted = train_converted + val_converted
    instr_lengths = [len(c["instruction"]) for c in all_converted]
    token_report = None
    if args.report_token_lengths:
        if not args.tokenizer_path:
            raise ValueError("--report-token-lengths 需要 --tokenizer-path")
        token_report = build_token_length_report(
            train_converted,
            val_converted,
            args.tokenizer_path,
            args.cutoff_len,
            train_samples=train_samples,
            val_samples=val_samples,
        )

    # ============================
    # 12. 保存 + 注册
    # ============================
    output_stem = args.output_name or "full"
    if val_converted:
        train_path = out_dir / f"{output_stem}_train.json"
        val_path = out_dir / f"{output_stem}_val.json"
        save_jsonl(train_converted, train_path)
        save_jsonl(val_converted, val_path)
        print(f"\n  ✅ 拆分为: 训练集 {len(train_converted)}条 + 验证集 {len(val_converted)}条")
        register_dataset(args.name, train_path, val_path, task_type)
    else:
        all_path = out_dir / f"{output_stem}.json"
        save_jsonl(train_converted, all_path)
        register_dataset(args.name, all_path, task_type=task_type)

    # ============================
    # 13. Conversion Report
    # ============================
    generate_conversion_report(
        mode=mode,
        source_files=args.input,
        seed=args.seed,
        groups_count=len(groups),
        report_data=train_report,
        instruction_lengths=instr_lengths,
    )

    # 保存 conversion report 到文件
    report_path = out_dir / "conversion_report.json"
    report_json = {
        "mode": mode,
        "source_files": args.input,
        "seed": args.seed,
        "strict_pairing": args.strict_pairing,
        "groups_count": len(groups),
        "train_group_count": len(train_groups),
        "val_group_count": len(val_groups),
        "train": train_report,
        "val": val_report,
        "instruction_length_stats": {
            "min": min(instr_lengths) if instr_lengths else 0,
            "max": max(instr_lengths) if instr_lengths else 0,
            "samples": len(instr_lengths),
        },
        "data_signature": compute_data_signature(all_samples),
        "manifest_hash": active_manifest.get("sha256") if active_manifest else None,
        "instruction_template_version": INSTRUCTION_TEMPLATE_VERSION,
        "instruction_template_hash": INSTRUCTION_TEMPLATE_HASH,
        "operation_description_hash": OPERATION_DESCRIPTION_HASH,
        "relation_effect_hash": RELATION_EFFECT_HASH,
        "ordered_sample_signature": compute_ordered_sample_signature(
            train_samples,
            val_samples,
        ),
        "converted_train_sha256": converted_rows_sha256(train_converted),
        "converted_val_sha256": (
            converted_rows_sha256(val_converted) if val_converted else None
        ),
        "token_length": token_report,
        "upper_bound": mode == "full_oracle",
        "shuffle_audit": shuffle_audit,
    }
    report_path.write_text(json.dumps(report_json, indent=2, ensure_ascii=False))
    print(f"  📊 转换报告: {report_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ 完成！数据集: {args.name}")
    print(f"  路径: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
