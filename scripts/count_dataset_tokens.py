#!/usr/bin/env python3
"""
count_dataset_tokens.py
=======================
统计一个数据集的：记录数、文本 token 总量（原始文本 + 完整训练样本两种口径）。

与 convert_nli_to_ft.py 保持一致的两类口径：
  * raw tokens —— 只对样本内容文本做 tokenize（无 instruction/chat 模板开销）
  * train tokens —— 按真实送入模型训练的格式（instruction→user、output→assistant）
      经 apply_chat_template 计数，等价于 convert 的 count_row_token_length()。
      对已经是 Alpaca 格式（含 instruction）的行逐字精确；对未转换的原始 NLI 行
      （premise/hypothesis，且无 MR 标记）按 RQ1 默认 none-mode + 模板 v2 估算。

记录格式自动识别：
  * converted —— 含 "instruction"（Alpaca）：两种口径都给（train 精确）
  * nli_raw   —— 含 premise+hypothesis，无 MR 标记：raw 精确；train 按 none-mode 估算
  * nli_raw_mr—— 含 premise+hypothesis 且带 MR 标记（mr_id/mr_type≠none）：
                 真实训练文本含 reference sample + operation 描述，无法从单文件还原，
                 只给 raw，train 留空并计数说明
  * other     —— 其余（自定义任务）：只给 raw（拼接所有字符串字段）

用法（llmtrain310 环境）：
    # 单个文件（JSON / JSONL / 多对象串联均可）
    python scripts/count_dataset_tokens.py data/nli/zaug/rq1_mnli/zaug_sample_s42.jsonl

    # 整个目录（聚合目录下所有样本文件，如 MR 数据集目录）
    python scripts/count_dataset_tokens.py data/nli/mr_test_data_merged

    # convert 输出目录（如 data/ft_datasets/<实验名>/）：自动只统计 full*.json
    # (Alpaca 样本)，跳过 conversion_report.json / sampled.json；并分开列出
    # full_train / full_val 各自的 token 总量
    python scripts/count_dataset_tokens.py data/ft_datasets/rq1_mr_mnlim_seed42

    # 同时比较多个数据集
    python scripts/count_dataset_tokens.py \
        data/nli/original_dataset/mnlim/train.json \
        data/nli/mettrain/mnlim_lr0.0051_gemma-3-4b-it-qat/augmented_data.json

    # 换 tokenizer（HF id 或本地路径）
    python scripts/count_dataset_tokens.py data/.../x.jsonl --tokenizer meta-llama/Llama-3.1-8B

    # 二分类任务（label 0/1 → entailment / not_entailment）
    python scripts/count_dataset_tokens.py data/.../rte.jsonl --binary
"""
__test__ = False

import argparse
import json
import statistics
import sys
from pathlib import Path

from project_runtime import PROJECT_ROOT

DEFAULT_TOKENIZER_ID = "google/gemma-3-4b-it"
DEFAULT_TOKENIZER_LOCAL = PROJECT_ROOT / "models" / "google" / "gemma-3-4b-it"

LABEL3_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_BINARY_NAMES = {0: "entailment", 1: "not_entailment"}

# ── 与 convert_nli_to_ft.py 相同的 none-mode instruction / 模板 v2（惰性导入，防漂移）──
_NLI_INSTRUCTION = None
_NLI_INSTRUCTION_BINARY = None


def _load_convert_constants():
    global _NLI_INSTRUCTION, _NLI_INSTRUCTION_BINARY
    if _NLI_INSTRUCTION is not None:
        return
    try:
        # 同目录的 convert 脚本，顶部 import 很轻（无 transformers/torch 副作用）
        from convert_nli_to_ft import (
            INSTRUCTION_NLI,
            INSTRUCTION_NLI_BINARY,
        )
        _NLI_INSTRUCTION = INSTRUCTION_NLI
        _NLI_INSTRUCTION_BINARY = INSTRUCTION_NLI_BINARY
    except ImportError as exc:  # pragma: no cover - 防御
        print(f"⚠️  无法导入 convert_nli_to_ft 常量（{exc}），回退为内嵌副本")
        _NLI_INSTRUCTION = (
            "Determine the natural language inference relation between the premise "
            "and hypothesis. Answer with exactly one label: entailment, neutral, "
            "or contradiction."
        )
        _NLI_INSTRUCTION_BINARY = (
            "Determine whether the premise entails the hypothesis. "
            "Answer with exactly one label: entailment or not_entailment."
        )


# ══════════════════════════════════════════════════════════════════════════
# 记录读取（JSON / JSONL / 同一行多对象串联 均支持）
# ══════════════════════════════════════════════════════════════════════════

def iter_records(text):
    """把文件文本解析为记录序列。

    依次尝试：(1) 整个文件是一个 JSON array/single-object；(2) 否则按行 / raw_decode
    处理 JSONL 与"每行多个拼接 JSON 对象"（convert 输出的 full.json 可能如此）。
    无法解析的行会被跳过并计数。
    """
    text = text.lstrip("﻿")
    text = text.strip()
    if not text:
        return
    try:
        data = json.loads(text)
        if isinstance(data, list):
            yield from (x for x in data if isinstance(x, dict))
            return
        if isinstance(data, dict):
            yield data
            return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    skipped = 0
    while True:
        while idx < n and text[idx] in " \t\r\n":
            idx += 1
        if idx >= n:
            break
        try:
            obj, idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                yield obj
        except json.JSONDecodeError:
            skipped += 1
            nxt = text.find("\n", idx)
            idx = n if nxt == -1 else nxt + 1
    if skipped:
        print(f"    ⚠️ 跳过 {skipped} 行无法解析的 JSON", file=sys.stderr)


def load_records(path):
    """读取单个文件，返回 (records, skipped_lines)。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return list(iter_records(text))


# ══════════════════════════════════════════════════════════════════════════
# 记录分类与文本提取
# ══════════════════════════════════════════════════════════════════════════

_MR_KEYS = ("mr_id", "mr_type")


def is_mr_row(rec):
    """行是否为 MR 变体（真实训练文本需转换上下文，无法从单文件还原）。

    mr_id 是权威标记：source 行也可能带 mr_type（如 inv），所以优先看 mr_id；
    仅当 mr_id 缺失时才退回看 mr_type。
    """
    if "mr_id" in rec:
        val = rec.get("mr_id")
        if val is None:
            return False
        return str(val).strip().lower() not in ("", "none", "source", "original")
    for key in ("mr_type",):
        val = rec.get(key)
        if val and str(val).strip().lower() not in ("", "none"):
            return True
    return False


def classify(rec):
    if "instruction" in rec and isinstance(rec.get("instruction"), str):
        return "converted"
    if "premise" in rec and "hypothesis" in rec:
        return "nli_raw_mr" if is_mr_row(rec) else "nli_raw"
    return "other"


def label_name(label, binary):
    if isinstance(label, bool):
        label = int(label)
    if isinstance(label, int):
        mapping = LABEL_BINARY_NAMES if binary else LABEL3_NAMES
        return mapping.get(label, str(label))
    return str(label)


def build_training_text(rec, binary):
    """把原始 NLI 行按 convert none-mode + 模板 v2 还原成 Alpaca 行。

    返回 (user_content, assistant_content) 或 None（无法还原）。
    """
    _load_convert_constants()
    nli_instruction = _NLI_INSTRUCTION_BINARY if binary else _NLI_INSTRUCTION
    input_text = (
        f"Premise: {str(rec.get('premise', '')).strip()}\n"
        f"Hypothesis: {str(rec.get('hypothesis', '')).strip()}"
    )
    output = label_name(rec.get("label", ""), binary)
    return f"{nli_instruction}\n\n{input_text}", output


def raw_text_of(rec, binary):
    """样本内容文本（不含 instruction/chat 模板开销），用于 raw 口径。

    - converted: instruction + input + output 拼接
    - nli_*:     premise + hypothesis（不含标签与 "Premise:" 前缀）
    - other:     递归拼接所有字符串字段
    """
    if classify(rec) == "converted":
        parts = [str(rec.get("instruction", ""))]
        if rec.get("input"):
            parts.append(str(rec["input"]))
        parts.append(str(rec.get("output", "")))
        return "\n".join(p for p in parts if p)
    if "premise" in rec and "hypothesis" in rec:
        return f"{str(rec.get('premise', '')).strip()}\n{str(rec.get('hypothesis', '')).strip()}"
    # other：收集所有字符串叶子
    leaves = []

    def collect(node):
        if isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)
        elif isinstance(node, str):
            leaves.append(node)

    collect(rec)
    return "\n".join(leaves)


def chat_template_tokens(tokenizer, user_content, assistant_content):
    """按训练格式（user→instruction+input, assistant→output）计数 token。

    与 convert_nli_to_ft.count_row_token_length 一致：优先 apply_chat_template；
    旧 tokenizer 无该方法时退化为纯文本拼接。
    """
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    try:
        return len(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
        )), "apply_chat_template"
    except (AttributeError, ValueError, NotImplementedError):
        text = f"{user_content}\n{assistant_content}"
        return len(tokenizer(text)["input_ids"]), "tokenizer_fallback"


def count_record(tokenizer, rec, binary):
    """统计单条记录，返回 dict。

    raw  为 None 表示该格式不适用；train 为 None 表示该行不计入训练口径（见 kind）。
    """
    kind = classify(rec)
    raw_len = train_len = None
    train_method = None

    if kind == "converted":
        inst = str(rec.get("instruction", ""))
        inp = str(rec.get("input", "")) if rec.get("input") else ""
        out = str(rec.get("output", ""))
        user_content = f"{inst}\n\n{inp}" if inp else inst
        raw_text = raw_text_of(rec, binary)
        raw_len = len(tokenizer(raw_text)["input_ids"]) if raw_text else None
        train_len, train_method = chat_template_tokens(tokenizer, user_content, out)
    elif kind in ("nli_raw", "nli_raw_mr"):
        text = f"{str(rec.get('premise', '')).strip()}\n{str(rec.get('hypothesis', '')).strip()}"
        raw_len = len(tokenizer(text)["input_ids"])
        if kind == "nli_raw":
            # source 行（mr_id 缺失或 none）：convert 一律用普通 NLI instruction（none-mode），
            # 此处估算即精确（original / zaug 训练同理）
            user_content, assistant = build_training_text(rec, binary)
            train_len, train_method = chat_template_tokens(tokenizer, user_content, assistant)
    else:
        text = raw_text_of(rec, binary)
        if text:
            raw_len = len(tokenizer(text)["input_ids"])

    return {
        "kind": kind,
        "raw": raw_len,
        "train": train_len,
        "train_method": train_method,
    }


# ══════════════════════════════════════════════════════════════════════════
# 汇总与输出
# ══════════════════════════════════════════════════════════════════════════

def summarize(values):
    """对 token 长度序列给出 total/mean/median/max 摘要（空序列返回 None）。"""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "total": sum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def fmt_summary(s):
    if s is None:
        return "  n/a"
    return (f"total={s['total']:,}  mean={s['mean']:,.1f}  "
            f"median={s['median']:,.0f}  max={s['max']:,}")


_METADATA_NAMES = {"conversion_report.json", "dataset_info.json"}
_CONVERT_SAMPLE_PREFIX = "full"  # convert 输出：full.json / full_train.json / full_val.json


def _select_files(path):
    """按目录内容选择"样本文件"，返回 (files, note, is_convert_dir)。

    - convert 输出目录（含 conversion_report.json 或 full*.json）：
      只取 full*.json（Alpaca 样本），排除 conversion_report.json / sampled.json /
      *_summary*.json 等元数据或转换前原始输入。
    - 普通数据目录：取全部 .json/.jsonl，排除 *_summary*.json（如 MR 目录的 summary 元数据）。
    """
    all_files = sorted(
        list(path.glob("*.json")) + list(path.glob("*.jsonl"))
    )
    names = {f.name for f in all_files}
    is_convert_dir = bool(
        names & _METADATA_NAMES
        or any(f.stem.startswith(_CONVERT_SAMPLE_PREFIX) and not f.stem.endswith("_report")
               for f in all_files)
    )

    dropped = []
    if is_convert_dir:
        sample = [f for f in all_files
                  if f.stem.startswith(_CONVERT_SAMPLE_PREFIX)
                  and not f.stem.endswith("_report")]
        if not sample:
            # 目录看起来是 convert 输出但还没有 full*.json（未完成/损坏），退回普通处理
            is_convert_dir = False
            sample = [f for f in all_files
                      if "summary" not in f.stem.lower() and f.name not in _METADATA_NAMES]
            dropped = [f.name for f in all_files if f not in sample]
        else:
            dropped = [f.name for f in all_files if f not in sample]
    else:
        sample = [f for f in all_files if "summary" not in f.stem.lower()]

    note = ""
    if is_convert_dir:
        note = f"（convert 输出目录：只统计 full*.json，跳过 {len(dropped)} 个其它文件：{', '.join(sorted(dropped)[:6])}{'…' if len(dropped) > 6 else ''}）"
    elif dropped:
        note = f"（跳过 {len(dropped)} 个 summary 元数据文件）"
    return sample, note, is_convert_dir


def process_path(path, tokenizer, binary, verbose_files=False):
    """处理一个文件或目录，打印汇总表。返回 (records_count, stats dict) 便于累加。"""
    path = Path(path)
    files = []
    label = str(path)
    if path.is_dir():
        files, note, is_convert_dir = _select_files(path)
        label = f"{path}  (目录, {len(files)} 样本文件){note}"
    elif path.is_file():
        files = [path]
        label = str(path)
    else:
        print(f"⚠️  路径不存在: {path}", file=sys.stderr)
        return 0, {}

    if not files:
        print(f"⚠️  {label}: 无样本 .json/.jsonl 文件", file=sys.stderr)
        return 0, {}

    raw_all, train_all = [], []
    counts = {"converted": 0, "nli_raw": 0, "nli_raw_mr": 0, "other": 0}
    methods = {}
    n_total = 0
    per_file = []  # (name, n, raw_n, raw_sum, train_n, train_sum)

    print(f"\n=== {label} ===")
    for f in files:
        recs = load_records(f)
        f_n = f_raw_n = f_raw_sum = f_train_n = f_train_sum = 0
        for rec in recs:
            n_total += 1
            f_n += 1
            r = count_record(tokenizer, rec, binary)
            counts[r["kind"]] += 1
            if r["raw"] is not None:
                raw_all.append(r["raw"])
                f_raw_n += 1
                f_raw_sum += r["raw"]
            if r["train"] is not None:
                train_all.append(r["train"])
                f_train_n += 1
                f_train_sum += r["train"]
                if r["train_method"]:
                    methods[r["train_method"]] = methods.get(r["train_method"], 0) + 1
        per_file.append((f.name, f_n, f_raw_n, f_raw_sum, f_train_n, f_train_sum))

    raw_sum = summarize(raw_all)
    train_sum = summarize(train_all)
    n_mr_skipped = counts["nli_raw_mr"]

    # 决定 train 口径说明
    if train_sum is None:
        if n_total > 0 and n_mr_skipped == n_total:
            train_desc = ("n/a —— 全部为 MR 变体行，真实训练文本含 reference sample + operation 描述，"
                          "无法从单文件还原；需先 convert 成 Alpaca 后再统计")
        else:
            train_desc = "n/a（不含可训练格式记录）"
    elif counts["converted"] == n_total:
        train_desc = f"chat-template（{n_total} 行均为已转换 Alpaca，精确）"
    else:
        parts = []
        if counts["converted"]:
            parts.append(f"{counts['converted']:,} 行已转换（精确）")
        if counts["nli_raw"]:
            parts.append(f"{counts['nli_raw']:,} 行 source/普通 NLI（none-mode 模板 v2 估算，original/zaug 即精确）")
        if n_mr_skipped:
            parts.append(f"{n_mr_skipped:,} 行 MR 变体未计入 train")
        train_desc = "chat-template —— " + "、".join(parts)

    print(f"  记录数    : {n_total:,}")
    if len(per_file) > 1:
        for name, f_n, f_raw_n, f_raw_sum, f_train_n, f_train_sum in per_file:
            raw_txt = f"raw={f_raw_sum:,}" if f_raw_n else "raw=—"
            train_txt = f"train={f_train_sum:,}" if f_train_n else "train=—"
            print(f"    · {name:<16} {f_n:>7,} 条   {raw_txt:<18} {train_txt}")
    print(f"  构成      : converted={counts['converted']:,}  "
          f"nli_raw(source)={counts['nli_raw']:,}  nli_raw(MR)={counts['nli_raw_mr']:,}  "
          f"other={counts['other']:,}")
    print(f"  raw tokens: {fmt_summary(raw_sum)}")
    print(f"  train     : {fmt_summary(train_sum)}  [{train_desc}]")
    if methods:
        print(f"  计数方式  : {', '.join(f'{k}×{v}' for k, v in sorted(methods.items()))}")

    stats = {
        "path": str(path),
        "n": n_total,
        "counts": counts,
        "raw": raw_sum,
        "train": train_sum,
        "train_desc": train_desc,
    }
    return n_total, stats


def main():
    parser = argparse.ArgumentParser(
        description="统计数据集记录数与 token 总量（raw 文本 + 训练格式两口径）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/count_dataset_tokens.py data/nli/zaug/rq1_mnli/zaug_sample_s42.jsonl\n"
            "  python scripts/count_dataset_tokens.py data/nli/mr_test_data_merged\n"
            "  python scripts/count_dataset_tokens.py data/ft_datasets/rq1_mnlim_seed42   # convert 输出目录\n"
            "  python scripts/count_dataset_tokens.py a.jsonl b.jsonl --tokenizer meta-llama/Llama-3.1-8B\n"
        ),
    )
    parser.add_argument("paths", nargs="+", help="数据集文件或目录（可多个）")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer 的 HF id 或本地路径（默认优先本地缓存 Gemma，否则 "
                             f"回退 {DEFAULT_TOKENIZER_ID}）")
    parser.add_argument("--binary", action="store_true",
                        help="二分类任务（label 0/1 → entailment / not_entailment），仅影响 raw-NLI 行的训练估算")
    parser.add_argument("--show-files", action="store_true",
                        help="目录模式下列出每个文件的行数")
    args = parser.parse_args()

    if args.tokenizer:
        tok_ref = args.tokenizer
        if not Path(tok_ref).is_absolute() and "/" not in tok_ref and "\\" not in tok_ref:
            cand = PROJECT_ROOT / tok_ref
            if cand.is_dir():
                tok_ref = str(cand)
    elif DEFAULT_TOKENIZER_LOCAL.is_dir():
        tok_ref = str(DEFAULT_TOKENIZER_LOCAL)
    else:
        tok_ref = DEFAULT_TOKENIZER_ID

    print(f"🔤 Tokenizer: {tok_ref}")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_ref, trust_remote_code=True)
    except Exception as exc:
        sys.exit(f"❌ 加载 tokenizer 失败（{tok_ref}）: {exc}")

    grand_n = 0
    grand_raw = grand_train = None
    all_stats = []
    for p in args.paths:
        n, stats = process_path(p, tokenizer, args.binary, args.show_files)
        grand_n += n
        all_stats.append(stats)

    if len(args.paths) > 1:
        # 汇总行
        raw_vals = [s["raw"]["total"] for s in all_stats if s.get("raw")]
        train_vals = [s["train"]["total"] for s in all_stats if s.get("train")]
        print("\n" + "─" * 60)
        print(f"总计: {grand_n:,} 条")
        if raw_vals:
            print(f"  raw tokens  合计: {sum(raw_vals):,}")
        if train_vals:
            print(f"  train tokens 合计: {sum(train_vals):,}")


if __name__ == "__main__":
    main()
