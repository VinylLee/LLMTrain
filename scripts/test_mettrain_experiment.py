#!/usr/bin/env python3
"""
测试微调后的 LoRA 模型：
  - Original 数据集
  - MR 数据集（聚合所有 MR 类型）
  - Merged 数据集（source + MR 混合，通过 is_source 区分）
保存 JSONL 结果到 output/experiments/<name>/tests/{original,mr,merged}/
"""
__test__ = False
import json, os, sys, time, torch, logging
from pathlib import Path

from project_runtime import PROJECT_ROOT, configure_console_encoding
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging
from peft import PeftModel
from inference_utils import decode_generated_continuations, unpack_generation_inputs

# 抑制烦人的生成参数警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

WORK_DIR = PROJECT_ROOT

# 原始数据集配置
ORIGINAL_DATASETS = {
    "snli":   {"dir": "original_dataset/snli",   "file": "test.json",      "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlim":  {"dir": "original_dataset/mnlim",  "file": "test.json",      "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlimm": {"dir": "original_dataset/mnlimm", "file": "test.json",      "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "sick":   {"dir": "original_dataset/sick",   "file": "test.json",      "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
}

# MR 测试数据集配置
MR_DATASETS = {
    "snli":   {"dir": "MR_testing/snli_test_MR",   "task": "3class", "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlim":  {"dir": "MR_testing/mnlim_test_MR",  "task": "3class", "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlimm": {"dir": "MR_testing/mnlimm_test_MR", "task": "3class", "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "sick":   {"dir": "MR_testing/sick_test_MR",   "task": "3class", "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
}

# Merged 测试数据集配置（source + MR 混合，通过 is_source 字段区分）
MERGED_DATASETS = {
    "snli":   "mr_test_data_merged/snli.jsonl",
    "mnlim":  "mr_test_data_merged/mnlim.jsonl",
    "mnlimm": "mr_test_data_merged/mnlimm.jsonl",
    "sick":   "mr_test_data_merged/sick.jsonl",
}

BINARY_PROMPT = "Determine whether the premise entails the hypothesis. Answer with exactly one label: entailment or not_entailment.\n\nPremise: {premise}\nHypothesis: {hypothesis}"
CLASS3_PROMPT = "Determine the natural language inference relation between the premise and hypothesis. Answer with exactly one label: entailment, neutral, or contradiction.\n\nPremise: {premise}\nHypothesis: {hypothesis}"


def load_model(base_model_name, lora_path):
    print(f"🔄 加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    if lora_path and os.path.exists(lora_path):
        print(f"🔄 加载 LoRA 适配器: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def call_model(model, tokenizer, prompt, max_tokens=10):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    model_inputs, input_length = unpack_generation_inputs(inputs)
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    return decode_generated_continuations(outputs, input_length, tokenizer)[0]


def call_model_batch(model, tokenizer, prompts, max_tokens=10):
    """批量贪婪生成。

    左 padding（Gemma 默认 padding_side='left'），配合 attention_mask，
    每条样本的生成结果与单条逐次生成完全一致（greedy + 无跨样本注意力）。
    每行最后 max_tokens 个 token 即为该样本的新生成内容。
    """
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, tokenize=False)
        for p in prompts
    ]
    enc = tokenizer(texts, padding=True, return_tensors="pt",
                    add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return decode_generated_continuations(outputs, enc["input_ids"].shape[1], tokenizer)


def normalize_prediction(pred, task, labels):
    """归一化预测结果"""
    pred = pred.strip().lower().rstrip(".!?,")

    # === 二分类模式 ===
    if task == "binary":
        # 直接匹配
        if pred in ("entailment", "not_entailment"):
            return pred
        # 关键词匹配（优先于部分匹配，避免 "entailment" in "not entailment" 误匹配）
        if "entail" in pred and "not" not in pred:
            return "entailment"
        if any(word in pred for word in ("not_entail", "not entail", "not entailment",
                                         "neutral", "contradict", "contradiction", "2")):
            return "not_entailment"
        # 简写映射
        if pred in ("entailed", "e", "0"):
            return "entailment"
        if pred in ("n", "1"):
            return "not_entailment"
        # 部分匹配（兜底）
        for label_id, label_name in labels.items():
            if label_name.lower() in pred or pred in label_name.lower():
                return label_name
        return pred

    # === 三分类模式 ===
    if pred in ("entailment", "neutral", "contradiction"):
        return pred
    for label_id, label_name in labels.items():
        if pred == label_name.lower():
            return label_name
        if label_name.lower() in pred or pred in label_name.lower():
            return label_name
    mapping = {
        "entailed": "entailment", "e": "entailment", "0": "entailment",
        "n": "neutral", "1": "neutral",
        "contradictory": "contradiction", "contradict": "contradiction",
        "c": "contradiction", "2": "contradiction",
        "not_entailment": "not_entailment", "not entailment": "not_entailment",
    }
    return mapping.get(pred, pred)


def load_jsonl_samples(filepath):
    """加载 JSONL 文件"""
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                samples.append(d)
            except json.JSONDecodeError:
                continue
    return samples


def collapse_label(gold_label, dataset_task, model_task):
    """将 gold label 折叠到模型训练任务的标签空间。

    当模型训练任务与测试集原生任务不同时，折叠标签使评估有意义：
    - 模型是二分类、测试集是三分类：neutral/contradiction → not_entailment
    """
    if model_task == "binary" and dataset_task == "3class":
        gold_lower = gold_label.strip().lower()
        if gold_lower in ("neutral", "contradiction", "contradictory"):
            return "not_entailment"
    return gold_label


def test_dataset(model, tokenizer, samples, ds_name, dataset_task, labels,
                 output_path, max_samples=None, model_task=None, batch_size=32,
                 test_type="original", mr_types=None):
    """测试一个数据集（批量推理），保存 JSONL 结果。

    Args:
        model_task: 模型训练时的任务类型 (binary/3class)。
                    None 表示与 dataset_task 相同（常规评估）。
                    不同时触发标签折叠和跨任务 prompt。
        batch_size: 批量推理大小，0 或负数表示逐条（兼容旧逻辑）。
        test_type: "original" 或 "mr"。
        mr_types: list[str] 与 samples 等长，每个样本的 MR 类型（仅 test_type="mr" 时）。
    """
    effective_task = model_task or dataset_task
    prompt_template = BINARY_PROMPT if effective_task == "binary" else CLASS3_PROMPT
    results = []
    correct = total = 0
    start_time = time.time()

    n = len(samples)
    if max_samples:
        n = min(n, max_samples)
    work = samples[:n]

    bs = batch_size if batch_size and batch_size > 0 else 1
    for start in range(0, n, bs):
        batch = work[start:start + bs]
        prompts = [prompt_template.format(premise=d["premise"], hypothesis=d["hypothesis"])
                   for d in batch]

        try:
            if bs == 1:
                preds_raw = [call_model(model, tokenizer, prompts[0])]
            else:
                preds_raw = call_model_batch(model, tokenizer, prompts)
        except Exception as e:
            print(f"  ⚠️  [{ds_name}] 批次起始 {start} 预测失败: {e}")
            preds_raw = ["ERROR"] * len(batch)

        for j, d in enumerate(batch):
            gold_raw = labels.get(d["label"], str(d["label"]))
            gold = collapse_label(gold_raw, dataset_task, model_task)
            pred = normalize_prediction(preds_raw[j], effective_task, labels)
            is_correct = (pred == gold.lower())

            # 构建带丰富元数据的结果
            if test_type == "merged":
                # Merged 模式：记录全部输入字段 + 预测字段
                result_entry = {
                    # 输入字段（完整保留）
                    "idx": d.get("idx", ""),
                    "premise": d.get("premise", ""),
                    "hypothesis": d.get("hypothesis", ""),
                    "mr_id": d.get("mr_id", ""),
                    "pair_id": d.get("pair_id", ""),
                    "label": d.get("label", -1),
                    "mr_type": d.get("mr_type", ""),
                    "is_source": d.get("is_source", False),
                    # 预测字段
                    "pred": pred,
                    "gold": gold,
                    "correct": is_correct,
                    # 元数据
                    "dataset": ds_name,
                    "test_type": "merged",
                }
            elif test_type == "mr":
                result_entry = {
                    "index": total + 1,
                    "dataset": ds_name,
                    "test_type": test_type,
                    "premise": d.get("premise", ""),
                    "hypothesis": d.get("hypothesis", ""),
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                }
                sample_idx = start + j
                # 优先使用数据中自带的 mr_type 字段（更干净），fallback 到文件名派生
                data_mr_type = d.get("mr_type", "")
                result_entry["mr_type"] = (
                    data_mr_type if data_mr_type
                    else (mr_types[sample_idx] if mr_types and sample_idx < len(mr_types) else "")
                )
                result_entry["pair_id"] = d.get("pair_id", "")
                result_entry["mr_category"] = d.get("mr_category", "")
            else:
                result_entry = {
                    "index": total + 1,
                    "dataset": ds_name,
                    "test_type": test_type,
                    "premise": d.get("premise", ""),
                    "hypothesis": d.get("hypothesis", ""),
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                }
                result_entry["pair_id"] = d.get("pair_id", "")
                result_entry["idx"] = d.get("idx", d.get("index", ""))

            results.append(result_entry)

            if is_correct:
                correct += 1
            total += 1

        if total % 500 < bs or total == n:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  [{ds_name}] {total}/{n}  acc={correct/total*100:.2f}%  ({rate:.1f} samples/s)")

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    acc = correct / total * 100 if total > 0 else 0
    return total, correct, acc, elapsed


def collect_mr_samples(mr_dir):
    """收集 MR 测试目录下所有 JSON 文件"""
    all_samples = []
    mr_types = []
    mr_dir_path = WORK_DIR / "data" / "nli" / mr_dir
    if not mr_dir_path.exists():
        print(f"  ⚠️  MR 目录不存在: {mr_dir_path}")
        return [], []

    for f in sorted(mr_dir_path.glob("*.json")):
        samples = load_jsonl_samples(f)
        # 提取 MR 类型
        mr_type = f.stem
        all_samples.extend(samples)
        mr_types.extend([mr_type] * len(samples))

    return all_samples, mr_types


def compute_msr(results):
    """计算 Metamorphic Satisfaction Rate (MSR)。

    基于 pair_id 分组 source 和 follow-up 预测，逐对检查 MR 输出关系是否满足。

    Args:
        results: list[dict], test_dataset() 返回的逐条结果（含 pred/gold/mr_type/is_source/pair_id）。

    Returns:
        dict: {
            "overall": {"satisfied": int, "total": int, "rate": float},
            "per_type": {"inv": {...}, "flip": {...}, "neutral": {...}},
        }
    """
    # Step 1: Group by pair_id, separate source from follow-ups
    pairs = {}
    for row in results:
        pid = row.get("pair_id", "")
        if pid == "":
            continue
        pid = int(pid)
        if pid not in pairs:
            pairs[pid] = {"source": None, "followups": []}
        if row.get("is_source"):
            pairs[pid]["source"] = row
        else:
            pairs[pid]["followups"].append(row)

    # Step 2: Count satisfactions per mr_type
    counters = {
        "overall": {"satisfied": 0, "total": 0},
    }
    for mr_type in ("inv", "flip", "neutral"):
        counters[mr_type] = {"satisfied": 0, "total": 0}

    for pid, pair_data in pairs.items():
        src = pair_data["source"]
        if src is None:
            continue
        src_pred = src.get("pred", "")

        for fup in pair_data["followups"]:
            mr = fup.get("mr_type", "")
            fup_pred = fup.get("pred", "")
            satisfied = False

            if mr == "inv":
                # Invariant: output must be identical
                satisfied = (fup_pred == src_pred)

            elif mr == "flip":
                # entailment <-> contradiction; neutral stays neutral
                if src_pred == "entailment":
                    satisfied = (fup_pred == "contradiction")
                elif src_pred == "contradiction":
                    satisfied = (fup_pred == "entailment")
                elif src_pred == "neutral":
                    satisfied = (fup_pred == "neutral")

            elif mr == "neutral":
                # Follow-up must be neutral regardless of source
                satisfied = (fup_pred == "neutral")

            # Increment counters
            counters["overall"]["total"] += 1
            if satisfied:
                counters["overall"]["satisfied"] += 1

            if mr in counters:
                counters[mr]["total"] += 1
                if satisfied:
                    counters[mr]["satisfied"] += 1

    # Step 3: Compute rates
    for key in counters:
        t = counters[key]["total"]
        s = counters[key]["satisfied"]
        counters[key]["rate"] = (s / t * 100) if t > 0 else 0.0

    return counters


def main():
    configure_console_encoding()
    os.chdir(WORK_DIR)
    import argparse
    parser = argparse.ArgumentParser(description="测试微调实验模型")
    parser.add_argument("--experiment", required=True, help="实验名")
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--lora", default=None, help="LoRA 权重路径（覆盖 --experiment）")
    parser.add_argument("--no-lora", action="store_true",
                        help="仅使用 base model，不加载 LoRA（zero-shot 推理）")
    parser.add_argument("--max-samples", type=int, default=None, help="每个数据集最大测试数")
    parser.add_argument("--skip-original", action="store_true", help="跳过 Original 测试")
    parser.add_argument("--skip-mr", action="store_true", help="跳过 MR 测试")
    parser.add_argument("--merged", action="store_true",
                        help="使用 Merged 测试数据（单一 JSONL，source+MR 混合，is_source 字段区分）。"
                             "计算 source accuracy、MR accuracy 和 MSR。")
    parser.add_argument("--merged-path", default=None,
                        help="覆盖 Merged 数据目录路径 (default: data/nli/mr_test_data_merged)")
    parser.add_argument("--model-task", default=None, choices=["binary", "3class"],
                        help="模型训练时的任务类型。与测试集任务不同时，自动折叠标签并调整prompt")
    parser.add_argument("--datasets", default=None,
                        help="逗号分隔的数据集名，只测这些（如 mnlim,mnlimm,sick,snli）；默认全部")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批量推理大小（0=逐条）；默认 32，可显著缩短测试时间")
    parser.add_argument("--output-root", default="output/experiments",
                        help="实验输出根目录；相对路径按仓库根目录解析")
    args = parser.parse_args()

    # 按指定的数据集过滤（默认不过滤，保留全部）
    wanted = set(d.strip() for d in args.datasets.split(",")) if args.datasets else None
    orig_ds = {k: v for k, v in ORIGINAL_DATASETS.items() if not wanted or k in wanted}
    mr_ds = {k: v for k, v in MR_DATASETS.items() if not wanted or k in wanted}
    merged_ds = {k: v for k, v in MERGED_DATASETS.items() if not wanted or k in wanted}

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = WORK_DIR / output_root
    lora_path = None if args.no_lora else (args.lora or str(output_root / args.experiment / "model"))
    exp_base = output_root / args.experiment
    tests_orig_dir = exp_base / "tests" / "original"
    tests_mr_dir = exp_base / "tests" / "mr"
    tests_merged_dir = exp_base / "tests" / "merged"

    model, tokenizer = load_model(args.base_model, lora_path)

    model_task = args.model_task
    if model_task:
        print(f"  ℹ️  模型训练任务: {model_task}（测试集任务不同时会自动折叠标签）")

    all_results = {}

    # === Original 测试 ===
    if not args.skip_original:
        print(f"\n{'='*70}")
        print(f"  📊 Original 数据集测试")
        print(f"{'='*70}\n")

        for ds_name, ds_info in orig_ds.items():
            filepath = WORK_DIR / "data" / "nli" / ds_info["dir"] / ds_info["file"]
            if not filepath.exists():
                print(f"  ⚠️  文件不存在: {filepath}")
                continue

            print(f"--- {ds_name.upper()} ({filepath}) ---")
            samples = load_jsonl_samples(filepath)
            print(f"  加载 {len(samples)} 条")

            out_path = tests_orig_dir / f"{ds_name}.jsonl"
            total, correct, acc, elapsed = test_dataset(
                model, tokenizer, samples, ds_name,
                ds_info["task"], ds_info["labels"],
                out_path, args.max_samples, model_task, args.batch_size,
                test_type="original",
            )

            print(f"  ✅ {ds_name}: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)")
            all_results[f"original/{ds_name}"] = {"total": total, "correct": correct, "acc": acc}

    # === MR 测试 ===
    if not args.skip_mr:
        print(f"\n{'='*70}")
        print(f"  📊 MR 数据集测试")
        print(f"{'='*70}\n")

        for ds_name, ds_info in mr_ds.items():
            print(f"--- {ds_name.upper()} (task={ds_info['task']}) ---")
            samples, mr_types = collect_mr_samples(ds_info["dir"])
            if not samples:
                print(f"  ⚠️  没有找到 MR 测试数据")
                continue

            print(f"  加载 {len(samples)} 条")

            out_path = tests_mr_dir / f"{ds_name}.jsonl"
            total, correct, acc, elapsed = test_dataset(
                model, tokenizer, samples, ds_name,
                ds_info["task"], ds_info["labels"],
                out_path, args.max_samples, model_task, args.batch_size,
                test_type="mr", mr_types=mr_types,
            )

            print(f"  ✅ {ds_name}: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)")
            all_results[f"mr/{ds_name}"] = {"total": total, "correct": correct, "acc": acc}

    # === Merged 测试（单次推理，source + MR 混合） ===
    if args.merged:
        print(f"\n{'='*70}")
        print(f"  📊 Merged 数据集测试（source + MR 混合，单次推理）")
        print(f"{'='*70}\n")

        merged_base = Path(args.merged_path) if args.merged_path else (WORK_DIR / "data" / "nli")

        for ds_name in sorted(merged_ds.keys()):
            filepath = merged_base / merged_ds[ds_name]
            if not filepath.exists():
                print(f"  ⚠️  文件不存在: {filepath}")
                continue

            print(f"--- {ds_name.upper()} ({filepath}) ---")
            samples = load_jsonl_samples(filepath)
            print(f"  加载 {len(samples)} 条")

            # Merged 数据统一按三分类处理
            labels = {0: "entailment", 1: "neutral", 2: "contradiction"}

            out_path = tests_merged_dir / f"{ds_name}.jsonl"
            total, correct, acc, elapsed = test_dataset(
                model, tokenizer, samples, ds_name,
                "3class", labels,
                out_path, args.max_samples, model_task, args.batch_size,
                test_type="merged",
            )

            # 分 source / MR 计算准确率
            source_total = source_correct = 0
            mr_total = mr_correct = 0
            for r_path in [out_path]:
                if r_path.exists():
                    with open(r_path, encoding="utf-8") as f:
                        for line in f:
                            row = json.loads(line.strip())
                            if row.get("is_source"):
                                source_total += 1
                                if row.get("correct"):
                                    source_correct += 1
                            else:
                                mr_total += 1
                                if row.get("correct"):
                                    mr_correct += 1

            source_acc = source_correct / source_total * 100 if source_total > 0 else 0
            mr_acc = mr_correct / mr_total * 100 if mr_total > 0 else 0

            # 计算 MSR
            merged_results = load_jsonl_samples(out_path) if out_path.exists() else []
            msr = compute_msr(merged_results) if merged_results else {}

            print(f"  ✅ {ds_name}: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)")
            print(f"     Source  : {source_correct}/{source_total} = {source_acc:.2f}%")
            print(f"     MR      : {mr_correct}/{mr_total} = {mr_acc:.2f}%")
            if msr:
                overall = msr.get("overall", {})
                print(f"     MSR     : {overall.get('satisfied', 0)}/{overall.get('total', 0)} = {overall.get('rate', 0):.2f}%")
                for mr_type in ("inv", "flip", "neutral"):
                    m = msr.get(mr_type, {})
                    if m.get("total", 0) > 0:
                        print(f"       {mr_type:<8}: {m.get('satisfied', 0)}/{m.get('total', 0)} = {m.get('rate', 0):.2f}%")

            all_results[f"merged/{ds_name}"] = {
                "total": total, "correct": correct, "acc": acc,
                "source_total": source_total, "source_correct": source_correct, "source_acc": source_acc,
                "mr_total": mr_total, "mr_correct": mr_correct, "mr_acc": mr_acc,
                "msr": msr,
            }

    # === 结果汇总 ===
    print(f"\n{'='*70}")
    print(f"  📊 测试结果汇总 - {args.experiment}")
    print(f"{'='*70}")

    for category in ["original/", "mr/", "merged/"]:
        cat_results = {k: v for k, v in all_results.items() if k.startswith(category)}
        if cat_results:
            print(f"\n  {category.upper().rstrip('/')}:")
            if category == "merged/":
                # Merged 汇总：source acc, MR acc, MSR
                for ds_key in sorted(cat_results.keys()):
                    r = cat_results[ds_key]
                    ds_short = ds_key.split("/")[1]
                    print(f"    {ds_short:<10} overall {r['correct']:>4}/{r['total']:<5} ({r['acc']:.2f}%)")
                    print(f"    {'':<10} source  {r.get('source_correct', 0):>4}/{r.get('source_total', 0):<5} ({r.get('source_acc', 0):.2f}%)")
                    print(f"    {'':<10} MR      {r.get('mr_correct', 0):>4}/{r.get('mr_total', 0):<5} ({r.get('mr_acc', 0):.2f}%)")
                    msr = r.get("msr", {})
                    overall = msr.get("overall", {})
                    if overall:
                        print(f"    {'':<10} MSR     {overall.get('satisfied', 0):>4}/{overall.get('total', 0):<5} ({overall.get('rate', 0):.2f}%)")
            else:
                for ds_key in sorted(cat_results.keys()):
                    r = cat_results[ds_key]
                    ds_short = ds_key.split("/")[1]
                    print(f"    {ds_short:<10} {r['correct']:>4}/{r['total']:<5} ({r['acc']:.2f}%)")
                total_c = sum(r["correct"] for r in cat_results.values())
                total_t = sum(r["total"] for r in cat_results.values())
                print(f"    {'总计':<10} {total_c:>4}/{total_t:<5} ({total_c/total_t*100:.2f}%)")

    if lora_path:
        print(f"\n  LoRA: {lora_path}")
    else:
        print(f"\n  Mode: zero-shot (base model only, no LoRA)")
    print(f"  Original: {tests_orig_dir}")
    print(f"  MR:       {tests_mr_dir}")
    if args.merged:
        print(f"  Merged:   {tests_merged_dir}")
    print()


if __name__ == "__main__":
    main()
