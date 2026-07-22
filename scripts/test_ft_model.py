#!/usr/bin/env python3
"""
测试微调后的 LoRA 模型在原始数据上的表现
用法:
  python scripts/test_ft_model.py \
    --base-model google/gemma-3-4b-it \
    --lora output/ft_gemma_output \
    --data-dir data/nli
"""
__test__ = False
import json, os, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")
os.chdir(WORK_DIR)

DATASETS = {
    "rte":    {"file": "data/nli/rte/test.json",     "task": "binary",  "labels": {0:"entailment", 1:"not_entailment"}},
    "snli":   {"file": "data/nli/rte/snlitest.json", "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlim":  {"file": "data/nli/mnlim/test.json",   "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlimm": {"file": "data/nli/mnlimm/test.json",  "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "sick":   {"file": "data/nli/sick/test.json",    "task": "3class",  "labels": {0:"entailment", 1:"neutral", 2:"contradiction"}},
}

BINARY_PROMPT = "Determine whether the premise entails the hypothesis. Answer with exactly one label: entailment or not_entailment.\n\nPremise: {premise}\nHypothesis: {hypothesis}"
CLASS3_PROMPT = "Determine the natural language inference relation between the premise and hypothesis. Answer with exactly one label: entailment, neutral, or contradiction.\n\nPremise: {premise}\nHypothesis: {hypothesis}"


def collapse_label(gold_label, dataset_task, model_task):
    """将 gold label 折叠到模型训练任务的标签空间。"""
    if model_task == "binary" and dataset_task == "3class":
        gold_lower = gold_label.strip().lower()
        if gold_lower in ("neutral", "contradiction", "contradictory"):
            return "not_entailment"
    return gold_label


def normalize_prediction(pred, task):
    """归一化模型输出，返回标准标签名。"""
    pred = pred.strip().lower().rstrip(".!?,")
    if task == "binary":
        if pred in ("entailment", "not_entailment"):
            return pred
        if pred in ("entailed", "e", "0"):
            return "entailment"
        # 三分类模型输出 → 映射到二分类空间
        if any(word in pred for word in ("neutral", "contradict", "contradiction", "2")):
            return "not_entailment"
        if "entail" in pred and "not" not in pred:
            return "entailment"
        if any(word in pred for word in ("not_entail", "not entail", "1")):
            return "not_entailment"
        return pred  # fallback
    else:
        # 3-class
        if pred in ("entailment", "neutral", "contradiction"):
            return pred
        mapping = {
            "entailed": "entailment", "e": "entailment", "0": "entailment",
            "n": "neutral", "1": "neutral",
            "contradictory": "contradiction", "contradict": "contradiction",
            "c": "contradiction", "2": "contradiction",
        }
        return mapping.get(pred, pred)

def load_model(base_model_name, lora_path):
    print(f"🔄 加载基础模型: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
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
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip().lower().rstrip(".!?,")
    return response


def test_dataset(model, tokenizer, ds_name, ds_info, max_samples=500, model_task=None):
    data_file = Path(ds_info["file"])
    if not data_file.exists():
        return 0, 0, 0
    dataset_task = ds_info["task"]  # "binary" 或 "3class"
    effective_task = model_task or dataset_task
    prompt_template = BINARY_PROMPT if effective_task == "binary" else CLASS3_PROMPT
    label_map = ds_info["labels"]
    correct = total = 0
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            d = json.loads(line)
            gold_raw = label_map.get(d["label"], "")
            gold = collapse_label(gold_raw, dataset_task, model_task)
            prompt = prompt_template.format(premise=d["premise"], hypothesis=d["hypothesis"])
            pred_raw = call_model(model, tokenizer, prompt)
            pred = normalize_prediction(pred_raw, effective_task)
            total += 1
            if pred == gold.lower():
                correct += 1
            elif total <= 5 and correct < total:
                print(f"  [{ds_name}] #{i+1} gold={gold} pred_raw='{pred_raw}' norm='{pred}'")
            if total % 100 == 0:
                print(f"  [{ds_name}] {total}/{max_samples} 正确:{correct}/{total} ({correct/total*100:.1f}%)")
    return total, correct, correct/total*100 if total else 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--experiment", default=None,
                        help="实验名 (例如 mettrain_gemma3_4b)，会自动拼接路径")
    parser.add_argument("--lora", default=None,
                        help="LoRA 权重路径（优先级高于 --experiment）")
    parser.add_argument("--samples", type=int, default=200, help="每个数据集测试样本数")
    parser.add_argument("--model-task", default=None, choices=["binary", "3class"],
                        help="模型训练时的任务类型。与测试集任务不同时自动折叠标签")
    args = parser.parse_args()

    # 解析 LoRA 路径：优先 --lora，其次 --experiment
    lora_path = args.lora
    if not lora_path and args.experiment:
        lora_path = f"output/experiments/{args.experiment}/model"
    if not lora_path:
        lora_path = "output/ft_gemma_output"

    model, tokenizer = load_model(args.base_model, lora_path)

    print(f"\n{'='*70}")
    print(f"  🔬 微调模型测试")
    print(f"  Base: {args.base_model}")
    print(f"  LoRA: {lora_path}")
    print(f"  每数据集: {args.samples} 条")
    print(f"{'='*70}\n")

    model_task = args.model_task
    if model_task:
        print(f"  ℹ️  模型训练任务: {model_task}（测试集任务不同时会自动折叠标签）")

    results = {}
    for ds_name, ds_info in DATASETS.items():
        print(f"\n--- {ds_name.upper()} (dataset_task={ds_info['task']}, model_task={model_task or '同测试集'}) ---")
        total, correct, acc = test_dataset(model, tokenizer, ds_name, ds_info, args.samples, model_task)
        results[ds_name] = {"total": total, "correct": correct, "acc": acc}
        print(f"  ✅ {ds_name}: {correct}/{total} = {acc:.1f}%")

    print(f"\n{'='*70}")
    print(f"  📊 测试结果汇总")
    print(f"{'='*70}")
    total_c = sum(r["correct"] for r in results.values())
    total_t = sum(r["total"] for r in results.values())
    for ds, r in results.items():
        print(f"  {ds:<10} {r['correct']:>4}/{r['total']:<5} ({r['acc']:.1f}%)")
    print(f"  {'总计':<10} {total_c:>4}/{total_t:<5} ({total_c/total_t*100:.1f}%)")
    print()

if __name__ == "__main__":
    main()
