#!/usr/bin/env python3
"""
Zero-shot merged test via OpenAI-compatible API (e.g. LM Studio).

Loads merged JSONL data (source + MR), calls the API for each example,
and saves results in the exact same JSONL format as test_mettrain_experiment.py --merged.
The output can be directly consumed by read_test_results() / compute_msr() in run_rq1_nli.py.

Usage:
    # LM Studio (default)
    python scripts/test_merged_via_api.py \
        --experiment rq1_zeroshot \
        --api-model llama-3.3-70b-instruct \
        --output-root RQ1/output/llama-3.3-70b-instruct

    # Custom API endpoint
    python scripts/test_merged_via_api.py \
        --experiment rq1_zeroshot \
        --api-url http://localhost:1234/v1/chat/completions \
        --api-key "" \
        --api-model llama-3.3-70b-instruct \
        --output-root RQ1/output/llama-3.3-70b-instruct \
        --datasets snli,mnlim \
        --max-samples 100
"""
__test__ = False

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# ── 项目路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from project_runtime import PROJECT_ROOT as PR

WORK_DIR = PR

# ── .env：让 DEEPSEEK_* / LMSTUDIO_* 等凭据在未 export 的 shell 下也可用 ──
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

# ── Merged 数据集配置（与 test_mettrain_experiment.py 保持一致） ──────────
MERGED_DATASETS = {
    "snli":   "mr_test_data_merged/snli.jsonl",
    "mnlim":  "mr_test_data_merged/mnlim.jsonl",
    "mnlimm": "mr_test_data_merged/mnlimm.jsonl",
    "sick":   "mr_test_data_merged/sick.jsonl",
}

CLASS3_PROMPT = (
    "Determine the natural language inference relation between the premise and hypothesis. "
    "Answer with exactly one label: entailment, neutral, or contradiction.\n\n"
    "Premise: {premise}\nHypothesis: {hypothesis}"
)


# ══════════════════════════════════════════════════════════════════════════════
# Prediction normalization (same logic as test_mettrain_experiment.py)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_prediction(pred, labels=None):
    """Normalize 3-class NLI prediction."""
    if labels is None:
        labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    pred = pred.strip().lower().rstrip(".!?,")

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


# ══════════════════════════════════════════════════════════════════════════════
# API call
# ══════════════════════════════════════════════════════════════════════════════

def call_api(api_url, api_key, model, prompt, max_retries=3, timeout=120, reasoning_effort=None):
    """Call OpenAI-compatible chat completions API. Returns (status_code, response_data)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    # reasoning 模型（如 deepseek-v4-flash）：reasoning_effort="none" 关闭思维链（reason off）
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            try:
                data = resp.json()
            except Exception:
                data = resp.text
            if resp.status_code == 200:
                return resp.status_code, data
            last_error = f"HTTP {resp.status_code}: {str(data)[:200]}"
        except requests.exceptions.ConnectionError:
            last_error = "ConnectionError"
        except requests.exceptions.Timeout:
            last_error = "Timeout"
        except requests.exceptions.RequestException as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            wait = 2 ** attempt
            time.sleep(wait)

    return 0, last_error


def extract_pred_text(resp):
    """Extract raw model output text from an OpenAI chat response."""
    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if choices:
            return (choices[0].get("message") or {}).get("content", "")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Main test logic
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(filepath):
    """Load a JSONL file, returning a list of dicts."""
    samples = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def test_merged_dataset(api_url, api_key, api_model, samples, ds_name,
                         output_path, max_samples=None, delay=0.0,
                         reasoning_effort=None):
    """Test a merged dataset via API, saving results in merged JSONL format.

    Returns (total, correct, acc, elapsed).
    """
    n = len(samples)
    if max_samples:
        n = min(n, max_samples)
    work = samples[:n]

    labels = {0: "entailment", 1: "neutral", 2: "contradiction"}
    results = []
    correct = total = 0
    start_time = time.time()

    for i, d in enumerate(work):
        prompt = CLASS3_PROMPT.format(premise=d["premise"], hypothesis=d["hypothesis"])

        status, resp = call_api(api_url, api_key, api_model, prompt,
                                reasoning_effort=reasoning_effort)

        if status == 200:
            raw_pred = extract_pred_text(resp)
            if raw_pred is None:
                raw_pred = ""
        else:
            raw_pred = "ERROR"
            if total < 3:  # Only print first few errors
                print(f"  ⚠️  [{ds_name}] API 调用失败 (idx={i}): {resp}")

        gold_raw = labels.get(d.get("label", -1), str(d.get("label", -1)))
        gold = gold_raw
        pred = normalize_prediction(raw_pred, labels)
        is_correct = (pred == gold.lower())

        result_entry = {
            # 输入字段（完整保留，与 test_mettrain_experiment.py --merged 格式一致）
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
        results.append(result_entry)

        if is_correct:
            correct += 1
        total += 1

        # Progress
        if total % 100 == 0 or total == n:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  [{ds_name}] {total}/{n}  acc={correct/total*100:.2f}%  ({rate:.1f} samples/s)", flush=True)

        if delay > 0:
            time.sleep(delay)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    acc = correct / total * 100 if total > 0 else 0
    return total, correct, acc, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def resolve_api_url(url_str):
    """Auto-append /v1/chat/completions if the URL looks like a base URL."""
    url_str = url_str.rstrip("/")
    if url_str.endswith(("chat/completions", "/generate", "/v1/completions")):
        return url_str
    if url_str.endswith("/v1"):
        return url_str + "/chat/completions"
    return url_str + "/v1/chat/completions"


def main():
    # 强制行缓冲，确保重定向到文件时进度实时可见
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Zero-shot merged test via OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--experiment", required=True, help="实验名（如 rq1_zeroshot）")
    parser.add_argument("--api-url", default=None,
                        help="API 完整 URL（默认从环境变量 LMSTUDIO_BASE_URL 读取）")
    parser.add_argument("--api-key", default=None,
                        help="API key（默认从环境变量 LMSTUDIO_KEY 读取；LM Studio 通常为空）")
    parser.add_argument("--api-model", required=True,
                        help="API 请求中的 model id（如 llama-3.3-70b-instruct）")
    parser.add_argument("--datasets", default=None,
                        help="逗号分隔的数据集名（默认全部: snli,mnlim,mnlimm,sick）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="每个数据集最大测试数（冒烟测试用）")
    parser.add_argument("--output-root", default="RQ1/output",
                        help="输出根目录（默认: RQ1/output）")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="每条请求间延迟（秒），用于速率限制（默认: 0）")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["none", "low", "medium", "high"],
                        help="reasoning_effort 参数发送给 API；none 关闭思维链（如 deepseek-v4-flash reason off）")
    parser.add_argument("--merged-data-dir", default=None,
                        help="覆盖 Merged 数据目录（默认: data/nli）")
    args = parser.parse_args()

    # Resolve API URL and key from environment (.env 已在模块顶部加载)
    # 优先 DeepSeek，其次 LM Studio（本地默认）
    api_key = args.api_key
    if api_key is None:
        api_key = (os.getenv("DEEPSEEK_API_KEY")
                   or os.getenv("DEEPSEEK_KEY")
                   or os.getenv("LMSTUDIO_KEY", ""))

    api_url = args.api_url
    if api_url is None:
        api_url = (os.getenv("DEEPSEEK_OPENAI_BASE_URL")
                   or os.getenv("DEEPSEEK_API_URL")
                   or os.getenv("DEEPSEEK_URL")
                   or os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234"))
    api_url = resolve_api_url(api_url)

    print(f"🌐 API endpoint: {api_url}")
    print(f"🤖 API model:    {args.api_model}")
    if args.reasoning_effort:
        print(f"🧠 Reasoning:    {args.reasoning_effort}")
    print(f"📊 Experiment:   {args.experiment}")

    # Dataset filter
    wanted = set(d.strip() for d in args.datasets.split(",")) if args.datasets else None
    merged_ds = {k: v for k, v in MERGED_DATASETS.items() if not wanted or k in wanted}

    # Output root
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = WORK_DIR / output_root

    # Merged data base dir
    merged_base = Path(args.merged_data_dir) if args.merged_data_dir else (WORK_DIR / "data" / "nli")

    tests_merged_dir = output_root / args.experiment / "tests" / "merged"

    all_results = {}

    for ds_name in sorted(merged_ds.keys()):
        filepath = merged_base / merged_ds[ds_name]
        if not filepath.exists():
            print(f"  ⚠️  文件不存在: {filepath}")
            continue

        print(f"\n--- {ds_name.upper()} ({filepath}) ---")
        samples = load_jsonl(filepath)
        print(f"  加载 {len(samples)} 条")

        out_path = tests_merged_dir / f"{ds_name}.jsonl"
        total, correct, acc, elapsed = test_merged_dataset(
            api_url, api_key, args.api_model, samples, ds_name,
            out_path, args.max_samples, args.delay,
            args.reasoning_effort,
        )

        # Split by is_source for summary
        source_total = source_correct = 0
        mr_total = mr_correct = 0
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
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

        print(f"  ✅ {ds_name}: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s)")
        print(f"     Source  : {source_correct}/{source_total} = {source_acc:.2f}%")
        print(f"     MR      : {mr_correct}/{mr_total} = {mr_acc:.2f}%")

        all_results[f"merged/{ds_name}"] = {
            "total": total, "correct": correct, "acc": acc,
            "source_total": source_total, "source_correct": source_correct, "source_acc": source_acc,
            "mr_total": mr_total, "mr_correct": mr_correct, "mr_acc": mr_acc,
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  📊 Merged Test Results — {args.experiment}")
    print(f"{'='*70}")
    for ds_key in sorted(all_results.keys()):
        r = all_results[ds_key]
        ds_short = ds_key.split("/")[1]
        print(f"  {ds_short:<10} overall {r['correct']:>5}/{r['total']:<5} ({r['acc']:.2f}%)")
        print(f"  {'':<10} source  {r['source_correct']:>5}/{r['source_total']:<5} ({r['source_acc']:.2f}%)")
        print(f"  {'':<10} MR      {r['mr_correct']:>5}/{r['mr_total']:<5} ({r['mr_acc']:.2f}%)")

    print(f"\n  Results saved to: {tests_merged_dir}")
    print()


if __name__ == "__main__":
    main()
