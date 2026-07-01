#!/usr/bin/env python3
"""
全模型原始数据集测试脚本
逐个加载 LM Studio 中的模型，在5个数据集的原始测试集上跑推理
完成后自动切换下一个模型
"""
import json, os, sys, time, requests, subprocess
from datetime import datetime
from pathlib import Path

WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")
os.chdir(WORK_DIR)

API_URL = "http://localhost:1234/v1/chat/completions"
LMS = "/home/ubuntu/.lmstudio/bin/lms"
DELAY = 0.5
MAX_RETRIES = 3

# 所有 LLM 模型（按顺序执行）
ALL_MODELS = [
    "llama-3.3-70b-instruct",
    "gemma-3-4b-it-qat",
]

# 5个数据集的原始测试文件
DATASETS = {
    "rte":    {"file": "data/nli/rte/test.json",     "binary": True,  "label_map": {0:"entailment", 1:"not_entailment"}},
    "snli":   {"file": "data/nli/rte/snlitest.json", "binary": False, "label_map": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlim":  {"file": "data/nli/mnlim/test.json",   "binary": False, "label_map": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "mnlimm": {"file": "data/nli/mnlimm/test.json",  "binary": False, "label_map": {0:"entailment", 1:"neutral", 2:"contradiction"}},
    "sick":   {"file": "data/nli/sick/test.json",    "binary": False, "label_map": {0:"entailment", 1:"neutral", 2:"contradiction"}},
}

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def switch_model(model_id):
    """卸载当前模型，异步加载新模型，轮询等待就绪"""
    log(f"切换模型至: {model_id}")
    # 用 API 获取当前模型列表，逐个卸载
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        if r.status_code == 200:
            for m in r.json()["data"]:
                mid = m["id"]
                if mid != model_id and "embed" not in mid:
                    subprocess.run([LMS, "unload", mid], capture_output=True, timeout=30)
                    log(f"  卸载旧模型: {mid}")
    except:
        subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=30)
    time.sleep(3)

    # 异步加载新模型（不阻塞脚本）
    subprocess.Popen([LMS, "load", model_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"  正在后台加载 {model_id}，轮询等待就绪...")

    # 轮询最多10分钟等待模型就绪
    for i in range(40):
        time.sleep(15)
        try:
            r = requests.get("http://localhost:1234/v1/models", timeout=5)
            if r.status_code == 200:
                loaded = [m["id"] for m in r.json()["data"]]
                if model_id in loaded:
                    log(f"  ✅ 模型 {model_id} 已就绪 (等待{ (i+1)*15 }秒)")
                    return True
        except:
            pass
        if i % 4 == 0:
            log(f"  等待模型加载中... ({ (i+1)*15 }秒)")

    log(f"  ⚠️ 模型 {model_id} 加载超时(>10min)")
    return False

def ensure_server():
    """确保 LM Studio 服务器在运行"""
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        if r.status_code == 200:
            models = [m["id"] for m in r.json()["data"]]
            log(f"  服务器正常，可用模型: {models}")
            return models
    except:
        log("  服务器未运行，尝试启动...")
        subprocess.run([LMS, "server", "start"], capture_output=True, timeout=30)
        time.sleep(5)
        try:
            r = requests.get("http://localhost:1234/v1/models", timeout=10)
            if r.status_code == 200:
                return [m["id"] for m in r.json()["data"]]
        except:
            log("  ❌ 服务器启动失败")
    return []

def call_llm(prompt, model, retry=0):
    try:
        r = requests.post(API_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.0,
        }, timeout=120)
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"].strip()
            return 200, content
        # 模型未加载错误 - 等30秒重试
        if r.status_code == 404 and retry < MAX_RETRIES:
            log(f"  模型未就绪，等待30秒重试...")
            time.sleep(30)
            return call_llm(prompt, model, retry+1)
        return r.status_code, None
    except requests.Timeout:
        if retry < MAX_RETRIES:
            log(f"  超时，重试 {retry+1}/{MAX_RETRIES}...")
            time.sleep(10)
            return call_llm(prompt, model, retry+1)
        return 0, "timeout"
    except Exception as e:
        return 0, str(e)

def build_prompt(premise, hypothesis, binary=False):
    if binary:
        return f"Determine whether the premise entails the hypothesis.\nAnswer with one of: entailment, not_entailment.\n\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
    else:
        return f"Determine the relationship between the premise and hypothesis.\nAnswer with one of: entailment, neutral, contradiction.\n\nPremise: {premise}\nHypothesis: {hypothesis}\nAnswer:"

def process_dataset(ds_name, ds_info, model):
    data_file = Path(ds_info["file"])
    if not data_file.exists():
        log(f"  [跳过] {data_file} 不存在")
        return 0, 0, 0

    safe_model = model.replace(" ", "_").replace("/", "_")
    out_dir = Path(f"output/original_data/{safe_model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ds_name}.jsonl"
    binary = ds_info["binary"]

    # 断点续跑：跳过已存在的 index
    existing = set()
    if out_file.exists():
        for line in open(out_file):
            try:
                existing.add(json.loads(line)["index"])
            except:
                pass

    total = 0
    processed = 0
    errors = 0

    with open(out_file, "a", encoding="utf-8") as f_out:
        with open(data_file) as f_in:
            for i, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                total += 1
                if i in existing:
                    continue

                premise = ex.get("premise", "")
                hypothesis = ex.get("hypothesis", "")
                label = ex.get("label")
                gold = ds_info["label_map"].get(label, str(label))
                prompt = build_prompt(premise, hypothesis, binary)
                ts = datetime.utcnow().isoformat() + "Z"

                status, pred = call_llm(prompt, model)
                record = {
                    "index": i, "timestamp": ts, "status": status,
                    "premise": premise, "hypothesis": hypothesis,
                    "gold_label": gold, "prediction": pred,
                    "label": label, "dataset": ds_name, "binary": binary,
                }
                if status == 200:
                    processed += 1
                else:
                    record["error"] = str(pred)
                    errors += 1

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                if i % 100 == 0:
                    log(f"  [{ds_name}] {i}/{total}条 成功:{processed} 失败:{errors}")

                time.sleep(DELAY)

    log(f"  [{ds_name}] 完成! {total}条 成功:{processed} 失败:{errors}")
    return total, processed, errors

def run_all():
    log("=" * 60)
    log("🏁 全模型 × 5数据集 原始测试启动")
    log(f"模型列表: {ALL_MODELS}")
    log("=" * 60)

    # 进度文件，用于断点续传
    progress_file = Path("output/original_data/_progress.json")
    progress = {"current_model_idx": 0, "current_ds_idx": 0}
    if progress_file.exists():
        try:
            progress = json.load(open(progress_file))
            log(f"检测到上次进度: 模型#{progress['current_model_idx']}, 数据集#{progress['current_ds_idx']}")
        except:
            pass

    start_model = progress["current_model_idx"]
    ds_keys = list(DATASETS.keys())

    for mi in range(start_model, len(ALL_MODELS)):
        model = ALL_MODELS[mi]
        log(f"\n{'='*60}")
        log(f"📦 模型 [{mi+1}/{len(ALL_MODELS)}]: {model}")
        log(f"{'='*60}")

        # 切换模型
        if not switch_model(model):
            log(f"❌ {model} 加载失败，跳过")
            continue

        # 确保服务器正常
        available = ensure_server()
        if model not in available:
            log(f"⚠️ {model} 不在已加载列表中，等待60秒...")
            time.sleep(60)
            available = ensure_server()

        start_ds = progress["current_ds_idx"] if mi == start_model else 0
        model_totals = {"total": 0, "ok": 0, "err": 0}

        for di in range(start_ds, len(ds_keys)):
            ds_name = ds_keys[di]
            log(f"\n  --- {ds_name.upper()} ---")
            t, ok, err = process_dataset(ds_name, DATASETS[ds_name], model)
            model_totals["total"] += t
            model_totals["ok"] += ok
            model_totals["err"] += err

            # 保存进度
            progress["current_model_idx"] = mi
            progress["current_ds_idx"] = di + 1
            json.dump(progress, open(progress_file, "w"))

        log(f"\n  ✅ {model} 全部完成: {model_totals['ok']}/{model_totals['total']} 条成功")

        # 重置数据集索引（下一个模型从0开始）
        progress["current_ds_idx"] = 0

    log(f"\n{'='*60}")
    log(f"🏆 全部完成！")
    log(f"{'='*60}")

if __name__ == "__main__":
    run_all()
