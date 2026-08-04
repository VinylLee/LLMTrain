#!/usr/bin/env python3
"""Generate comprehensive accuracy summary for all experiments across phases."""
import json, os, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

def compute_accuracy(jsonl_path):
    if not jsonl_path.exists():
        return None, 0
    correct = total = 0
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("correct"):
                correct += 1
            total += 1
    return correct, total

def print_table(title, data, col_keys, row_label="Mode_Seed"):
    if not data:
        return
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    header = f"  {row_label:<35}"
    for ck in col_keys:
        header += f" {ck:>15}"
    print(header)
    print(f"  {'-'*35}{' '}{'-'*15 * len(col_keys)}")
    for row_name in sorted(data.keys()):
        row_data = data[row_name]
        line = f"  {row_name:<35}"
        for ck in col_keys:
            val = row_data.get(ck)
            if val is not None and val[1] > 0:
                acc = val[0] / val[1] * 100
                line += f" {acc:>14.2f}%"
            else:
                line += f" {'N/A':>15}"
        print(line)

def parse_name(exp_name):
    """Parse readable label from experiment name."""
    parts = exp_name.split("_")
    seed = None
    for p in parts:
        if p.startswith("seed") and p[4:].isdigit():
            seed = p
    # MR-as-Instruction modes
    mode_map = {
        "none": "none", "pairop": "pair_operation", "pair_operation": "pair_operation",
        "shuffled": "shuffled_operation", "shuffled_operation": "shuffled_operation",
        "mrop": "operation_only", "operation_only": "operation_only",
        "paironly": "pair_only", "pair_only": "pair_only",
        "fulloracle": "full_oracle", "full_oracle": "full_oracle",
    }
    for p in parts:
        if p in mode_map:
            return mode_map[p], seed
    # Phase B: extract train source (mettrain/original) + dataset + target
    # e.g. mettrain_mnlim_4413_gemma3_4b -> mettrain_mnlim4413
    # e.g. original_snli_5340_gemma3_4b -> original_snli5340
    train_type = "met" if "mettrain" in exp_name else "orig"
    ds_map = {"mnlim": "MNLI", "mnlimm": "MNLIMM", "sick": "SICK", "snli": "SNLI"}
    dataset = None
    target = None
    for i, p in enumerate(parts):
        if p in ds_map:
            dataset = ds_map[p]
            # target is usually next part
            if i+1 < len(parts) and parts[i+1].isdigit():
                target = parts[i+1]
    if dataset and target:
        return f"{train_type}_{dataset}{target}", seed
    if dataset:
        return f"{train_type}_{dataset}", seed
    return exp_name.rsplit("_seed", 1)[0] if seed else exp_name, seed

def process_experiments(exp_root):
    """Process all experiment dirs under a root."""
    orig_data = defaultdict(dict)
    mr_data = defaultdict(dict)
    progress_file = exp_root / "_progress.json"
    if not progress_file.exists():
        return orig_data, mr_data

    progress = json.loads(progress_file.read_text(encoding="utf-8"))
    for exp_name in sorted(progress):
        exp_dir = exp_root / exp_name
        if not exp_dir.is_dir():
            continue
        mode, seed = parse_name(exp_name)
        if not seed:
            continue
        label = f"{mode}_{seed}" if mode else exp_name
        # Original tests
        orig_dir = exp_dir / "tests" / "original"
        if orig_dir.is_dir():
            for f in orig_dir.glob("*.jsonl"):
                ds = f.stem
                correct, total = compute_accuracy(f)
                if total > 0:
                    orig_data[label][ds] = (correct, total)
        # MR tests
        mr_dir = exp_dir / "tests" / "mr"
        if mr_dir.is_dir():
            for f in mr_dir.glob("*.jsonl"):
                ds = f.stem
                correct, total = compute_accuracy(f)
                if total > 0:
                    mr_data[label][ds] = (correct, total)
    return orig_data, mr_data

# ── Phase A: MR-as-Instruction ──
phase_a_roots = [
    ROOT / "output" / "experiments" / "gemma3_4b_mrinstr_full_multiseed_v1",
    ROOT / "output" / "experiments" / "gemma3_4b_mrinstr_exploratory_full_seed42_v1",
    ROOT / "output" / "experiments" / "gemma3_4b_mrinstr_snli_full_multiseed_v1",
    ROOT / "output" / "experiments" / "gemma3_4b_mrinstr_sick_small_multiseed_v1",
    ROOT / "output" / "experiments" / "gemma3_4b_mrinstr_sick_full_multiseed_v1",
]
for exp_root in phase_a_roots:
    orig_data, mr_data = process_experiments(exp_root)
    if not orig_data and not mr_data:
        continue
    print(f"\n{'#'*100}")
    print(f"# Phase A (MR-as-Instruction): {exp_root.name}")
    print(f"{'#'*100}")
    if orig_data:
        datasets = sorted(set(d for v in orig_data.values() for d in v))
        print_table("Original 数据集准确率", orig_data, datasets)
    if mr_data:
        datasets = sorted(set(d for v in mr_data.values() for d in v))
        print_table("MR 变体数据集准确率", mr_data, datasets)

# ── Phase B: MetTrain main pipeline ──
exp_root = ROOT / "output" / "experiments"
orig_data, mr_data = process_experiments(exp_root)
# Filter out MR-as-Instruction experiments that live in sub-dirs
if orig_data or mr_data:
    print(f"\n{'#'*100}")
    print(f"# Phase B (MetTrain 主流水线): {exp_root}")
    print(f"{'#'*100}")
    if orig_data:
        datasets = sorted(set(d for v in orig_data.values() for d in v))
        print_table("Original 数据集准确率", orig_data, datasets)
    if mr_data:
        datasets = sorted(set(d for v in mr_data.values() for d in v))
        print_table("MR 变体数据集准确率", mr_data, datasets)

print(f"\n{'='*100}")
print("  全量汇总完成")
print(f"{'='*100}")
