#!/usr/bin/env python3
"""
从 augmented_data.json 中按 pair_id 为单位随机抽取数据，
直到总条数达到 target 为止。

用法:
  python scripts/sample_mettrain_pairid.py \
    --input data/nli/mettrain/rte_lr0.8_gemma-3-4b-it-qat/augmented_data.json \
    --target 2490 \
    --seed 123 \
    --output data/nli/mettrain/rte_lr0.8_gemma-3-4b-it-qat/augmented_data_sampled_seed123.json
"""
import json
import random
import argparse
import hashlib
from pathlib import Path
from collections import Counter


SIGNATURE_FIELDS = (
    "pair_id", "mr_id", "premise", "hypothesis", "label", "idx", "id", "_source",
)


def canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_data_signature(samples):
    canonical = [
        {key: sample.get(key) for key in SIGNATURE_FIELDS if key in sample}
        for sample in samples
    ]
    canonical.sort(key=lambda row: json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    return canonical_sha256(canonical)


def build_sampling_report(
    source_path,
    sampled_path,
    source_samples,
    selected_samples,
    selected_pair_ids,
    seed,
    target,
    stratify,
):
    source_groups = group_by_pairid(source_samples)
    sampled_groups = group_by_pairid(selected_samples)
    return {
        "schema_version": 1,
        "seed": seed,
        "target_samples": target,
        "stratify": bool(stratify),
        "source_file": str(source_path),
        "source_file_sha256": file_sha256(source_path),
        "source_data_signature": compute_data_signature(source_samples),
        "source_sample_count": len(source_samples),
        "source_group_count": len(source_groups),
        "sampled_file": str(sampled_path),
        "sampled_file_sha256": file_sha256(sampled_path),
        "sample_data_signature": compute_data_signature(selected_samples),
        "selected_sample_count": len(selected_samples),
        "selected_group_count": len(sampled_groups),
        "selected_pair_ids_sha256": canonical_sha256(selected_pair_ids),
        "label_distribution": {
            str(label): count
            for label, count in sorted(Counter(
                sample.get("label") for sample in selected_samples
            ).items(), key=lambda item: str(item[0]))
        },
        "sampled_group_size_distribution": {
            str(size): count
            for size, count in sorted(Counter(
                len(group) for group in sampled_groups.values()
            ).items())
        },
    }


def load_jsonl(filepath):
    """加载 JSONL 文件，返回列表"""
    samples = []
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                samples.append(d)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  第 {i} 行 JSON 解析失败: {e}")
    return samples


def group_by_pairid(samples):
    """按 pair_id 分组"""
    groups = {}
    for s in samples:
        pid = s["pair_id"]
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(s)
    return groups


def get_group_primary_label(group):
    """通过多数投票确定组的首要标签（用于分层）。平局时取 label 值最小的。"""
    labels = [s["label"] if isinstance(s["label"], int) else 0 for s in group]
    counter = Counter(labels)
    max_count = max(counter.values())
    candidates = [l for l, c in counter.items() if c == max_count]
    return min(candidates)


def main():
    parser = argparse.ArgumentParser(description="从 JSONL 文件中随机采样数据")
    parser.add_argument("--input", "-i", required=True, help="输入 JSONL 文件")
    parser.add_argument("--target", "-t", type=int, required=True, help="目标条数")
    parser.add_argument("--seed", "-s", type=int, default=123, help="随机种子")
    parser.add_argument("--output", "-o", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--stratify", action="store_true",
                        help="分层采样：按标签类别均衡采样（pair_id 分组时按组内多数标签分层）")
    parser.add_argument(
        "--report-output",
        default=None,
        help="sampling report 路径（默认与 sampled file 同目录的 sampling_report.json）",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  📊 随机采样")
    print(f"  输入: {args.input}")
    print(f"  目标: {args.target} 条")
    print(f"  种子: {args.seed}")
    print(f"  模式: {'分层采样' if args.stratify else '简单随机'}")
    print(f"{'='*60}\n")

    # 1. 加载数据
    print("📥 加载数据...")
    samples = load_jsonl(args.input)
    print(f"  共加载 {len(samples)} 条")
    if not samples:
        print("  ❌ 没有有效样本，退出")
        return

    # 2. 检查是否有 pair_id 字段，没有则自动分配
    has_pair_id = any("pair_id" in s for s in samples)
    auto_id_counter = [0]  # mutable for closure

    def ensure_pair_id(s):
        if "pair_id" not in s:
            auto_id_counter[0] += 1
            s["pair_id"] = f"auto_{auto_id_counter[0]}"

    if not has_pair_id:
        print(f"  ℹ️  数据无 pair_id 字段，为每行分配唯一ID（逐行独立采样）")
        for s in samples:
            ensure_pair_id(s)
    else:
        missing = sum(1 for s in samples if "pair_id" not in s)
        if missing > 0:
            print(f"  ℹ️  部分行缺少 pair_id ({missing}条)，为其分配唯一ID")
            for s in samples:
                ensure_pair_id(s)

    # 3. 按 pair_id 分组
    print("\n🔗 按 pair_id 分组...")
    groups = group_by_pairid(samples)
    pair_ids = list(groups.keys())
    print(f"  共 {len(pair_ids)} 个唯一 pair_id")

    # 统计分布
    dist = Counter(len(groups[pid]) for pid in pair_ids)
    print(f"  分布: {dict(sorted(dist.items()))}")

    # 4. 采样
    if args.stratify:
        # 分层采样：按组的主要标签分层
        print(f"\n🔀 分层采样 (按组内多数标签分层)...")
        strata = {}  # label -> [pair_ids]
        for pid in pair_ids:
            pl = get_group_primary_label(groups[pid])
            if pl not in strata:
                strata[pl] = []
            strata[pl].append(pid)

        num_strata = len(strata)
        print(f"  共 {num_strata} 个分层: {sorted(strata.keys())}")
        for s in sorted(strata.keys()):
            print(f"    标签 {s}: {len(strata[s])} 组 (总群组 {sum(len(groups[pid]) for pid in strata[s])} 条)")

        # 每层独立采样，目标平均分配
        base_target = args.target // num_strata
        remainder = args.target % num_strata

        selected = []
        selected_pair_ids = []
        stratum_labels = sorted(strata.keys())
        for si, s in enumerate(stratum_labels):
            target_s = base_target + (1 if si < remainder else 0)
            stratum_pids = strata[s][:]
            random.shuffle(stratum_pids)
            print(f"\n  🎲 标签 {s}: 目标 {target_s} 条")
            cumulative = 0
            for pid in stratum_pids:
                entries = groups[pid]
                selected.extend(entries)
                cumulative += len(entries)
                selected_pair_ids.append(pid)
                if cumulative >= target_s:
                    break
            print(f"     实际采样 {cumulative} 条 ({len([p for p in stratum_pids if p in selected_pair_ids and p in strata[s]])} 组)")
    else:
        # 原始逻辑：简单随机
        print(f"\n🎲 随机打乱 (seed={args.seed}) 并选取...")
        random.shuffle(pair_ids)

        selected = []
        cumulative = 0
        selected_pair_ids = []
        for pid in pair_ids:
            entries = groups[pid]
            selected.extend(entries)
            cumulative += len(entries)
            selected_pair_ids.append(pid)
            if cumulative >= args.target:
                break

    # 5. 输出统计
    print(f"\n📊 采样结果:")
    print(f"  选取 pair_id 数: {len(selected_pair_ids)}")
    print(f"  总条数: {len(selected)}")
    print(f"  目标: {args.target}")
    print(f"  达成: {'✅' if len(selected) >= args.target else '❌'}")

    # 标签分布
    label_dist = Counter(s["label"] for s in selected)
    print(f"  标签分布: {dict(sorted(label_dist.items()))}")

    # 检查是否所有被选中 pair_id 的条目都在
    check_ok = True
    for pid in selected_pair_ids:
        g = groups[pid]
        for entry in g:
            if entry not in selected:
                print(f"  ⚠️  pair_id {pid} 的条目未完全选中!")
                check_ok = False
    if check_ok:
        print(f"  ✅ 所有选中 pair_id 的条目均已完整包含")

    # 6. 保存
    print(f"\n💾 保存到: {args.output}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✅ 保存完成 ({len(selected)} 条)")

    # 额外保存 pair_id 列表方便追溯
    pid_path = out_path.with_suffix(".pair_ids.json")
    with open(pid_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "target": args.target, "pair_ids": selected_pair_ids}, f, indent=2)
    print(f"  📝 pair_id 列表已保存: {pid_path}")

    report_path = Path(args.report_output) if args.report_output else out_path.parent / "sampling_report.json"
    report = build_sampling_report(
        source_path=Path(args.input),
        sampled_path=out_path,
        source_samples=samples,
        selected_samples=selected,
        selected_pair_ids=selected_pair_ids,
        seed=args.seed,
        target=args.target,
        stratify=args.stratify,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  📊 sampling report 已保存: {report_path}")


if __name__ == "__main__":
    main()
