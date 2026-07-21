#!/usr/bin/env python3
"""Recompute multi-seed experiment reports from per-row ``correct`` values."""

import argparse
import json
import statistics
import sys
from pathlib import Path


WORK_DIR = Path("/home/ubuntu/LLMTrain/LLMTrain")


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else WORK_DIR / path


def read_result(path):
    total = correct = malformed = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            total += 1
            if row.get("correct") is True:
                correct += 1
    return correct, total, malformed


def collect(config, output_root, seeds):
    collected = {}
    missing = []
    malformed = []
    test_sets = config.get("test_sets", {})

    for experiment in config["experiments"]:
        base_name = experiment["name"]
        collected[base_name] = {}
        for seed in seeds:
            seed_name = f"{base_name}_seed{seed}"
            collected[base_name][seed] = {}
            for test_type in ("original", "mr"):
                for dataset in test_sets.get(test_type, {}):
                    path = output_root / seed_name / "tests" / test_type / f"{dataset}.jsonl"
                    key = (test_type, dataset)
                    if not path.is_file():
                        missing.append(path)
                        continue
                    correct, total, bad = read_result(path)
                    collected[base_name][seed][key] = (correct, total)
                    if bad:
                        malformed.append((path, bad))

    return collected, missing, malformed


def column_label(test_type, dataset):
    prefix = "Orig" if test_type == "original" else "MR"
    names = {"mnlim": "MNLIm", "mnlimm": "MNLImm", "sick": "SICK", "snli": "SNLI"}
    return f"{prefix} {names.get(dataset, dataset.upper())}"


def render_report(config, collected, seeds, missing, malformed):
    test_columns = [
        (test_type, dataset)
        for test_type in ("original", "mr")
        for dataset in config.get("test_sets", {}).get(test_type, {})
    ]
    expected_files = len(config["experiments"]) * len(seeds) * len(test_columns)
    present_files = expected_files - len(missing)

    lines = [
        "# 实验结果汇总",
        "",
        "数据来源：各实验 `tests/{original,mr}/*.jsonl`；所有统计均由逐行 `correct` 字段重新计算。",
        "",
        f"- 模型：`{config['model']}`",
        f"- 请求模型：`{config.get('requested_model', config['model'])}`",
        f"- 随机种子：{', '.join(map(str, seeds))}",
        f"- seed 口径：{config.get('seed_scope', '未记录')}",
        "- 每个单元格：`正确数/总数 (准确率)`",
        "- 均值±标准差：跨 seed 准确率的算术平均值与样本标准差（分母 n-1）",
        f"- 完整性：{present_files}/{expected_files} 个测试文件存在；JSON 解析异常文件 {len(malformed)} 个",
        "",
    ]

    for experiment in config["experiments"]:
        name = experiment["name"]
        lines.extend([
            f"## {name}",
            "",
            "| Seed | " + " | ".join(column_label(*key) for key in test_columns) + " |",
            "|---:|" + "---:|" * len(test_columns),
        ])

        accuracies = {key: [] for key in test_columns}
        for seed in seeds:
            cells = []
            for key in test_columns:
                value = collected[name][seed].get(key)
                if value is None:
                    cells.append("—")
                    continue
                correct, total = value
                accuracy = correct / total * 100 if total else 0.0
                accuracies[key].append(accuracy)
                cells.append(f"{correct}/{total} ({accuracy:.2f}%)")
            lines.append(f"| {seed} | " + " | ".join(cells) + " |")

        mean_cells = []
        for key in test_columns:
            values = accuracies[key]
            if len(values) == len(seeds) and len(values) > 1:
                mean_cells.append(f"**{statistics.mean(values):.2f}±{statistics.stdev(values):.2f}%**")
            elif values:
                mean_cells.append(f"**{statistics.mean(values):.2f}%**")
            else:
                mean_cells.append("—")
        lines.extend(["| **均值±标准差** | " + " | ".join(mean_cells) + " |", ""])

    if missing:
        lines.extend(["## 缺失文件", ""])
        lines.extend(f"- `{path.relative_to(WORK_DIR)}`" for path in missing)
        lines.append("")
    if malformed:
        lines.extend(["## JSON 解析异常", ""])
        lines.extend(f"- `{path.relative_to(WORK_DIR)}`：{count} 行" for path, count in malformed)
        lines.append("")

    return "\n".join(lines)


def render_comparison(config, llama_results, gemma_results, seeds):
    test_columns = [
        (test_type, dataset)
        for test_type in ("original", "mr")
        for dataset in config.get("test_sets", {}).get(test_type, {})
    ]
    lines = [
        "# Gemma-3-4B 与 Llama-3.2-3B 对比",
        "",
        "数值为三个相同数据采样 seed 的准确率均值；Δ = Llama − Gemma（百分点）。",
        "",
        "| 实验 | 测试集 | Gemma | Llama | Δ |",
        "|---|---|---:|---:|---:|",
    ]

    for experiment in config["experiments"]:
        llama_name = experiment["name"]
        gemma_name = llama_name.replace("_llama32_3b", "_gemma3_4b")
        for key in test_columns:
            llama_accs = []
            gemma_accs = []
            for seed in seeds:
                llama_value = llama_results.get(llama_name, {}).get(seed, {}).get(key)
                gemma_value = gemma_results.get(gemma_name, {}).get(seed, {}).get(key)
                if llama_value and gemma_value:
                    llama_accs.append(llama_value[0] / llama_value[1] * 100)
                    gemma_accs.append(gemma_value[0] / gemma_value[1] * 100)
            if len(llama_accs) != len(seeds):
                continue
            llama_mean = statistics.mean(llama_accs)
            gemma_mean = statistics.mean(gemma_accs)
            lines.append(
                f"| {llama_name} | {column_label(*key)} | {gemma_mean:.2f}% | "
                f"{llama_mean:.2f}% | {llama_mean - gemma_mean:+.2f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--comparison-root", default=None)
    parser.add_argument("--comparison-output", default=None)
    args = parser.parse_args()

    config = json.loads(resolve_path(args.config).read_text())
    output_root = resolve_path(args.output_root or config.get("output_root", "output/experiments"))
    output_path = resolve_path(args.output or output_root / "result.md")

    results, missing, malformed = collect(config, output_root, args.seeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(config, results, args.seeds, missing, malformed))
    print(f"报告: {output_path}")

    if args.comparison_root:
        gemma_config = json.loads(json.dumps(config))
        for experiment in gemma_config["experiments"]:
            experiment["name"] = experiment["name"].replace("_llama32_3b", "_gemma3_4b")
        gemma_results, gemma_missing, gemma_malformed = collect(
            gemma_config, resolve_path(args.comparison_root), args.seeds
        )
        comparison_output = resolve_path(
            args.comparison_output or output_root / "comparison_gemma3_4b_vs_llama32_3b.md"
        )
        comparison_output.write_text(render_comparison(config, results, gemma_results, args.seeds))
        print(f"对比: {comparison_output}")
        missing.extend(gemma_missing)
        malformed.extend(gemma_malformed)

    if (missing or malformed) and not args.allow_incomplete:
        print(f"不完整: missing={len(missing)}, malformed_files={len(malformed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
