#!/usr/bin/env python3
"""整理 NLI 实验结果：把 test_llm.py 的逐用例结果规范化，合并 MR，并生成逐数据集准确率摘要。

输入目录（test_llm.py --test-type original|mr 的输出）:
    output/<model>/<OFFICIAL_DATASET>/original/<DATASET>_original.jsonl
    output/<model>/<OFFICIAL_DATASET>/MR/<mr>.jsonl

本脚本:
    1. 规范化每条用例（缺失的 premise/gold/pred/correct 从 meta/example/response 回填），
       保证每一条用例都有 pred 与 correct —— 精确到每一个用例。
    2. MR：把每个 <mr>.jsonl 规范化后合并成 <DATASET>_MR_all.jsonl（新增 uid 保证全局唯一）。
    3. 生成 output/<model>/SUMMARY.md：按数据集×测试类型准确率 + 按 MR 细分 + 未计分统计。

幂等：可重复运行，只回填/重写。

用法:
    python3 scripts/organize_nli_experiment.py --model DeepSeek-v4-flash-0731 --output-dir output
    python3 scripts/organize_nli_experiment.py --model DeepSeek-v4-flash-0731 --dry-run
"""
import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 复用 test_llm.py 的解析函数（scripts 不是包，用 importlib 按仓库既有模式加载），
# 保证 pred/correct 的解析口径与 test_llm.py 完全一致。
_SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('test_llm', _SCRIPT_DIR / 'test_llm.py')
_tl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tl)

DISPLAY_NAMES = _tl.DISPLAY_NAMES
normalize_label = _tl.normalize_label
normalize_prediction_3 = _tl.normalize_prediction_3
normalize_prediction_binary = _tl.normalize_prediction_binary
extract_pred_text = _tl.extract_pred_text

DEFAULT_DATASETS = ('snli', 'mnlim', 'mnlimm', 'sick')


def read_records(path):
    """读 JSONL，忽略空行与解析失败的行。"""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_records(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def canonical_record(rec, ds, display, test_type, model):
    """规范化一条用例：缺失字段从 meta/example/response 回填，保证 pred/correct 齐备。"""
    rec = dict(rec)
    example = rec.get('example') if isinstance(rec.get('example'), dict) else {}
    meta = rec.get('meta') if isinstance(rec.get('meta'), dict) else {}

    rec.setdefault('model', model)
    rec.setdefault('dataset', ds)
    rec.setdefault('dataset_display', display)
    rec.setdefault('test_type', test_type)

    if rec.get('premise') is None:
        rec['premise'] = meta.get('premise') or example.get('premise')
    if rec.get('hypothesis') is None:
        rec['hypothesis'] = meta.get('hypothesis') or example.get('hypothesis')

    if rec.get('gold') is None:
        gold = meta.get('gold_label')
        if gold is None:
            gold = example.get('label')
        rec['gold'] = normalize_label(gold)

    if rec.get('idx') is None and 'idx' in example:
        rec['idx'] = example['idx']

    if test_type == 'mr':
        rec.setdefault('mr_type', example.get('mr_type'))
        rec.setdefault('mr_category', example.get('mr_category'))

    if rec.get('pred') is None and rec.get('status') == 200:
        pred_text = extract_pred_text(rec.get('response'))
        if pred_text is not None:
            if meta.get('task') == 'nli-binary':
                rec['pred'] = normalize_prediction_binary(pred_text)
            else:
                rec['pred'] = normalize_prediction_3(pred_text)

    if rec.get('correct') is None and rec.get('pred') is not None and rec.get('gold') is not None:
        rec['correct'] = (rec['pred'] == rec['gold'])

    return rec


def acc_stats(records):
    """按口径统计：total 仅含 status==200、gold 存在、pred 可解析的用例。"""
    scored = [r for r in records
              if r.get('status') == 200 and r.get('gold') is not None and r.get('pred') is not None]
    total = len(scored)
    correct = sum(1 for r in scored if r.get('correct') is True)
    http_errors = sum(1 for r in records if r.get('status') != 200)
    unparsed = sum(1 for r in records
                   if r.get('status') == 200 and r.get('gold') is not None and r.get('pred') is None)
    return {
        'total': total,
        'correct': correct,
        'acc': (correct / total) if total else 0.0,
        'http_errors': http_errors,
        'unparsed': unparsed,
    }


def process_original(orig_dir, display, ds, model, dry_run):
    """规范化并回写 original 数据集文件，返回统计。"""
    target = os.path.join(orig_dir, f'{display}_original.jsonl')
    if not os.path.exists(target):
        jsonls = sorted(f for f in os.listdir(orig_dir) if f.endswith('.jsonl'))
        if not jsonls:
            return None
        target = os.path.join(orig_dir, jsonls[0])
    records = [canonical_record(r, ds, display, 'original', model) for r in read_records(target)]
    stats = acc_stats(records)
    if not dry_run:
        write_records(target, records)
    return {'ds': ds, 'display': display, 'test_type': 'original', 'file': target,
            'records': records, 'stats': stats}


def process_mr(mr_dir, display, ds, model, dry_run):
    """规范化每个 MR 文件，合并成 <DISPLAY>_MR_all.jsonl，返回统计。"""
    merged_name = f'{display}_MR_all.jsonl'
    merged_path = os.path.join(mr_dir, merged_name)
    mr_files = sorted(f for f in os.listdir(mr_dir)
                      if f.endswith('.jsonl') and f != merged_name)

    if not mr_files and os.path.exists(merged_path):
        # 只有合并文件时（例如中断后重跑）：读合并文件统计，不回写。
        records = [canonical_record(r, ds, display, 'mr', model) for r in read_records(merged_path)]
        return {'ds': ds, 'display': display, 'test_type': 'mr', 'per_mr': [],
                'merged_file': merged_path, 'merged_records': records,
                'stats': acc_stats(records)}

    per_mr = []
    all_records = []
    uid = 0
    for fname in mr_files:
        mr = fname[:-len('.jsonl')]
        path = os.path.join(mr_dir, fname)
        recs = [canonical_record(r, ds, display, 'mr', model) for r in read_records(path)]
        for r in recs:
            uid += 1
            r['uid'] = uid
        stats = acc_stats(recs)
        if not dry_run:
            write_records(path, recs)
        per_mr.append({'mr': mr, 'file': path, 'records': recs, 'stats': stats})
        all_records.extend(recs)

    merged_stats = acc_stats(all_records)
    if not dry_run and all_records:
        write_records(merged_path, all_records)
    return {'ds': ds, 'display': display, 'test_type': 'mr', 'per_mr': per_mr,
            'merged_file': merged_path, 'merged_records': all_records, 'stats': merged_stats}


def _pct(total, correct):
    if not total:
        return '-'
    return f'{correct / total * 100:.2f}%'


def _first_mr_category(records):
    for r in records:
        if r.get('mr_category'):
            return r['mr_category']
    return '-'


def build_summary(model, datasets, results):
    """生成 SUMMARY.md 文本。"""
    lines = []
    lines.append(f'# NLI Evaluation Summary — {model}')
    lines.append('')
    lines.append(f'- Model: `{model}`')
    lines.append(f'- Generated: {datetime.utcnow().isoformat()}Z')
    lines.append('- Datasets: ' + ', '.join(DISPLAY_NAMES.get(d, d) for d in datasets))
    lines.append('- 口径: accuracy = correct / total；total 仅含 HTTP 200 且 gold 存在、'
                 'pred 可解析的用例；HTTP≠200 与解析失败在末尾单独统计。')
    lines.append('')

    # 1) 按数据集 × 测试类型
    lines.append('## Accuracy by dataset × test type')
    lines.append('')
    lines.append('| Dataset | Test type | Correct | Total | Accuracy |')
    lines.append('|---|---|---:|---:|---:|')
    for res in results:
        s = res['stats']
        lines.append(f"| {res['display']} | {res['test_type']} | {s['correct']} | {s['total']} "
                     f"| {_pct(s['total'], s['correct'])} |")
    lines.append('')

    # 2) 按 MR 细分
    mr_results = [r for r in results if r['test_type'] == 'mr']
    if mr_results:
        lines.append('## Per-MR accuracy')
        lines.append('')
        for res in mr_results:
            lines.append(f"### {res['display']} (MR)")
            lines.append('')
            lines.append('| MR type | Category | Correct | Total | Accuracy |')
            lines.append('|---|---|---:|---:|---:|')
            for p in res['per_mr']:
                s = p['stats']
                cat = _first_mr_category(p['records'])
                lines.append(f"| {p['mr']} | {cat} | {s['correct']} | {s['total']} "
                             f"| {_pct(s['total'], s['correct'])} |")
            lines.append('')

    # 3) 未计分统计
    lines.append('## Skipped / not scored')
    lines.append('')
    lines.append('| Dataset | Test type | HTTP != 200 | Unparseable pred |')
    lines.append('|---|---|---:|---:|')
    for res in results:
        s = res['stats']
        lines.append(f"| {res['display']} | {res['test_type']} | {s['http_errors']} | {s['unparsed']} |")
    lines.append('')

    return '\n'.join(lines)


def print_console(results):
    print()
    print('Dataset   Test type   Correct/Total     Accuracy')
    print('-' * 50)
    for res in results:
        s = res['stats']
        print(f"{res['display']:8s}  {res['test_type']:10s}  "
              f"{s['correct']:>6d}/{s['total']:<6d}   {_pct(s['total'], s['correct'])}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Organize NLI experiment results into per-example data + SUMMARY.md')
    parser.add_argument('--model', default='DeepSeek-v4-flash-0731', help='LLM folder name under output/')
    parser.add_argument('--output-dir', default='output', help='output directory')
    parser.add_argument('--run-id', default=None, help='optional run-id subdirectory above the model folder')
    parser.add_argument('--datasets', default=','.join(DEFAULT_DATASETS),
                        help='comma-separated dataset keys (default snli,mnlim,mnlimm,sick)')
    parser.add_argument('--dry-run', action='store_true',
                        help='only print what would be organized/summarized, write nothing')
    args = parser.parse_args()

    if args.run_id:
        root = os.path.join(args.output_dir, args.run_id, args.model)
    else:
        root = os.path.join(args.output_dir, args.model)
    datasets = tuple(d.strip() for d in args.datasets.split(',') if d.strip())

    results = []
    for ds in datasets:
        display = DISPLAY_NAMES.get(ds, ds)
        orig_dir = os.path.join(root, display, 'original')
        mr_dir = os.path.join(root, display, 'MR')
        if os.path.isdir(orig_dir):
            res = process_original(orig_dir, display, ds, args.model, args.dry_run)
            if res:
                results.append(res)
        if os.path.isdir(mr_dir):
            res = process_mr(mr_dir, display, ds, args.model, args.dry_run)
            if res:
                results.append(res)

    if not results:
        print(f'No result files found under {root}', file=sys.stderr)
        sys.exit(1)

    md = build_summary(args.model, datasets, results)

    if args.dry_run:
        print(md)
        print_console(results)
        print('[dry-run] nothing was written.')
        return

    summary_path = os.path.join(root, 'SUMMARY.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'SUMMARY.md written to {summary_path}')
    print_console(results)


if __name__ == '__main__':
    main()
