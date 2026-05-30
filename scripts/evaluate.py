#!/usr/bin/env python3
"""Evaluate NLI predictions from test_deepseek.py output JSONL files.

Supports both 3-class (entailment/neutral/contradiction) and binary
(entailment/not_entailment for RTE) evaluation.  Task type is auto-detected
from the meta.task field in each entry.

Usage:
    python scripts/evaluate.py output/results.jsonl
    python scripts/evaluate.py output/results.jsonl --group-by _source mr_type
    python scripts/evaluate.py output/results.jsonl --report report.txt
"""
import argparse
import json
import sys
import os
import re
from collections import Counter, defaultdict
from datetime import datetime


# --- label sets ---
LABELS_3 = ['entailment', 'neutral', 'contradiction']
LABELS_2 = ['entailment', 'not_entailment']


def get_labels(entry):
    """Return the label set appropriate for the entry's task."""
    task = entry.get('meta', {}).get('task', 'nli')
    return LABELS_2 if task == 'nli-binary' else LABELS_3


# --- prediction parsing ---

def normalize_prediction_3(text):
    if not isinstance(text, str):
        return None
    t = text.strip().lower().rstrip('.!?')
    # normalized-space for word-boundary checks (turn underscores/punctuation into spaces)
    t_space = re.sub(r'[^a-z0-9]+', ' ', t).strip()
    # direct exact matches (either raw or normalized)
    if t in LABELS_3 or t_space in LABELS_3:
        return t_space if t_space in LABELS_3 else t
    # Reject binary-classification patterns (not_entailment / not entailment)
    # — these don't belong in 3-class output. Check BEFORE the word-boundary
    # search so 'entailment' inside 'not entailment' isn't misclassified.
    if re.search(r'\bnot[ _]entail', t_space):
        return None
    # word-boundary search to avoid matching 'entailment' inside 'not_entailment'
    found = [lab for lab in LABELS_3 if re.search(r"\b" + re.escape(lab) + r"\b", t_space)]
    if found:
        return found[0]
    mapping = {'entailment': 'entailment', 'entailed': 'entailment', 'e': 'entailment',
               '0': 'entailment',
               'neutral': 'neutral', 'n': 'neutral', '1': 'neutral',
               'contradiction': 'contradiction', 'contradictory': 'contradiction',
               'c': 'contradiction', 'contradict': 'contradiction', '2': 'contradiction'}
    # keep underscores for multi-word normalized tokens (e.g. 'not_entailment')
    t_clean = re.sub(r'[^a-z0-9]+', '_', t).strip('_')
    return mapping.get(t_clean, None)


def normalize_prediction_binary(text):
    """Binary: map model output to entailment / not_entailment."""
    if not isinstance(text, str):
        return None
    t = text.strip().lower().rstrip('.!?')
    t_space = re.sub(r'[^a-z0-9]+', ' ', t).strip()
    # Direct exact matches (handle both underscore and space forms)
    if t in ('entailment', 'not_entailment') or t_space in ('entailment', 'not entailment'):
        return 'not_entailment' if (t == 'not_entailment' or t_space == 'not entailment') else 'entailment'
    # Detect entailment short forms
    if t == 'entailment' or t in ('entailed', 'e', '0') or t_space in ('entailed', 'e', '0'):
        return 'entailment'
    # Detect not_entailment — look for whole-word matches (handles 'not entailment')
    for lab in ('not_entailment', 'neutral', 'contradiction', 'contradictory', 'contradict'):
        lab_check = lab.replace('_', ' ')
        if re.search(r"\b" + re.escape(lab_check) + r"\b", t_space):
            return 'not_entailment'
    # Numeric / compact mapping
    mapping = {'entailment': 'entailment', 'entailed': 'entailment', 'e': 'entailment',
               '0': 'entailment',
               'neutral': 'not_entailment', 'n': 'not_entailment', '1': 'not_entailment',
               'contradiction': 'not_entailment', 'contradictory': 'not_entailment',
               'c': 'not_entailment', 'contradict': 'not_entailment', '2': 'not_entailment',
               'not_entailment': 'not_entailment', 'not entailment': 'not_entailment'}
    t_clean = re.sub(r'[^a-z0-9]+', '_', t).strip('_')
    if t_clean in mapping:
        return mapping[t_clean]
    # Flexible fallback checks using word tokens
    if re.search(r"\\bnot\\b", t_space) and re.search(r"\\bentail\\b", t_space):
        return 'not_entailment'
    if re.search(r"\\bentail\\b", t_space):
        return 'entailment'
    return None


def extract_prediction(entry):
    resp = entry.get('response')
    pred_text = None
    if isinstance(resp, dict):
        choices = resp.get('choices', [])
        if choices:
            pred_text = choices[0].get('message', {}).get('content', '')
    elif isinstance(resp, str):
        pred_text = resp
    else:
        return None

    labels = get_labels(entry)
    if labels == LABELS_2:
        return normalize_prediction_binary(pred_text)
    return normalize_prediction_3(pred_text)


# --- metrics ---

def compute_metrics(golds, preds, labels):
    total = len(golds)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    acc = correct / total if total else 0

    metrics = {}
    for label in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        metrics[label] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'support': sum(1 for g in golds if g == label)}

    macro_f1 = sum(metrics[l]['f1'] for l in labels) / len(labels)
    macro_precision = sum(metrics[l]['precision'] for l in labels) / len(labels)
    macro_recall = sum(metrics[l]['recall'] for l in labels) / len(labels)

    return {
        'accuracy': acc,
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
        'f1_macro': macro_f1,
        'per_class': metrics,
        'labels': labels,
        'total': total,
        'correct': correct,
    }


def pct(v):
    return f'{v * 100:.2f}%'


# --- report builders ---

def build_breakdown(entries, golds, preds, group_keys):
    groups = defaultdict(lambda: {'golds': [], 'preds': [], 'entries': []})
    for entry, g, p in zip(entries, golds, preds):
        key = tuple(str(entry['example'].get(k, '?')) for k in group_keys)
        groups[key]['golds'].append(g)
        groups[key]['preds'].append(p)
        groups[key]['entries'].append(entry)
    rows = []
    for key in sorted(groups):
        g = groups[key]['golds']
        p = groups[key]['preds']
        correct = sum(1 for a, b in zip(g, p) if a == b)
        total = len(g)
        rows.append((', '.join(key), correct, total, pct(correct / total)))
    return rows


def build_token_usage(entries):
    total_prompt = 0
    total_completion = 0
    total_cached = 0
    count = 0
    for e in entries:
        usage = e.get('response', {}).get('usage', {}) if isinstance(e.get('response'), dict) else {}
        if usage:
            total_prompt += usage.get('prompt_tokens', 0)
            total_completion += usage.get('completion_tokens', 0)
            total_cached += usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
            count += 1
    return {'count': count, 'prompt': total_prompt, 'completion': total_completion, 'cached': total_cached, 'total': total_prompt + total_completion}


def build_report(args, entries, valid, metrics, errors, token_usage, breakdown_rows):
    labels = metrics['labels']
    task_name = 'Binary (entailment / not_entailment)' if labels == LABELS_2 else '3-class'

    lines = []
    lines.append('=' * 64)
    lines.append('  NLI Evaluation Report')
    lines.append('=' * 64)
    lines.append(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'  Input:     {args.input}')
    lines.append(f'  Task:      {task_name}')
    lines.append(f'  Group by:  {", ".join(args.group_by) if args.group_by else "none"}')
    lines.append('')

    # 1. API Status
    lines.append('─' * 64)
    lines.append('  1. API Status Summary')
    lines.append('─' * 64)
    statuses = Counter(e['status'] for e in entries)
    for status, count in sorted(statuses.items()):
        lines.append(f'     HTTP {status}: {count} ({pct(count / len(entries))})')
    lines.append('')

    # 2. Overall Metrics
    lines.append('─' * 64)
    lines.append('  2. Overall Metrics')
    lines.append('─' * 64)
    lines.append(f'     Total samples:  {metrics["total"]}')
    lines.append(f'     Correct:        {metrics["correct"]}')
    lines.append(f'     Accuracy:       {pct(metrics["accuracy"])}')
    lines.append('')
    lines.append(f'     {"":24s} {"Precision":>10s} {"Recall":>10s} {"F1":>10s} {"Support":>8s}')
    lines.append(f'     {"─" * 62}')
    for label in labels:
        m = metrics['per_class'][label]
        lines.append(f'     {label:24s} {pct(m["precision"]):>10s} {pct(m["recall"]):>10s} {pct(m["f1"]):>10s} {m["support"]:>8d}')
    lines.append(f'     {"─" * 62}')
    lines.append(f'     {"macro":24s} {pct(metrics["precision_macro"]):>10s} {pct(metrics["recall_macro"]):>10s} {pct(metrics["f1_macro"]):>10s}')
    lines.append('')

    # 3. Confusion Matrix
    lines.append('─' * 64)
    lines.append('  3. Confusion Matrix')
    lines.append('─' * 64)
    header = '     ' + ''.join(f'{l:>14s}' for l in labels)
    lines.append(header)
    for true in labels:
        row = f'     {true:>14s}'
        for pred in labels:
            count = sum(1 for g, p in zip(metrics.get('_golds', []), metrics.get('_preds', [])) if g == true and p == pred)
            row += f'{count:>14d}'
        lines.append(row)
    lines.append('')

    # 4. Error Cases
    lines.append('─' * 64)
    lines.append(f'  4. Error Cases (total: {len(errors)})')
    lines.append('─' * 64)
    if not errors:
        lines.append('     No errors!')
    else:
        for i, (entry, gold, pred) in enumerate(errors, 1):
            src = entry['example'].get('_source', '?')
            mr = entry['example'].get('mr_type', '?')
            prem = entry['meta']['premise']
            hypo = entry['meta']['hypothesis']
            lines.append(f'     #{i}  gold={gold}  pred={pred}  [{src}/{mr}]')
            lines.append(f'       P: {prem}')
            lines.append(f'       H: {hypo}')
            lines.append('')
    lines.append('')

    # 5. Breakdown
    if breakdown_rows:
        lines.append('─' * 64)
        lines.append('  5. Breakdown by Group')
        lines.append('─' * 64)
        for key, correct, total, acc_str in breakdown_rows:
            lines.append(f'     {key:48s}  {correct}/{total} = {acc_str}')
        lines.append('')

    # 6. Token Usage
    if token_usage['count']:
        lines.append('─' * 64)
        lines.append('  6. Token Usage')
        lines.append('─' * 64)
        lines.append(f'     Samples with usage data: {token_usage["count"]}')
        lines.append(f'     Total prompt tokens:     {token_usage["prompt"]}')
        lines.append(f'     Total completion tokens: {token_usage["completion"]}')
        lines.append(f'     Total cached tokens:     {token_usage["cached"]}')
        lines.append(f'     Total tokens:            {token_usage["total"]}')
        lines.append(f'     Avg per sample:          {token_usage["total"] / token_usage["count"]:.1f}')
        lines.append('')

    lines.append('=' * 64)
    return '\n'.join(lines)


def detect_task_label(entries):
    """Detect whether the file contains binary or 3-class entries by scanning meta.task."""
    for e in entries:
        task = e.get('meta', {}).get('task', '')
        if task == 'nli-binary':
            return LABELS_2
    return LABELS_3


def main():
    parser = argparse.ArgumentParser(description='Evaluate NLI prediction results')
    parser.add_argument('input', help='Path to output JSONL file')
    parser.add_argument('--group-by', nargs='+', default=None,
                        help='Group metrics by fields (e.g. _source mr_type)')
    parser.add_argument('--report', '-o', default=None,
                        help='Path to write report file (default: auto-name based on input)')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f'Loaded {len(entries)} entries from {args.input}')

    valid = [e for e in entries if e.get('status') == 200]
    valid = [e for e in valid if 'gold_label' in e.get('meta', {})]
    labels = detect_task_label(entries)
    skipped_generic = len([e for e in entries if e.get('status') == 200]) - len(valid)
    if skipped_generic:
        print(f'  (skipping {skipped_generic} non-NLI / no-gold-label entries)')

    golds = [e['meta']['gold_label'] for e in valid]
    preds = [extract_prediction(e) for e in valid]

    parsed = [(e, g, p) for e, g, p in zip(valid, golds, preds) if p is not None]
    unparsed = len(golds) - len(parsed)

    if not parsed:
        print('No valid entries to evaluate.')
        sys.exit(1)

    entries_ok, golds_ok, preds_ok = zip(*parsed)
    metrics = compute_metrics(golds_ok, preds_ok, labels)
    metrics['_golds'] = golds_ok
    metrics['_preds'] = preds_ok

    errors = [(e, g, p) for e, g, p in zip(entries_ok, golds_ok, preds_ok) if g != p]
    token_usage = build_token_usage(entries_ok)

    breakdown_rows = []
    if args.group_by:
        breakdown_rows = build_breakdown(entries_ok, golds_ok, preds_ok, args.group_by)

    report = build_report(args, entries, valid, metrics, errors, token_usage, breakdown_rows)

    report_path = args.report
    if not report_path:
        base = os.path.splitext(os.path.basename(args.input))[0]
        report_path = os.path.join(os.path.dirname(args.input) or '.', f'{base}_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'\nReport written to: {report_path}')

    print(f'\n  Accuracy:      {pct(metrics["accuracy"])}  ({metrics["correct"]}/{metrics["total"]})')
    print(f'  Macro F1:      {pct(metrics["f1_macro"])}')
    print(f'  Errors:        {len(errors)}')
    if errors:
        for e, g, p in errors[:3]:
            mr = e['example'].get('mr_type', '?')
            print(f'    gold={g} pred={p} [{mr}]')
        if len(errors) > 3:
            print(f'    ... and {len(errors) - 3} more (see report)')
    print(f'  Total tokens:  {token_usage["total"]}')
    if token_usage['count']:
        print(f'  Cost est.:     ${token_usage["total"] * 0.15 / 1e6:.4f} (at $0.15/M tok)')


if __name__ == '__main__':
    main()
