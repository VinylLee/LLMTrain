#!/usr/bin/env python3
"""Compute MRV (Mutation Rate Value = error_count / total) per MR and per dataset."""
import json
import os
import re
import sys
from collections import OrderedDict

LABELS_3 = ['entailment', 'neutral', 'contradiction']
LABELS_2 = ['entailment', 'not_entailment']


def normalize_prediction_3(text):
    if not isinstance(text, str):
        return None
    t = text.strip().lower().rstrip('.!?')
    if t in LABELS_3:
        return t
    found = [lab for lab in LABELS_3 if lab in t]
    if found:
        return found[0]
    mapping = {'entailment': 'entailment', 'entailed': 'entailment', 'e': 'entailment',
               '0': 'entailment',
               'neutral': 'neutral', 'n': 'neutral', '1': 'neutral',
               'contradiction': 'contradiction', 'contradictory': 'contradiction',
               'c': 'contradiction', 'contradict': 'contradiction', '2': 'contradiction'}
    t_clean = re.sub(r'[^a-z0-9]', '', t)
    return mapping.get(t_clean, None)


def normalize_prediction_binary(text):
    if not isinstance(text, str):
        return None
    t = text.strip().lower().rstrip('.!?')
    if t in ('entailment', 'not_entailment'):
        return t
    if t == 'entailment' or t in ('entailed', 'e', '0'):
        return 'entailment'
    for lab in ('not_entailment', 'neutral', 'contradiction', 'contradictory', 'contradict'):
        if lab in t:
            return 'not_entailment'
    mapping = {'entailment': 'entailment', 'entailed': 'entailment', 'e': 'entailment',
               '0': 'entailment',
               'neutral': 'not_entailment', 'n': 'not_entailment', '1': 'not_entailment',
               'contradiction': 'not_entailment', 'contradictory': 'not_entailment',
               'c': 'not_entailment', 'contradict': 'not_entailment', '2': 'not_entailment',
               'not_entailment': 'not_entailment', 'not entailment': 'not_entailment'}
    t_clean = re.sub(r'[^a-z0-9]', '', t).replace(' ', '_')
    if t_clean in mapping:
        return mapping[t_clean]
    if 'not' in t and 'entail' in t:
        return 'not_entailment'
    if 'entail' in t:
        return 'entailment'
    return None


def extract_prediction(entry):
    """Extract the model's prediction from an output entry."""
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

    task = entry.get('meta', {}).get('task', 'nli')
    if task == 'nli-binary':
        return normalize_prediction_binary(pred_text)
    return normalize_prediction_3(pred_text)


def compute_mrv(base_dir):
    """Walk the output directory and compute MRV for each MR and dataset."""
    results = OrderedDict()
    dataset_totals = OrderedDict()

    datasets = sorted([d for d in os.listdir(base_dir)
                       if os.path.isdir(os.path.join(base_dir, d))])

    for ds in datasets:
        ds_dir = os.path.join(base_dir, ds)
        mr_files = sorted([f for f in os.listdir(ds_dir)
                          if f.endswith('.jsonl') and not f.endswith('_all.jsonl')])

        ds_total = 0
        ds_errors = 0
        ds_mrs = OrderedDict()

        for mr_file in mr_files:
            mr = mr_file.replace('.jsonl', '')
            mr_path = os.path.join(ds_dir, mr_file)
            total = 0
            errors = 0

            with open(mr_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get('status') != 200:
                        continue
                    gold = entry.get('meta', {}).get('gold_label')
                    if gold is None:
                        continue

                    pred = extract_prediction(entry)
                    if pred is None:
                        continue

                    total += 1
                    if gold != pred:
                        errors += 1

            if total > 0:
                mrv = errors / total
                ds_mrs[mr] = {'total': total, 'errors': errors, 'mrv': mrv}
                ds_total += total
                ds_errors += errors

        results[ds] = ds_mrs
        dataset_totals[ds] = {'total': ds_total, 'errors': ds_errors,
                              'mrv': ds_errors / ds_total if ds_total > 0 else 0}

    return results, dataset_totals


def print_table(results, dataset_totals):
    """Print MRV results as formatted tables."""
    # Calculate column widths
    all_mrs = set()
    for ds_mrs in results.values():
        all_mrs.update(ds_mrs.keys())
    all_mrs = sorted(all_mrs)

    col_w = max(26, max(len(mr) for mr in all_mrs) + 2)
    ds_w = max(8, max(len(ds) for ds in results.keys()) + 2)

    # Header
    header = f'{"Dataset":<{ds_w}}  {"Total":>6}  {"Errors":>6}  {"MRV":>8}'
    for mr in all_mrs:
        header += f'  {mr:>{col_w}}'
    print(header)
    print('-' * len(header))

    for ds in results:
        dt = dataset_totals[ds]
        line = f'{ds:<{ds_w}}  {dt["total"]:>6}  {dt["errors"]:>6}  {dt["mrv"]:>8.4f}'
        for mr in all_mrs:
            if mr in results[ds]:
                r = results[ds][mr]
                line += f'  {r["mrv"]:>{col_w}.4f} ({r["errors"]}/{r["total"]})'
            else:
                line += f'  {"—":>{col_w}}'
        print(line)

    # Overall
    grand_total = sum(dt['total'] for dt in dataset_totals.values())
    grand_errors = sum(dt['errors'] for dt in dataset_totals.values())
    grand_mrv = grand_errors / grand_total if grand_total > 0 else 0
    print('-' * len(header))
    print(f'{"ALL":<{ds_w}}  {grand_total:>6}  {grand_errors:>6}  {grand_mrv:>8.4f}')

    print()
    print('MRV = errors / total cases (higher = more mistakes by the model)')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute MRV (error rate) per MR and dataset')
    parser.add_argument('input_dir', nargs='?', default='output',
                        help='Path to model output directory (default: output)')
    parser.add_argument('--output', '-o', default=None,
                        help='Save report to file')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f'Directory not found: {args.input_dir}')
        sys.exit(1)

    results, dataset_totals = compute_mrv(args.input_dir)

    # Capture printed output
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_table(results, dataset_totals)
    sys.stdout = old_stdout
    output_text = buf.getvalue()

    print(output_text)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f'Saved to: {args.output}')


if __name__ == '__main__':
    main()
