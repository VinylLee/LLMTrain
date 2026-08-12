#!/usr/bin/env python3
"""Run LLM inference on NLI dataset JSON/JSONL files (original + MR) and save per-example results.

Supports two output layouts:

* ``--test-type original|mr``  — the experiment layout. Output goes under
  ``output/<llm>/<OFFICIAL_DATASET>/original|MR/<file>.jsonl`` with one record
  per test case carrying ``pred``/``correct`` (parsed from the response).
* ``--test-type auto`` (default) — legacy layout for the old wrappers
  (``output/[<run-id>/]<llm>/<dataset>/<mr>.jsonl``), preserved for backward
  compatibility.
"""
import os
import sys
import json
import re
import time
import argparse
from datetime import datetime
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import requests


# The 4 NLI datasets in scope for the DeepSeek-v4-flash experiment.
ALLOWED_DATASETS = ('snli', 'mnlim', 'mnlimm', 'sick')
DISPLAY_NAMES = {'snli': 'SNLI', 'mnlim': 'MNLIm', 'mnlimm': 'MNLImm', 'sick': 'SICK'}

# Real metamorphic relations. Note: 'summary' is deliberately NOT here — it is
# the name of the per-directory metadata file ({ds}_test_summary_*.json), not a
# real MR type, and must be skipped.
KNOWN_MRS = [
    'adding_contradiction', 'antonym_substitution', 'conditional_clause',
    'negation_flip', 'pronoun_substitution', 'synonym_replacement',
    'uninformative', 'voice_switch'
]

LABELS_3 = ['entailment', 'neutral', 'contradiction']
LABELS_2 = ['entailment', 'not_entailment']


def parse_dataset_mr(filepath):
    """Extract (dataset_key, mr_type, test_type) from a data file path.

    test_type is one of 'original', 'mr', 'skip'.

    Path examples:
        data/nli/original_dataset/snli/test.json                      -> ('snli', None, 'original')
        data/nli/MR_testing/snli_test_MR/snli_test_negation_flip_20260119_202923.json
                                                                      -> ('snli', 'negation_flip', 'mr')
        data/nli/MR_testing/snli_test_MR/snli_test_summary_20260119_202923.json
                                                                      -> ('snli', None, 'skip')
    """
    fname = os.path.splitext(os.path.basename(filepath))[0]
    parts = filepath.replace(os.sep, '/').split('/')

    ds = ''
    for part in parts:
        for suffix in ('_test_MR', '_test_mr'):
            if part.endswith(suffix):
                ds = part[:-len(suffix)]
                break
        if ds:
            break

    if ds:
        # MR path
        if 'summary' in fname:
            return ds, None, 'skip'
        mr = ''
        for known in sorted(KNOWN_MRS, key=len, reverse=True):
            if known in fname:
                mr = known
                break
        if not mr:
            return ds, None, 'skip'
        return ds, mr, 'mr'

    # Original path: only a file literally named test.json under a dataset dir
    # (drops train.json / validation.json / dataset_info.json and rte scratch
    # files like mnlimmtest.json / "test copy.json").
    if fname == 'test':
        parent = parts[-2] if len(parts) >= 2 else ''
        return parent, None, 'original'

    return '', None, 'skip'


def discover_nli_files(data_dir, datasets=None, test_type=None):
    """Yield (path, ds, mr, test_type) for in-scope NLI files.

    When ``test_type`` is given (not None) the scan is restricted to that test
    type AND to ``datasets`` (defaulting to ALLOWED_DATASETS), which excludes
    rte, train/validation, dataset_info and summary metadata. When ``test_type``
    is None (legacy auto mode) every parseable NLI file is yielded (no dataset
    filter) — keeps the old wrappers working.
    """
    if test_type is not None and datasets is None:
        datasets = ALLOWED_DATASETS
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.lower().endswith('.json'):
                continue
            path = os.path.join(root, f)
            ds, mr, tt = parse_dataset_mr(path)
            if tt == 'skip':
                continue
            if test_type is not None and tt != test_type:
                continue
            if test_type is not None and datasets is not None and ds not in datasets:
                continue
            yield path, ds, mr, tt


def compute_output_path(output_dir, run_id, llm, ds, mr, test_type, new_layout):
    """Return (out_path, out_dir) for one input file (does NOT mkdir).

    ``new_layout`` (True when ``--test-type original|mr``) uses the nested
    experiment layout carrying model / official dataset name / test type.
    Legacy (False) keeps the old flat layout.
    """
    safe_llm = llm.replace(' ', '_')
    if run_id:
        base = os.path.join(output_dir, run_id, safe_llm)
    else:
        base = os.path.join(output_dir, safe_llm)

    if new_layout:
        display = DISPLAY_NAMES.get(ds, ds)
        if test_type == 'original':
            out_path = os.path.join(base, display, 'original', f'{display}_original.jsonl')
        else:  # mr
            out_path = os.path.join(base, display, 'MR', f'{mr}.jsonl')
    elif test_type == 'mr':
        # Legacy flat layout
        out_path = os.path.join(base, ds, f'{mr}.jsonl')
    else:
        # original (or fallback): a sane single file per dataset
        out_path = os.path.join(base, ds, f'{ds}.jsonl')
    return out_path, os.path.dirname(out_path)


def load_env(path=None, api_url_env=None, api_key_env=None):
    if load_dotenv and path:
        load_dotenv(path)

    if api_key_env:
        api_key = os.getenv(api_key_env)
    else:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")

    if api_url_env:
        api_url = os.getenv(api_url_env)
    else:
        api_url = (
            os.getenv("DEEPSEEK_API_URL")
            or os.getenv("DEEPSEEK_URL")
            or os.getenv("DEEPSEEK_OPENAI_BASE_URL")
            or os.getenv("DEEPSEEK_ANTHROPIC_BASE_URL")
        )
    # Auto-append chat completions path if URL looks like a base URL
    if api_url and not api_url.rstrip('/').endswith(('chat/completions', '/generate', '/v1/completions')):
        base = api_url.rstrip('/')
        if base.endswith('/v1'):
            api_url = base + '/chat/completions'
        else:
            api_url = base + '/v1/chat/completions'
    return api_url or None, api_key


def guess_text_field(example):
    for k in ('input', 'prompt', 'text', 'source'):
        if isinstance(example, dict) and k in example:
            return example[k]
    return json.dumps(example, ensure_ascii=False)


def extract_nli_fields(example):
    if not isinstance(example, dict):
        return None, None, None

    premise_keys = ('premise', 'sentence1', 'sentence1_text', 'text_a')
    hypo_keys = ('hypothesis', 'sentence2', 'sentence2_text', 'text_b')
    label_keys = ('label', 'gold_label', 'annotator_labels')

    premise = None
    hypothesis = None
    label = None

    for k in premise_keys:
        if k in example:
            premise = example[k]
            break
    for k in hypo_keys:
        if k in example:
            hypothesis = example[k]
            break
    for k in label_keys:
        if k in example:
            label = example[k]
            break

    if isinstance(label, list) and label:
        label = label[0]

    return premise, hypothesis, label


def normalize_label(label):
    if label is None:
        return None
    s = str(label).strip().lower()
    mapping = {
        'entailment': 'entailment', 'entailed': 'entailment', 'e': 'entailment', '0': 'entailment',
        'neutral': 'neutral', 'n': 'neutral', '1': 'neutral',
        'contradiction': 'contradiction', 'contradictory': 'contradiction', 'c': 'contradiction', 'contradict': 'contradiction', '2': 'contradiction',
    }
    return mapping.get(s, s)


def normalize_label_binary(label):
    """RTE-style: 0 = entailment, everything else = not_entailment."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in ('entailment', 'entailed', 'e', '0'):
        return 'entailment'
    return 'not_entailment'


def build_nli_prompt(premise, hypothesis, binary=False):
    if binary:
        return (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Question: Is the hypothesis entailed by the premise? "
            "Answer ONLY with one of: entailment, not_entailment."
        )
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Question: Is the hypothesis entailed by the premise? "
        "Answer ONLY with one of: entailment, neutral, contradiction."
    )


def detect_binary_dataset(path):
    """Returns True if the path indicates a binary NLI dataset (RTE)."""
    parts = path.lower().replace(os.sep, '/').split('/')
    for p in parts:
        if 'rte' in p:
            return True
    return False


def extract_pred_text(resp):
    """Raw model output text from an OpenAI dict response or a plain string."""
    if isinstance(resp, dict):
        choices = resp.get('choices') or []
        if choices:
            return (choices[0].get('message') or {}).get('content')
    elif isinstance(resp, str):
        return resp
    return None


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
    # Detect not_entailment — look for whole-word matches
    for lab in ('not_entailment', 'neutral', 'contradiction', 'contradictory', 'contradict'):
        lab_check = lab.replace('_', ' ')
        if re.search(r"\b" + re.escape(lab_check) + r"\b", t_space):
            return 'not_entailment'
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
    if re.search(r'\bnot\b', t_space) and re.search(r'\bentail\b', t_space):
        return 'not_entailment'
    if re.search(r'\bentail\b', t_space):
        return 'entailment'
    return None


def call_llm(api_url, api_key, llm, prompt, max_retries=3, reasoning_effort=None):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {
        'model': llm,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
    }
    # For reasoning models (e.g. deepseek-v4-flash) sending reasoning_effort="none"
    # disables chain-of-thought, cutting output tokens from ~50-3100 to 1.
    if reasoning_effort:
        payload['reasoning_effort'] = reasoning_effort

    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=120)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return 0, f'ConnectionError after {max_retries} retries'
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return 0, f'Timeout after {max_retries} retries'
        except requests.exceptions.RequestException as e:
            return 0, str(e)


def load_jsonl(path):
    """Load a file that may be JSON (single object/array) or JSONL (one object per line)."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return []

    # Try parsing as a single JSON document first (array or object)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for k in ('data', 'examples', 'tests'):
                if k in data and isinstance(data[k], list):
                    return data[k]
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: treat as JSONL (one JSON object per line)
    examples = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'  [warn] skipping unparseable line: {e}', file=sys.stderr)
    return examples


def output_complete(out_path, source_path, max_samples=None):
    """True if out_path exists and already holds >= the expected record count.

    Used by ``--resume`` to skip input files that were fully processed in an
    earlier (possibly interrupted) run.
    """
    if not os.path.exists(out_path):
        return False
    expected = len(load_jsonl(source_path))
    if max_samples is not None:
        expected = min(expected, max_samples)
    if expected <= 0:
        return False
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            actual = sum(1 for _ in f)
    except OSError:
        return False
    return actual >= expected


def process_file(path, ds, mr, test_type, api_url, api_key, llm, api_model,
                 output_dir, run_id, delay, new_layout, max_samples=None,
                 reasoning_effort=None):
    out_path, out_dir = compute_output_path(output_dir, run_id, llm, ds, mr,
                                            test_type, new_layout)
    os.makedirs(out_dir, exist_ok=True)

    is_binary = detect_binary_dataset(path)
    examples = load_jsonl(path)
    if not examples:
        print(f'  [warn] no examples loaded from {path}', file=sys.stderr)
        return

    if max_samples is not None:
        examples = examples[:max_samples]

    with open(out_path, 'w', encoding='utf-8') as out_f:
        for i, ex in enumerate(examples, 1):
            premise, hypothesis, label = extract_nli_fields(ex)
            if premise is not None and hypothesis is not None:
                prompt = build_nli_prompt(premise, hypothesis, binary=is_binary)
                task = 'nli-binary' if is_binary else 'nli'
                gold = normalize_label_binary(label) if is_binary else normalize_label(label)
                meta = {'task': task, 'premise': premise, 'hypothesis': hypothesis, 'gold_label': gold}
            else:
                prompt = guess_text_field(ex)
                meta = {'task': 'generic'}
                gold = None

            ts = datetime.utcnow().isoformat() + 'Z'
            pred = None
            correct = None
            status = None
            resp = None
            error = None
            try:
                status, resp = call_llm(api_url, api_key, api_model, prompt,
                                        reasoning_effort=reasoning_effort)
                if status == 200:
                    pred_text = extract_pred_text(resp)
                    if pred_text is not None:
                        pred = (normalize_prediction_binary(pred_text) if is_binary
                                else normalize_prediction_3(pred_text))
                        if pred is not None and gold is not None:
                            correct = (pred == gold)
            except Exception as e:
                status = 'error'
                error = str(e)

            record = {
                'index': i,
                'idx': ex.get('idx') if isinstance(ex, dict) else None,
                'model': llm,
                'api_model': api_model,
                'dataset': ds,
                'dataset_display': DISPLAY_NAMES.get(ds, ds),
                'test_type': test_type,
                'mr_type': mr if test_type == 'mr' else None,
                'mr_category': (ex.get('mr_category')
                                if test_type == 'mr' and isinstance(ex, dict) else None),
                'premise': premise,
                'hypothesis': hypothesis,
                'gold': gold,
                'pred': pred,
                'correct': correct,
                'status': status,
                'timestamp': ts,
                'example': ex,
                'meta': meta,
                'response': resp,
            }
            if error:
                record['error'] = error
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            out_f.flush()
            time.sleep(delay)

    print(f'Wrote results: {out_path}')


def check_api_url(url):
    if not url:
        return False
    domain = urlparse(url).hostname or ''
    if domain.endswith('.example') or domain.endswith('.test') or domain.endswith('.invalid'):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Run LLM inference on NLI datasets (original + MR)')
    parser.add_argument('--env', default='.env', help='path to .env file')
    parser.add_argument('--llm', default=os.getenv('LLM_NAME', 'deepseek'),
                        help='LLM folder/display name under output/')
    parser.add_argument('--api-model', default=None,
                        help='actual model id sent to the API (default: --llm); '
                             'use this when the folder name differs from the API model id')
    parser.add_argument('--data-dir', default='data', help='data directory')
    parser.add_argument('--output-dir', default='output', help='output directory')
    parser.add_argument('--run-id', default=None,
                        help='run ID / subdirectory name (default: none, the model name is the top-level folder)')
    parser.add_argument('--delay', type=float, default=0.1, help='delay between requests (s)')
    parser.add_argument('--test-type', choices=['auto', 'original', 'mr'], default='auto',
                        help="test type to run; 'auto' keeps the legacy flat layout, "
                             "'original'/'mr' use the nested experiment layout")
    parser.add_argument('--datasets', default=None,
                        help='comma-separated dataset keys to run '
                             '(default: snli,mnlim,mnlimm,sick when --test-type is set)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='only process the first N samples per file (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='only print the discovered files and their target paths, do not call the API')
    parser.add_argument('--resume', action='store_true',
                        help='skip input files whose target output file already exists and is complete')
    parser.add_argument('--reasoning-effort', default=None,
                        choices=['none', 'low', 'medium', 'high'],
                        help='reasoning effort sent to the API; "none" disables chain-of-thought '
                             'for reasoning models like deepseek-v4-flash (huge token/cost saving)')
    parser.add_argument('--api-url-env', default=None,
                        help='env var name for API URL (e.g. BAILIAN_BASE_URL, LMSTUDIO_BASE_URL)')
    parser.add_argument('--api-key-env', default=None,
                        help='env var name for API key (e.g. BAILIAN_API_KEY, LMSTUDIO_KEY)')
    args = parser.parse_args()

    new_layout = args.test_type != 'auto'
    tt = None if args.test_type == 'auto' else args.test_type
    datasets = tuple(args.datasets.split(',')) if args.datasets else (
        ALLOWED_DATASETS if new_layout else None)

    if load_dotenv:
        load_dotenv(args.env)

    files = list(discover_nli_files(args.data_dir, datasets=datasets, test_type=tt))
    if not files:
        print('No matching JSON files found under', args.data_dir)
        sys.exit(1)

    if args.dry_run:
        for p, ds, mr, ftt in files:
            out_path, _ = compute_output_path(args.output_dir, args.run_id, args.llm,
                                              ds, mr, ftt, new_layout)
            print(f'{ftt:8s} {ds:8s} {mr or "-":22s} -> {out_path}')
        print(f'\n[dry-run] {len(files)} file(s) discovered. No API calls made.')
        return

    api_url, api_key = load_env(args.env, args.api_url_env, args.api_key_env)
    env_path = args.env

    if not api_url:
        print(f'Error: API URL not found.')
        if args.api_url_env:
            print(f'  The env var "{args.api_url_env}" is not set in {env_path}.')
        else:
            print(f'  Please set DEEPSEEK_API_URL in {env_path} or environment.')
            print(f'  Or use --api-url-env to specify a different env var.')
            print(f'  Example: --api-url-env BAILIAN_BASE_URL')
        sys.exit(2)

    if not api_key:
        print(f'Warning: No API key found{" (" + args.api_key_env + ")" if args.api_key_env else ""}.')
        print(f'  Requests may fail if the endpoint does not require authentication.')

    if not check_api_url(api_url):
        print(f'Warning: "{api_url}" looks like a placeholder (uses a reserved domain).')
        ans = input('  Continue anyway? [y/N] ').strip().lower()
        if ans not in ('y', 'yes'):
            sys.exit(1)

    api_model = args.api_model or args.llm

    for p, ds, mr, ftt in files:
        out_path, _ = compute_output_path(args.output_dir, args.run_id, args.llm,
                                          ds, mr, ftt, new_layout)
        if args.resume and output_complete(out_path, p, args.max_samples):
            print(f'  [resume] skip (complete): {out_path}')
            continue
        print('Processing', p)
        process_file(p, ds, mr, ftt, api_url, api_key, args.llm, api_model,
                     args.output_dir, args.run_id, args.delay, new_layout,
                     max_samples=args.max_samples, reasoning_effort=args.reasoning_effort)


if __name__ == '__main__':
    main()
