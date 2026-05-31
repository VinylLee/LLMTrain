#!/usr/bin/env python3
"""Run LLM inference on dataset JSON/JSONL files and save responses."""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

import requests


KNOWN_MRS = [
    'adding_contradiction', 'antonym_substitution', 'conditional_clause',
    'negation_flip', 'pronoun_substitution', 'synonym_replacement',
    'uninformative', 'voice_switch', 'summary'
]


def parse_dataset_mr(filepath):
    """Extract (dataset_name, mr_type) from a data file path.

    Path example: data/nli/rte_test_MR/rte_test_pronoun_substitution_20260119_213432.json
    Returns: ('rte', 'pronoun_substitution')
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

    mr = ''
    for known in sorted(KNOWN_MRS, key=len, reverse=True):
        if known in fname:
            mr = known
            break

    return ds, mr


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


def discover_json_files(data_dir):
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.json'):
                yield os.path.join(root, f)


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


def call_llm(api_url, api_key, llm, prompt, max_retries=3):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {
        'model': llm,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
    }

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


def process_file(path, api_url, api_key, llm, output_dir, run_id, delay):
    ds, mr = parse_dataset_mr(path)
    safe_llm = llm.replace(' ', '_')

    out_dir = os.path.join(output_dir, run_id, safe_llm, ds)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{mr}.jsonl')

    is_binary = detect_binary_dataset(path)
    examples = load_jsonl(path)
    if not examples:
        print(f'  [warn] no examples loaded from {path}', file=sys.stderr)
        return

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

            ts = datetime.utcnow().isoformat() + 'Z'
            try:
                status, resp = call_llm(api_url, api_key, llm, prompt)
                record = {'index': i, 'timestamp': ts, 'status': status, 'example': ex, 'meta': meta, 'response': resp}
            except Exception as e:
                record = {'index': i, 'timestamp': ts, 'status': 'error', 'example': ex, 'meta': meta, 'error': str(e)}
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
    parser = argparse.ArgumentParser(description='Run LLM inference on datasets')
    parser.add_argument('--env', default='.env', help='path to .env file')
    parser.add_argument('--llm', default=os.getenv('LLM_NAME', 'deepseek'), help='LLM model name to use')
    parser.add_argument('--data-dir', default='data', help='data directory')
    parser.add_argument('--output-dir', default='output', help='output directory')
    parser.add_argument('--run-id', default=None, help='run ID / subdirectory name (auto if not set)')
    parser.add_argument('--delay', type=float, default=0.1, help='delay between requests (s)')
    parser.add_argument('--api-url-env', default=None,
                        help='env var name for API URL (e.g. BAILIAN_BASE_URL, LMSTUDIO_BASE_URL)')
    parser.add_argument('--api-key-env', default=None,
                        help='env var name for API key (e.g. BAILIAN_API_KEY, LMSTUDIO_KEY)')
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

    if load_dotenv:
        load_dotenv(args.env)

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

    files = list(discover_json_files(args.data_dir))
    if not files:
        print('No JSON files found under', args.data_dir)
        sys.exit(1)

    for p in files:
        print('Processing', p)
        process_file(p, api_url, api_key, args.llm, args.output_dir, run_id, args.delay)


if __name__ == '__main__':
    main()
