"""Unit tests for the DeepSeek-v4-flash NLI experiment code (test_llm.py + organize_nli_experiment.py)."""
import json

import pytest

from scripts import test_llm
from scripts import organize_nli_experiment as org


# --- parse_dataset_mr -------------------------------------------------------

def test_parse_dataset_mr_original():
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/snli/test.json') == ('snli', None, 'original')
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/mnlim/test.json') == ('mnlim', None, 'original')


def test_parse_dataset_mr_skips_non_test_original_files():
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/snli/train.json')[2] == 'skip'
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/sick/validation.json')[2] == 'skip'
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/sick/dataset_info.json')[2] == 'skip'
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/rte/mnlimmtest.json')[2] == 'skip'
    assert test_llm.parse_dataset_mr('data/nli/original_dataset/rte/test.json') == ('rte', None, 'original')


def test_parse_dataset_mr_mr_path():
    path = 'data/nli/MR_testing/snli_test_MR/snli_test_negation_flip_20260119_202923.json'
    assert test_llm.parse_dataset_mr(path) == ('snli', 'negation_flip', 'mr')


def test_parse_dataset_mr_skips_summary_metadata():
    path = 'data/nli/MR_testing/snli_test_MR/snli_test_summary_20260119_202923.json'
    assert test_llm.parse_dataset_mr(path)[2] == 'skip'


# --- discover_nli_files -----------------------------------------------------

def test_discover_original_exactly_four_datasets():
    files = list(test_llm.discover_nli_files('data/nli/original_dataset', test_type='original'))
    paths = {p for p, *_ in files}
    assert len(files) == 4, paths
    assert all(tt == 'original' for _, _, _, tt in files)


def test_discover_mr_31_files_no_summary():
    # snli/mnlim/mnlimm: 8 MR types each; sick: 7 (no negation_flip) -> 31 data files.
    files = list(test_llm.discover_nli_files('data/nli/MR_testing', test_type='mr'))
    assert len(files) == 31
    assert all('summary' not in path for path, _, _, _ in files)
    assert all(tt == 'mr' for _, _, _, tt in files)
    assert all(ds != 'rte' for _, ds, _, _ in files)


def test_discover_dataset_filter():
    files = list(test_llm.discover_nli_files('data/nli/original_dataset',
                                             datasets=('snli', 'sick'), test_type='original'))
    assert {ds for _, ds, _, _ in files} == {'snli', 'sick'}
    assert len(files) == 2


# --- compute_output_path ----------------------------------------------------

def test_compute_output_path_new_layout_original():
    path, _ = test_llm.compute_output_path('output', None, 'DeepSeek-v4-flash-0731',
                                           'snli', None, 'original', new_layout=True)
    assert path == 'output/DeepSeek-v4-flash-0731/SNLI/original/SNLI_original.jsonl'


def test_compute_output_path_new_layout_mr():
    path, _ = test_llm.compute_output_path('output', None, 'DeepSeek-v4-flash-0731',
                                           'sick', 'synonym_replacement', 'mr', new_layout=True)
    assert path == 'output/DeepSeek-v4-flash-0731/SICK/MR/synonym_replacement.jsonl'


def test_compute_output_path_with_run_id():
    path, _ = test_llm.compute_output_path('output', 'run20260804', 'Model A',
                                           'mnlim', None, 'original', new_layout=True)
    assert path == 'output/run20260804/Model_A/MNLIm/original/MNLIm_original.jsonl'


def test_compute_output_path_legacy():
    path, _ = test_llm.compute_output_path('output', None, 'deepseek', 'snli',
                                           'negation_flip', 'mr', new_layout=False)
    assert path == 'output/deepseek/snli/negation_flip.jsonl'


# --- process_file -----------------------------------------------------------

def _write_input(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def _mk_example(label, mr_type=None, mr_category=None, idx=None):
    ex = {'premise': 'A man is playing guitar.', 'hypothesis': 'Nobody is playing guitar.',
          'label': label}
    if mr_type:
        ex['mr_type'] = mr_type
    if mr_category:
        ex['mr_category'] = mr_category
    if idx is not None:
        ex['idx'] = idx
    return ex


def test_process_file_parses_pred_and_correct(tmp_path, monkeypatch):
    monkeypatch.setattr(test_llm, 'call_llm', lambda *a, **k: (
        200, {'choices': [{'message': {'content': 'contradiction'}}]}))
    inp = tmp_path / 'snli_test_negation_flip_20260119_202923.json'
    _write_input(inp, [_mk_example(2, 'negation_flip', 'flip'),
                       _mk_example(0, 'negation_flip', 'flip')])
    test_llm.process_file(str(inp), 'snli', 'negation_flip', 'mr', 'http://x', 'k',
                          'DeepSeek-v4-flash-0731', 'deepseek-v4-flash',
                          str(tmp_path), None, 0.0, new_layout=True)
    out_path = tmp_path / 'DeepSeek-v4-flash-0731' / 'SNLI' / 'MR' / 'negation_flip.jsonl'
    assert out_path.exists()
    rows = [json.loads(l) for l in out_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(rows) == 2
    # gold contradiction, model predicts contradiction -> correct True
    assert rows[0]['pred'] == 'contradiction' and rows[0]['correct'] is True
    # gold entailment, model predicts contradiction -> correct False
    assert rows[1]['pred'] == 'contradiction' and rows[1]['correct'] is False
    assert rows[0]['mr_type'] == 'negation_flip' and rows[0]['mr_category'] == 'flip'
    assert rows[0]['test_type'] == 'mr' and rows[0]['dataset_display'] == 'SNLI'


def test_output_complete(tmp_path):
    src = tmp_path / 'src.json'
    _write_input(src, [_mk_example(0), _mk_example(1), _mk_example(2)])
    out = tmp_path / 'out.jsonl'

    assert test_llm.output_complete(str(out), str(src)) is False  # missing

    out.write_text('{}\n{}\n{}\n', encoding='utf-8')
    assert test_llm.output_complete(str(out), str(src)) is True

    out.write_text('{}\n{}\n', encoding='utf-8')
    assert test_llm.output_complete(str(out), str(src)) is False  # incomplete

    # capped by max_samples: 2 output rows is enough for max_samples=2
    assert test_llm.output_complete(str(out), str(src), max_samples=2) is True


def test_process_file_passes_reasoning_effort(tmp_path, monkeypatch):
    captured = {}

    def fake_call_llm(api_url, api_key, model, prompt, max_retries=3, reasoning_effort=None):
        captured['reasoning_effort'] = reasoning_effort
        return 200, {'choices': [{'message': {'content': 'contradiction'}}]}

    monkeypatch.setattr(test_llm, 'call_llm', fake_call_llm)
    inp = tmp_path / 'test.json'
    _write_input(inp, [_mk_example(2)])
    test_llm.process_file(str(inp), 'snli', None, 'original', 'http://x', 'k',
                          'DeepSeek-v4-flash-0731', 'deepseek-v4-flash',
                          str(tmp_path), None, 0.0, new_layout=True, reasoning_effort='none')
    assert captured.get('reasoning_effort') == 'none'


def test_process_file_non200_and_unparseable(tmp_path, monkeypatch):
    monkeypatch.setattr(test_llm, 'call_llm', lambda *a, **k: (500, 'boom'))
    inp = tmp_path / 'test.json'
    _write_input(inp, [_mk_example(2)])
    test_llm.process_file(str(inp), 'snli', None, 'original', 'http://x', 'k',
                          'DeepSeek-v4-flash-0731', 'deepseek-v4-flash',
                          str(tmp_path), None, 0.0, new_layout=True)
    out_path = tmp_path / 'DeepSeek-v4-flash-0731' / 'SNLI' / 'original' / 'SNLI_original.jsonl'
    rows = [json.loads(l) for l in out_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert rows[0]['status'] == 500
    assert rows[0]['pred'] is None and rows[0]['correct'] is None


# --- organize_nli_experiment ------------------------------------------------

def _mk_record(mr_type, with_pred):
    rec = {
        'index': 1, 'model': 'TestModel', 'dataset': 'snli', 'dataset_display': 'SNLI',
        'test_type': 'mr' if mr_type else 'original',
        'mr_type': mr_type, 'mr_category': 'flip' if mr_type else None,
        'premise': 'A man is playing guitar.', 'hypothesis': 'Nobody is playing guitar.',
        'gold': 'contradiction',
        'pred': 'contradiction' if with_pred else None,
        'correct': True if with_pred else None,
        'status': 200, 'timestamp': '2026-01-01T00:00:00Z',
        'example': {'premise': 'A man is playing guitar.', 'hypothesis': 'Nobody is playing guitar.',
                    'label': 2, 'mr_type': mr_type, 'mr_category': 'flip'},
        'meta': {'task': 'nli', 'premise': 'A man is playing guitar.',
                 'hypothesis': 'Nobody is playing guitar.', 'gold_label': 'contradiction'},
        'response': {'choices': [{'message': {'content': 'contradiction'}}]},
    }
    return rec


def test_organize_backfills_missing_pred_and_writes_merged(tmp_path):
    root = tmp_path / 'TestModel'
    orig_dir = root / 'SNLI' / 'original'
    mr_dir = root / 'SNLI' / 'MR'
    orig_dir.mkdir(parents=True)
    mr_dir.mkdir(parents=True)

    (orig_dir / 'SNLI_original.jsonl').write_text(
        json.dumps(_mk_record(None, with_pred=False), ensure_ascii=False) + '\n', encoding='utf-8')
    (mr_dir / 'negation_flip.jsonl').write_text(
        json.dumps(_mk_record('negation_flip', with_pred=False), ensure_ascii=False) + '\n', encoding='utf-8')

    orig = org.process_original(str(orig_dir), 'SNLI', 'snli', 'TestModel', dry_run=False)
    mr = org.process_mr(str(mr_dir), 'SNLI', 'snli', 'TestModel', dry_run=False)

    assert orig['stats']['total'] == 1 and orig['stats']['correct'] == 1
    assert mr['stats']['total'] == 1 and mr['stats']['correct'] == 1
    assert mr['per_mr'][0]['stats']['correct'] == 1

    # merged file exists and records carry uid
    merged_path = mr_dir / 'SNLI_MR_all.jsonl'
    assert merged_path.exists()
    merged_rows = [json.loads(l) for l in merged_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    assert len(merged_rows) == 1 and merged_rows[0]['uid'] == 1

    md = org.build_summary('TestModel', ('snli',), [orig, mr])
    assert '# NLI Evaluation Summary — TestModel' in md
    assert '| SNLI | original | 1 | 1 |' in md
    assert '| negation_flip | flip | 1 | 1 |' in md
