import sys
import importlib.util
from pathlib import Path

# load scripts/evaluate.py as a module (works even if scripts isn't a package)
spec = importlib.util.spec_from_file_location('evaluate', Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate.py')
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
normalize_prediction_3 = evaluate.normalize_prediction_3
normalize_prediction_binary = evaluate.normalize_prediction_binary


def run():
    failures = []

    try:
        assert normalize_prediction_3('not_entailment') is None
        assert normalize_prediction_3('not entailment') is None
        assert normalize_prediction_3('entailment') == 'entailment'
        assert normalize_prediction_3('Entailed.') == 'entailment'
        print('test_normalize_prediction_3 passed')
    except AssertionError:
        failures.append('test_normalize_prediction_3')

    try:
        assert normalize_prediction_binary('not_entailment') == 'not_entailment'
        assert normalize_prediction_binary('not entailment') == 'not_entailment'
        assert normalize_prediction_binary('NOT entailment!') == 'not_entailment'
        assert normalize_prediction_binary('entailment') == 'entailment'
        assert normalize_prediction_binary('Entailed') == 'entailment'
        assert normalize_prediction_binary('neutral') == 'not_entailment'
        assert normalize_prediction_binary('2') == 'not_entailment'
        assert normalize_prediction_binary('0') == 'entailment'
        print('test_normalize_prediction_binary passed')
    except AssertionError:
        failures.append('test_normalize_prediction_binary')

    if failures:
        print('\nFailures:', failures)
        sys.exit(1)
    print('\nAll tests passed')


if __name__ == '__main__':
    run()
