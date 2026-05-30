import pytest

from scripts.evaluate import normalize_prediction_3, normalize_prediction_binary


def test_normalize_prediction_3_not_entailment_not_misclassified():
    # ensure 'not entailment' variants are NOT misclassified as 'entailment' in 3-class
    assert normalize_prediction_3('not_entailment') is None
    assert normalize_prediction_3('not entailment') is None
    # positive checks
    assert normalize_prediction_3('entailment') == 'entailment'
    assert normalize_prediction_3('Entailed.') == 'entailment'


def test_normalize_prediction_binary_variants():
    # binary mapping: various forms should map correctly
    assert normalize_prediction_binary('not_entailment') == 'not_entailment'
    assert normalize_prediction_binary('not entailment') == 'not_entailment'
    assert normalize_prediction_binary('NOT entailment!') == 'not_entailment'
    assert normalize_prediction_binary('entailment') == 'entailment'
    assert normalize_prediction_binary('Entailed') == 'entailment'
    assert normalize_prediction_binary('neutral') == 'not_entailment'
    assert normalize_prediction_binary('2') == 'not_entailment'
    assert normalize_prediction_binary('0') == 'entailment'
