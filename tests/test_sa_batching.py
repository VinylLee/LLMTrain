"""Offline tests for batched generation and batched validator probes.

No model is loaded: a fake tokenizer/model pair stands in for Transformers so the
padding, decoding and scatter logic can be asserted directly.
"""

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import generate_sa_mr as engine
import sa_mr_catalog as catalog
from sa_validators import GeneratorSelfValidator, ScriptedValidator, ValidationVerdict


# --------------------------------------------------------------------------- #
# Fakes standing in for Transformers
# --------------------------------------------------------------------------- #

class _Ids(list):
    """List of token id rows that also exposes a tensor-like ``shape``."""

    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)


class _Encoding(dict):
    def to(self, device):
        self.device = device
        return self


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Cuda:
    @staticmethod
    def manual_seed_all(seed):
        _Cuda.last_seed = seed


class _Torch:
    cuda = _Cuda()

    def __init__(self):
        self.seeds = []

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def inference_mode(self):
        return _Context()


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def __init__(self):
        self.padding_side_seen_during_template = []
        self.batch_sizes = []

    def apply_chat_template(self, chunk, **kwargs):
        self.padding_side_seen_during_template.append(self.padding_side)
        self.batch_sizes.append(len(chunk))
        assert kwargs.get("padding") is True
        return _Encoding(input_ids=_Ids([[7] * 5 for _ in chunk]))

    def decode(self, ids, skip_special_tokens=True):
        return f"decoded-{len(ids)}"


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        batch = len(kwargs["input_ids"])
        self.calls.append(kwargs)
        # Echo back prompt tokens plus two generated ones per row.
        return [[0] * 5 + [9, 9] for _ in range(batch)]


class FakeInner:
    """Mimics the subset of ``HFTextGenerator`` that ``BatchedGenerator`` uses."""

    def __init__(self, temperature=0.8, top_p=0.95):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.torch = _Torch()
        self.device = type("D", (), {"type": "cuda"})()
        self.max_new_tokens = 128
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, messages, seed):
        return "single"


# --------------------------------------------------------------------------- #
# BatchedGenerator
# --------------------------------------------------------------------------- #

def test_batched_generator_chunks_by_batch_size():
    inner = FakeInner()
    batched = engine.BatchedGenerator(inner, batch_size=2)
    prompts = [[{"role": "user", "content": f"p{i}"}] for i in range(5)]

    results = batched.generate_batch(prompts, [1, 2, 3, 4, 5])

    assert len(results) == 5
    assert inner.tokenizer.batch_sizes == [2, 2, 1]
    assert len(inner.model.calls) == 3


def test_batched_generator_uses_left_padding_and_restores_it():
    inner = FakeInner()
    batched = engine.BatchedGenerator(inner, batch_size=4)
    batched.generate_batch([[{"role": "user", "content": "x"}]], [1])

    assert inner.tokenizer.padding_side_seen_during_template == ["left"]
    assert inner.tokenizer.padding_side == "right", "padding side must be restored"


def test_batched_generator_strips_the_prompt_from_each_output():
    inner = FakeInner()
    batched = engine.BatchedGenerator(inner, batch_size=2)
    results = batched.generate_batch(
        [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], [1, 2]
    )
    assert results == ["decoded-2", "decoded-2"]


def test_batched_generator_samples_when_temperature_is_positive():
    inner = FakeInner(temperature=0.8)
    engine.BatchedGenerator(inner, batch_size=2).generate_batch(
        [[{"role": "user", "content": "a"}]], [1]
    )
    call = inner.model.calls[0]
    assert call["do_sample"] is True
    assert call["temperature"] == 0.8
    assert call["top_p"] == 0.95


def test_batched_generator_is_greedy_when_temperature_is_zero():
    inner = FakeInner(temperature=0.0)
    engine.BatchedGenerator(inner, batch_size=2).generate_batch(
        [[{"role": "user", "content": "a"}]], [1]
    )
    call = inner.model.calls[0]
    assert call["do_sample"] is False
    assert "temperature" not in call


def test_batched_generator_seeds_from_the_first_task_in_the_batch():
    inner = FakeInner()
    engine.BatchedGenerator(inner, batch_size=4).generate_batch(
        [[{"role": "user", "content": str(i)}] for i in range(3)], [11, 22, 33]
    )
    assert inner.torch.seeds == [11]


def test_batched_generator_forwards_the_sampling_knobs_to_the_inner_model():
    inner = FakeInner()
    batched = engine.BatchedGenerator(inner, batch_size=2)
    batched.temperature = 0.0
    batched.top_p = 1.0
    assert inner.temperature == 0.0
    assert inner.top_p == 1.0


# --------------------------------------------------------------------------- #
# TorchDynamo recompilation ceiling
# --------------------------------------------------------------------------- #

def _torch_with_limit(limit):
    """A stand-in for ``torch`` mirroring its real ``_dynamo.config`` object chain.

    Returns the config object the helper will actually mutate, so assertions read
    the same attribute the production code writes.
    """
    config = type("Config", (), {"cache_size_limit": limit})()
    dynamo = type("Dynamo", (), {"config": config})()
    torch_module = type("Torch", (), {"_dynamo": dynamo})()
    return torch_module, config


def test_relax_dynamo_cache_limit_raises_a_low_ceiling():
    torch_module, config = _torch_with_limit(8)
    assert engine.relax_dynamo_cache_limit(torch_module) == engine.DYNAMO_CACHE_SIZE_LIMIT
    assert config.cache_size_limit == engine.DYNAMO_CACHE_SIZE_LIMIT


def test_relax_dynamo_cache_limit_never_lowers_an_existing_high_ceiling():
    torch_module, config = _torch_with_limit(4096)
    engine.relax_dynamo_cache_limit(torch_module)
    assert config.cache_size_limit == 4096


def test_relax_dynamo_cache_limit_tolerates_missing_dynamo():
    class Bare:
        pass

    assert engine.relax_dynamo_cache_limit(Bare()) is None
    assert engine.relax_dynamo_cache_limit(None) is None


def test_batched_generator_relaxes_the_ceiling_on_construction():
    """Regression: variable batch shapes used to abort long runs mid-flight."""
    inner = FakeInner()
    torch_module, config = _torch_with_limit(8)
    inner.torch = torch_module
    engine.BatchedGenerator(inner, batch_size=4)
    assert config.cache_size_limit == engine.DYNAMO_CACHE_SIZE_LIMIT


# --------------------------------------------------------------------------- #
# Fallbacks
# --------------------------------------------------------------------------- #

def test_generate_batch_falls_back_to_sequential_calls():
    class Plain:
        def __init__(self):
            self.calls = 0

        def generate(self, messages, seed):
            self.calls += 1
            return f"r{seed}"

    plain = Plain()
    prompts = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    assert engine.generate_batch(plain, prompts, [1, 2]) == ["r1", "r2"]
    assert plain.calls == 2


def test_predict_many_falls_back_to_sequential_predictions():
    validator = ScriptedValidator({"a": "positive", "b": "negative"}, validator_id="v")
    verdicts = engine.predict_many(validator, ["a", "b"], seed=0)
    assert [verdict.predicted_label for verdict in verdicts] == ["positive", "negative"]


# --------------------------------------------------------------------------- #
# Batched validator probes
# --------------------------------------------------------------------------- #

class RecordingValidator:
    """Records how many texts it was asked about, and in how many calls."""

    def __init__(self, labels_by_text, validator_id="recording"):
        self.labels_by_text = labels_by_text
        self.validator_id = validator_id
        self.batch_calls = []

    def predict(self, text, seed):
        return ValidationVerdict(self.validator_id, self.labels_by_text.get(text))

    def predict_many(self, texts, seed):
        self.batch_calls.append(list(texts))
        return [
            ValidationVerdict(self.validator_id, self.labels_by_text.get(text)) for text in texts
        ]


def test_collect_verdicts_batched_deduplicates_identical_probes():
    """Every flip group re-checks its source, so sources must be sent once."""
    validator = RecordingValidator({"SRC": "positive", "F1": "negative", "F2": "negative"})
    flip = catalog.get_mr("sa_controlled_negation_flip")
    requests = [
        {"group_id": "g1", "definition": flip, "source_text": "SRC", "followup_text": "F1"},
        {"group_id": "g2", "definition": flip, "source_text": "SRC", "followup_text": "F2"},
    ]
    result = engine.collect_verdicts_batched({"generator": validator}, requests, seed=0)

    assert len(validator.batch_calls) == 1, "probes must be dispatched in one batch"
    sent = validator.batch_calls[0]
    assert sorted(sent) == ["F1", "F2", "SRC"], "the shared source is sent exactly once"

    assert result["g1"]["generator"]["source"].predicted_label == "positive"
    assert result["g1"]["generator"]["followup"].predicted_label == "negative"
    assert result["g2"]["generator"]["followup"].predicted_label == "negative"


def test_collect_verdicts_batched_skips_source_probes_for_preserve_mrs():
    validator = RecordingValidator({"SRC": "positive", "F1": "positive"})
    preserve = catalog.get_mr("sa_tense_shift")
    engine.collect_verdicts_batched(
        {"generator": validator},
        [{"group_id": "g1", "definition": preserve, "source_text": "SRC", "followup_text": "F1"}],
        seed=0,
    )
    assert validator.batch_calls[0] == ["F1"]


def test_collect_verdicts_batched_covers_every_configured_validator():
    a = RecordingValidator({"F1": "positive"}, validator_id="a")
    b = RecordingValidator({"F1": "negative"}, validator_id="b")
    preserve = catalog.get_mr("sa_tense_shift")
    result = engine.collect_verdicts_batched(
        {"a": a, "b": b},
        [{"group_id": "g1", "definition": preserve, "source_text": "S", "followup_text": "F1"}],
        seed=0,
    )
    assert result["g1"]["a"]["followup"].predicted_label == "positive"
    assert result["g1"]["b"]["followup"].predicted_label == "negative"


def test_collect_verdicts_batched_handles_no_requests():
    assert engine.collect_verdicts_batched({"g": RecordingValidator({})}, [], seed=0) == {}


# --------------------------------------------------------------------------- #
# GeneratorSelfValidator.predict_many
# --------------------------------------------------------------------------- #

class _BatchCapableGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.batch_calls = 0

    def generate(self, messages, seed):
        return self.responses.pop(0)

    def generate_batch(self, prompts, seeds):
        self.batch_calls += 1
        return [self.responses.pop(0) for _ in prompts]


def test_generator_self_validator_predict_many_uses_one_batch():
    generator = _BatchCapableGenerator(
        [
            json.dumps({"predicted_label": "positive", "unambiguous": True, "reason": "r"}),
            json.dumps({"predicted_label": "negative", "unambiguous": True, "reason": "r"}),
        ]
    )
    validator = GeneratorSelfValidator(generator)
    verdicts = validator.predict_many(["a", "b"], seed=3)

    assert generator.batch_calls == 1
    assert [verdict.predicted_label for verdict in verdicts] == ["positive", "negative"]


def test_generator_self_validator_predict_many_marks_unparseable_rows():
    generator = _BatchCapableGenerator(["not json", "not json either"])
    verdicts = GeneratorSelfValidator(generator).predict_many(["a", "b"], seed=3)
    assert all(not verdict.usable for verdict in verdicts)


def test_generator_self_validator_predict_many_falls_back_without_batching():
    class Plain:
        def __init__(self):
            self.responses = [
                json.dumps({"predicted_label": "positive", "unambiguous": True, "reason": "r"})
            ]

        def generate(self, messages, seed):
            return self.responses.pop(0)

    verdicts = GeneratorSelfValidator(Plain()).predict_many(["a"], seed=3)
    assert verdicts[0].predicted_label == "positive"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_batch_size_defaults_to_eight():
    args = engine.build_parser().parse_args(
        ["--input", "x", "--dataset", "imdb", "--dry-run"]
    )
    assert args.batch_size == 8


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="--batch-size"):
        engine.parse_args(
            ["--input", "x", "--dataset", "imdb", "--dry-run", "--batch-size", "0"]
        )


def test_report_records_the_batch_size(tmp_path):
    import generate_sa_mr

    source = tmp_path / "s.jsonl"
    source.write_text(
        json.dumps(
            {
                "source_id": "sst2-test-000001",
                "text": "The director was brilliant and the cast delivered outstanding work here.",
                "label": "positive",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "o.jsonl"

    class Gen:
        def __init__(self, **kwargs):
            pass

        def generate(self, messages, seed):
            prompt = messages[-1]["content"]
            return json.dumps({"text": prompt.split("Original text:\n")[1].split("\n\n")[0].replace(" was ", " is ")})

    code = generate_sa_mr.main(
        [
            "--input", str(source), "--dataset", "sst2", "--output", str(output),
            "--mrs", "sa_tense_shift", "--num-samples", "1", "--seed", "42",
            "--label-strategy", "source", "--offline", "--batch-size", "4",
        ],
        generator_factory=lambda **kwargs: Gen(**kwargs),
        validator_factory=lambda args, gen, ref: {
            "generator": ScriptedValidator(default="positive", validator_id="generator")
        },
    )
    assert code == 0
    report = json.loads(Path(str(output) + ".report.json").read_text(encoding="utf-8"))
    assert report["generation_parameters"]["batch_size"] == 4
    assert report["generation_parameters"]["verifier_temperature"] == 0.0
