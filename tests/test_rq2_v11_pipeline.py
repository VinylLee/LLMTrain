"""CPU-only safety gates for the versioned RQ2 evaluation queue."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from RQ2 import run_rq2_v11 as pipeline


def test_config_keeps_scientific_base_and_scope():
    config, base = pipeline.load_config(pipeline.DEFAULT_CONFIG)
    assert config["seeds"] == [42, 43, 44]
    assert config["dataset"] == "snli"
    assert len(base["modes"]) == 14
    assert base["instruction_template_version"] == 4
    assert not base["allow_partial_control_coverage"]
    assert config["output_root"] != config["adapter_root"]
    assert base["training"]["batch"] == 4
    assert base["training"]["grad_accum"] == 8


def test_gpu_selection_requires_budget_plus_spare_and_prefers_free_memory():
    devices = [dict(index=0, free_mib=19000, utilization=0),
               dict(index=1, free_mib=22618, utilization=100),
               dict(index=2, free_mib=24000, utilization=80)]
    assert pipeline.choose_gpu(devices, 18432, 4096)["index"] == 2
    assert pipeline.choose_gpu(devices[:1], 16384, 4096) is None
    assert pipeline.choose_gpu([], 1, 1) is None


def test_identity_allows_only_pre_gpu_resource_preparation_revisions():
    old = dict(base_config_sha256="base", test_sha256="test", training_sha256="train", run_config_sha256="first")
    current = dict(old, run_config_sha256="second")
    pipeline.validate_run_identity(old, current, gpu_started=False)
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.validate_run_identity(old, current, gpu_started=True)
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.validate_run_identity(old, dict(current, test_sha256="changed"), gpu_started=False)
    with pytest.raises(ValueError, match="identity changed"):
        pipeline.validate_run_identity(old, dict(old, code_sha256={"test_script": "changed"}), gpu_started=True)


def test_isolated_registry_points_to_exact_v4_payloads(tmp_path):
    registry = pipeline.registry_entries(tmp_path / "converted_v4", 43, ["none", "pair_wrong_relation"])
    assert len(registry) == 4
    entry = registry["rq2_snli_none_seed43"]
    assert entry["file_name"] == str((tmp_path / "converted_v4/rq2_snli_none_seed43/full_train.json").resolve())
    assert entry["columns"] == {"prompt": "instruction", "query": "input", "response": "output"}
    assert registry["rq2_snli_none_seed43_val"]["file_name"].endswith("full_val.json")


def test_checkpoint_recovery_skips_latest_incomplete_checkpoint(tmp_path):
    older = tmp_path / "checkpoint-100"
    latest = tmp_path / "checkpoint-200"
    for directory in (older, latest):
        directory.mkdir()
        pipeline.save_json(directory / "trainer_state.json", {"global_step": int(directory.name.split("-")[-1])})
    for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (older / name).write_bytes(b"cpu fixture")
    assert pipeline.complete_checkpoint(tmp_path) == older
    (older / "rng_state.pth").unlink()
    assert pipeline.complete_checkpoint(tmp_path) is None


def test_invalid_attempts_are_preserved_not_deleted(tmp_path):
    output = tmp_path / "snli.jsonl"
    output.write_text("partial prediction", encoding="utf-8")
    destination = Path(pipeline.quarantine([output], tmp_path / "failed_attempts", "partial write"))
    assert not output.exists()
    assert (destination / "snli.jsonl").read_text() == "partial prediction"
    assert pipeline.read_json(destination / "reason.json")["reason"] == "partial write"


def test_gpu_worker_admission_requeues_before_model_load(monkeypatch):
    import types
    fake_cuda = types.SimpleNamespace(device_count=lambda: 1, mem_get_info=lambda _: (1024, 46068 * 1024**2))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    with pytest.raises(SystemExit) as exc:
        pipeline.worker("inference", 16384, 4096, [])
    assert exc.value.code == 75


def test_seed42_review_gate_never_prepares_or_trains_later_seeds():
    class FakePipeline:
        cfg = {"seeds": [42, 43, 44]}
        base = {"modes": ["none", "full_oracle"]}

        def __init__(self):
            self.prepared = []
            self.evaluated = []
            self.events = []

        def preflight(self):
            pass

        def prepare_seed(self, seed):
            self.prepared.append(seed)

        def train(self, *args, **kwargs):
            raise AssertionError("No training allowed through seed42 review gate")

        def evaluate(self, seed, mode, **kwargs):
            self.evaluated.append((seed, mode))

        def summarize(self, seed):
            assert seed == 42

        def event(self, status, **kwargs):
            self.events.append(status)

    fake = FakePipeline()
    pipeline.Pipeline.run(fake, seed42_only=True)
    assert fake.prepared == [42]
    assert all(seed == 42 for seed, _ in fake.evaluated)
    assert fake.events[-1] == "seed42_review_gate"


def examples():
    source = dict(idx=0, pair_id="a", premise="A dog runs.", hypothesis="An animal runs.",
                  mr_id="none", mr_type="inv", label=0, is_source=True)
    follow = dict(idx=1, pair_id="a", premise="A dog runs.", hypothesis="No animal runs.",
                  mr_id="composite_flip", mr_type="flip", component_mrs=["pronoun_substitution", "negation_flip"],
                  label=2, is_source=False)
    return [source, follow]


def predictions(items):
    return [{**r, "gold": pipeline.LABELS[r["label"]], "pred": pipeline.LABELS[r["label"]],
             "correct": True, "dataset": "snli", "test_type": "merged"} for r in items]


def test_test_schema_pairing_provenance_and_gold_relation():
    report = pipeline.validate_test(examples(), [])
    assert report["sources"] == report["followups"] == 1
    assert not report["semantic_human_validation_claimed"]
    invalid = examples()
    invalid[1]["label"] = 1
    with pytest.raises(ValueError, match="Gold relation"):
        pipeline.validate_test(invalid, [])
    with pytest.raises(ValueError, match="overlap"):
        pipeline.validate_test(examples(), examples())
    with pytest.raises(ValueError, match="exactly one source"):
        pipeline.validate_test(examples()[1:], [])


def test_representative_smoke_covers_mrs_and_preserves_whole_groups():
    items = examples()
    items += [dict(items[0], pair_id="b", idx=2),
              dict(items[1], pair_id="b", idx=3, mr_id="negation_flip", component_mrs=None)]
    selected = pipeline.representative_smoke(items)
    assert {r["mr_id"] for r in selected} == {r["mr_id"] for r in items}
    assert selected == pipeline.representative_smoke(items)
    assert all(sum(r["is_source"] for r in selected if r["pair_id"] == group) == 1
               for group in {r["pair_id"] for r in selected})


def test_valid_predictions_require_correct_denominator(tmp_path):
    output = tmp_path / "predictions.jsonl"
    pipeline.write_rows(output, predictions(examples()))
    report = pipeline.validate_predictions(output, examples())
    assert report["rows"] == 2
    assert report["invalid_predictions"] == 0


@pytest.mark.parametrize("failure", ["missing", "text", "target", "error", "correct", "dataset"])
def test_predictions_fail_closed(tmp_path, failure):
    actual = predictions(examples())
    if failure == "missing":
        actual.pop()
    elif failure == "text":
        actual[1]["premise"] = "changed"
    elif failure == "target":
        actual[1]["gold"] = "neutral"
    elif failure == "error":
        actual[1]["pred"] = "ERROR"
    elif failure == "correct":
        actual[1]["correct"] = False
    else:
        actual[1]["dataset"] = "sick"
    output = tmp_path / "predictions.jsonl"
    pipeline.write_rows(output, actual)
    with pytest.raises(ValueError):
        pipeline.validate_predictions(output, examples())


def adapter_fixture(tmp_path):
    config, base = pipeline.load_config(pipeline.DEFAULT_CONFIG)
    exp = tmp_path / "rq2_snli_none_seed42"
    (exp / "model").mkdir(parents=True)
    (exp / "model/adapter_model.safetensors").write_bytes(b"fake adapter for CPU test")
    pipeline.save_json(exp / "model/adapter_config.json", {"r": 8})
    pipeline.save_json(exp / "model/trainer_state.json", {"global_step": 480, "max_steps": 480, "epoch": 3.0})
    pipeline.save_json(exp / "experiment_meta.json", {
        "seed": 42, "mode": "none", "model": base["model"]["hub_id"], "smoke": False,
        "mr_design": {**pipeline.mode_design_meta("none"), "template_version": 4},
        "training": base["training"], "train_input": base["data"]["training_input"],
    })
    return exp, base


def test_adapter_must_exist_match_identity_and_finish_training(tmp_path):
    exp, base = adapter_fixture(tmp_path)
    assert pipeline.validate_adapter(exp, base, 42, "none")["epoch"] == 3.0
    with pytest.raises(ValueError, match="identity"):
        pipeline.validate_adapter(exp, base, 43, "none")
    pipeline.save_json(exp / "model/trainer_state.json", {"global_step": 20, "max_steps": 480, "epoch": 0.1})
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.validate_adapter(exp, base, 42, "none")


def test_missing_adapter_cannot_fall_back_to_base_model(tmp_path):
    exp, base = adapter_fixture(tmp_path)
    (exp / "model/adapter_model.safetensors").unlink()
    with pytest.raises(ValueError, match="Missing adapter"):
        pipeline.validate_adapter(exp, base, 42, "none")


def test_source_only_groups_not_in_paired_denominator(tmp_path):
    items = examples() + [dict(examples()[0], idx=2, pair_id="source_only")]
    assert pipeline.validate_test(items, [])["source_only_groups"] == 1
    output = tmp_path / "predictions.jsonl"
    pipeline.write_rows(output, predictions(items))
    assert pipeline.validate_predictions(output, items)["joint_diagnostics"]["source_only_groups"] == 1
