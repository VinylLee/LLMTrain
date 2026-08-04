from pathlib import Path
import sys
import json

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_batch_experiments as runner
from run_batch_experiments import (
    build_run_signature,
    build_yaml,
    enforce_stage2_training_gate,
    initialize_existing_cohort_metadata,
    validate_cohort_artifacts,
)


def experiment(mode="none", seed=42):
    return {"name": "pilot_seed42", "task_type": "nli", "train_data": "x",
            "target": 4413, "seed": seed, "mr_instruction_mode": mode,
            "eval_strategy": "steps", "eval_steps": 50,
            "ft_params": {"lr": 3e-4, "epochs": 3, "rank": 8,
                          "batch": 4, "grad_accum": 8}}


def test_pilot_yaml_has_eval_and_seeds(tmp_path):
    text = build_yaml(experiment(), "model", "gemma", tmp_path)
    assert f"dataset_dir: {(runner.WORK_DIR / 'data').as_posix()}" in text
    assert "eval_dataset: pilot_seed42_val" in text
    assert "eval_strategy: \"steps\"" in text
    assert "seed: 42" in text and "data_seed: 42" in text


def test_exploratory_yaml_uses_max_steps_instead_of_full_epochs(tmp_path):
    exp = experiment()
    exp["ft_params"]["max_steps"] = 20
    text = build_yaml(exp, "model", "gemma", tmp_path)
    assert "max_steps: 20" in text
    assert "num_train_epochs:" not in text


def test_yaml_resumes_only_from_latest_complete_checkpoint(tmp_path):
    model_dir = tmp_path / "pilot_seed42" / "model"
    valid = model_dir / "checkpoint-300"
    partial = model_dir / "checkpoint-350"
    valid.mkdir(parents=True)
    partial.mkdir()
    for name in (
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (valid / name).write_bytes(b"complete")
    (valid / "trainer_state.json").write_text(
        json.dumps({"global_step": 300}), encoding="utf-8"
    )
    (partial / "trainer_state.json").write_text(
        json.dumps({"global_step": 350}), encoding="utf-8"
    )

    text = build_yaml(experiment(), "model", "gemma", tmp_path)
    assert f"resume_from_checkpoint: {valid.as_posix()}" in text
    assert partial.as_posix() not in text


def test_run_cmd_passes_argv_and_environment_without_shell(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    assert runner.run_cmd(["python", "script.py", "value with spaces"],
                          "cross-platform", cwd=tmp_path, env=env)
    assert captured["cmd"] == ["python", "script.py", "value with spaces"]
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["env"] == env
    assert "shell" not in captured["kwargs"]


def test_workspace_relative_uses_portable_posix_separators(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path)
    path = tmp_path / "data" / "ft_datasets" / "sampled.json"
    assert runner.workspace_relative(path) == "data/ft_datasets/sampled.json"


def test_load_config_uses_utf8_on_windows(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"label": "EXPLORATORY — 人工验证未通过"}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert runner.load_config(config_path)["label"] == "EXPLORATORY — 人工验证未通过"


def test_run_signature_is_deterministic_and_sensitive():
    config = {"model": "model", "template": "gemma", "generation": {"do_sample": False}}
    first = build_run_signature(experiment(), config, "abc", "data", "manifest", {"test": "hash"})
    second = build_run_signature(experiment(), config, "abc", "data", "manifest", {"test": "hash"})
    assert first == second
    changed = build_run_signature(experiment("pair_only"), config, "abc", "data", "manifest", {"test": "hash"})
    assert changed["run_signature"] != first["run_signature"]
    gated = build_run_signature(
        experiment(), config, "abc", "data", "manifest", {"test": "hash"},
        {"sha256": "gate-decision"},
    )
    assert gated["run_signature"] != first["run_signature"]


def test_run_signature_covers_all_seed_roles():
    payload = build_run_signature(experiment(seed=44), {
        "model": "m",
        "template": "t",
        "instruction_template_version": 2,
    },
                                  "sha", "d", "m", {})["run_signature_payload"]
    assert set(payload["seeds"]) == {"sampling", "split", "shuffle", "trainer", "data"}
    assert set(payload["seeds"].values()) == {44}
    assert payload["instruction_template"] == {"version": 2}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def cohort_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "")
    source_rows = [
        {"pair_id": "p1", "premise": "a", "hypothesis": "b", "label": 0, "mr_id": "none"},
        {"pair_id": "p1", "premise": "a", "hypothesis": "c", "label": 2, "mr_id": "adding_contradiction"},
        {"pair_id": "p2", "premise": "d", "hypothesis": "e", "label": 1, "mr_id": "none"},
    ]
    source_path = tmp_path / "source.jsonl"
    write_jsonl(source_path, source_rows)
    cohort_dir = tmp_path / "data" / "ft_datasets" / "_cohorts" / "pilot" / "seed_42"
    sampled_path = cohort_dir / "sampled.json"
    write_jsonl(sampled_path, source_rows[:2])
    (cohort_dir / "sampled.pair_ids.json").write_text(
        json.dumps({"seed": 42, "target": 2, "pair_ids": ["p1"]}),
        encoding="utf-8",
    )
    exp = {
        "name": "pilot_none_seed42",
        "cohort_id": "pilot",
        "seed": 42,
        "target": 2,
        "stratify": False,
        "train_data": "source.jsonl",
        "mr_instruction_mode": "none",
    }
    return exp, cohort_dir, sampled_path


def test_cohort_reuse_requires_metadata(tmp_path, monkeypatch):
    exp, cohort_dir, _ = cohort_fixture(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="cohort 复用缺少审计文件"):
        validate_cohort_artifacts(exp, cohort_dir)


def test_cohort_metadata_initialization_and_tamper_detection(tmp_path, monkeypatch):
    exp, cohort_dir, sampled_path = cohort_fixture(tmp_path, monkeypatch)
    result = initialize_existing_cohort_metadata(exp, cohort_dir, "abc123")
    assert result["selected_group_count"] == 1
    assert (cohort_dir / "cohort_meta.json").exists()
    assert (cohort_dir / "sampling_report.json").exists()
    validate_cohort_artifacts(exp, cohort_dir)

    rows = [json.loads(line) for line in sampled_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["hypothesis"] = "tampered"
    write_jsonl(sampled_path, rows)
    with pytest.raises(ValueError, match="sampled group 未完整复用 source group"):
        validate_cohort_artifacts(exp, cohort_dir)


def test_stage2_gate_requires_pass_open_and_matching_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path)
    conversion = {
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "converted_train_sha256": "train",
        "converted_val_sha256": "val",
    }
    exp = {
        "seed": 42,
        "cohort_id": "pilot",
        "mr_instruction_mode": "none",
    }
    report = {
        "stage2_status": "PASS",
        "training_gate": "OPEN",
        "automated_checks_status": "PASS",
        "seed": 42,
        "cohort_id": "pilot",
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "instruction_template_version": 2,
        "mode_reports": {
            "none": {
                "converted_train_sha256": "train",
                "converted_val_sha256": "val",
            },
        },
    }
    report_path = tmp_path / "stage2.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    config = {
        "instruction_template_version": 2,
        "stage2_gate": {"required": True, "report": "stage2.json"},
    }
    decision = enforce_stage2_training_gate(config, exp, conversion)
    assert decision["decision"] == "STAGE2_PASS"
    assert decision["stage2_report"]["stage2_status"] == "PASS"
    assert decision["stage2_report"]["training_gate"] == "OPEN"
    assert decision["sha256"]

    report["training_gate"] = "BLOCKED"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Stage 2 training gate blocked"):
        enforce_stage2_training_gate(config, exp, conversion)


def test_stage2_gate_allows_narrow_audited_exploratory_override(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path)
    conversion = {
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "converted_train_sha256": "train",
        "converted_val_sha256": "val",
    }
    exp = {
        "seed": 42,
        "cohort_id": "pilot",
        "mr_instruction_mode": "none",
        "ft_params": {"max_steps": 20},
    }
    report = {
        "stage2_status": "FAIL",
        "training_gate": "BLOCKED",
        "automated_checks_status": "PASS",
        "seed": 42,
        "cohort_id": "pilot",
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "instruction_template_version": 2,
        "mode_reports": {
            mode: {
                "converted_train_sha256": "train",
                "converted_val_sha256": "val",
            }
            for mode in ("none", "pair_only")
        },
    }
    report_path = tmp_path / "stage2.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    evidence_path = tmp_path / "quick_audit.md"
    evidence_path.write_text("quick audit evidence", encoding="utf-8")
    config = {
        "instruction_template_version": 2,
        "output_root": "output/exploratory",
        "research_status": {
            "classification": "exploratory_pilot",
            "human_validation_status": "QUICK_AUDIT_ONLY_NOT_PASSED",
            "confirmatory_use_allowed": False,
        },
        "stage2_gate": {
            "required": True,
            "report": "stage2.json",
            "exploratory_override": {
                "enabled": True,
                "evidence": "quick_audit.md",
                "evidence_sha256": runner.file_sha256(evidence_path),
                "allowed_seeds": [42],
                "allowed_modes": ["none"],
                "required_output_root": "output/exploratory",
                "max_training_steps": 20,
            },
        },
    }

    decision = enforce_stage2_training_gate(config, exp, conversion)
    assert decision["decision"] == "EXPLORATORY_OVERRIDE"
    assert decision["human_validation_status"] == "QUICK_AUDIT_ONLY_NOT_PASSED"
    assert decision["confirmatory_use_allowed"] is False
    assert decision["stage2_report"]["stage2_status"] == "FAIL"
    assert decision["stage2_report"]["training_gate"] == "BLOCKED"
    assert decision["evidence"]["sha256"] == runner.file_sha256(evidence_path)

    with pytest.raises(RuntimeError, match="mode 'pair_only' is outside exploratory scope"):
        enforce_stage2_training_gate(
            config,
            {**exp, "mr_instruction_mode": "pair_only"},
            conversion,
        )

    with pytest.raises(RuntimeError, match="exploratory output root mismatch"):
        enforce_stage2_training_gate(
            config,
            exp,
            conversion,
            output_root=tmp_path / "output" / "wrong",
        )

    with pytest.raises(RuntimeError, match="exploratory max_steps invalid"):
        enforce_stage2_training_gate(
            config,
            {**exp, "ft_params": {"max_steps": 21}},
            conversion,
        )

    evidence_path.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence SHA-256 mismatch"):
        enforce_stage2_training_gate(config, exp, conversion)


def test_exploratory_override_allows_bounded_epoch_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path)
    conversion = {
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "converted_train_sha256": "train",
        "converted_val_sha256": "val",
    }
    exp = {
        "seed": 42,
        "cohort_id": "pilot",
        "mr_instruction_mode": "pair_operation",
        "ft_params": {"epochs": 3.0},
    }
    report = {
        "stage2_status": "FAIL",
        "training_gate": "BLOCKED",
        "automated_checks_status": "PASS",
        "seed": 42,
        "cohort_id": "pilot",
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "instruction_template_version": 2,
        "mode_reports": {
            "pair_operation": {
                "converted_train_sha256": "train",
                "converted_val_sha256": "val",
            },
        },
    }
    (tmp_path / "stage2.json").write_text(json.dumps(report), encoding="utf-8")
    evidence_path = tmp_path / "quick_audit.md"
    evidence_path.write_text("quick audit evidence", encoding="utf-8")
    config = {
        "instruction_template_version": 2,
        "output_root": "output/exploratory-full",
        "research_status": {
            "classification": "exploratory_pilot",
            "human_validation_status": "QUICK_AUDIT_ONLY_NOT_PASSED",
            "confirmatory_use_allowed": False,
        },
        "stage2_gate": {
            "required": True,
            "report": "stage2.json",
            "exploratory_override": {
                "enabled": True,
                "evidence": "quick_audit.md",
                "evidence_sha256": runner.file_sha256(evidence_path),
                "allowed_seeds": [42],
                "allowed_modes": ["pair_operation"],
                "required_output_root": "output/exploratory-full",
                "max_training_epochs": 3.0,
            },
        },
    }

    decision = enforce_stage2_training_gate(config, exp, conversion)
    assert decision["decision"] == "EXPLORATORY_OVERRIDE"
    assert decision["max_training_epochs"] == 3.0
    assert "max_training_steps" not in decision

    with pytest.raises(RuntimeError, match="exploratory epochs invalid"):
        enforce_stage2_training_gate(
            config,
            {**exp, "ft_params": {"epochs": 3.1}},
            conversion,
        )
    with pytest.raises(RuntimeError, match="exploratory epochs invalid"):
        enforce_stage2_training_gate(
            config,
            {**exp, "ft_params": {"epochs": 3.0, "max_steps": 20}},
            conversion,
        )


def test_exploratory_override_rejects_false_human_validation_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORK_DIR", tmp_path)
    evidence_path = tmp_path / "audit.md"
    evidence_path.write_text("audit", encoding="utf-8")
    report = {
        "stage2_status": "FAIL",
        "training_gate": "BLOCKED",
        "automated_checks_status": "PASS",
        "seed": 42,
        "cohort_id": "pilot",
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "instruction_template_version": 2,
        "mode_reports": {
            "none": {
                "converted_train_sha256": "train",
                "converted_val_sha256": "val",
            },
        },
    }
    (tmp_path / "stage2.json").write_text(json.dumps(report), encoding="utf-8")
    config = {
        "instruction_template_version": 2,
        "output_root": "output/exploratory",
        "research_status": {
            "classification": "exploratory_pilot",
            "human_validation_status": "PASS",
            "confirmatory_use_allowed": False,
        },
        "stage2_gate": {
            "required": True,
            "report": "stage2.json",
            "exploratory_override": {
                "enabled": True,
                "evidence": "audit.md",
                "evidence_sha256": runner.file_sha256(evidence_path),
                "allowed_seeds": [42],
                "allowed_modes": ["none"],
                "required_output_root": "output/exploratory",
                "max_training_steps": 20,
            },
        },
    }
    conversion = {
        "data_signature": "data",
        "manifest_hash": "manifest",
        "ordered_sample_signature": "order",
        "converted_train_sha256": "train",
        "converted_val_sha256": "val",
    }
    with pytest.raises(RuntimeError, match="QUICK_AUDIT_ONLY_NOT_PASSED"):
        enforce_stage2_training_gate(
            config,
            {
                "seed": 42,
                "cohort_id": "pilot",
                "mr_instruction_mode": "none",
                "ft_params": {"max_steps": 20},
            },
            conversion,
        )


def test_exploratory_summary_has_non_confirmatory_banner(tmp_path):
    output_root = tmp_path / "output"
    exp_dir = output_root / "pilot_seed42"
    (exp_dir / "tests" / "original").mkdir(parents=True)
    (exp_dir / "experiment_meta.json").write_text(
        json.dumps({
            "task_type": "nli",
            "research_status": {
                "classification": "exploratory_pilot",
                "human_validation_status": "QUICK_AUDIT_ONLY_NOT_PASSED",
                "confirmatory_use_allowed": False,
            },
        }),
        encoding="utf-8",
    )
    write_jsonl(
        exp_dir / "tests" / "original" / "mnlim.jsonl",
        [{"correct": True}],
    )
    summary_path = output_root / "SUMMARY.md"
    runner.collect_summary(output_root, summary_path)
    summary = summary_path.read_text(encoding="utf-8")
    assert "EXPLORATORY PILOT" in summary
    assert "Human Validation 未通过" in summary
    assert "不得作为确认性证据" in summary
