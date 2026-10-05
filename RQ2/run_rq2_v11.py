#!/usr/bin/env python3
"""Versioned SNLI evaluation and serial seed completion on a shared GPU.

No instruction construction or metric definition is duplicated here. CPU
audits precede each seed. GPU workers see exactly one UUID, have an allocator
budget, and start only when that budget plus a spare margin is available.
Failures stop the queue; exit code alone never marks a prediction complete.
"""

import argparse
import collections
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from RQ2.run_rq2_snli import build_yaml, experiments  # noqa: E402
from RQ2.scripts.summarize_rq2_snli import summarize_one  # noqa: E402
from convert_nli_to_ft import (  # noqa: E402
    build_pair_groups, canonical_sha256, compute_data_signature,
    compute_ordered_sample_signature, flatten_groups, load_split_manifest,
    sort_samples_by_stable_key, validate_manifest_for_groups,
)
from mr_instruction_design import mode_design_meta  # noqa: E402
from project_runtime import build_subprocess_env, resolve_model_reference  # noqa: E402
from metamorphic_metrics import compute_joint_correctness, compute_msr  # noqa: E402

DEFAULT_CONFIG = "RQ2/configs/rq2_snli_config_v4_test_v1_1.json"
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
INPUT_FIELDS = ("idx", "premise", "hypothesis", "mr_id", "pair_id", "label", "mr_type", "is_source")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(p)


def sha256(p):
    digest = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(p):
    with Path(p).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(p, items):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_config(p):
    cfg = read_json(path(p))
    base = read_json(path(cfg["base_config"]))
    require(cfg["dataset"] == "snli" and cfg["seeds"] == [42, 43, 44], "Scope must remain SNLI / 42,43,44")
    require(base["instruction_template_version"] == 4 and len(base["modes"]) == 14, "Expected v4 / 14 modes")
    require(base["require_composite_provenance"] and not base["allow_partial_control_coverage"], "Strict controls required")
    require(path(cfg["output_root"]) != path(cfg["adapter_root"]), "Evaluation root must not overwrite model root")
    require(path(cfg["test_snapshot_root"]) != ROOT / "RQ2/data/test", "Keep old snapshot untouched")
    require(path(cfg["adapter_root"]) == path(base["output_root"]), "Adapter root must match formal base config")
    require(base["training"]["max_new_tokens"] == 10, "Legacy test script uses max_new_tokens=10")
    require(all(cfg["gpu_memory"][k] > 0 for k in ("inference_budget_mib", "training_budget_mib", "spare_mib", "poll_seconds")), "Invalid memory budget")
    return cfg, base


def validate_run_identity(previous, current, gpu_started):
    """Permit preparation-only scheduling revisions, never a mixed GPU run."""
    if previous == current:
        return
    scientific_keys = ("base_config_sha256", "test_sha256", "training_sha256")
    require(not gpu_started and all(previous[k] == current[k] for k in scientific_keys),
            "Run identity changed; do not mix versions")


def code_identity():
    files = (Path(__file__).resolve(), ROOT / "scripts/mr_instruction_design.py",
             ROOT / "scripts/convert_nli_to_ft.py", ROOT / "scripts/test_mettrain_experiment.py",
             ROOT / "scripts/metamorphic_metrics.py", ROOT / "RQ2/scripts/convert_snli_rq2.py",
             ROOT / "RQ2/run_rq2_snli.py")
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def model_identity(model):
    directory = path(model).resolve()
    require(directory.is_dir(), "Offline base model must exist locally")
    names = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json")
    require((directory / "config.json").is_file() and (directory / "tokenizer_config.json").is_file(), "Missing model/tokenizer metadata")
    return {"resolved_snapshot": str(directory), "metadata_sha256": {n: sha256(directory / n) for n in names if (directory / n).exists()}}


def registry_entries(converted_dir, seed, modes):
    registry = {}
    for mode in modes:
        name = f"rq2_snli_{mode}_seed{seed}"
        for suffix, filename in (("", "full_train.json"), ("_val", "full_val.json")):
            registry[name + suffix] = {"file_name": str((Path(converted_dir) / name / filename).resolve()),
                                      "formatting": "alpaca", "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
    return registry


def complete_checkpoint(model_dir):
    checkpoints = sorted(Path(model_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]), reverse=True)
    for candidate in checkpoints:
        if not all((candidate / name).is_file() for name in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")):
            continue
        try:
            state = read_json(candidate / "trainer_state.json")
        except (ValueError, OSError):
            continue
        if state.get("global_step") == int(candidate.name.split("-")[-1]) and state["global_step"] > 0:
            return candidate
    return None


def quarantine(files, destination, reason):
    destination = Path(destination) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for source in files:
        source = Path(source)
        if source.exists():
            target = destination / source.name
            require(not target.exists(), "Quarantine filename collision")
            shutil.move(str(source), str(target))
            moved.append({"original": str(source), "preserved": str(target)})
    save_json(destination / "reason.json", {"time": stamp(), "reason": reason, "files": moved})
    return str(destination)


def choose_gpu(devices, budget, spare):
    eligible = [d for d in devices if d["free_mib"] >= budget + spare]
    return max(eligible, key=lambda d: (d["free_mib"], -d["utilization"], -d["index"])) if eligible else None


def gpu_snapshot():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    devices = []
    for line in result.strip().splitlines():
        index, uuid, total, free, utilization = [item.strip() for item in line.split(",")]
        devices.append(dict(index=int(index), uuid=uuid, total_mib=int(total), free_mib=int(free), utilization=int(utilization)))
    return devices


def validate_test(items, training):
    require(bool(items), "Empty test set")
    grouped = collections.defaultdict(list)
    for r in items:
        require(all(k in r for k in INPUT_FIELDS), "Incomplete test schema")
        require(type(r["label"]) is int and r["label"] in LABELS, "Invalid label")
        require(type(r["is_source"]) is bool and r["pair_id"] not in (None, ""), "Invalid pairing")
        require(bool(r["premise"].strip()) and bool(r["hypothesis"].strip()), "Empty text")
        require((r["mr_id"] == "none") == r["is_source"], "Source/MR conflict")
        grouped[str(r["pair_id"])].append(r)
    for members in grouped.values():
        sources = [r for r in members if r["is_source"]]
        require(len(sources) == 1, "Test group does not have exactly one source")
        y = sources[0]["label"]
        for r in members:
            if r["is_source"]:
                continue
            relation = r["mr_type"]
            require(relation in ("inv", "flip", "neutral"), "Unknown relation")
            require((relation == "inv" and r["label"] == y) or
                    (relation == "flip" and y == 0 and r["label"] == 2) or
                    (relation == "neutral" and y == 0 and r["label"] == 1), "Gold relation conflict")
            if r["mr_id"].startswith("composite_"):
                require(isinstance(r.get("component_mrs"), list) and bool(r["component_mrs"]), "Missing test composite trace")
    text_key = lambda r: (r["premise"].strip(), r["hypothesis"].strip())
    overlap = {text_key(r) for r in training} & {text_key(r) for r in items}
    require(not overlap, f"Exact train/test text overlap: {len(overlap)}")
    return {
        "rows": len(items), "sources": sum(r["is_source"] for r in items),
        "followups": sum(not r["is_source"] for r in items), "groups": len(grouped),
        "source_only_groups": sum(all(r["is_source"] for r in g) for g in grouped.values()),
        "mr_counts": dict(collections.Counter(r["mr_id"] for r in items)),
        "exact_train_test_overlap": 0, "gold_relation_conflicts": 0,
        "semantic_human_validation_claimed": False,
    }


def representative_smoke(items):
    """Whole groups: one of every MR plus the 32 longest follow-up inputs."""
    grouped = collections.defaultdict(list)
    selected = set()
    seen = set()
    for r in items:
        grouped[str(r["pair_id"])].append(r)
        if r["mr_id"] not in seen:
            selected.add(str(r["pair_id"]))
            seen.add(r["mr_id"])
    longest = sorted((r for r in items if not r["is_source"]),
                     key=lambda r: len(r["premise"]) + len(r["hypothesis"]), reverse=True)[:32]
    selected.update(str(r["pair_id"]) for r in longest)
    return [r for r in items if str(r["pair_id"]) in selected]


def validate_predictions(p, expected):
    actual = rows(p)
    require(len(actual) == len(expected), f"Prediction count mismatch: {len(actual)} != {len(expected)}")
    for i, (r, original) in enumerate(zip(actual, expected)):
        require(all(r.get(k) == original[k] for k in INPUT_FIELDS), f"Input changed at row {i}")
        require(r.get("dataset") == "snli" and r.get("test_type") == "merged", "Wrong evaluation identity")
        require(r.get("pred") in LABELS.values(), f"Invalid/ERROR prediction at row {i}: {r.get('pred')}")
        require(r.get("gold") == LABELS[original["label"]], f"Gold changed at row {i}")
        require(type(r.get("correct")) is bool and r["correct"] == (r["pred"] == r["gold"]), f"Incorrect correctness field at {i}")
    joint = compute_joint_correctness(actual)
    require(joint["overall"]["total"] == sum(not r["is_source"] for r in expected), "Joint denominator mismatch")
    return {"rows": len(actual), "invalid_predictions": 0, "sha256": sha256(p), "joint_diagnostics": joint["diagnostics"]}


def validate_conversion(directory, cfg, sampled, manifest, seed, mode):
    report = read_json(directory / "conversion_report.json")
    groups = build_pair_groups(sampled, ["sampled"] * len(sampled))
    train_ids, val_ids = validate_manifest_for_groups(manifest, groups, seed, cfg["data"]["val_ratio"], compute_data_signature(sampled))
    for group in groups:
        for sample in group["samples"]:
            sample["_group_key"] = group["group_key"]
    group_map = {g["group_key"]: g for g in groups}
    train = sort_samples_by_stable_key(flatten_groups([group_map[k] for k in train_ids]))
    val = sort_samples_by_stable_key(flatten_groups([group_map[k] for k in val_ids]))
    require(report["seed"] == seed and report["canonical_mode"] == mode and report["instruction_template_version"] == 4, "Conversion identity mismatch")
    require(report["mr_design"] == {**mode_design_meta(mode), "template_version": 4}, "Design metadata mismatch")
    require(report["data_signature"] == compute_data_signature(sampled) and report["manifest_sha256"] == manifest["sha256"], "Cohort/manifest mismatch")
    require(report["ordered_sample_signature"] == compute_ordered_sample_signature(train, val), "Sample order mismatch")
    for filename, key in (("full_train.json", "train_json_sha256"), ("full_val.json", "val_json_sha256")):
        require(canonical_sha256((directory / filename).read_text(encoding="utf-8")) == report[key], "Converted payload hash mismatch")
    require(len(rows(directory / "full_train.json")) == len(train) == report["train_sample_count"], "Train count mismatch")
    require(len(rows(directory / "full_val.json")) == len(val) == report["val_sample_count"], "Validation count mismatch")
    integrity = report["design_integrity"]
    for key in ("pair_fallback_count", "missing_operation_count", "missing_relation_count", "missing_source_label_count", "composite_operation_fallback_count", "relation_type_mismatch_count"):
        require(integrity[key] == 0, f"Design integrity failure: {key}")
    if mode == "pair_wrong_operation_relation_matched":
        for k in ("wrong_operation_relation_matched_ineligible", "wrong_operation_trace_identity_collision_count", "wrong_operation_text_unchanged_count"):
            require(integrity[k] == 0, f"Strict Operation failure: {k}")
        for k in ("strict_relation_matched_wrong_operation_coverage", "wrong_operation_same_relation_kind_rate", "wrong_operation_same_arity_rate"):
            require(integrity[k] == 1.0, f"Strict Operation failure: {k}")
    if mode == "pair_wrong_relation":
        require(integrity["wrong_relation_identity_collision_count"] == integrity["wrong_relation_text_unchanged_count"] == 0, "Wrong Relation collision")
        require(integrity["wrong_relation_deranged_rate"] == 1.0, "Wrong Relation not 100% deranged")
    require(report["require_composite_provenance"], "Provenance requirement disabled")
    return report


def validate_adapter(exp_dir, base, seed, mode, smoke=False):
    meta = read_json(exp_dir / "experiment_meta.json")
    require(meta["seed"] == seed and meta["mode"] == mode and meta["model"] == base["model"]["hub_id"], "Adapter identity mismatch")
    require(meta["smoke"] is smoke and meta["mr_design"] == {**mode_design_meta(mode), "template_version": 4}, "Adapter template/smoke mismatch")
    require(meta["training"] == base["training"] and meta["train_input"] == base["data"]["training_input"], "Adapter training definition mismatch")
    adapter = exp_dir / "model/adapter_model.safetensors"
    require(adapter.is_file() and adapter.stat().st_size > 0, f"Missing adapter: {adapter}")
    adapter_cfg = read_json(exp_dir / "model/adapter_config.json")
    require(adapter_cfg["r"] == base["training"]["rank"], "Adapter rank mismatch")
    state = read_json(exp_dir / "model/trainer_state.json")
    require(state["global_step"] == state["max_steps"] and state["global_step"] > 0, "Training incomplete")
    require(state["global_step"] == 20 if smoke else state["epoch"] >= base["training"]["epochs"] - 1e-6, "Wrong training duration")
    return {"path": str(adapter), "sha256": sha256(adapter), "global_step": state["global_step"], "epoch": state["epoch"]}


class Pipeline:
    def __init__(self, config):
        self.config_path = path(config)
        self.cfg, self.base = load_config(config)
        self.root = path(self.cfg["output_root"])
        self.model_root = path(self.cfg["adapter_root"])
        self.snapshot = path(self.cfg["test_snapshot_root"])
        self.test = self.snapshot / "mr_test_data_merged/snli.jsonl"
        self.state_path = self.root / "_progress.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {"jobs": {}, "events": []}
        self.model = resolve_model_reference(self.base["model"]["hub_id"], self.base["model"].get("local_path"))
        self.env = build_subprocess_env(cuda="", offline=True, torch_compile_disable=True, extra={"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})

    def event(self, status, **fields):
        entry = {"time": stamp(), "status": status, **fields}
        self.state["current"] = entry
        self.state["events"].append(entry)
        save_json(self.state_path, self.state)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def command(self, arguments, label, env=None, gpu_kind=None):
        original_arguments = [str(v) for v in arguments]
        if gpu_kind:
            require(code_identity() == self.inputs["code_sha256"], "Code changed during an active GPU run")
        logfile = self.root / "logs" / f"{label}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        while True:
            arguments = original_arguments
            if gpu_kind:
                budget = self.cfg["gpu_memory"][f"{gpu_kind}_budget_mib"]
                while True:
                    snapshot = gpu_snapshot()
                    device = choose_gpu(snapshot, budget, self.cfg["gpu_memory"]["spare_mib"])
                    if device:
                        break
                    self.event("waiting_for_shared_gpu", job=label, budget_mib=budget, gpus=snapshot)
                    time.sleep(self.cfg["gpu_memory"]["poll_seconds"])
                env = build_subprocess_env(cuda=device["uuid"], offline=True, torch_compile_disable=True, extra={"TORCHDYNAMO_DISABLE": "1", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false", "FORCE_TORCHRUN": "0"})
                self.event("gpu_selected", job=label, gpu=device, budget_mib=budget, spare_mib=self.cfg["gpu_memory"]["spare_mib"])
                arguments = [sys.executable, str(Path(__file__).resolve()), "--worker", gpu_kind, "--budget-mib", str(budget), "--spare-mib", str(self.cfg["gpu_memory"]["spare_mib"]), "--", *original_arguments]
            self.event("running", job=label, command=arguments, log=str(logfile))
            with logfile.open("a", encoding="utf-8") as stream:
                stream.write(f"\n{stamp()} {subprocess.list2cmdline(arguments)}\n")
                stream.flush()
                output_offset = stream.tell()
                result = subprocess.run(arguments, cwd=ROOT, env=env or self.env, stdout=stream, stderr=subprocess.STDOUT)
            if gpu_kind and result.returncode == 75:
                self.event("shared_gpu_admission_retry", job=label)
                time.sleep(self.cfg["gpu_memory"]["poll_seconds"])
                continue
            require(result.returncode == 0, f"{label} exited {result.returncode}; inspect {logfile}")
            return logfile, output_offset

    def preflight(self):
        source = path(self.cfg["test_input"])
        require(sha256(source) == self.cfg["test_sha256"], "Uploaded V1.1 test hash changed")
        if self.test.exists():
            require(sha256(self.test) == self.cfg["test_sha256"], "Versioned snapshot differs")
        else:
            self.test.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, self.test)
        self.test_rows = rows(self.test)
        self.train_rows = rows(path(self.base["data"]["training_input"]))
        report = validate_test(self.test_rows, self.train_rows)
        frozen = {
            "base_config_sha256": sha256(path(self.cfg["base_config"])),
            "run_config_sha256": sha256(self.config_path), "test_sha256": sha256(self.test),
            "training_sha256": sha256(path(self.base["data"]["training_input"])),
            "code_sha256": code_identity(),
            "base_model": model_identity(self.model),
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "llamafactory")},
        }
        manifest_path = self.root / "run_manifest.json"
        if manifest_path.exists():
            run_manifest = read_json(manifest_path)
            validate_run_identity(run_manifest["inputs"], frozen,
                                  any(e["status"] == "gpu_selected" for e in self.state["events"]))
            if run_manifest["inputs"] != frozen:
                run_manifest.setdefault("pre_gpu_config_revisions", []).append({"time": stamp(), "previous_inputs": run_manifest["inputs"], "previous_resource_policy": run_manifest["resource_policy"]})
                run_manifest["inputs"] = frozen
                run_manifest["resource_policy"] = self.cfg["gpu_memory"]
        else:
            run_manifest = {"inputs": frozen, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "created": stamp(), "scope": "SNLI; 14 modes; seed42 reevaluation -> seed43 -> seed44", "resource_policy": self.cfg["gpu_memory"], "checkpoint_steps": self.cfg["checkpoint_steps"], "scientific_training": self.base["training"]}
        run_manifest.setdefault("executions", []).append({"time": stamp(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner_sha256": sha256(Path(__file__).resolve())})
        save_json(manifest_path, run_manifest)
        self.inputs = frozen
        save_json(self.root / "audits/test_v1_1.json", report)
        self.command([sys.executable, ROOT / "RQ2/scripts/audit_snli_rq2.py", "--train", path(self.base["data"]["training_input"]), "--test", self.test, "--output", self.root / "audits/snli_preflight.json"], "cpu_source_disjoint_audit")
        require(read_json(self.root / "audits/snli_preflight.json")["passed"], "Source-disjoint audit failed")
        smoke = representative_smoke(self.test_rows)
        smoke_file = self.root / "smoke_data/mr_test_data_merged/snli.jsonl"
        if smoke_file.exists():
            require(rows(smoke_file) == smoke, "Smoke set changed")
        else:
            write_rows(smoke_file, smoke)
        self.smoke_rows = smoke
        self.event("cpu_preflight_passed", test_report=report, smoke_rows=len(smoke),
                   git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   code_sha256=frozen["code_sha256"])

    def prepare_seed(self, seed):
        cohort = path(self.base["cohorts_dir"]) / f"snli_seed{seed}"
        sampled = cohort / "sampled.json"
        manifest = cohort / "split_manifest.json"
        if not sampled.exists():
            require(seed != 42, "Do not reconstruct historical seed42 cohort")
            self.command([sys.executable, ROOT / "scripts/sample_mettrain_pairid.py", "--input", path(self.base["data"]["training_input"]), "--target", self.base["data"]["target_samples"], "--seed", seed, "--output", sampled, "--report-output", cohort / "sampling_report.json"], f"cpu_sample_seed{seed}")
        sampling = read_json(cohort / "sampling_report.json")
        require(sampling["seed"] == seed and sampling["sampled_file_sha256"] == sha256(sampled), "Sampled cohort identity mismatch")
        require(sampling["source_file_sha256"] == self.inputs["training_sha256"], "Training source changed since sampling")
        sampled_rows = rows(sampled)
        require(sampling["sample_data_signature"] == compute_data_signature(sampled_rows), "Sampled data signature mismatch")
        self.command([sys.executable, ROOT / "scripts/audit_rq2_v4_provenance.py", "--input", sampled, "--seed", seed, "--json"], f"cpu_provenance_all_seed{seed}")
        for mode in self.base["modes"]:
            name = f"rq2_snli_{mode}_seed{seed}"
            directory = path(self.base["converted_dir"]) / name
            if not (directory / "conversion_report.json").exists():
                require(seed != 42, f"Missing historical seed42 conversion: {name}")
                args = [sys.executable, ROOT / "RQ2/scripts/convert_snli_rq2.py", "--input", sampled, "--name", name, "--mode", mode, "--seed", seed, "--output-dir", self.base["converted_dir"], "--manifest", manifest, "--val-ratio", self.base["data"]["val_ratio"], "--cutoff-len", self.base["data"]["cutoff_len"], "--tokenizer-path", self.model, "--instruction-template-version", "4", "--require-composite-provenance"]
                if not manifest.exists():
                    args.append("--write-manifest")
                self.command(args, f"cpu_convert_{name}")
            validate_conversion(directory, self.base, sampled_rows, load_split_manifest(manifest), seed, mode)
        logfile, offset = self.command([sys.executable, ROOT / "scripts/audit_rq2_v4_provenance.py", "--input", sampled, "--manifest", manifest, "--seed", seed, "--val-ratio", self.base["data"]["val_ratio"], "--json"], f"cpu_provenance_train_seed{seed}")
        # The audit CLI prints JSON but does not fail on strict-control infeasibility.
        with logfile.open() as stream:
            stream.seek(offset)
            content = stream.read()
        audit = json.loads(content[content.index("{"):])
        require(audit["formal_v4_ready"], "Training provenance audit blocked")
        require(audit["strict_wrong_operation_feasibility"]["strict_relation_matched_wrong_operation_coverage"] == 1.0, "Strict control coverage is not complete")
        save_json(self.root / "audits" / f"provenance_train_seed{seed}.json", audit)
        # Inspect the actual train split (not a donor pool containing validation).
        train_input = self.root / "inspection_data" / f"seed{seed}" / "sampled.json"
        groups = build_pair_groups(sampled_rows, ["sampled"] * len(sampled_rows))
        train_ids, _ = validate_manifest_for_groups(load_split_manifest(manifest), groups, seed,
                                                  self.base["data"]["val_ratio"], compute_data_signature(sampled_rows))
        group_map = {g["group_key"]: g for g in groups}
        train_rows = sort_samples_by_stable_key(flatten_groups([group_map[k] for k in train_ids]))
        write_rows(train_input, [{k: v for k, v in r.items() if not k.startswith("_")} for r in train_rows])
        composite = next(r for r in train_rows if r["mr_id"] == "composite_flip")
        logfile, offset = self.command([sys.executable, ROOT / "scripts/inspect_rq2_instructions.py", "--input", train_input,
                                       "--pair-id", composite["pair_id"], "--mr-id", "composite_flip", "--template-version", "4",
                                       "--seed", seed, "--json"], f"cpu_instruction_inspection_seed{seed}")
        with logfile.open() as stream:
            stream.seek(offset)
            inspection = json.loads(stream.read())
        save_json(self.root / "audits" / f"instruction_inspection_seed{seed}.json", inspection)
        registry_path = self.root / "training_registry" / f"seed{seed}" / "dataset_info.json"
        expected_registry = registry_entries(path(self.base["converted_dir"]), seed, self.base["modes"])
        if registry_path.exists():
            require(read_json(registry_path) == expected_registry, "Isolated training registry changed")
        else:
            save_json(registry_path, expected_registry)
        if seed == 42:
            adapter_identities = {mode: validate_adapter(self.model_root / f"rq2_snli_{mode}_seed{seed}", self.base, seed, mode) for mode in self.base["modes"]}
            require(all(Path(read_json(self.model_root / f"rq2_snli_{mode}_seed{seed}/experiment_meta.json")["model_path"]).resolve() == Path(self.inputs["base_model"]["resolved_snapshot"]) for mode in self.base["modes"]), "Seed42 base model revision mismatch")
            save_json(self.root / "audits/seed42_adapter_inventory.json", adapter_identities)
        self.event("seed_cpu_audits_passed", seed=seed)

    def train(self, seed, mode, smoke=False):
        name = f"rq2_snli_{mode}_seed{seed}"
        destination = self.root / "training_smoke" if smoke else self.model_root
        exp_dir = destination / name
        adapter = exp_dir / "model/adapter_model.safetensors"
        conversion = path(self.base["converted_dir"]) / name / "conversion_report.json"
        registry = self.root / "training_registry" / f"seed{seed}" / "dataset_info.json"
        require(read_json(registry) == registry_entries(path(self.base["converted_dir"]), seed, self.base["modes"]), "Training registry drift")
        conversion_report = read_json(conversion)
        for filename, key in (("full_train.json", "train_json_sha256"), ("full_val.json", "val_json_sha256")):
            require(canonical_sha256((conversion.parent / filename).read_text(encoding="utf-8")) == conversion_report[key], "Training payload changed after audit")
        job_identity = {"seed": seed, "mode": mode, "smoke": smoke, "conversion_sha256": sha256(conversion), "base_config_sha256": self.inputs["base_config_sha256"], "registry_sha256": sha256(registry), "base_model": self.inputs["base_model"], "code_sha256": self.inputs["code_sha256"], "checkpoint_steps": 20 if smoke else self.cfg["checkpoint_steps"]}
        identity_path = exp_dir / "v11_training_provenance.json"
        if identity_path.exists():
            require(read_json(identity_path) == job_identity, "Cannot resume changed training job")
        if adapter.exists():
            try:
                identity = validate_adapter(exp_dir, self.base, seed, mode, smoke)
            except (ValueError, OSError) as exc:
                # Only our identified, checkpointed job may repair an interrupted save.
                require(seed != 42 and identity_path.exists() and complete_checkpoint(exp_dir / "model") is not None,
                        f"Incomplete/unidentified adapter cannot be recovered automatically: {exc}")
                preserved = quarantine([adapter, exp_dir / "model/adapter_config.json", exp_dir / "model/trainer_state.json"],
                                       self.root / "failed_attempts" / name / "final_save", str(exc))
                self.event("interrupted_final_save_preserved", job=name, preserved=preserved)
            else:
                self.event("verified_existing_adapter", job=name, smoke=smoke, adapter=identity)
                return
        require(seed != 42, "Seed42 retraining prohibited in V1.1 reevaluation")
        if not identity_path.exists():
            require(not list((exp_dir / "model").glob("checkpoint-*")), "Unidentified checkpoint; do not resume")
            if (exp_dir / "experiment_meta.json").exists():
                save_json(self.root / "audits" / f"{name}_previous_failed_meta.json", read_json(exp_dir / "experiment_meta.json"))
            save_json(identity_path, job_identity)
        metadata = {"experiment": name, "model": self.base["model"]["hub_id"], "model_path": self.model, "mode": mode, "mr_design": {**mode_design_meta(mode), "template_version": 4}, "seed": seed, "train_input": self.base["data"]["training_input"], "target_samples": self.base["data"]["target_samples"], "training": self.base["training"], "smoke": smoke, "created": stamp(), "runtime_checkpoint_steps": job_identity["checkpoint_steps"]}
        save_json(exp_dir / "experiment_meta.json", metadata)
        runtime_cfg = copy.deepcopy(self.base)
        runtime_cfg["training"]["save_steps"] = self.cfg["checkpoint_steps"]
        yaml_text = build_yaml(experiments([seed], [mode])[0], runtime_cfg, destination, smoke)
        yaml_text = yaml_text.replace(f"dataset_dir: {(ROOT / 'RQ2/data').as_posix()}\n", f"dataset_dir: {registry.parent.as_posix()}\n")
        require(f"dataset_dir: {registry.parent.as_posix()}\n" in yaml_text, "Training did not use isolated registry")
        checkpoints = sorted((exp_dir / "model").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            candidate = complete_checkpoint(exp_dir / "model")
            require(candidate is not None, "No complete checkpoint recovery state")
            yaml_text += f"resume_from_checkpoint: {candidate.as_posix()}\n"
        yaml_path = self.root / "generated_configs" / f"{name}{'_smoke' if smoke else ''}.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml_text, encoding="utf-8")
        self.command([str(yaml_path)], f"{'smoke' if smoke else 'train'}_{name}", gpu_kind="training")
        identity = validate_adapter(exp_dir, self.base, seed, mode, smoke)
        if not smoke:
            self.state["jobs"].setdefault(name, {})["finetune"] = {"status": "completed", "adapter": identity}
            old_progress_path = self.model_root / "_progress.json"
            old_progress = read_json(old_progress_path) if old_progress_path.exists() else {}
            old_progress.setdefault(name, {})["finetune"] = "completed"
            save_json(old_progress_path, old_progress)
        self.event("training_verified", job=name, smoke=smoke, adapter=identity)

    def evaluate(self, seed, mode, smoke=False, smoke_adapter=False):
        require(sha256(self.test) == self.inputs["test_sha256"], "Frozen test snapshot changed")
        name = f"rq2_snli_{mode}_seed{seed}"
        adapter_root = self.root / "training_smoke" if smoke_adapter else self.model_root
        adapter = validate_adapter(adapter_root / name, self.base, seed, mode, smoke_adapter)
        destination = self.root / ("smoke_predictions_training" if smoke_adapter else "smoke_predictions") if smoke else self.root
        expected = self.smoke_rows if smoke else self.test_rows
        data_root = self.root / "smoke_data" if smoke else self.snapshot
        prediction = destination / name / "tests/merged/snli.jsonl"
        identity = {"test_version": self.cfg["test_version"], "test_sha256": sha256(data_root / "mr_test_data_merged/snli.jsonl"), "adapter": adapter, "seed": seed, "mode": mode, "smoke": smoke, "batch_size": self.base["training"]["test_batch_size"], "max_new_tokens": 10, "base_model": self.inputs["base_model"], "code_sha256": self.inputs["code_sha256"], "packages": self.inputs["packages"]}
        eval_meta = destination / name / "evaluation_meta.json"
        if eval_meta.exists():
            require(read_json(eval_meta) == identity, "Evaluation identity changed")
        else:
            require(not prediction.exists(), "Unidentified existing predictions; refuse overwrite")
            save_json(eval_meta, identity)
            save_json(destination / name / "experiment_meta.json", read_json(adapter_root / name / "experiment_meta.json"))
        if prediction.exists():
            try:
                validate_predictions(prediction, expected)
            except (ValueError, OSError) as exc:
                preserved = quarantine([prediction], self.root / "failed_attempts" / name / "predictions", str(exc))
                self.event("invalid_prediction_attempt_preserved", job=name, preserved=preserved)
        try:
            if not prediction.exists():
                self.command([ROOT / "scripts/test_mettrain_experiment.py", "--experiment", name, "--base-model", self.model, "--lora", (adapter_root / name / "model"), "--output-root", destination, "--datasets", "snli", "--batch-size", self.base["training"]["test_batch_size"], "--merged", "--skip-original", "--skip-mr", "--merged-path", data_root], f"{'smoke_eval_training' if smoke_adapter else 'smoke_eval' if smoke else 'eval'}_{name}", gpu_kind="inference")
            validated = validate_predictions(prediction, expected)
        except (ValueError, OSError) as exc:
            if prediction.exists():
                preserved = quarantine([prediction], self.root / "failed_attempts" / name / "predictions", str(exc))
                self.event("failed_prediction_attempt_preserved", job=name, preserved=preserved)
            raise
        if not smoke:
            self.state["jobs"].setdefault(name, {})["test"] = {"status": "completed", "validation": validated, "identity": identity}
        self.event("predictions_verified", job=name, smoke=smoke, smoke_adapter=smoke_adapter, validation=validated)

    def summarize(self, seed):
        self.command([sys.executable, ROOT / "RQ2/scripts/summarize_rq2_snli.py", "--output-root", self.root], f"summarize_after_seed{seed}")
        records = []
        for name, job in sorted(self.state["jobs"].items()):
            if job.get("test", {}).get("status") != "completed":
                continue
            prediction = self.root / name / "tests/merged/snli.jsonl"
            record = summarize_one(prediction)
            identity = job["test"]["identity"]
            record.update(seed=identity["seed"], mode=identity["mode"], mr_design=mode_design_meta(identity["mode"]))
            results = rows(prediction)
            sources = [r for r in results if r["is_source"]]
            record["breakdown"] = {}
            for category, followups in (("atomic", [r for r in results if not r["is_source"] and not r["mr_id"].startswith("composite_")]), ("composite", [r for r in results if not r["is_source"] and r["mr_id"].startswith("composite_")])):
                correct = sum(r["correct"] for r in followups)
                record["breakdown"][category] = {"accuracy": {"correct": correct, "total": len(followups), "rate": 100 * correct / len(followups) if followups else None}, "msr": compute_msr(sources + followups)["overall"], "joint": compute_joint_correctness(sources + followups)["overall"]}
            records.append(record)
        aggregates = {}
        for mode in self.base["modes"]:
            selected = [r for r in records if r["mode"] == mode]
            metrics = {}
            for key in ("source_accuracy", "mr_accuracy", "msr", "joint_correctness"):
                values = [r[key]["rate"] for r in selected]
                metrics[key] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None} if values else {}
            aggregates[mode] = {"seeds": [r["seed"] for r in selected], "metrics": metrics}
        report = {"test_version": "V1.1", "test_sha256": self.inputs["test_sha256"], "complete": len(records) == 42, "records": records, "aggregates": aggregates, "notes": ["MSR uses unchanged scripts/metamorphic_metrics.py semantics.", "Source-only groups are excluded only from paired metrics, not source accuracy.", "V1.1 automated filtering is not human semantic validation.", "Old and V1.1 follow-up populations differ; changes are not model-improvement estimates."]}
        save_json(self.root / "v11_report.json", report)
        save_json(self.root / f"seed{seed}_review.json", {**report, "records": [r for r in records if r["seed"] == seed]})
        self.event("seed_complete", seed=seed, verified_result_files=len(records), final_complete=report["complete"])

    def run(self, prepare_only=False):
        self.preflight()
        for seed in self.cfg["seeds"]:
            self.prepare_seed(seed)
            if prepare_only:
                return
            if seed == 42:
                for mode in ("none", "full_oracle"):
                    self.evaluate(seed, mode, smoke=True)
            else:
                for mode in ("full_oracle", "pair_wrong_operation_relation_matched", "pair_wrong_relation") if seed == 43 else ("full_oracle",):
                    self.train(seed, mode, smoke=True)
                    self.evaluate(seed, mode, smoke=True, smoke_adapter=True)
            for mode in self.base["modes"]:
                if seed != 42:
                    self.train(seed, mode)
                self.evaluate(seed, mode)
            self.summarize(seed)
        self.event("completed", seed_order=self.cfg["seeds"], verified_results=42)


def worker(kind, budget, spare, arguments):
    import torch
    require(torch.cuda.device_count() == 1, "Worker must see exactly one GPU")
    try:
        free, total = torch.cuda.mem_get_info(0)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"GPU admission deferred before model load: {exc}", flush=True)
            raise SystemExit(75)
        raise
    if free < (budget + spare) * 1024**2:
        print("Shared GPU free memory shrank before model load; requeue", flush=True)
        raise SystemExit(75)
    torch.cuda.set_per_process_memory_fraction(budget * 1024**2 / total, 0)
    print(json.dumps({"worker": kind, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "allocator_budget_mib": budget, "free_mib_at_start": free // 1024**2}), flush=True)
    if kind == "training":
        sys.argv = ["llamafactory", "train", *arguments]
        runpy.run_module("llamafactory.cli", run_name="__main__")
    else:
        script, *argv = arguments
        sys.argv = [script, *argv]
        runpy.run_path(script, run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true", help="CPU checks and seed42 audit only; no GPU work")
    parser.add_argument("--worker", choices=("training", "inference"), help=argparse.SUPPRESS)
    parser.add_argument("--budget-mib", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--spare-mib", type=int, default=4096, help=argparse.SUPPRESS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.budget_mib, args.spare_mib, args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
        return
    pipeline = Pipeline(args.config)
    pipeline.root.mkdir(parents=True, exist_ok=True)
    with (pipeline.root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline.event("started", pid=os.getpid(), prepare_only=args.prepare_only)
        try:
            pipeline.run(args.prepare_only)
        except Exception as exc:
            pipeline.event("failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
