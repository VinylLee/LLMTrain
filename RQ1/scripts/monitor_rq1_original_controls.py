#!/usr/bin/env python3
"""Hourly read-only status monitor for the two RQ1 Original-SNLI controls."""

import fcntl
import json
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DIR = (
    PROJECT_ROOT / "RQ1" / "output" / "gemma-3-4b-it"
    / "snli_original_controls_monitor_v1"
)
ROOTS = {
    "original_2033": (
        PROJECT_ROOT / "RQ1" / "output" / "gemma-3-4b-it"
        / "snli_original_2033_multiseed_v1"
    ),
    "original_2033_tokenmatched": (
        PROJECT_ROOT / "RQ1" / "output" / "gemma-3-4b-it"
        / "snli_original_2033_tokenmatched_multiseed_v1"
    ),
}
NAME_TEMPLATES = {
    "original_2033": "rq1_original_snli_2033_gemma-3-4b-it_seed{seed}",
    "original_2033_tokenmatched": (
        "rq1_original_snli_2033_gemma-3-4b-it_tokenmatched_seed{seed}"
    ),
}
SEEDS = (42, 43, 44)


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def latest_trainer_record(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def relevant_processes():
    result = subprocess.run(
        [
            "pgrep",
            "-af",
            (
                "rq1_original_snli_2033|RQ1/run_rq1_nli.py|"
                "run_rq1_original_controls_tests_when_gpu_free"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "monitor_rq1_original_controls.py" not in line
    ]


def gpu_status():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            rows.append(
                {
                    "index": int(fields[0]),
                    "memory_used_mib": int(fields[1]),
                    "memory_total_mib": int(fields[2]),
                    "utilization_percent": int(fields[3]),
                }
            )
    return rows


def collect():
    processes = relevant_processes()
    experiments = {}
    finetune_completed = 0
    merged_completed = 0
    for group, output_root in ROOTS.items():
        progress = load_json(output_root / "_progress.json", {})
        for seed in SEEDS:
            name = NAME_TEMPLATES[group].format(seed=seed)
            stages = progress.get(name, {})
            model_dir = output_root / name / "model"
            trainer = latest_trainer_record(model_dir / "trainer_log.jsonl")
            adapter_complete = (
                (model_dir / "adapter_model.safetensors").is_file()
                and (model_dir / "adapter_config.json").is_file()
            )
            merged_files = sorted((output_root / name / "tests" / "merged").glob("*.jsonl"))
            if stages.get("finetune") == "completed" and adapter_complete:
                finetune_completed += 1
            if stages.get("test_merged") == "completed" and len(merged_files) == 4:
                merged_completed += 1
            if stages.get("test_merged") == "completed":
                state = "test_complete"
            elif stages.get("finetune") == "completed":
                state = "training_complete"
            elif trainer.get("current_steps"):
                state = "training"
            elif stages.get("convert") == "completed":
                state = "prepared"
            else:
                state = "not_prepared"
            experiments[name] = {
                "group": group,
                "seed": seed,
                "state": state,
                "stages": stages,
                "current_step": trainer.get("current_steps"),
                "total_steps": trainer.get("total_steps"),
                "percentage": trainer.get("percentage"),
                "last_loss": trainer.get("loss"),
                "remaining_time": trainer.get("remaining_time"),
                "adapter_complete": adapter_complete,
                "merged_test_files": len(merged_files),
            }

    if merged_completed == 6:
        overall = "all_complete"
    elif (
        any("run_rq1_original_controls_tests_when_gpu_free" in item for item in processes)
        and not any("--steps test_merged" in item for item in processes)
    ):
        overall = "queued_waiting_for_gpu"
    elif processes:
        overall = "running"
    elif finetune_completed == 6:
        overall = "training_complete_awaiting_test"
    else:
        overall = "incomplete_no_process"
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "overall_status": overall,
        "finetune_completed": finetune_completed,
        "finetune_total": 6,
        "merged_test_completed": merged_completed,
        "merged_test_total": 6,
        "experiments": experiments,
        "relevant_processes": processes,
        "gpus": gpu_status(),
    }


def render_latest(record):
    lines = [
        "# RQ1 Original-SNLI hourly status",
        "",
        f"Timestamp: {record['timestamp']}",
        f"Overall: {record['overall_status']}",
        (
            f"Fine-tune complete: {record['finetune_completed']}/"
            f"{record['finetune_total']}"
        ),
        (
            f"Merged test complete: {record['merged_test_completed']}/"
            f"{record['merged_test_total']}"
        ),
        "",
        "| Experiment | State | Step | Adapter | Merged files |",
        "|---|---|---:|---:|---:|",
    ]
    for name, item in record["experiments"].items():
        step = "—"
        if item["current_step"] is not None:
            step = f"{item['current_step']}/{item['total_steps']}"
        lines.append(
            f"| {name} | {item['state']} | {step} | "
            f"{str(item['adapter_complete']).lower()} | {item['merged_test_files']}/4 |"
        )
    return chr(10).join(lines) + chr(10)


def main():
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MONITOR_DIR / ".monitor.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = collect()
        with (MONITOR_DIR / "hourly_status.jsonl").open(
            "a", encoding="utf-8", newline=chr(10)
        ) as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + chr(10))
        (MONITOR_DIR / "latest.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + chr(10),
            encoding="utf-8",
        )
        (MONITOR_DIR / "latest.md").write_text(
            render_latest(record),
            encoding="utf-8",
        )
        print(json.dumps(record, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()
