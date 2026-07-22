import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from project_runtime import PROJECT_ROOT, build_subprocess_env, resolve_model_reference


def test_project_root_is_derived_from_script_location():
    assert PROJECT_ROOT == ROOT
    assert (PROJECT_ROOT / "experiments_config.json").is_file()


def test_subprocess_environment_is_cross_platform():
    env = build_subprocess_env(
        cuda="0",
        offline=True,
        torch_compile_disable=True,
        extra={"LLMTRAIN_TEST_VALUE": 7},
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["TORCH_COMPILE_DISABLE"] == "1"
    assert env["LLMTRAIN_TEST_VALUE"] == "7"
    if os.name == "nt":
        assert env["PYTHONUTF8"] == "1"
        assert env["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
    assert all(isinstance(key, str) and isinstance(value, str)
               for key, value in env.items())


def test_active_configs_use_cuda_zero():
    for name in (
        "experiments_config.json",
        "experiments_config_llama32_3b.json",
        "experiments_config_mrinstr_pilot.json",
    ):
        config = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert config["cuda"] == "0"


def test_model_reference_prefers_existing_local_directory(tmp_path):
    local_model = tmp_path / "model"
    local_model.mkdir()
    assert resolve_model_reference("org/model", local_model) == str(local_model.resolve())
    assert resolve_model_reference("org/model", tmp_path / "missing") == "org/model"


def test_active_scripts_do_not_contain_remote_workspace_path():
    remote_path = "/home/ubuntu/LLMTrain/LLMTrain"
    for path in (ROOT / "scripts").glob("*"):
        if path.suffix not in {".py", ".sh"}:
            continue
        assert remote_path not in path.read_text(encoding="utf-8")
