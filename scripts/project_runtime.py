"""Cross-platform runtime helpers shared by project scripts."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_console_encoding():
    """Use UTF-8 for project CLI output on Windows consoles."""
    if os.name != "nt":
        return
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def build_subprocess_env(
    *,
    cuda=None,
    offline=None,
    torch_compile_disable=False,
    extra=None,
):
    """Return a child-process environment without shell-specific prefixes."""
    env = os.environ.copy()
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
    if offline is not None:
        value = "1" if offline else "0"
        env["HF_HUB_OFFLINE"] = value
        env["TRANSFORMERS_OFFLINE"] = value
    if torch_compile_disable:
        env["TORCH_COMPILE_DISABLE"] = "1"
    if os.name == "nt":
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def apply_offline_mode(enabled=True):
    """Configure Hugging Face offline flags for the current process."""
    value = "1" if enabled else "0"
    os.environ["HF_HUB_OFFLINE"] = value
    os.environ["TRANSFORMERS_OFFLINE"] = value


def resolve_model_reference(hub_id, local_path=None):
    """Prefer an existing project-local model, otherwise keep the Hub id."""
    if local_path:
        candidate = Path(local_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_dir():
            return str(candidate.resolve())
    return hub_id
