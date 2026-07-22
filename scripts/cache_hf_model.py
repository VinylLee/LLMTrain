#!/usr/bin/env python3
"""Download a Hugging Face model into the standard cache and verify it offline."""

import argparse
import os
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoConfig, AutoTokenizer

from project_runtime import PROJECT_ROOT, apply_offline_mode, configure_console_encoding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def default_local_dir(model):
    return PROJECT_ROOT / "models" / Path(*model.split("/"))


def cache_and_verify(model, revision=None, local_dir=None, verify_only=False):
    token = os.environ.get("HF_TOKEN")
    snapshot_path = Path(local_dir).resolve() if local_dir else default_local_dir(model)
    if verify_only:
        if not snapshot_path.is_dir():
            raise SystemExit(f"Local model directory does not exist: {snapshot_path}")
    else:
        apply_offline_mode(False)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_path = Path(snapshot_download(
                repo_id=model,
                revision=revision,
                local_dir=snapshot_path,
                token=token,
            )).resolve()
        except GatedRepoError as exc:
            raise SystemExit(
                f"Cannot access gated model {model}. Activate the intended environment, "
                "accept the model license, run `hf auth login`, and retry."
            ) from exc

    apply_offline_mode(True)
    config = AutoConfig.from_pretrained(snapshot_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    return {
        "model": model,
        "snapshot_path": str(snapshot_path),
        "model_type": config.model_type,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "offline_verified": True,
    }


def main():
    configure_console_encoding()
    args = parse_args()
    result = cache_and_verify(
        args.model,
        revision=args.revision,
        local_dir=args.local_dir,
        verify_only=args.verify_only,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
