#!/usr/bin/env python3
"""Download the official IndexTTS2 checkpoint bundle into the shared model dir."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf


DEFAULT_REPO_ID = "IndexTeam/IndexTTS-2"


def build_required_paths(model_dir: Path) -> list[Path]:
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return [config_path]

    cfg = OmegaConf.load(config_path)
    required = [
        config_path,
        model_dir / cfg.gpt_checkpoint,
        model_dir / cfg.s2mel_checkpoint,
        model_dir / cfg.bigvgan_checkpoint,
        model_dir / cfg.dataset["bpe_model"],
        model_dir / cfg.emo_matrix,
        model_dir / cfg.spk_matrix,
        model_dir / cfg.w2v_stat,
        model_dir / cfg.qwen_emo_path,
    ]
    return required


def needs_download(model_dir: Path) -> bool:
    return any(not path.exists() for path in build_required_paths(model_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IndexTTS2 models into a local directory")
    parser.add_argument("--model-dir", default="/app/shared/models/checkpoints", help="Target model directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repository to download")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    if not needs_download(model_dir):
        print(f"[models] Required model files already present in {model_dir}")
        return

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(f"[models] Downloading {args.repo_id} into {model_dir}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(model_dir),
        token=token,
        resume_download=True,
    )

    if needs_download(model_dir):
        raise RuntimeError(f"Download completed but required model files are still missing in {model_dir}")

    print(f"[models] Model download complete: {model_dir}")


if __name__ == "__main__":
    main()
