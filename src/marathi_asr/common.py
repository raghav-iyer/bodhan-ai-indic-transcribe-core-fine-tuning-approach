"""Small dependency-free configuration and artifact helpers."""

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_local_token(path=".env"):
    """Load only HF_TOKEN; never evaluate shell syntax or include it in artifacts."""
    secret_file = Path(path)
    if os.environ.get("HF_TOKEN") or not secret_file.exists():
        return
    for line in secret_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "HF_TOKEN":
            value = value.strip().strip("\"'")
            if value:
                os.environ["HF_TOKEN"] = value
            return


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for section in ("model", "data"):
        if not re.fullmatch(r"[0-9a-f]{40}", cfg[section]["revision"]):
            raise ValueError(f"{section}.revision must be an immutable Hugging Face commit SHA")
    if cfg["data"]["language"] != "mr" or cfg["data"]["sample_rate"] != 16000:
        raise ValueError("This assignment pipeline requires Marathi and 16 kHz audio")
    if cfg["data"]["text_field"] != "raw_transcription" or cfg["data"]["pnc"] != "yes":
        raise ValueError("The documented transcript contract is raw_transcription with pnc=yes")
    for split, limit in cfg["data"]["limits"].items():
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError(f"data.limits.{split} must be a positive integer or null (all)")
    if cfg["training"]["mode"] not in ("decoder_top", "decoder", "full", "decoder_lora"):
        raise ValueError("Unknown training mode")
    for key in ("max_steps", "batch_size", "accumulate_grad_batches", "val_check_interval"):
        if cfg["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    lora = cfg["training"].get("lora", {})
    rank = lora.get("rank", 8)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("training.lora.rank must be a positive integer")
    alpha, dropout = lora.get("alpha", 16), lora.get("dropout", 0.05)
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("LoRA requires positive finite alpha and dropout in [0, 1)")
    if not math.isfinite(cfg["training"]["learning_rate"]) or cfg["training"]["learning_rate"] <= 0:
        raise ValueError("training.learning_rate must be positive and finite")
    return cfg


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    # Never persist environment variables or auth tokens.
    result = {"time_utc": utc_now(), "python": sys.version, "platform": platform.platform()}
    result["pip_freeze"] = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return result


def manifest_path(cfg, split):
    return Path(cfg["data"]["directory"]) / f"{split}.jsonl"
