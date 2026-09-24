"""Collect submission evidence using an explicit allowlist, excluding secrets/data."""

import json
import tarfile
from pathlib import Path

from .common import sha256, write_json


def package(cfg):
    run = Path(cfg["training"]["directory"]).resolve()
    status = json.loads((run / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("Only a completed training run can be packaged for submission")
    required = [run / "model.nemo", run / "canary_multilingual_tokenizer.py",
                run / "evaluation/test/comparison.json", run / "evaluation/test/report.md",
                run / "checkpoints/last.ckpt", run / "environment.json", run / "losses.jsonl"]
    if cfg["training"].get("mode") == "decoder_lora":
        required.extend([run / "adapters.pt", run / "adapters.json"])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required submission evidence: {path}")
    # Allow only artifacts created by this pipeline. Never recursively archive the project root.
    files = set(required)
    for name in ("config.json", "parameters.json", "token_audit.json", "data_hashes.json", "nemo_config.yaml", "status.json", "initialization.json"):
        files.add(run / name)
    for folder in ("evaluation", "csv", "tensorboard", "logs", "checkpoints"):
        files.update(p for p in (run / folder).rglob("*") if p.is_file())
    files.update(p for p in run.glob("*.md") if "license" in p.name.lower())
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(run):
            raise ValueError(f"Refusing external/symlink artifact: {path}")
        if path.name == ".env" or path.name.startswith(".env."):
            raise ValueError("Refusing to package a secrets file")
    index = {str(p.relative_to(run)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(files)}
    write_json(run / "artifact_index.json", index)
    destination = Path("dist")
    destination.mkdir(exist_ok=True)
    archive = destination / f"{run.name}-artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(files | {run / "artifact_index.json"}):
            tar.add(path, arcname=f"{run.name}/{path.relative_to(run)}", recursive=False)
        # Preparation evidence, without redistributing the corpus.
        tar.add(Path(cfg["data"]["directory"]) / "dataset.json", arcname=f"{run.name}/dataset.json")
    print(f"Artifact archive: {archive.resolve()}")
    return archive


def source_bundle():
    """Portable source bundle for the recipient's PC; includes no assignment PDF or credentials."""
    destination = Path("dist")
    destination.mkdir(exist_ok=True)
    archive = destination / "bodhan-marathi-asr-source.tar.gz"
    files = [Path(x) for x in ("README.md", "APPROACH.md", "pyproject.toml", ".gitignore", ".env.example")]
    files.extend(Path('.').glob('requirements*.txt'))
    for directory in ("src", "configs", "scripts", "tests"):
        files.extend(p for p in Path(directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts
                     and not any(part.endswith(".egg-info") for part in p.parts)
                     and p.suffix in (".py", ".json", ".sh"))
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(files):
            if path.is_symlink():
                raise ValueError(f"Refusing symlink in source bundle: {path}")
            tar.add(path, arcname=f"bodhan-marathi-asr/{path}", recursive=False)
    return archive


def package_experiment(cfg):
    """Package only the recognized outputs of a completed A/B/(C) comparison."""
    root = Path(cfg["experiment"]["directory"]).resolve()
    status = json.loads((root / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError("Only a completed experiment can be packaged")
    required = [root / name for name in ("status.json", "decision.json", "comparison.json", "comparison.md")]
    strategies = status.get("strategies", [])
    if strategies not in (["a_decoder", "b_lora"], ["a_decoder", "b_lora", "c_decoder_lora"]):
        raise ValueError("Completed experiment must contain A and B, and optionally C")
    comparison = json.loads((root / "comparison.json").read_text())
    if comparison.get("decision_sha256") != sha256(root / "decision.json"):
        raise ValueError("Experiment decision differs from the completed comparison")
    for name in ["baseline", *strategies]:
        label = "baseline" if name == "baseline" else "finetuned"
        required += [root / name / "evaluation" / split / label / filename
                     for split in ("validation", "test") for filename in ("metrics.json", "predictions.jsonl")]
        if name == "baseline":
            continue
        run = root / name
        required += [run / filename for filename in (
            "model.nemo", "canary_multilingual_tokenizer.py", "status.json", "config.json",
            "parameters.json", "initialization.json", "environment.json", "losses.jsonl", "checkpoints/last.ckpt")]
        run_status = json.loads((run / "status.json").read_text())
        digest = sha256(run / "model.nemo")
        if (run_status.get("status") != "completed" or run_status.get("export_sha256") != digest
                or comparison["models"][name]["checkpoint_sha256"] != digest):
            raise ValueError(f"Missing or changed completed model: {name}")
        if name in ("b_lora", "c_decoder_lora"):
            required += [run / "adapters.pt", run / "adapters.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Keep generated settings and evidence, never arbitrary project-root files.
    allowed_suffixes = {".json", ".jsonl", ".md", ".yaml", ".log", ".csv", ".ckpt", ".nemo", ".pt"}
    files = set(required)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing external/symlink artifact: {path}")
        if path.name.startswith(".env"):
            continue
        if (path.suffix in allowed_suffixes or path.name.startswith("events.out.tfevents.")
                or path.name == "canary_multilingual_tokenizer.py"):
            files.add(path)
    write_json(root / "artifact_index.json", {
        str(p.relative_to(root)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(files) if p.name != "artifact_index.json"
    })
    files.add(root / "artifact_index.json")
    destination = Path("dist")
    destination.mkdir(exist_ok=True)
    archive = destination / f"{root.name}-artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(files):
            tar.add(path, arcname=f"{root.name}/{path.relative_to(root)}", recursive=False)
        dataset = Path(cfg["data"]["directory"]) / "dataset.json"
        tar.add(dataset, arcname=f"{root.name}/dataset.json", recursive=False)
    return archive
