import tarfile
from pathlib import Path

import pytest

from marathi_asr.artifacts import package, package_experiment, source_bundle
from marathi_asr.common import sha256, write_json


def test_source_bundle_excludes_credentials_data_and_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("README.md", "APPROACH.md", "pyproject.toml", "requirements-gpu.txt", ".gitignore", ".env.example"):
        Path(name).write_text("safe public content")
    Path(".env").write_text("HF_TOKEN=secret-test-value")
    Path("data").mkdir()
    Path("data/audio.wav").write_bytes(b"private audio")
    Path("models").mkdir()
    Path("models/base.nemo").write_bytes(b"weights")
    Path("src").mkdir()
    Path("src/example.py").write_text("print('hello')")
    Path("src/example.egg-info").mkdir()
    Path("src/example.egg-info/PKG-INFO").write_text("stale metadata")
    result = source_bundle()
    with tarfile.open(result) as tar:
        names = tar.getnames()
        assert "bodhan-marathi-asr/.env.example" in names
        assert "bodhan-marathi-asr/APPROACH.md" in names
        assert not any("egg-info" in n for n in names)
        assert all(not n.endswith("/.env") for n in names)
        assert not any("/data/" in n or "/models/" in n for n in names)
        for member in tar.getmembers():
            assert b"secret-test-value" not in tar.extractfile(member).read()


def test_incomplete_run_cannot_produce_submission_bundle(tmp_path):
    write_json(tmp_path / "status.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completed"):
        package({"training": {"directory": str(tmp_path)}})


def test_incomplete_experiment_cannot_be_packaged(tmp_path):
    write_json(tmp_path / "status.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completed"):
        package_experiment({"experiment": {"directory": str(tmp_path)}})


def test_experiment_bundle_includes_lineage_and_refuses_missing_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "comparison"
    strategies = ["a_decoder", "b_lora", "c_decoder_lora"]
    write_json(root / "status.json", {"status": "completed", "strategies": strategies})
    write_json(root / "decision.json", {"selected_before_test": True})
    (root / "comparison.md").write_text("Controlled fixture, not a training result.")
    (root / ".env").write_text("HF_TOKEN=never-archive-this")
    records = {}
    for name in ["baseline", *strategies]:
        label = "baseline" if name == "baseline" else "finetuned"
        for split in ("validation", "test"):
            write_json(root / name / "evaluation" / split / label / "metrics.json", {})
            (root / name / "evaluation" / split / label / "predictions.jsonl").write_text("{}\n")
        if name == "baseline":
            continue
        for filename in ("model.nemo", "canary_multilingual_tokenizer.py", "config.json", "parameters.json",
                         "initialization.json", "environment.json", "losses.jsonl", "checkpoints/last.ckpt"):
            path = root / name / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"controlled fixture")
        digest = sha256(root / name / "model.nemo")
        write_json(root / name / "status.json", {"status": "completed", "export_sha256": digest})
        records[name] = {"checkpoint_sha256": digest}
        if name != "a_decoder":
            (root / name / "adapters.pt").write_bytes(b"adapter fixture")
            write_json(root / name / "adapters.json", {"base_checkpoint": "exact starting checkpoint"})
    write_json(root / "comparison.json", {"decision_sha256": sha256(root / "decision.json"), "models": records})
    write_json(tmp_path / "data/dataset.json", {})
    cfg = {"experiment": {"directory": str(root)}, "data": {"directory": str(tmp_path / "data")}}
    archive = package_experiment(cfg)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        assert "comparison/a_decoder/model.nemo" in names
        assert "comparison/c_decoder_lora/adapters.pt" in names
        assert "comparison/c_decoder_lora/initialization.json" in names
        assert not any(n.endswith("/.env") for n in names)
    (root / "c_decoder_lora/adapters.pt").unlink()
    with pytest.raises(FileNotFoundError):
        package_experiment(cfg)
