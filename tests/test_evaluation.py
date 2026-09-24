"""Exercise evaluation/report artifacts with controlled hypotheses, not ASR claims."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from marathi_asr.common import load_config, read_jsonl, write_jsonl
from marathi_asr.evaluate import compare, evaluate


def test_evaluation_and_report_contract(tmp_path, monkeypatch):
    cfg = load_config("configs/marathi.json")
    cfg["data"]["directory"] = str(tmp_path / "data")
    cfg["training"]["directory"] = str(tmp_path / "run")
    cfg["model"]["directory"] = str(tmp_path / "model")
    model_file = Path(cfg["model"]["directory"]) / cfg["model"]["checkpoint"]
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"test-fixture-not-a-model")
    tuned_file = tmp_path / "fixture.nemo"
    tuned_file.write_bytes(b"second-fixture-not-a-model")
    write_jsonl(Path(cfg["data"]["directory"]) / "test.jsonl", [
        {"id": "one", "source_id": "1", "audio_sha256": "audio1", "audio_filepath": "fixture.wav",
         "duration": 1.0, "text": "नमस्कार महाराष्ट्र"}])
    monkeypatch.setattr("marathi_asr.model.restore", lambda *a, **kw: torch.nn.Identity())
    monkeypatch.setattr("marathi_asr.model.configure_decoding", lambda model: None)
    monkeypatch.setattr("marathi_asr.model.transcribe", lambda *a, **kw: ["नमस्कार"])
    evaluate(cfg, "baseline", device="cpu")
    monkeypatch.setattr("marathi_asr.model.transcribe", lambda *a, **kw: ["नमस्कार महाराष्ट्र"])
    evaluate(cfg, "finetuned", checkpoint=str(tuned_file), device="cpu")
    result = compare(cfg)
    assert result["normalized_wer_delta_tuned_minus_baseline"] == -0.5
    report = Path(cfg["training"]["directory"]) / "evaluation/test/report.md"
    assert "नमस्कार महाराष्ट्र" in report.read_text()
    metrics = report.parent / "finetuned/metrics.json"
    saved = json.loads(metrics.read_text())
    saved["decoding"] = "different"
    metrics.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="contract differs"):
        compare(cfg)
