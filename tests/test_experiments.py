"""Controlled orchestration fixtures verify selection policy, never claim ASR accuracy."""

import copy
import json
from pathlib import Path

import pytest

from marathi_asr import experiments
from marathi_asr.common import read_jsonl, sha256, write_json, write_jsonl
from marathi_asr.metrics import aggregate, score_pair


def validation(wer):
    return {"split": "validation", "normalized": {"wer": wer}}


@pytest.mark.parametrize("a,b,expected", [(0.1, 0.2, True), (0.2, 0.2, False), (0.3, 0.2, False)])
def test_conditional_gate_is_strict(a, b, expected):
    assert experiments.should_run_conditional(validation(a), validation(b)) is expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, None, "0.1", True])
def test_invalid_validation_scores_fail(bad):
    with pytest.raises(ValueError, match="invalid"):
        experiments.should_run_conditional(validation(bad), validation(0.5))


def test_test_metrics_cannot_select_or_gate():
    metrics = {"split": "test", "normalized": {"wer": 0.1}}
    with pytest.raises(ValueError, match="never test"):
        experiments.should_run_conditional(metrics, validation(0.2))
    with pytest.raises(ValueError, match="never test"):
        experiments.select_winner({"baseline": validation(0.2), "a_decoder": metrics})
    assert experiments.select_winner({"baseline": validation(0.1), "a_decoder": validation(0.1)}) == "baseline"


@pytest.fixture
def cfg(tmp_path):
    return {
        "seed": 42,
        "model": {"directory": str(tmp_path / "model"), "checkpoint": "nemo/original.nemo"},
        "data": {"directory": str(tmp_path / "data")},
        "training": {"directory": str(tmp_path / "unused"), "mode": "decoder_top", "decoder_layers": 2,
                     "max_steps": 2, "init_checkpoint": "must-not-leak-into-A-or-B.nemo",
                     "lora": {"rank": 8, "alpha": 16, "dropout": 0.05}},
        "evaluation": {"bootstrap_samples": 20},
        "experiment": {"directory": str(tmp_path / "comparison"), "conditional_decoder_lora": True},
    }


def fake_stages(monkeypatch, cfg, validation_errors=None, test_errors=None, tamper=None):
    """Produce internally consistent real metrics from synthetic text, without model imports."""
    validation_errors = validation_errors or {"baseline": 3, "a_decoder": 1, "b_lora": 2, "c_decoder_lora": 0}
    test_errors = test_errors or {"baseline": 3, "a_decoder": 0, "b_lora": 1, "c_decoder_lora": 2}
    calls = []
    root = Path(cfg["experiment"]["directory"])

    def run(config_path, arguments, log_path):
        current = json.loads(config_path.read_text())
        output = Path(current["training"]["directory"])
        name, stage = output.name, arguments[0]
        calls.append({"name": name, "stage": stage, "arguments": arguments, "cfg": current})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Controlled test fixture, no model was executed.\n")
        if stage == "preflight":
            write_json(output / "preflight.json", {"ready_to_train": True})
        elif stage == "prepare":
            for split in ("train", "validation", "test"):
                write_jsonl(Path(current["data"]["directory"]) / f"{split}.jsonl", [{
                    "id": split, "source_id": split, "audio_sha256": f"audio-{split}",
                    "audio_filepath": "fixture.wav", "duration": 1.0, "text": "एक दोन तीन चार"}])
        elif stage == "download-model":
            original = Path(current["model"]["directory"]) / current["model"]["checkpoint"]
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b"original-fixture-not-a-model")
        elif stage == "train":
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / "model.nemo"
            checkpoint.write_bytes(name.encode())
            initial = Path(current["training"].get("init_checkpoint") or
                           Path(current["model"]["directory"]) / current["model"]["checkpoint"])
            write_json(output / "status.json", {"status": "completed", "export_sha256": sha256(checkpoint),
                                                 "initial_checkpoint_sha256": sha256(initial),
                                                 "optimizer_steps": 2, "training_wall_seconds": 1.0})
            write_json(output / "parameters.json", {"trainable_parameters": 12, "total_parameters": 100,
                                                     "mode": current["training"]["mode"]})
        elif stage == "evaluate":
            label = arguments[arguments.index("--label") + 1]
            split = arguments[arguments.index("--split") + 1]
            if split == "test":
                # The first test prediction cannot exist until selection is persisted.
                decision = json.loads((root / "decision.json").read_text())
                assert decision["selected_before_test"] is True
                assert all(metrics["split"] == "validation" for metrics in decision["validation"].values())
            checkpoint = (Path(arguments[arguments.index("--checkpoint") + 1]) if label == "finetuned"
                          else Path(current["model"]["directory"]) / current["model"]["checkpoint"])
            manifest = Path(current["data"]["directory"]) / f"{split}.jsonl"
            source = read_jsonl(manifest)[0]
            errors = (validation_errors if split == "validation" else test_errors)[name]
            hypothesis = " ".join(source["text"].split()[errors:])
            predictions = [{"id": source["id"], "audio_sha256": source["audio_sha256"],
                            "reference": source["text"], "prediction": hypothesis,
                            "scores": score_pair(source["text"], hypothesis)}]
            metadata = {**aggregate(predictions), "label": label, "split": split,
                        "manifest_sha256": sha256(manifest), "checkpoint_sha256": sha256(checkpoint),
                        "decoding": "identical decoding fixture", "wall_seconds": 1.0, "real_time_factor": 1.0}
            if tamper is not None:
                tamper(name, split, metadata)
            folder = output / "evaluation" / split / label
            write_json(folder / "metrics.json", metadata)
            write_jsonl(folder / "predictions.jsonl", predictions)
        return 2.0

    monkeypatch.setattr(experiments, "_run_stage", run)
    return calls


def test_full_experiment_preserves_selection_and_checkpoint_lineage(cfg, monkeypatch):
    untouched = copy.deepcopy(cfg)
    calls = fake_stages(monkeypatch, cfg)
    root = experiments.run_experiment(cfg)
    result = json.loads((root / "comparison.json").read_text())
    decision = result["decision"]
    assert decision["selected_model"] == "c_decoder_lora"
    assert decision["conditional"]["run"] is True
    assert result["models"]["a_decoder"]["test"]["normalized"]["wer"] == 0
    assert result["models"]["c_decoder_lora"]["test"]["normalized"]["wer"] == 0.5
    assert sha256(root / "decision.json") == result["decision_sha256"]
    assert cfg == untouched
    training = {call["name"]: call["cfg"]["training"] for call in calls if call["stage"] == "train"}
    assert "init_checkpoint" not in training["a_decoder"]
    assert "init_checkpoint" not in training["b_lora"]
    assert training["c_decoder_lora"]["init_checkpoint"] == str(root / "a_decoder" / "model.nemo")
    assert training["a_decoder"]["mode"] == "decoder_top"
    assert training["b_lora"]["mode"] == training["c_decoder_lora"]["mode"] == "decoder_lora"
    assert {call["cfg"]["training"]["max_steps"] for call in calls if call["stage"] == "train"} == {2}
    for split in ("validation", "test"):
        baseline_calls = [call for call in calls if call["name"] == "baseline" and
                          call["stage"] == "evaluate" and split in call["arguments"]]
        assert len(baseline_calls) == 1
    test_start = next(i for i, call in enumerate(calls) if "test" in call["arguments"])
    assert all(call["stage"] == "evaluate" and "test" in call["arguments"] for call in calls[test_start:])
    assert set(result["paired_test_vs_baseline"]) == {"a_decoder", "b_lora", "c_decoder_lora"}
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "completed"
    assert set(status["strategies"]) == {"a_decoder", "b_lora", "c_decoder_lora"}
    assert "continuation experiment" in (root / "comparison.md").read_text()
    with pytest.raises(ValueError, match="already used"):
        experiments.run_experiment(cfg)
    assert sha256(root / "decision.json") == result["decision_sha256"]


@pytest.mark.parametrize("a,b,enabled", [(2, 2, True), (2, 1, True), (1, 2, False)])
def test_conditional_stage_is_skipped_on_tie_loss_or_disabled(cfg, monkeypatch, a, b, enabled):
    cfg["experiment"]["conditional_decoder_lora"] = enabled
    calls = fake_stages(monkeypatch, cfg, {"baseline": 0, "a_decoder": a, "b_lora": b})
    root = experiments.run_experiment(cfg)
    decision = json.loads((root / "decision.json").read_text())
    assert decision["conditional"]["run"] is False
    assert decision["selected_model"] == "baseline"
    assert not any(call["name"] == "c_decoder_lora" for call in calls)


@pytest.mark.parametrize("key", ["manifest_sha256", "checkpoint_sha256", "decoding", "split"])
def test_mismatched_validation_contract_fails_before_gate_or_test(cfg, monkeypatch, key):
    def tamper(name, split, metadata):
        if name == "b_lora":
            metadata[key] = "different"

    calls = fake_stages(monkeypatch, cfg, tamper=tamper)
    with pytest.raises(ValueError, match="contract differs"):
        experiments.run_experiment(cfg)
    assert not any("test" in call["arguments"] for call in calls)
    root = Path(cfg["experiment"]["directory"])
    assert not (root / "decision.json").exists()
    assert json.loads((root / "status.json").read_text())["status"] == "failed"


def test_saved_metrics_must_match_predictions(cfg, monkeypatch):
    def tamper(name, split, metadata):
        if name == "a_decoder":
            metadata["normalized"]["wer"] = 0.123

    calls = fake_stages(monkeypatch, cfg, tamper=tamper)
    with pytest.raises(ValueError, match="differs from predictions"):
        experiments.run_experiment(cfg)
    assert not any("test" in call["arguments"] for call in calls)


def test_failed_preflight_stops_all_expensive_stages(cfg, monkeypatch):
    calls = []

    def failed_preflight(config_path, arguments, log_path):
        calls.append(arguments)
        current = json.loads(config_path.read_text())
        write_json(Path(current["training"]["directory"]) / "preflight.json", {"ready_to_train": False})
        return 1

    monkeypatch.setattr(experiments, "_run_stage", failed_preflight)
    with pytest.raises(RuntimeError, match="Preflight"):
        experiments.run_experiment(cfg)
    assert calls == [["preflight"]]
