import copy
import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from marathi_asr.common import load_config, read_jsonl, write_jsonl
from marathi_asr.data import convert_audio, prepare, validate


def wav_bytes(index=1, rate=22050, channels=2):
    signal = (0.2 * np.sin(2 * np.pi * (170 + index * 40) * np.arange(rate) / rate)).astype(np.float32)
    if channels == 2:
        signal = np.column_stack([signal, signal * 0.5])
    stream = io.BytesIO()
    sf.write(stream, signal, rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def test_resampling_and_silence():
    audio = convert_audio(io.BytesIO(wav_bytes()))
    assert audio.ndim == 1 and len(audio) == 16000
    silence = io.BytesIO()
    sf.write(silence, np.zeros(16000), 16000, format="WAV")
    silence.seek(0)
    with pytest.raises(ValueError, match="silent"):
        convert_audio(silence)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cfg = load_config("configs/marathi.json")
    cfg["data"]["directory"] = str(tmp_path / "prepared")
    cfg["data"]["limits"] = {key: None for key in ("train", "validation", "test")}
    files = {}
    for i, split in enumerate(("train", "validation", "test")):
        records = [{"id": i, "path": f"source-{i}.wav", "audio": {"bytes": wav_bytes(i), "path": "irrelevant.wav"},
                    "raw_transcription": f"नमस्कार महाराष्ट्र {i}", "gender": 0}]
        # Same audio should be rejected, regardless of transcript or metadata ID.
        records.append({**records[0], "id": 100 + i})
        path = tmp_path / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        files[split] = str(path)
    def download(repo, filename, **kwargs):
        split = Path(filename).name.split("-")[0]
        return files[split]
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    report = prepare(cfg)
    return cfg, report


def test_prepare_round_trip_and_duplicate_policy(prepared):
    cfg, report = prepared
    assert validate(cfg)["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["splits"]["train"]["rejection_reasons"]["duplicate_audio"] == 1
    assert prepare(cfg)["data_config"] == cfg["data"]  # safe reuse


def test_audio_tampering_detected(prepared):
    cfg, _ = prepared
    row = read_jsonl(Path(cfg["data"]["directory"]) / "test.jsonl")[0]
    sf.write(row["audio_filepath"], np.ones(16000) * 0.3, 16000, subtype="PCM_16")
    with pytest.raises(ValueError, match="changed"):
        validate(cfg)


def test_split_leakage_detected(prepared):
    cfg, _ = prepared
    root = Path(cfg["data"]["directory"])
    row = read_jsonl(root / "train.jsonl")[0]
    row["id"] = "unique-but-leaked"
    write_jsonl(root / "test.jsonl", [row])
    with pytest.raises(ValueError, match="Duplicate audio"):
        validate(cfg)


def test_invalid_task_tag_detected(prepared):
    cfg, _ = prepared
    path = Path(cfg["data"]["directory"]) / "test.jsonl"
    rows = read_jsonl(path)
    rows[0]["target_lang"] = "hi"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="target_lang"):
        validate(cfg)


def test_reprepare_changed_policy_requires_new_directory(prepared):
    cfg, _ = prepared
    cfg["data"]["max_train_duration"] = 15
    with pytest.raises(ValueError, match="config changed"):
        prepare(cfg)
