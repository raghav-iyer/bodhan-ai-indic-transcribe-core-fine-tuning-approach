"""Read pinned HF Parquet files directly, without remote dataset scripts/codecs."""

import hashlib
import io
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from .common import manifest_path, read_jsonl, sha256, write_json, write_jsonl
from .text import clean_transcript, normalize_for_scoring

SPLITS = ("train", "validation", "test")
PROMPT = {
    "source_lang": "mr", "target_lang": "mr", "task": "asr", "taskname": "asr",
    "pnc": "yes", "itn": "no", "timestamp": "no", "diarize": "no",
    "emotion": "<|emo:undefined|>", "decodercontext": "",
}


def convert_audio(source, sample_rate=16000):
    """Read, mix to mono, resample, and reject unusable audio without truncating it."""
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("empty_or_nonfinite_audio")
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor)
    if np.max(np.abs(audio)) > 1.01:
        raise ValueError("out_of_range_audio")
    if float(np.sqrt(np.mean(audio ** 2))) < 1e-5:
        raise ValueError("silent_audio")
    return audio.astype(np.float32)


def _prepare_record(item, spec, split, audio_dir, all_hashes, all_texts):
    """Check one source recording, save its WAV, and return one manifest row.

    A manifest is a JSON-lines list linking each audio file to its transcript.
    Shared hash/text dictionaries keep the train, validation and test splits apart.
    """
    max_duration = spec["max_train_duration"] if split == "train" else spec["max_eval_duration"]
    text = clean_transcript(item[spec["text_field"]])
    if not text or not normalize_for_scoring(text):
        raise ValueError("empty_transcript")
    if not any("\u0900" <= c <= "\u097f" for c in text):
        raise ValueError("no_devanagari")
    audio_field = item["audio"]
    if not audio_field.get("bytes"):
        raise ValueError("Parquet audio is not embedded; refusing an unverified external path")
    audio = convert_audio(io.BytesIO(audio_field["bytes"]), spec["sample_rate"])
    duration = len(audio) / spec["sample_rate"]
    if not spec["min_duration"] <= duration <= max_duration:
        raise ValueError("duration_outside_policy")
    # Hash the canonical PCM waveform, independent of the source container.
    pcm = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
    digest = hashlib.sha256(pcm.tobytes()).hexdigest()
    if digest in all_hashes:
        raise ValueError("duplicate_audio")
    text_key = normalize_for_scoring(text)
    if text_key in all_texts and all_texts[text_key] != split:
        raise ValueError("cross_split_transcript_overlap")
    utterance_id = f"{split}-{digest[:20]}"
    audio_path = audio_dir / f"{utterance_id}.wav"
    sf.write(audio_path, pcm, spec["sample_rate"], subtype="PCM_16")
    all_hashes[digest] = split
    all_texts[text_key] = split
    return {
        "id": utterance_id, "audio_filepath": str(audio_path), "duration": duration,
        "text": text, **PROMPT, "pnc": spec["pnc"],
        "source_id": str(item["id"]), "source_path": item.get("path"),
        "audio_sha256": digest, "gender": item.get("gender"),
    }


def prepare(cfg):
    """Download each source split and keep the first eligible recordings."""
    from huggingface_hub import hf_hub_download

    spec = cfg["data"]
    root = Path(spec["directory"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "dataset.json").exists():
        import json
        existing = json.loads((root / "dataset.json").read_text())
        if existing["data_config"] != spec:
            raise ValueError("Data config changed. Choose a new data.directory to preserve the previous split contract.")
        validate(cfg)
        print(f"Reusing validated data at {root}")
        return existing

    report = {"data_config": spec, "selection": "first eligible rows in pinned Parquet order", "splits": {}}
    all_hashes, all_texts = {}, {}
    for split in SPLITS:
        filename = f"parquet-data/{spec['subset']}/{split}-00000-of-00001.parquet"
        local = hf_hub_download(spec["repo_id"], filename, repo_type="dataset", revision=spec["revision"],
                                cache_dir=".cache/huggingface/hub")
        rows, rejected = [], []
        limit = spec["limits"][split]
        audio_dir = root / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        count_seen = 0
        for batch in pq.ParquetFile(local).iter_batches(batch_size=16):
            for item in batch.to_pylist():
                count_seen += 1
                try:
                    record = _prepare_record(item, spec, split, audio_dir, all_hashes, all_texts)
                    rows.append(record)
                except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
                    rejected.append({"row": count_seen, "source_id": str(item.get("id")), "reason": str(exc)})
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
        if not rows:
            raise ValueError(f"No usable {split} rows; inspect the source data and audio policy")
        write_jsonl(manifest_path(cfg, split), rows)
        write_jsonl(root / f"{split}.rejected.jsonl", rejected)
        report["splits"][split] = {
            "utterances": len(rows), "hours": sum(x["duration"] for x in rows) / 3600,
            "rows_examined": count_seen, "rejected": len(rejected),
            "rejection_reasons": dict(Counter(x["reason"] for x in rejected)),
            "manifest_sha256": sha256(manifest_path(cfg, split)),
            "parquet_file": filename, "parquet_sha256": sha256(local),
            "duration_seconds": {"min": min(x["duration"] for x in rows), "max": max(x["duration"] for x in rows)},
        }
        print(f"{split}: {len(rows)} utterances, {report['splits'][split]['hours']:.2f} hours; {len(rejected)} rejected")
    report["audit"] = validate(cfg)
    report["limitations"] = [
        "Speaker IDs are unavailable in this source; speaker separation was not independently verified.",
        "Bodhan pretraining overlap with FLEURS is unknown; this is an adaptation exercise, not a novel benchmark.",
        "Duration filtering and any subset caps define a restricted evaluation population.",
    ]
    write_json(root / "dataset.json", report)
    return report


def validate(cfg):
    """Recheck saved audio and detect any leakage between dataset splits."""
    ids, audio_hashes, text_splits = set(), {}, {}
    counts = {}
    spec = cfg["data"]
    for split in SPLITS:
        rows = read_jsonl(manifest_path(cfg, split))
        if not rows:
            raise ValueError(f"Empty {split} manifest")
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate utterance ID: {row['id']}")
            ids.add(row["id"])
            for key, value in PROMPT.items():
                if row.get(key) != (spec["pnc"] if key == "pnc" else value):
                    raise ValueError(f"Invalid {key} in {row['id']}")
            text = clean_transcript(row["text"])
            if not text or text != row["text"]:
                raise ValueError(f"Invalid transcript in {row['id']}")
            audio, rate = sf.read(row["audio_filepath"], dtype="int16", always_2d=True)
            info = sf.info(row["audio_filepath"])
            if rate != 16000 or audio.shape[1] != 1 or info.subtype != "PCM_16":
                raise ValueError(f"Expected mono 16 kHz PCM_16: {row['id']}")
            duration = len(audio) / rate
            maximum = spec["max_train_duration"] if split == "train" else spec["max_eval_duration"]
            if not spec["min_duration"] <= duration <= maximum or abs(duration - row["duration"]) > 1 / rate:
                raise ValueError(f"Duration mismatch/policy violation: {row['id']}")
            digest = hashlib.sha256(audio.astype("<i2").tobytes()).hexdigest()
            if digest != row["audio_sha256"]:
                raise ValueError(f"Audio changed since preparation: {row['id']}")
            if digest in audio_hashes:
                raise ValueError(f"Duplicate audio across {audio_hashes[digest]} and {split}")
            audio_hashes[digest] = split
            text_key = normalize_for_scoring(text)
            if text_key in text_splits and text_splits[text_key] != split:
                raise ValueError("Transcript leakage across splits")
            text_splits[text_key] = split
        counts[split] = len(rows)
    return {"valid": True, "split_counts": counts, "audio_overlap": 0, "normalized_text_overlap": 0}
