"""Identical decoding and held-out examples for base/fine-tuned comparison."""

import json
import time
from pathlib import Path

from .common import manifest_path, read_jsonl, sha256, write_json, write_jsonl
from .metrics import aggregate, paired_comparison, score_pair


def _wait_for_device(device):
    """Finish pending GPU work before reading the wall-clock timer."""
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def evaluate(cfg, label, checkpoint=None, split="test", device=None):
    """Reload one saved model, generate transcripts, and write its measured errors."""
    import torch
    from .model import configure_decoding, restore, transcribe
    from .device import check_device, prepare_for_device
    training_device = cfg["training"].get("accelerator", "cuda")
    evaluation_device = cfg["evaluation"].get("device", training_device)
    device = check_device(device or evaluation_device)

    if label not in ("baseline", "finetuned"):
        raise ValueError("Label must be baseline or finetuned")
    if label == "finetuned" and checkpoint is None:
        raise ValueError("Fine-tuned evaluation requires --checkpoint")
    if label == "baseline" and checkpoint is not None:
        raise ValueError("Baseline always uses the pinned original model")
    manifest = manifest_path(cfg, split)
    rows = read_jsonl(manifest)
    model = restore(cfg, checkpoint=checkpoint)
    prepare_for_device(model, device)
    configure_decoding(model)
    model.to(device).eval()
    output_directory = Path(cfg["training"]["directory"]) / "evaluation" / split / label
    output_directory.mkdir(parents=True, exist_ok=True)
    predictions = []
    batch_size = cfg["evaluation"]["batch_size"]
    _wait_for_device(device)
    start = time.perf_counter()
    # FP32 inference is deliberately shared across baseline and fine-tuned model.
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = transcribe(model, [row["audio_filepath"] for row in batch], batch_size)
            for row, hypothesis in zip(batch, texts):
                predictions.append({
                    "id": row["id"], "source_id": row["source_id"], "audio_sha256": row["audio_sha256"],
                    "duration": row["duration"], "reference": row["text"], "prediction": hypothesis,
                    "scores": score_pair(row["text"], hypothesis),
                })
    _wait_for_device(device)
    elapsed = time.perf_counter() - start
    model_path = Path(checkpoint) if checkpoint else Path(cfg["model"]["directory"]) / cfg["model"]["checkpoint"]
    write_jsonl(output_directory / "predictions.jsonl", predictions)
    scores = aggregate(predictions)
    write_json(output_directory / "metrics.json", {
        **scores, "label": label, "split": split, "manifest_sha256": sha256(manifest),
        "checkpoint_sha256": sha256(model_path), "decoding": "beam_size=1; mr->mr; pnc=yes; noitn; fp32",
        "wall_seconds": elapsed, "real_time_factor": elapsed / sum(r["duration"] for r in rows),
        "timing_note": "Includes decoding and scoring, excludes model loading; no warmup excluded.",
        "device": device,
    })
    print(f"{label}: normalized WER = {scores['normalized']['wer']:.4f}")
    return output_directory


def compare(cfg, split="test"):
    """Compare original and adapted predictions from one run, using identical data."""
    root = Path(cfg["training"]["directory"]) / "evaluation" / split
    paths = [root / name for name in ("baseline", "finetuned")]
    metadata = [json.loads((p / "metrics.json").read_text()) for p in paths]
    for key in ("manifest_sha256", "decoding", "split"):
        if metadata[0][key] != metadata[1][key]:
            raise ValueError(f"Evaluation contract differs: {key}")
    predictions = [read_jsonl(p / "predictions.jsonl") for p in paths]
    result = paired_comparison(*predictions, samples=cfg["evaluation"]["bootstrap_samples"], seed=cfg["seed"])
    write_json(root / "comparison.json", {**result, "baseline": metadata[0], "finetuned": metadata[1]})
    baseline_metrics, tuned_metrics = metadata
    content = (
        "# Marathi ASR evaluation\n\n"
        f"Evaluated {tuned_metrics['utterances']} held-out {split} utterances with the same decoding and text policy.\n\n"
        "| Model | Normalized WER | Normalized CER | Raw WER |\n|---|---:|---:|---:|\n"
        f"| Baseline | {baseline_metrics['normalized']['wer']:.2%} | {baseline_metrics['normalized']['cer']:.2%} | {baseline_metrics['raw']['wer']:.2%} |\n"
        f"| Fine-tuned | {tuned_metrics['normalized']['wer']:.2%} | {tuned_metrics['normalized']['cer']:.2%} | {tuned_metrics['raw']['wer']:.2%} |\n\n"
        f"WER change (fine-tuned minus baseline): {result['normalized_wer_delta_tuned_minus_baseline']:+.2%}. "
        "A negative change indicates improvement; a positive change indicates regression.\n\n"
        f"Paired bootstrap interval (ratios): {result['paired_bootstrap_95pct_ci']}. "
        "This small adaptation experiment does not establish performance on unseen domains. "
        "Pretraining overlap is unknown. CER counts Unicode code points, not grapheme clusters.\n\n"
        "## Largest fine-tuned errors\n\n"
    )
    worst = sorted(predictions[1], key=lambda r: r["scores"]["normalized"]["word_errors"], reverse=True)[:10]
    for row in worst:
        content += f"- `{row['id']}`\n  - Reference: {row['reference']}\n  - Prediction: {row['prediction']}\n"
    (root / "report.md").write_text(content, encoding="utf-8")
    return result
