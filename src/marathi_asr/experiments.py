"""Compare A and B, optionally train C, choose on validation, then test all models.

Every candidate, including the original model, is evaluated on the same data.
Training and evaluation run in separate processes to release device memory.
"""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import manifest_path, read_jsonl, sha256, utc_now, write_json
from .metrics import aggregate, paired_comparison, score_pair


NAMES = {
    "baseline": "Original model",
    "a_decoder": "A: selected decoder weights",
    "b_lora": "B: decoder LoRA from original model",
    "c_decoder_lora": "C: decoder LoRA from A",
}


def _wer(metrics):
    """Selection accepts only finite validation scores; test scores are never inputs."""
    if metrics.get("split") != "validation":
        raise ValueError("Model selection requires validation metrics, never test metrics")
    value = metrics.get("normalized", {}).get("wer")
    if not _valid_error_rate(value):
        raise ValueError("Missing or invalid normalized validation WER")
    return value


def _valid_error_rate(value):
    """Reject missing, nonnumeric, negative, or infinite WER/CER values."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def should_run_conditional(a_metrics, b_metrics):
    """A strict A < B gate; ties do not trigger another training stage."""
    return _wer(a_metrics) < _wer(b_metrics)


def select_winner(validation):
    """Prefer the earlier, simpler candidate on an exact tie (baseline first)."""
    if "baseline" not in validation:
        raise ValueError("Model selection must include the original baseline")
    candidates = [name for name in NAMES if name in validation]
    return min(candidates, key=lambda name: _wer(validation[name]))


def _conditional_decision(cfg, a_metrics, b_metrics):
    """C receives another training stage only when A beats B on validation WER."""
    a_is_better = should_run_conditional(a_metrics, b_metrics)
    enabled = cfg["experiment"].get("conditional_decoder_lora", True)
    if not enabled:
        reason = "Disabled in configuration."
    elif a_is_better:
        reason = "A has strictly lower validation WER than B."
    else:
        reason = "A does not have strictly lower validation WER than B; ties skip C."
    return {
        "enabled": enabled,
        "run": enabled and a_is_better,
        "rule": "A normalized validation WER < B normalized validation WER",
        "reason": reason,
    }


def _run_stage(config_path, arguments, log_path):
    """Subprocess boundaries release model/optimizer device memory between stages."""
    command = [sys.executable, "-m", "marathi_asr", "--config", str(config_path), *arguments]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {' '.join(arguments)}; log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"Stage {' '.join(arguments)} failed ({exit_code}); inspect {log_path}")
    return time.perf_counter() - started


def _source_hashes():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


def _read_evaluation(cfg, label, split, checkpoint_hash, decoding=None):
    """Check the saved evaluation, then recompute its scores from the predictions.

    This prevents comparisons between different data, checkpoints, or decoding
    settings, and catches saved metrics that do not match their predictions.
    """
    directory = Path(cfg["training"]["directory"]) / "evaluation" / split / label
    metadata = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    expected = {
        "split": split,
        "manifest_sha256": sha256(manifest_path(cfg, split)),
        "checkpoint_sha256": checkpoint_hash,
        "label": label,
    }
    if decoding is not None:
        expected["decoding"] = decoding
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Evaluation contract differs: {key}")
    if not isinstance(metadata.get("decoding"), str) or not metadata["decoding"]:
        raise ValueError("Missing evaluation decoding contract")
    predictions = read_jsonl(directory / "predictions.jsonl")
    examples_by_id = {row["id"]: row for row in read_jsonl(manifest_path(cfg, split))}
    prediction_ids = {row["id"] for row in predictions}
    if not predictions or len(prediction_ids) != len(predictions):
        raise ValueError("Missing or duplicate evaluation predictions")
    if prediction_ids != examples_by_id.keys():
        raise ValueError("Predictions do not cover the evaluation manifest")
    for row in predictions:
        source = examples_by_id[row["id"]]
        if row["reference"] != source["text"] or row["audio_sha256"] != source["audio_sha256"]:
            raise ValueError(f"Mismatched evaluation example {row['id']}")
        row["scores"] = score_pair(row["reference"], row["prediction"])
    recomputed = aggregate(predictions)
    if metadata.get("utterances") != recomputed["utterances"]:
        raise ValueError("Evaluation utterance count differs from predictions")
    for style in ("raw", "normalized"):
        for metric in ("wer", "cer"):
            value = metadata.get(style, {}).get(metric)
            if not _valid_error_rate(value):
                raise ValueError(f"Missing or invalid {style} {metric}")
            if not math.isclose(value, recomputed[style][metric], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Saved {style} {metric} differs from predictions")
    return metadata, predictions


def _training_record(run_cfg, elapsed):
    """Load a completed training run and verify its exported checkpoint."""
    root = Path(run_cfg["training"]["directory"])
    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"Training did not complete: {root}")
    checkpoint = root / "model.nemo"
    if status.get("export_sha256") != sha256(checkpoint):
        raise ValueError(f"Export differs from completed training artifact: {checkpoint}")
    parameters = json.loads((root / "parameters.json").read_text(encoding="utf-8"))
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": status["export_sha256"],
        "parameters": parameters,
        "training": status,
        "training_process_wall_seconds": elapsed,
    }


def _write_report(root, result):
    """Write a readable summary; comparison.json retains all detailed results."""
    decision = result["decision"]
    lines = [
        "# Marathi adaptation comparison",
        "",
        f"Selected using validation only: **{NAMES[decision['selected_model']]}**.",
        "",
        "Test results are reported after this decision and do not change it. Lower WER/CER is better.",
        "",
        "| Model | Validation WER | Test WER | Test CER | Raw test WER | Raw test CER | "
        "Trainable parameters | Training seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in result["models"].items():
        validation = record["validation"]["normalized"]
        normalized_test = record["test"]["normalized"]
        raw_test = record["test"]["raw"]
        parameter_count = record.get("parameters", {}).get("trainable_parameters", 0)
        training_seconds = record.get("training", {}).get("training_wall_seconds", 0)
        cells = [
            NAMES[name],
            f"{validation['wer']:.2%}",
            f"{normalized_test['wer']:.2%}",
            f"{normalized_test['cer']:.2%}",
            f"{raw_test['wer']:.2%}",
            f"{raw_test['cer']:.2%}",
            f"{parameter_count:,}",
            f"{training_seconds:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    conditional_status = "run" if decision["conditional"]["run"] else "skipped"
    lines.extend([
        "",
        f"Conditional stage C: **{conditional_status}**. " + decision["conditional"]["reason"],
        "",
        "A and B use the same data and optimizer-step budget and start from the original model. "
        "C starts from A and receives another training budget, so it is a continuation experiment, "
        "not a comparison with equal total training. Its row shows only the additional stage's training time. "
        "Training seconds include fitting and final validation; exclude checkpoint export. "
        "Full subprocess and inference timings are in comparison.json.",
        "",
        "## Paired test comparisons against the original model",
        "",
        "| Candidate | WER change | 95% bootstrap interval |",
        "|---|---:|---|",
    ])
    for name, comparison in result["paired_test_vs_baseline"].items():
        wer_change = comparison["normalized_wer_delta_tuned_minus_baseline"]
        interval = comparison["paired_bootstrap_95pct_ci"]
        interval_text = "unavailable"
        if interval is not None:
            interval_text = f"[{interval[0]:+.2%}, {interval[1]:+.2%}]"
        lines.append(f"| {NAMES[name]} | {wer_change:+.2%} | {interval_text} |")
    lines.extend([
        "",
        "Negative WER change favors the adapted model. Intervals use paired utterance resampling; "
        "they ignore dependence between recordings of the same speaker or sentence. "
        "A small validation set can give an unstable selection. CER counts Unicode code points, "
        "not grapheme clusters. Pretraining overlap is unknown.",
        "",
        "Exact data, checkpoint and code hashes, timings, parameter counts, raw/normalized metrics, "
        "and the frozen selection decision are in comparison.json and decision.json.",
        "",
    ])
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(cfg):
    """Run A/B, optionally continue A with LoRA, then evaluate all on the same test set.

    Return the experiment directory. A fresh directory is required. Failed/previous
    runs are never silently reused: change experiment.directory to retry.
    """
    root = Path(cfg["experiment"]["directory"]).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Experiment directory already used; choose a new experiment.directory")
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes()
    write_json(root / "input_config.json", cfg)
    write_json(root / "status.json", {"status": "running", "started_at": utc_now()})
    stages = []
    manifest_hashes = {}
    run_configs = {}
    config_paths = {}
    config_hashes = {}

    def save_run_config(name, mode, initial_checkpoint=None):
        """Give each strategy its own output folder and explicit starting model."""
        run_cfg = copy.deepcopy(cfg)
        run_cfg["training"].update(directory=str(root / name), mode=mode)
        # A and B always start from the original model, even if the input config
        # contains a checkpoint from an earlier standalone training run.
        run_cfg["training"].pop("init_checkpoint", None)
        if initial_checkpoint is not None:
            run_cfg["training"]["init_checkpoint"] = initial_checkpoint
        config_path = root / "configs" / f"{name}.json"
        write_json(config_path, run_cfg)
        run_configs[name] = run_cfg
        config_paths[name] = config_path
        config_hashes[name] = sha256(config_path)
        return run_cfg

    def ensure_unchanged():
        """Do not mix results if code, data lists, or settings change mid-run."""
        if source_hashes != _source_hashes():
            raise ValueError("Pipeline source changed during the experiment")
        for split, expected_hash in manifest_hashes.items():
            if sha256(manifest_path(cfg, split)) != expected_hash:
                raise ValueError("Data manifests changed during the experiment")
        for name, expected_hash in config_hashes.items():
            if sha256(config_paths[name]) != expected_hash:
                raise ValueError("Run configuration changed during the experiment")

    def launch(name, *arguments):
        ensure_unchanged()
        log_path = root / "logs" / f"{len(stages) + 1:02d}-{name}-{arguments[0]}.log"
        elapsed = _run_stage(config_paths[name], list(arguments), log_path)
        stages.append({
            "model": name,
            "arguments": list(arguments),
            "wall_seconds": elapsed,
            "log": str(log_path),
        })
        write_json(root / "stages.json", stages)
        ensure_unchanged()
        return elapsed

    try:
        # 1. Check the machine, prepare shared data, and download the original model.
        baseline_cfg = save_run_config("baseline", "decoder_top")
        for stage in ("preflight", "prepare", "validate", "download-model"):
            launch("baseline", stage)
            if stage == "preflight":
                report = json.loads((root / "baseline" / "preflight.json").read_text())
                if report.get("ready_to_train") is not True:
                    raise RuntimeError("Preflight did not pass; inspect baseline/preflight.json")
            if stage == "validate":
                for split in ("train", "validation", "test"):
                    manifest_hashes[split] = sha256(manifest_path(cfg, split))

        # 2. Measure the original model on validation data. It can remain the winner.
        original_checkpoint = Path(cfg["model"]["directory"]) / cfg["model"]["checkpoint"]
        records = {
            "baseline": {
                "checkpoint": str(original_checkpoint.resolve()),
                "checkpoint_sha256": sha256(original_checkpoint),
            }
        }
        launch("baseline", "evaluate", "--label", "baseline", "--split", "validation")
        baseline_metrics, _ = _read_evaluation(
            baseline_cfg, "baseline", "validation", records["baseline"]["checkpoint_sha256"]
        )
        records["baseline"]["validation"] = baseline_metrics
        decoding = baseline_metrics["decoding"]

        def train_and_validate(name, mode, initial_checkpoint=None):
            """Train once, verify its starting checkpoint, then score validation."""
            run_cfg = save_run_config(name, mode, initial_checkpoint)
            elapsed = launch(name, "train")
            record = _training_record(run_cfg, elapsed)
            source_name = "a_decoder" if initial_checkpoint else "baseline"
            expected_initial_hash = records[source_name]["checkpoint_sha256"]
            if record["training"].get("initial_checkpoint_sha256") != expected_initial_hash:
                raise ValueError("Training initialization differs from the declared source checkpoint")
            if record["training"].get("optimizer_steps") != cfg["training"]["max_steps"]:
                raise ValueError("Training did not use the declared optimizer-step budget")
            launch(
                name, "evaluate", "--label", "finetuned",
                "--checkpoint", record["checkpoint"], "--split", "validation",
            )
            record["validation"], _ = _read_evaluation(
                run_cfg, "finetuned", "validation", record["checkpoint_sha256"], decoding
            )
            records[name] = record

        # 3. Compare the two independent approaches, both starting from the original.
        train_and_validate("a_decoder", "decoder_top")
        train_and_validate("b_lora", "decoder_lora")

        # 4. Continue A with LoRA only if A beat B on validation WER.
        conditional = _conditional_decision(
            cfg, records["a_decoder"]["validation"], records["b_lora"]["validation"]
        )
        write_json(root / "conditional_decision.json", conditional)
        if conditional["run"]:
            train_and_validate("c_decoder_lora", "decoder_lora", records["a_decoder"]["checkpoint"])

        # 5. Save the winner before any model sees the test set.
        validation = {name: record["validation"] for name, record in records.items()}
        winner = select_winner(validation)
        ensure_unchanged()
        checkpoint_hashes = {name: record["checkpoint_sha256"] for name, record in records.items()}
        decision = {
            "selected_model": winner,
            "selected_checkpoint": records[winner]["checkpoint"],
            "selection_metric": "normalized validation WER",
            "tie_break": "baseline, A, B, C order",
            "selected_before_test": True,
            "decided_at": utc_now(),
            "conditional": conditional,
            "validation": validation,
            "manifest_sha256": manifest_hashes,
            "decoding": decoding,
            "source_sha256": source_hashes,
            "checkpoint_sha256": checkpoint_hashes,
            "config_sha256": config_hashes,
        }
        write_json(root / "decision.json", decision)
        decision_hash = sha256(root / "decision.json")

        # 6. Evaluate every available model on the same held-out test set.
        test_predictions = {}
        for name, record in records.items():
            label = "baseline" if name == "baseline" else "finetuned"
            arguments = ["evaluate", "--label", label, "--split", "test"]
            if name != "baseline":
                arguments += ["--checkpoint", record["checkpoint"]]
            launch(name, *arguments)
            record["test"], test_predictions[name] = _read_evaluation(
                run_configs[name], label, "test", record["checkpoint_sha256"], decoding
            )
            if sha256(root / "decision.json") != decision_hash:
                raise ValueError("Selection decision changed after test evaluation started")

        # 7. Report test comparisons without changing the already saved winner.
        paired_test_results = {}
        for name in records:
            if name == "baseline":
                continue
            paired_test_results[name] = paired_comparison(
                test_predictions["baseline"],
                test_predictions[name],
                samples=cfg["evaluation"]["bootstrap_samples"],
                seed=cfg["seed"],
            )
        result = {
            "decision": decision,
            "decision_sha256": decision_hash,
            "models": records,
            "paired_test_vs_baseline": paired_test_results,
            "stages": stages,
            "note": "C, when run, has A's training plus an additional LoRA stage; "
                    "test never gates training or selection.",
        }
        write_json(root / "comparison.json", result)
        _write_report(root, result)
        write_json(root / "status.json", {
            "status": "completed",
            "finished_at": utc_now(),
            "strategies": [name for name in records if name != "baseline"],
            "selected_model": winner,
            "comparison": str(root / "comparison.json"),
        })
        print(f"Validation selected {NAMES[winner]}. Results: {root / 'comparison.md'}")
        return root
    except Exception as exc:
        write_json(root / "status.json", {
            "status": "failed",
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise
