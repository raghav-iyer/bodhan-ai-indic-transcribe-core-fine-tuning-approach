"""Single-device fine-tuning with a native NeMo/Lightning training loop."""

import json
import math
from pathlib import Path
from time import perf_counter

from .common import environment, manifest_path, read_jsonl, sha256, utc_now, write_json
from .data import validate


def _check_resume(cfg, output, resume, initial_hash):
    """Resume only when the config, data, and starting model are unchanged."""
    config_path = output / "config.json"
    if resume is not None and not config_path.is_file():
        raise ValueError("Resume requires the original run directory and its provenance files")
    if not config_path.exists():
        return
    if resume is None:
        raise ValueError("Run directory already used. Pick a new training.directory or pass --resume last.ckpt")

    original_config = json.loads(config_path.read_text())
    # A resumed scheduler retains its original decay schedule. To extend the
    # training budget, start a new run with init_checkpoint instead.
    if original_config != cfg:
        raise ValueError("Resume requires the exact original config, including max_steps")
    saved_hashes = json.loads((output / "data_hashes.json").read_text())
    for split, saved_hash in saved_hashes.items():
        if saved_hash != sha256(manifest_path(cfg, split)):
            raise ValueError("Manifests changed since the checkpoint; refusing unsafe resume")
    initialization = json.loads((output / "initialization.json").read_text())
    if initialization["checkpoint_sha256"] != initial_hash:
        raise ValueError("Initial checkpoint changed since this run; refusing unsafe resume")
    if Path(resume).resolve().parent != (output / "checkpoints").resolve():
        raise ValueError("Resume checkpoint must belong to this run's checkpoints directory")


def _audit_transcript_lengths(model, cfg):
    """Reject text that would be empty or truncated by the pretrained decoder."""
    decoder_config = model.cfg.transf_decoder.get("config_dict", {})
    max_sequence_length = decoder_config.get("max_sequence_length", 1024)
    audit = {}
    for split in ("train", "validation", "test"):
        recordings = read_jsonl(manifest_path(cfg, split))
        lengths = [
            len(model.tokenizer.text_to_ids(recording["text"], lang_id="mr"))
            for recording in recordings
        ]
        # Reserve 32 tokens for the fixed task prompt and start/end tokens.
        if not min(lengths) or max(lengths) + 32 > max_sequence_length:
            raise ValueError(
                f"{split} transcript exceeds the pretrained decoder token budget; "
                "do not silently truncate"
            )
        audit[split] = {
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "decoder_limit": max_sequence_length,
        }
    return audit


def _loader_config(cfg, split, device):
    """Build fresh data settings instead of reusing the publisher's paths/filters."""
    training = cfg["training"]
    return {
        "use_lhotse": True,
        "manifest_filepath": str(manifest_path(cfg, split).resolve()),
        "sample_rate": 16000,
        "batch_size": training["batch_size"],
        "batch_duration": None,
        "quadratic_duration": None,
        "use_bucketing": False,
        "shuffle": split == "train",
        "num_workers": training["num_workers"],
        "pin_memory": device == "cuda",
        "text_field": "text",
        "lang_field": "target_lang",
        "seed": cfg["seed"],
        "drop_last": False,
    }


def final_checkpoint_is_better(validation_results, previous_best_score):
    """Compare the endpoint with periodic validation using the same WER metric."""
    if len(validation_results) != 1 or "val_wer" not in validation_results[0]:
        raise ValueError("Final validation must report one val_wer score")
    final_score = float(validation_results[0]["val_wer"])
    if not math.isfinite(final_score) or final_score < 0:
        raise FloatingPointError("Final validation WER is not finite and nonnegative")
    if previous_best_score is None:
        return True, final_score
    previous_score = float(previous_best_score)
    if not math.isfinite(previous_score) or previous_score < 0:
        raise FloatingPointError("Previous best validation WER is invalid")
    # Retain the earlier checkpoint on a tie.
    return final_score < previous_score, min(final_score, previous_score)


def train(cfg, resume=None):
    """Train one strategy, select its best validation checkpoint, and export it."""
    import torch

    from .device import check_device, device_info, prepare_for_device

    # Fail before importing NeMo if the requested device is unavailable.
    device = check_device(cfg["training"].get("accelerator", "cuda"))
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    from omegaconf import OmegaConf, open_dict
    from .model import (
        configure_decoding,
        configure_trainable,
        export_support,
        keep_frozen_modules_in_eval,
        restore,
    )

    # 1. Validate the inputs and protect existing runs from accidental overwrite.
    validate(cfg)
    training = cfg["training"]
    output = Path(training["directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_checkpoint = Path(cfg["model"]["directory"]) / cfg["model"]["checkpoint"]
    initial_checkpoint = Path(training.get("init_checkpoint") or base_checkpoint).resolve()
    initial_hash = sha256(initial_checkpoint)
    _check_resume(cfg, output, resume, initial_hash)
    pl.seed_everything(cfg["seed"], workers=True)
    precision = training["precision"]
    if precision == "auto":
        if device == "cuda":
            precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
        else:
            precision = "32-true"

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{step:06d}-{val_wer:.4f}",
    )

    # Lightning calls these methods during fit(); NeMo still computes the ASR loss.
    class TrainingEvidence(Callback):
        """Record loss and verify that training produces finite, real updates."""

        def __init__(self):
            self.losses = []
            self.watched_name = None
            self.initial_values = None
            self.nonzero_gradient_steps = 0

        def on_train_start(self, trainer, model):
            # B starts at zero; its gradient can be nonzero on the very first step.
            # A starts random but its first-step gradient is zero when B is zero.
            candidates = [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.ndim > 1
            ]
            self.watched_name, parameter = candidates[0]
            for name, candidate in candidates:
                if name.endswith(".lora_B"):
                    self.watched_name, parameter = name, candidate
                    break
            # A small snapshot is enough to verify an update without copying the model.
            self.initial_values = parameter.detach().flatten()[:8192].float().cpu().clone()

        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            keep_frozen_modules_in_eval(model)

        def on_before_optimizer_step(self, trainer, model, optimizer):
            gradients = {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            }
            # Aggregate checks before transferring one Boolean to the host; a
            # separate GPU synchronization per adapter would slow training down.
            if not gradients:
                raise FloatingPointError("Missing or non-finite trainable gradients")
            finite_checks = [torch.isfinite(gradient).all() for gradient in gradients.values()]
            if not torch.stack(finite_checks).all():
                raise FloatingPointError("Missing or non-finite trainable gradients")
            watched_gradient = gradients.get(self.watched_name)
            if watched_gradient is not None and torch.count_nonzero(watched_gradient).item():
                self.nonzero_gradient_steps += 1

        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            value = float(loss.detach().float().cpu())
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Non-finite training loss")
            self.losses.append(value)
            model.log("observed_train_loss", value, on_step=True, on_epoch=False)
            record = {"global_step": trainer.global_step, "batch_idx": batch_idx, "loss": value}
            with (output / "losses.jsonl").open("a") as loss_file:
                loss_file.write(json.dumps(record) + "\n")

    # 2. Create the trainer, restore the model, and choose which weights can learn.
    evidence = TrainingEvidence()
    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else device,
        devices=1,
        precision=precision,
        max_steps=training["max_steps"],
        max_epochs=-1,
        accumulate_grad_batches=training["accumulate_grad_batches"],
        gradient_clip_val=1.0,
        # Integer interval counts microbatches, not optimizer steps. May cross epoch boundaries.
        val_check_interval=training["val_check_interval"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=1,
        log_every_n_steps=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step"), evidence],
        logger=[
            CSVLogger(str(output), name="csv"),
            TensorBoardLogger(str(output), name="tensorboard"),
        ],
        default_root_dir=str(output),
    )
    model = restore(cfg, checkpoint=initial_checkpoint, trainer=trainer)
    prepare_for_device(model, device)
    parameters = configure_trainable(
        model, training["mode"], training.get("decoder_layers", 2), training.get("lora")
    )
    configure_decoding(model)

    # 3. Check transcript lengths, then configure data and the optimizer.
    token_audit = _audit_transcript_lengths(model, cfg)
    model.setup_training_data(OmegaConf.create(_loader_config(cfg, "train", device)))
    model.setup_validation_data(OmegaConf.create(_loader_config(cfg, "validation", device)))
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create({
            "name": "adamw",
            "lr": training["learning_rate"],
            "betas": [0.9, 0.98],
            "weight_decay": training["weight_decay"],
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": training["warmup_steps"],
                "min_lr": training["learning_rate"] * 0.1,
                "max_steps": training["max_steps"],
            },
        })
    # NeMo configure_optimizers constructs optimizer/scheduler from cfg.optim during fit.

    # 4. Record the exact inputs so results can be reproduced and safely resumed.
    write_json(output / "config.json", cfg)
    write_json(output / "parameters.json", parameters)
    write_json(output / "token_audit.json", token_audit)
    write_json(output / "environment.json", {
        **environment(), **device_info(device), "precision": precision,
    })
    write_json(output / "data_hashes.json", {
        split: sha256(manifest_path(cfg, split)) for split in ("train", "validation", "test")
    })
    write_json(output / "initialization.json", {
        "checkpoint": str(initial_checkpoint),
        "checkpoint_sha256": initial_hash,
        "model_repo_id": cfg["model"]["repo_id"],
        "model_revision": cfg["model"]["revision"],
    })
    OmegaConf.save(model.cfg, output / "nemo_config.yaml")
    write_json(output / "status.json", {"status": "training", "started_at": utc_now(), "resume_from": resume})
    try:
        # 5. Train and confirm that at least one watched parameter actually changed.
        training_started = perf_counter()
        trainer.fit(model, ckpt_path=resume)
        if trainer.global_step < training["max_steps"] or not evidence.losses:
            raise RuntimeError("Trainer stopped before the requested optimizer steps")
        watched_parameter = dict(model.named_parameters())[evidence.watched_name]
        final_values = watched_parameter.detach().flatten()[:8192].float().cpu()
        max_parameter_change = float((final_values - evidence.initial_values).abs().max())
        if not torch.isfinite(watched_parameter).all():
            raise FloatingPointError("Watched trainable weights are non-finite")
        if max_parameter_change == 0 or not evidence.nonzero_gradient_steps:
            raise RuntimeError(
                "Watched trainable weights had no gradient or update; "
                "refusing to claim a successful fine-tune"
            )

        # 6. Select the checkpoint with the lowest validation WER. Never use test WER.
        # Guarantee final validation when training stops before its periodic interval.
        final_validation = trainer.validate(model)
        training_wall_seconds = perf_counter() - training_started
        use_final, selected_score = final_checkpoint_is_better(
            final_validation, checkpoint_callback.best_model_score
        )
        if use_final:
            # validate() runs outside FITTING, so Lightning does not update its
            # checkpoint callback. Persist the endpoint and its callback state
            # explicitly, including the best score a later resume must retain.
            best_checkpoint = str(output / f"checkpoints/best-final-{trainer.global_step:06d}.ckpt")
            score_tensor = torch.tensor(selected_score)
            checkpoint_callback.best_model_path = best_checkpoint
            checkpoint_callback.best_model_score = score_tensor
            checkpoint_callback.current_score = score_tensor
            checkpoint_callback.best_k_models = {best_checkpoint: score_tensor}
            checkpoint_callback.kth_best_model_path = best_checkpoint
            checkpoint_callback.kth_value = score_tensor
            trainer.save_checkpoint(best_checkpoint)
        else:
            best_checkpoint = checkpoint_callback.best_model_path
            if not best_checkpoint:
                raise RuntimeError("Best validation score has no corresponding checkpoint")
        checkpoint_callback.last_model_path = str(output / "checkpoints/last.ckpt")
        trainer.save_checkpoint(checkpoint_callback.last_model_path)
        if not use_final:
            # Loading our own trusted Lightning checkpoint; do not accept arbitrary external pickle files.
            state = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=True)
            del state
        # 7. Export the selected model; keep separate LoRA weights before merging them.
        if training["mode"] == "decoder_lora":
            from .lora import merge_decoder_lora, save_adapters
            # Save the selected checkpoint's adapters before merging. Their exact
            # base differs between the original-model B run and the A-initialized C run.
            save_adapters(model, output, initial_checkpoint, initial_hash)
            merge_decoder_lora(model)
        exported = output / "model.nemo"
        model.save_to(str(exported))
        export_support(cfg, output)
        write_json(output / "status.json", {
            "status": "completed",
            "finished_at": utc_now(),
            "optimizer_steps": trainer.global_step,
            "first_observed_loss": evidence.losses[0],
            "last_observed_loss": evidence.losses[-1],
            "watched_parameter": evidence.watched_name,
            "max_abs_parameter_update": max_parameter_change,
            "nonzero_gradient_steps": evidence.nonzero_gradient_steps,
            "initial_checkpoint_sha256": initial_hash,
            "training_wall_seconds": training_wall_seconds,
            "training_wall_seconds_scope": "trainer.fit plus final validation; excludes export",
            "export_selection": "final validation WER" if use_final else "best periodic validation WER",
            "selected_validation_wer": selected_score,
            "final_validation_wer": float(final_validation[0]["val_wer"]),
            "best_checkpoint": best_checkpoint,
            "export_sha256": sha256(exported),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "device": device,
        })
    except Exception as exc:
        write_json(output / "status.json", {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise
    return output
