"""Bodhan's custom tokenizer registration and native NeMo checkpoint loading."""

import importlib.util
import shutil
import sys
from pathlib import Path

from .common import sha256, write_json


def download_model(cfg, code_only=False):
    from huggingface_hub import snapshot_download

    spec = cfg["model"]
    # Only the custom tokenizer is needed by our native NeMo loader. The
    # publisher's separate inference/demo scripts are not used by this pipeline.
    download_files = ["nemo/canary_multilingual_tokenizer.py", "*icense*.md", "*License*.md", "LICENSE*", "README.md"]
    if not code_only:
        download_files.append(spec["checkpoint"])
    try:
        root = Path(snapshot_download(
            repo_id=spec["repo_id"], revision=spec["revision"], local_dir=spec["directory"],
            allow_patterns=download_files,
        )).resolve()
    except Exception as exc:
        raise RuntimeError(
            "Model download failed. Accept access at https://huggingface.co/bodhan-ai/indic-transcribe-core "
            "and run `hf auth login` (or set HF_TOKEN) on the training PC. Do not commit credentials."
        ) from exc
    checkpoint = root / spec["checkpoint"]
    if not code_only and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    write_json(root / "provenance.json", {
        "repo_id": spec["repo_id"], "revision": spec["revision"],
        "checkpoint_sha256": sha256(checkpoint) if checkpoint.is_file() else None,
        "tokenizer_sha256": sha256(root / "nemo/canary_multilingual_tokenizer.py"),
    })
    return root


def register_tokenizer(source):
    """Register the publisher's module without modifying the installed NeMo package."""
    import nemo.collections.common.tokenizers as parent

    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing Bodhan tokenizer module: {path}; run download-model first")
    name = "nemo.collections.common.tokenizers.canary_multilingual_tokenizer"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(parent, "canary_multilingual_tokenizer", module)


def restore(cfg, checkpoint=None, trainer=None):
    import torch
    from nemo.collections.asr.models import EncDecMultiTaskModel
    from omegaconf import OmegaConf, open_dict

    model_root = Path(cfg["model"]["directory"]).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint else model_root / cfg["model"]["checkpoint"]
    bundled_tokenizer = checkpoint.parent / "canary_multilingual_tokenizer.py"
    source = bundled_tokenizer if bundled_tokenizer.exists() else model_root / "nemo/canary_multilingual_tokenizer.py"
    register_tokenizer(source)
    saved = EncDecMultiTaskModel.restore_from(str(checkpoint), return_config=True, map_location=torch.device("cpu"))
    with open_dict(saved):
        # Clear publisher training paths and prevent an unrelated timestamp model download.
        saved.train_ds = None
        saved.validation_ds = None
        saved.test_ds = None
        saved.optim = None
        saved.restore_timestamps_model = False
        saved.log_prediction = False
        saved.use_loss_mask_for_prompt = True
        saved.multitask_metrics_cfg = OmegaConf.create({"log_predictions": False, "metrics": {
            "wer": {"_target_": "nemo.collections.asr.metrics.WER", "constraint": ".source_lang==.target_lang"}
        }})
    model = EncDecMultiTaskModel.restore_from(
        str(checkpoint), override_config_path=saved, map_location=torch.device("cpu"), trainer=trainer,
    )
    if model.prompt_format not in ("canary", "canary2"):
        raise ValueError(f"Unexpected prompt format {model.prompt_format}; inspect the pinned checkpoint")
    # Explicit Marathi routing must work before allocating GPU memory or beginning training.
    token_ids = model.tokenizer.text_to_ids("नमस्कार महाराष्ट्र", lang_id="mr")
    if not token_ids:
        raise ValueError("Marathi tokenizer produced no tokens")
    return model


def configure_trainable(model, mode, decoder_layers=2, lora=None):
    """Selective tuning leaves the tied vocabulary embeddings frozen in decoder_top mode."""
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "full")
    lora_metadata = None
    if mode == "decoder_lora":
        from .lora import inject_decoder_lora
        settings = {"rank": 8, "alpha": 16, "dropout": 0.05, **(lora or {})}
        targets = inject_decoder_lora(model, **settings)
        model._decoder_lora_targets = targets
        lora_metadata = {**settings, "target_modules": targets}
    elif mode == "decoder":
        for module in (model.transf_decoder, model.log_softmax):
            module.requires_grad_(True)
    elif mode == "decoder_top":
        # NeMo's get_transformer wrapper exposes .decoder.layers; fail if architecture differs.
        decoder = getattr(model.transf_decoder, "decoder", model.transf_decoder)
        layers = getattr(decoder, "layers", None)
        if layers is None or not 1 <= decoder_layers <= len(layers):
            raise ValueError("Cannot select decoder layers; inspect model.transf_decoder.named_modules()")
        for layer in layers[-decoder_layers:]:
            layer.requires_grad_(True)
        final_norm = getattr(decoder, "final_layer_norm", None)
        if final_norm is not None:
            final_norm.requires_grad_(True)
    elif mode != "full":
        raise ValueError(f"Unsupported mode {mode}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise ValueError("No trainable parameters")
    result = {"total_parameters": total, "trainable_parameters": trainable, "fraction_trainable": trainable / total,
            "mode": mode, "decoder_layers": decoder_layers,
            "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad]}
    if lora_metadata is not None:
        result["lora"] = lora_metadata
    return result


def keep_frozen_modules_in_eval(model):
    # model.train() recursively changes frozen BatchNorm/dropout too; reset all fully frozen subtrees.
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and not any(p.requires_grad for p in parameters):
            module.eval()
    if getattr(model, "_decoder_lora_targets", None):
        # In B/C only adapter dropout is active. Fixed decoder dropout should not
        # add noise to a frozen base; eval mode still allows gradients through it.
        from .lora import adapter_modules
        model.transf_decoder.eval()
        for module in adapter_modules(model).values():
            module.train()


def export_support(cfg, destination):
    root = Path(cfg["model"]["directory"])
    destination = Path(destination)
    shutil.copy2(root / "nemo/canary_multilingual_tokenizer.py", destination / "canary_multilingual_tokenizer.py")
    for path in root.glob("*.md"):
        if "license" in path.name.lower():
            shutil.copy2(path, destination / path.name)


def configure_decoding(model):
    from omegaconf import OmegaConf, open_dict
    decoding = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(decoding):
        decoding.strategy = "beam"
        if "beam" not in decoding:
            decoding.beam = {}
        decoding.beam.beam_size = 1
    model.change_decoding_strategy(decoding)


def transcribe(model, paths, batch_size=1):
    from .data import PROMPT
    prompt = {key: value for key, value in PROMPT.items() if key != "taskname"}
    if model.prompt_format == "canary":
        prompt = {key: prompt[key] for key in ("source_lang", "target_lang", "task", "pnc")}
    predictions = model.transcribe(paths, batch_size=batch_size, return_hypotheses=True, num_workers=0, **prompt)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    texts = [item if isinstance(item, str) else item.text for item in predictions]
    if len(texts) != len(paths) or any(not isinstance(t, str) for t in texts):
        raise ValueError("ASR output count/type differs from requested audio")
    return texts
