"""LoRA for the native NeMo decoder, with ordinary Linear layers at export.

No encoder layers are removed. These adapters implement W x + (alpha/r) B A x;
the pretrained Linear weight and bias stay frozen during adapter training.
"""

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import sha256, write_json


class LoRALinear(nn.Module):
    """Keep a pretrained projection and learn a small correction beside it."""

    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA requires an ordinary torch.nn.Linear projection")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA alpha must be positive and dropout must be in [0, 1)")
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        # A compresses the input to `rank` values; B expands it to the output size.
        # Starting B at zero makes the initial correction exactly zero.
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.train(base.training)

    def forward(self, inputs):
        compressed = F.linear(self.dropout(inputs), self.lora_A)
        correction = F.linear(compressed, self.lora_B)
        return self.base(inputs) + self.scaling * correction

    @torch.no_grad()
    def merged(self):
        # Accumulate in FP32 even if the runtime model uses lower precision.
        delta = self.lora_B.float() @ self.lora_A.float()
        self.base.weight.add_((self.scaling * delta).to(self.base.weight.dtype))
        return self.base


def _decoder_targets(model):
    """Find the native attention projections before replacing any of them."""
    decoder = getattr(model.transf_decoder, "decoder", model.transf_decoder)
    layers = getattr(decoder, "layers", None)
    if layers is None or not len(layers):
        raise ValueError("Cannot find NeMo decoder layers for LoRA")
    prefix = next(
        (name for name, module in model.named_modules() if module is decoder), None
    )
    if prefix is None:
        raise ValueError("Decoder must be a registered model module")
    targets = []
    for index, layer in enumerate(layers):
        # NeMo calls self-attention the first sublayer and cross-attention the second.
        for attention in ("first_sub_layer", "second_sub_layer"):
            parent = getattr(layer, attention, None)
            for projection in ("query_net", "key_net", "value_net", "out_projection"):
                module = getattr(parent, projection, None)
                name = f"{prefix}.layers.{index}.{attention}.{projection}"
                if not isinstance(module, nn.Linear):
                    raise ValueError(
                        f"Expected a native NeMo Linear at {name}; "
                        "architecture changed or LoRA already installed"
                    )
                targets.append((name, parent, projection, module))
    return targets


def inject_decoder_lora(model, rank=8, alpha=16, dropout=0.05):
    """Target Q/K/V/output in every decoder self- and cross-attention block.

    These are the actual projection names in NeMo's TransformerDecoderBlock and
    MultiHeadAttention. Validate the complete architecture before modifying it.
    """
    targets = _decoder_targets(model)
    # Construct first: invalid hyperparameters cannot leave a partially modified model.
    replacements = []
    for name, parent, projection, base in targets:
        adapter = LoRALinear(base, rank, alpha, dropout)
        replacements.append((name, parent, projection, adapter))
    for _, parent, projection, adapter in replacements:
        setattr(parent, projection, adapter)
    return [name for name, _, _, _ in targets]


def adapter_modules(model):
    """Return installed adapters, indexed by their full model paths."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def save_adapters(model, destination, base_checkpoint, base_sha256):
    """Save tensor-only adapters and enough metadata to identify their exact base."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    modules = adapter_modules(model)
    if not modules:
        raise ValueError("No LoRA adapters to save")
    tensors = {}
    targets = {}
    for name, module in modules.items():
        for key in ("lora_A", "lora_B"):
            tensors[f"{name}.{key}"] = getattr(module, key).detach().cpu().contiguous()
        targets[name] = {
            "rank": module.rank,
            "alpha": module.alpha,
            "dropout": module.dropout.p,
            "in_features": module.base.in_features,
            "out_features": module.base.out_features,
        }
    path = destination / "adapters.pt"
    torch.save(tensors, path)
    metadata = {
        "format_version": 1,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "base_checkpoint_sha256": base_sha256,
        "weights": path.name,
        "weights_sha256": sha256(path),
        "targets": targets,
    }
    write_json(destination / "adapters.json", metadata)
    return metadata


def _validate_saved_target(name, base, metadata, tensors):
    """Check one saved adapter without changing the model."""
    required_fields = {"rank", "alpha", "dropout", "in_features", "out_features"}
    if not isinstance(metadata, dict) or set(metadata) != required_fields:
        raise ValueError(f"Invalid adapter target metadata: {name}")

    rank = metadata["rank"]
    alpha = metadata["alpha"]
    dropout = metadata["dropout"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("Adapter rank must be a positive integer")
    valid_alpha = isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0
    valid_dropout = isinstance(dropout, (int, float)) and 0 <= dropout < 1
    if not valid_alpha or not valid_dropout:
        raise ValueError("Invalid adapter alpha or dropout")
    if metadata["in_features"] != base.in_features or metadata["out_features"] != base.out_features:
        raise ValueError(f"Adapter projection shape mismatch: {name}")

    expected_shapes = {
        "lora_A": (rank, base.in_features),
        "lora_B": (base.out_features, rank),
    }
    for key, expected_shape in expected_shapes.items():
        tensor = tensors[f"{name}.{key}"]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_shape
            or not tensor.is_floating_point()
        ):
            raise ValueError(f"Adapter tensor shape or dtype mismatch: {name}.{key}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite adapter tensor: {name}.{key}")
    return {"rank": rank, "alpha": alpha, "dropout": dropout}


def load_adapters(model, directory, base_checkpoint):
    """Load into a native model freshly restored from ``base_checkpoint``.

    Both files' hashes and every target, setting, and tensor shape are checked
    before the model is modified. Never load A-initialized adapters onto the
    original Bodhan checkpoint: their base SHA is intentionally different.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "adapters.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("weights") != "adapters.pt":
        raise ValueError("Unsupported adapter artifact format")
    if metadata.get("base_checkpoint_sha256") != sha256(base_checkpoint):
        raise ValueError("Adapter base checkpoint SHA256 mismatch; restore the exact training base")
    weights = directory / "adapters.pt"
    if metadata.get("weights_sha256") != sha256(weights):
        raise ValueError("Adapter weights SHA256 mismatch")
    targets = _decoder_targets(model)
    saved_targets = metadata.get("targets", {})
    if set(saved_targets) != {name for name, _, _, _ in targets}:
        raise ValueError("Adapter targets do not match the native decoder")
    tensors = torch.load(weights, map_location="cpu", weights_only=True)
    expected_keys = {
        f"{name}.{key}"
        for name, _, _, _ in targets
        for key in ("lora_A", "lora_B")
    }
    if not isinstance(tensors, dict) or set(tensors) != expected_keys:
        raise ValueError("Adapter tensor names do not match the declared targets")
    settings = None
    for name, _, _, base in targets:
        target_settings = _validate_saved_target(name, base, saved_targets[name], tensors)
        if settings is not None and target_settings != settings:
            raise ValueError("All decoder adapters must have the same rank, alpha, and dropout")
        settings = target_settings

    # Only install adapters after every file, setting and tensor passes validation.
    model.requires_grad_(False)
    names = inject_decoder_lora(model, **settings)
    model._decoder_lora_targets = names
    with torch.no_grad():
        for name, module in adapter_modules(model).items():
            for key in ("lora_A", "lora_B"):
                getattr(module, key).copy_(tensors[f"{name}.{key}"])
    return metadata


def merge_decoder_lora(model):
    """Replace adapters with merged native Linear modules for portable .nemo export."""
    modules = adapter_modules(model)
    if not modules:
        raise ValueError("No LoRA adapters to merge")
    for name, module in modules.items():
        parent_name, key = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), key, module.merged())
    if hasattr(model, "_decoder_lora_targets"):
        del model._decoder_lora_targets
    return list(modules)
