"""Explicit device selection; do not infer CUDA requirements from a Mac sandbox."""


def check_device(device):
    import torch
    if device not in ("cuda", "mps", "cpu"):
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Set training.accelerator to mps for Apple Silicon, or cpu.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Run from a normal macOS terminal (a sandbox may hide Metal), or choose cpu.")
    return device


def device_info(device):
    import torch
    result = {"device": device, "torch": torch.__version__, "cuda": torch.version.cuda}
    if device == "cuda":
        result.update(gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    elif device == "mps":
        result.update(gpu="Apple Metal", mps_allocated_bytes=torch.mps.current_allocated_memory())
    return result


def prepare_for_device(model, device):
    if device == "mps":
        import torch
        # NeMo's validation loss accumulator defaults to float64, unsupported by MPS.
        # Module.to(dtype=...) also converts TorchMetrics states/defaults while leaving integer counters intact.
        model.to(dtype=torch.float32)
    return model
