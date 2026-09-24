"""Command line entry points; GPU imports are lazy so local checks stay lightweight."""

import argparse
import json
import platform
from pathlib import Path

from .common import load_config, load_local_token, write_json


def preflight(cfg):
    import importlib.metadata
    import importlib.util
    import os
    import shutil
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url

    report = {"platform": platform.platform(), "free_disk_gib": shutil.disk_usage('.').free / 1024**3,
              "token_configured": bool(os.environ.get("HF_TOKEN")), "gpu_ready": False}
    if report["token_configured"]:
        try:
            HfApi().whoami(token=os.environ["HF_TOKEN"])
            report["token_valid"] = True
        except Exception as exc:
            report["token_valid"] = False
            report["token_error_type"] = type(exc).__name__
            cause = exc.__cause__ or exc
            response = getattr(cause, "response", None)
            report["token_error_status"] = getattr(response, "status_code", None)
    if importlib.util.find_spec("torch"):
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["mps_available"] = torch.backends.mps.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            report["gpu_ready"] = True
        target = cfg["training"].get("accelerator", "cuda")
        report["requested_device"] = target
        report["device_available"] = (target == "cpu" or (target == "mps" and report["mps_available"]) or
                                      (target == "cuda" and report["cuda_available"]))
        report["gpu_ready"] = target != "cpu" and report["device_available"]
        if target == "mps" and report["device_available"]:
            report["gpu"] = "Apple Metal"
    try:
        report["nemo_toolkit"] = importlib.metadata.version("nemo_toolkit")
    except importlib.metadata.PackageNotFoundError:
        report["nemo_toolkit"] = None
    spec = cfg["model"]
    try:
        metadata = get_hf_file_metadata(hf_hub_url(spec["repo_id"], spec["checkpoint"], revision=spec["revision"]),
                                       token=os.environ.get("HF_TOKEN"))
        report["model_access"] = "approved"
        report["checkpoint_bytes"] = metadata.size
    except Exception as exc:
        report["model_access"] = "unavailable"
        report["access_error_type"] = type(exc).__name__
    report["ready_to_train"] = bool(report.get("device_available", False) and report["nemo_toolkit"] and
                                    report["model_access"] == "approved" and report["free_disk_gib"] >= 30)
    write_json(Path(cfg["training"]["directory"]) / "preflight.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Marathi Bodhan ASR fine-tuning pipeline")
    parser.add_argument("--config", default="configs/marathi.json", help="Paths inside config are relative to working directory")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "validate", "preflight", "package", "bundle-source", "experiment", "package-experiment"):
        commands.add_parser(name)
    download = commands.add_parser("download-model")
    download.add_argument("--code-only", action="store_true", help="Inspect the small publisher helper files without downloading weights")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--resume", help="Trusted Lightning checkpoint from the same run")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--label", choices=("baseline", "finetuned"), required=True)
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--split", choices=("validation", "test"), default="test")
    evaluation.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    load_local_token()
    cfg = load_config(args.config)
    if cfg["training"].get("accelerator") == "mps" or getattr(args, "device", None) == "mps":
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if args.command == "preflight":
        preflight(cfg)
    elif args.command == "experiment":
        from .experiments import run_experiment
        run_experiment(cfg)
    elif args.command == "package-experiment":
        from .artifacts import package_experiment
        print(package_experiment(cfg))
    elif args.command in ("prepare", "validate"):
        from . import data
        print(json.dumps(getattr(data, args.command)(cfg), ensure_ascii=False, indent=2))
    elif args.command == "download-model":
        from .model import download_model
        print(download_model(cfg, code_only=args.code_only))
    elif args.command == "train":
        from .train import train
        train(cfg, resume=args.resume)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(cfg, args.label, args.checkpoint, args.split, args.device)
    elif args.command == "compare":
        from .evaluate import compare
        print(json.dumps(compare(cfg, args.split), indent=2))
    elif args.command == "package":
        from .artifacts import package
        package(cfg)
    elif args.command == "bundle-source":
        from .artifacts import source_bundle
        print(source_bundle())


if __name__ == "__main__":
    main()
