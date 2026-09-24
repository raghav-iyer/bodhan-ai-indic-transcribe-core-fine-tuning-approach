#!/usr/bin/env bash
set -euo pipefail

# Run from repository root on Linux x86_64 / WSL2 with an NVIDIA CUDA GPU.
python3 - <<'PY'
import platform, sys
assert platform.system() == 'Linux', 'Use Linux or WSL2 for CUDA; use bootstrap_mac.sh on Apple Silicon.'
assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Use Python 3.10, 3.11, or 3.12.'
PY
nvidia-smi
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-gpu.txt
python -m pip install -e '.[test]'
python -m pip check
python - <<'PY'
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
assert torch.cuda.is_available(), 'PyTorch cannot access the GPU. Check the NVIDIA driver.'
print('NeMo import succeeded; GPU:', torch.cuda.get_device_name(0))
PY
printf '\nEnvironment ready. Activate with: source .venv-gpu/bin/activate\n'
