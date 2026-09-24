#!/usr/bin/env bash
set -euo pipefail
# Choose Python 3.10-3.12; system Python 3.14 is not supported by this pinned stack.
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" - <<'PY'
import platform, sys
assert platform.system() == 'Darwin' and platform.machine() == 'arm64', 'This setup is for Apple Silicon.'
assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Use Python 3.10-3.12.'
PY
"$PYTHON_BIN" -m venv .venv-mac
source .venv-mac/bin/activate
python -m pip install -r requirements-mac.txt -e '.[test]'
python -m pip check
PYTORCH_ENABLE_MPS_FALLBACK=1 python - <<'PY'
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
assert torch.backends.mps.is_available(), 'Metal unavailable: run from a normal macOS terminal.'
print('NeMo import and MPS device checks passed.')
PY
printf '\nActivate with: source .venv-mac/bin/activate\n'
