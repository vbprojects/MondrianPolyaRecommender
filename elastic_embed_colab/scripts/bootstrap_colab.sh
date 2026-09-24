#!/usr/bin/env bash
set -euo pipefail
# Run from any directory in a Colab GPU runtime. Preserve Colab's CUDA Torch.
BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$BUNDLE_ROOT"
COLAB_PYTHON="${COLAB_PYTHON:-python}"
"$COLAB_PYTHON" -m pip install -e '.[colab]'
"$COLAB_PYTHON" -m pip install 'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@954438d1fcb5e1bb05c22f9908de9c5c2df74ae5'
"$COLAB_PYTHON" -m pip freeze > /tmp/elastic-embed-colab-pip-freeze.txt
"$COLAB_PYTHON" - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('BF16:', torch.cuda.is_bf16_supported())
if not torch.cuda.is_available():
    raise SystemExit('Select a GPU runtime in Colab before running FLA/teacher jobs.')
PY
