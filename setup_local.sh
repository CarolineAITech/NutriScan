#!/usr/bin/env bash
# One-time local setup for NutriScan Milestone 1.
# Creates a virtual environment at ~/nutriscan-env and installs everything.
# Safe to re-run.
set -euo pipefail

ENV_DIR="${HOME}/nutriscan-env"

echo "==> Creating virtual environment at ${ENV_DIR}"
python3 -m venv "${ENV_DIR}"

echo "==> Upgrading pip"
"${ENV_DIR}/bin/pip" install --upgrade pip

echo "==> Installing PyTorch (CPU build - this machine has no CUDA GPU)"
"${ENV_DIR}/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing the rest"
"${ENV_DIR}/bin/pip" install "transformers>=4.57" pillow numpy pandas matplotlib \
    seaborn scipy scikit-learn tqdm ipywidgets jupyterlab nbconvert

echo
echo "==> Verifying"
"${ENV_DIR}/bin/python" - <<'PY'
import torch, transformers, pandas, sklearn, scipy, seaborn, PIL
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  CUDA GPU     {torch.cuda.is_available()}  (False is expected on this machine)")
print(f"  CPU threads  {torch.get_num_threads()}")
PY

echo
echo "Done. Start the notebook with:"
echo "    ./run_local.sh"
