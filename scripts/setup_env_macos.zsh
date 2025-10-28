#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${0}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT_DIR}"

PY="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[setup] creating venv at ${VENV_DIR}"
${PY} -m venv "${VENV_DIR}"

echo "[setup] activating venv"
source "${VENV_DIR}/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip

echo "[setup] installing requirements"
pip install -r recognition/hipmri_unet_prostate_ayman/requirements.txt

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY

echo "[setup] done"
