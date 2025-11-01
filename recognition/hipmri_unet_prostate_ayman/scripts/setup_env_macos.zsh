#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

PY="${PYTHON_BIN:-python3}"
DEFAULT_VENV="${PROJECT_ROOT}/.venv"
VENV_DIR="${VENV_DIR:-${DEFAULT_VENV}}"

echo "[setup] creating venv at ${VENV_DIR}"
${PY} -m venv "${VENV_DIR}"

echo "[setup] activating venv"
source "${VENV_DIR}/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip

echo "[setup] installing requirements"
pip install -r "${PROJECT_ROOT}/requirements.txt"

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
PY

echo "[setup] done"
