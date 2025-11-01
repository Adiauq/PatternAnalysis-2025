#!/usr/bin/env zsh
set -euo pipefail

# Usage:
#   LABEL=3 OUT=outputs/hipmri_unet_label3_mps DATA_ROOT=~/datasets/hipmri2d \
#   ./scripts/train_label.zsh

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

: ${DATA_ROOT:="${HOME}/datasets/hipmri2d"}
: ${LABEL:=5}
: ${OUT:="outputs/hipmri_unet_label${LABEL}_mps"}
: ${EPOCHS:=10}
: ${BS:=8}
: ${SIZE:=256}
: ${BASE:=32}
: ${LR:=3e-4}
: ${SEED:=42}
: ${THRESHOLDS:="0.35,0.4,0.45,0.5"}

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found at ${VENV_DIR}. Run scripts/setup_env_macos.zsh first." >&2
  exit 1
fi
source "${VENV_DIR}/bin/activate"

python -m recognition.hipmri_unet_prostate_ayman.train \
  --train_root "${DATA_ROOT}/train" \
  --val_root   "${DATA_ROOT}/val" \
  --test_root  "${DATA_ROOT}/test" \
  --out        "${OUT}" \
  --epochs "${EPOCHS}" --bs "${BS}" --size "${SIZE}" --base "${BASE}" \
  --lr "${LR}" --seed "${SEED}" --label "${LABEL}" --thresholds "${THRESHOLDS}"
