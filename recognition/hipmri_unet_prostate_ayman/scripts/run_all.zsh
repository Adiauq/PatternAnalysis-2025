#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

"${PROJECT_ROOT}/scripts/setup_env_macos.zsh"

DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/hipmri2d}"
LABEL="${LABEL:-5}"
OUT="${OUT:-outputs/hipmri_unet_label${LABEL}_mps}"

DATA_ROOT="${DATA_ROOT}" LABEL="${LABEL}" OUT="${OUT}" "${PROJECT_ROOT}/scripts/train_label.zsh"

LABEL="${LABEL}" OUT="${OUT}" "${PROJECT_ROOT}/scripts/predict_example.zsh"

if [[ -f "${OUT}/metrics.json" ]]; then
  echo "\n[run_all] metrics:"
  cat "${OUT}/metrics.json"
fi
