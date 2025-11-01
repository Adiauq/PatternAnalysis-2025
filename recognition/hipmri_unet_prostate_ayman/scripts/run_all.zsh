#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

./scripts/setup_env_macos.zsh

DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/hipmri2d}"
OUT="${OUT:-outputs/hipmri_unet_label5_mps}"

DATA_ROOT="${DATA_ROOT}" OUT="${OUT}" ./scripts/train_label5.zsh

./scripts/predict_example.zsh

if [[ -f "${OUT}/metrics.json" ]]; then
  echo "\n[run_all] metrics:"
  cat "${OUT}/metrics.json"
fi
