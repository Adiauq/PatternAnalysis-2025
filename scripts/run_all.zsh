#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${0}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT_DIR}"

./scripts/setup_env_macos.zsh

DATA_ROOT="${DATA_ROOT:-${HOME}/datasets/hipmri2d}" \
OUT="${OUT:-outputs/hipmri_unet_label5_mps}" \
./scripts/train_label5.zsh

./scripts/predict_example.zsh

if [[ -f "${OUT:-outputs/hipmri_unet_label5_mps}/metrics.json" ]]; then
  echo "\n[run_all] metrics:"
  cat "${OUT:-outputs/hipmri_unet_label5_mps}/metrics.json"
fi
