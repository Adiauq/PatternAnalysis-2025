#!/usr/bin/env zsh
set -euo pipefail

# Usage:
#   DATA_ROOT=~/datasets/hipmri2d \
#   OUT=outputs/hipmri_unet_label5_mps \
#   ./scripts/train_label5.zsh

ROOT_DIR="$(cd -- "$(dirname -- "${0}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT_DIR}"

: ${DATA_ROOT:="${HOME}/datasets/hipmri2d"}
: ${OUT:="outputs/hipmri_unet_label5_mps"}
: ${EPOCHS:=10}
: ${BS:=8}
: ${SIZE:=256}
: ${BASE:=32}
: ${LR:=3e-4}
: ${SEED:=42}
: ${LABEL:=5}
: ${THRESHOLDS:="0.35,0.4,0.45,0.5"}

source .venv/bin/activate

python -m recognition.hipmri_unet_prostate_ayman.train \
  --train_root "${DATA_ROOT}/train" \
  --val_root   "${DATA_ROOT}/val" \
  --test_root  "${DATA_ROOT}/test" \
  --out        "${OUT}" \
  --epochs "${EPOCHS}" --bs "${BS}" --size "${SIZE}" --base "${BASE}" \
  --lr "${LR}" --seed "${SEED}" --label "${LABEL}" --thresholds "${THRESHOLDS}"
