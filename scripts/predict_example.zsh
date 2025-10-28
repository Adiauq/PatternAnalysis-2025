#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${0}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT_DIR}"
source .venv/bin/activate

: ${CKPT:="outputs/hipmri_unet_label5_mps/best.pt"}
: ${INPUT:="${HOME}/datasets/hipmri2d/test/img/case_040_week_0_slice_2.nii.gz"}
: ${OUT_DIR:="outputs/preds"}
: ${SIZE:=256}
: ${BASE:=32}
: ${TTA:=false}
: ${THRESH:=0.5}

ARGS=(
  --ckpt "${CKPT}"
  --inputs "${INPUT}"
  --out "${OUT_DIR}"
  --size "${SIZE}"
  --base "${BASE}"
)

if [[ "${TTA}" == true ]]; then
  ARGS+=(--tta)
fi

if [[ -n "${THRESH}" ]]; then
  ARGS+=(--thresh "${THRESH}")
fi

python -m recognition.hipmri_unet_prostate_ayman.predict "${ARGS[@]}"

COUNT=$(ls -1 "${OUT_DIR}"/*_pred.png 2>/dev/null | wc -l | tr -d ' ')
echo "[predict] wrote ${COUNT:-0} file(s) to ${OUT_DIR}"
