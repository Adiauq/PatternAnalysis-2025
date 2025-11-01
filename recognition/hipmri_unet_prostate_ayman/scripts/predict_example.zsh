#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
source "${VENV_DIR}/bin/activate"

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
