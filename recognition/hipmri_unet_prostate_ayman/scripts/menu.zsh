#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

PS3=$'\nSelect an action: '
select opt in "Setup env" "Run tests" "Train" "Predict" "Run all" "Quit"; do
  case $REPLY in
    1) "${PROJECT_ROOT}/scripts/setup_env_macos.zsh" ;;
    2) "${PROJECT_ROOT}/scripts/run_tests.sh" ;;
    3)
      read "label?Enter label to train (default 5): "
      label=${label:-5}
      read "epochs?Epochs (default 10): "
      epochs=${epochs:-10}
      read "base?Base channels (default 32): "
      base=${base:-32}
      read "lr?Learning rate (default 3e-4): "
      lr=${lr:-3e-4}
      read "bs?Batch size (default 8): "
      bs=${bs:-8}
      read "size?Image size (default 256): "
      size=${size:-256}
      read "thresh?Threshold sweep (default 0.35,0.4,0.45,0.5): "
      thresh=${thresh:-0.35,0.4,0.45,0.5}
      LABEL="$label" EPOCHS="$epochs" BASE="$base" LR="$lr" BS="$bs" SIZE="$size" THRESHOLDS="$thresh" "${PROJECT_ROOT}/scripts/train_label.zsh"
      ;;
    4)
      read "label?Enter label for inference (default 5): "
      label=${label:-5}
      read "thresh?Threshold (press Enter to use checkpoint/default): "
      thresh=${thresh:-}
      read "tta?Enable TTA? (y/N): "
      if [[ "${tta:l}" == "y" ]]; then
        LABEL="$label" THRESH="$thresh" TTA=true "${PROJECT_ROOT}/scripts/predict_example.zsh"
      else
        LABEL="$label" THRESH="$thresh" "${PROJECT_ROOT}/scripts/predict_example.zsh"
      fi
      ;;
    5)
      read "label?Enter label for run-all (default 5): "
      label=${label:-5}
      LABEL="$label" "${PROJECT_ROOT}/scripts/run_all.zsh"
      ;;
    6) exit 0 ;;
    *) echo "Invalid choice";;
  esac
done
