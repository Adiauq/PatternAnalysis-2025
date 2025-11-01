#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${0}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_ROOT}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

PS3=$'\nSelect an action: '
select opt in "Setup env" "Train (label=5)" "Predict example" "Run all" "Quit"; do
  case $REPLY in
    1) ./scripts/setup_env_macos.zsh ;;
    2) ./scripts/train_label5.zsh ;;
    3) ./scripts/predict_example.zsh ;;
    4) ./scripts/run_all.zsh ;;
    5) exit 0 ;;
    *) echo "Invalid choice";;
  esac
done
