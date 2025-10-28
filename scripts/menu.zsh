#!/usr/bin/env zsh
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${0}")/.." >/dev/null 2>&1 && pwd)"
cd "${ROOT_DIR}"

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
