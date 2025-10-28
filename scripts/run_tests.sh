#!/usr/bin/env bash
set -euo pipefail

# Simple smoke-test script for the HipMRI U-Net project.
# - Installs pinned dependencies.
# - Runs the unit tests shipped with the project.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/recognition/hipmri_unet_prostate_ayman"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

python -m pytest \
  "${ROOT_DIR}/tests/test_forward.py" \
  "${ROOT_DIR}/tests/test_metric.py"

echo "All smoke tests completed successfully."
