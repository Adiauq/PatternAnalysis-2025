#!/usr/bin/env bash
set -euo pipefail

# Simple smoke-test script for the HipMRI U-Net project.
# - Installs pinned dependencies.
# - Runs the unit tests shipped with the project.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

cd "${REPO_ROOT}"
python -m pytest "${PROJECT_ROOT}/tests"

echo "All smoke tests completed successfully."
