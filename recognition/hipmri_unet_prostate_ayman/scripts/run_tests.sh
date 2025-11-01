#!/usr/bin/env bash
set -euo pipefail

# Smoke-test helper for the HipMRI U-Net project.
# - Optionally refreshes dependencies (default).
# - Executes the pytest suite using the project virtual environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PY_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[run_tests] Python interpreter not found at ${PY_BIN}." >&2
  echo "             Run scripts/setup_env_macos.zsh first." >&2
  exit 1
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "${PY_BIN}" -m pip install --upgrade pip
  "${PY_BIN}" -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi

cd "${PROJECT_ROOT}"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PY_BIN}" -m pytest tests

echo "All tests completed successfully."
