#!/usr/bin/env bash
set -euo pipefail

MODE="remote"   # remote|full
PYVER="${PYVER:-3.12}"
REQ_FILE="requirements.txt"

usage() {
  echo "Usage: $0 [--mode remote|full] [--python MAJOR.MINOR]"
  echo "Examples:"
  echo "  $0 --mode remote --python 3.12   # API-only (no torch)"
  echo "  $0 --mode full   --python 3.12   # Install torch for local models"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --python|--py|--python-version) PYVER="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if ! command -v "python${PYVER}" >/dev/null 2>&1; then
  echo "python${PYVER} not found. Install via pyenv/conda/brew and re-run."
  exit 1
fi

"python${PYVER}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

if [[ "$MODE" == "remote" ]]; then
  REQ_FILE="requirements-remote.txt"
  echo "Installing API-only deps from $REQ_FILE"
  pip install -r "$REQ_FILE"
else
  echo "Installing base deps from requirements.txt"
  pip install -r requirements.txt || true

  echo "Installing torch for platform=$(uname -s)/arch=$(uname -m)"
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64)
      # Apple Silicon: CPU/MPS wheels from PyPI
      pip install --extra-index-url https://download.pytorch.org/whl cpuonly || true
      pip install torch --upgrade
      ;;
    Darwin-x86_64|Linux-x86_64|Linux-amd64)
      # CPU wheels
      pip install torch --index-url https://download.pytorch.org/whl/cpu --upgrade
      ;;
    *)
      # Fallback: try default PyPI
      pip install torch --upgrade || {
        echo "Torch install failed. Check https://pytorch.org/get-started/locally/ for your platform."
        exit 1
      }
      ;;
  esac
fi

echo "Done. Activate with: source .venv/bin/activate"
