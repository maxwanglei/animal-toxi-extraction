#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="./data"
OUT_DIR="./toxicity_output"

print_usage() {
  echo "Usage: $0 [--data-dir DIR] [--out DIR] -- [EXTRA_CLI_ARGS...]"
  echo "Example:"
  echo "  $0 --data-dir ./data --out ./toxicity_output -- --provider openai-compatible --model meta-llama-3-70b-instruct"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --out)
      OUT_DIR="$2"; shift 2;;
    --help|-h)
      print_usage; exit 0;;
    --)
      shift; break;;
    *)
      echo "Unknown arg: $1"; print_usage; exit 1;;
  esac
done

EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"

# Activate venv if present
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

shopt -s nullglob
XMLS=("$DATA_DIR"/*.xml)
if [[ ${#XMLS[@]} -eq 0 ]]; then
  echo "No XML files found in $DATA_DIR"
  exit 1
fi

for xml in "${XMLS[@]}"; do
  echo "Processing: $xml"
  python -m scripts.cli --xml "$xml" --out "$OUT_DIR" "${EXTRA_ARGS[@]}"
done
