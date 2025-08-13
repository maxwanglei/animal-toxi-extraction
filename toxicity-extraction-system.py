"""
LLM-based Toxicity Data Extraction from PMC XML Files
Extracts lab test results and organ injury data from animal studies

This file has been refactored into a package. Please use `scripts/cli.py`.
Keeping this thin wrapper for backward compatibility.
"""

import os
import sys
import textwrap


def main():
    """Deprecated: Use scripts/cli.py instead."""
    # Print quickstart guidance when no args are provided.
    if len(sys.argv) <= 1:
        msg = textwrap.dedent("""
        Note: This wrapper now delegates to scripts/cli.py.

        Quickstart to test locally with Meta Llama API:
          1) Create venv and install deps:
             python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
             # Or API-only: pip install -r requirements-remote.txt
          2) Put XMLs under ./data (e.g., data/sample.xml).
          3) Export your API settings:
             export LLAMA_API_KEY="your_api_key"
             export LLAMA_BASE_URL="https://api.llama.com/compat/v1/"
             export MODEL_NAME="Llama-4-Maverick-17B-128E-Instruct-FP8"
          4) Run a smoke test:
             python -m scripts.cli --provider meta-llama --model "$MODEL_NAME" --xml data/sample.xml --out ./toxicity_output
          5) Run a batch:
             python -m scripts.cli --provider meta-llama --model "$MODEL_NAME" --xml data/*.xml --out ./toxicity_output

        Tip: You can override the base URL with --base-url if needed.
        """).strip()
        print(msg, file=sys.stderr)
        if sys.version_info >= (3, 13):
            print("Detected Python 3.13. If torch installation fails, use Python 3.10–3.12 or install requirements-remote.txt for API-only.", file=sys.stderr)
    from scripts.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
