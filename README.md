# Animal Toxicity Extraction

LLM-based extraction of lab test results and organ injury data from PMC XML animal studies.

## Overview

- `toxicity_extraction/` – package with models, parsers, LLM providers, extractor, and pipeline
- `scripts/cli.py` – command-line entrypoint
- `toxicity-extraction-system.py` – thin wrapper that calls the CLI (for backward compatibility)

## Install

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: If you plan to use Local LLMs, ensure you have a compatible PyTorch and Transformers installation for your platform.

Note on HPC: create/prepare the virtual environment on a login node, not a compute node. If compute nodes have no internet, install dependencies on the login node and reuse the same .venv in jobs.

### Python and PyTorch compatibility

- Recommended Python: 3.10–3.12. PyTorch wheels may be unavailable on Python 3.13 and older macOS versions.
- If you only use a remote Llama API (no local models), you can skip installing PyTorch.

API-only install (no torch):
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-remote.txt
```

Full install (with torch for local models):
```bash
bash scripts/setup_env.sh --mode full
```

If setup_env.sh is not suitable for your cluster/mac, see inline comments in that script and the PyTorch website for platform-specific commands.

## Usage

Run the CLI on one or more PMC XML files:

```bash
python -m scripts.cli --xml path/to/pmc1.xml path/to/pmc2.xml \
  --out ./toxicity_output \
  --model-path /path/to/llama-model
```

Outputs CSVs per input and combined CSVs in the output directory.

### Running on a folder of XMLs (data/)

If your XML files are in a data/ folder, you can use shell globbing:

```bash
python -m scripts.cli --xml data/*.xml \
  --out ./toxicity_output \
  --model-path /path/to/llama-model
```

- The CLI accepts multiple --xml inputs; the shell expands data/*.xml into a list.
- Outputs are per-input CSVs and combined CSVs under --out.

Alternatively, use the batch helper script (see scripts/run_batch.sh below).

### Using a remote Llama API

If using a hosted Llama API (OpenAI-compatible), configure environment variables:

```bash
# Example; adjust to your provider
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.your-llama-endpoint.com/v1"
# Optional: model name if needed by your provider
export MODEL_NAME="meta-llama-3-70b-instruct"
```

Then run:

```bash
python -m scripts.cli --xml data/*.xml \
  --out ./toxicity_output \
  --provider openai-compatible \
  --model $MODEL_NAME
```

Tips:
- Set low concurrency to respect rate limits; if the pipeline supports it, use flags like --max-concurrency 1 and enable retries.
- If your providers.py uses different env vars (e.g., LLAMA_API_KEY), export those instead.

### Using a remote Llama API (no local models)

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.your-llama-endpoint.com/v1"
export MODEL_NAME="meta-llama-3-70b-instruct"
python -m scripts.cli --xml data/sample.xml --out ./toxicity_output \
  --provider openai-compatible --model "$MODEL_NAME"
```

### Using Meta Llama API (OpenAI-compatible)

```bash
export LLAMA_API_KEY="your_api_key"
export LLAMA_BASE_URL="https://api.llama.com/compat/v1/"
export MODEL_NAME="Llama-4-Maverick-17B-128E-Instruct-FP8"

python -m scripts.cli \
  --provider meta-llama \
  --model "$MODEL_NAME" \
  --xml data/sample.xml \
  --out ./toxicity_output
```

- You can override the base URL with --base-url if needed.
- For batch: python -m scripts.cli --provider meta-llama --model "$MODEL_NAME" --xml data/*.xml --out ./toxicity_output

### Troubleshooting installs

- Error: "No matching distribution found for torch"
  - Use Python 3.10–3.12 (e.g., via pyenv or conda).
  - On macOS Intel: install CPU wheels with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  - On Apple Silicon (arm64): `pip install torch` (CPU/MPS).
  - Or skip torch entirely with `pip install -r requirements-remote.txt` when using a remote API.

### Batch helper script

A convenience wrapper is provided to iterate over data/*.xml:

```bash
bash scripts/run_batch.sh \
  --data-dir ./data \
  --out ./toxicity_output \
  -- \
  --provider openai-compatible --model $MODEL_NAME
```

Everything after -- is passed directly to python -m scripts.cli.

### Submitting on an HPC cluster (SLURM)

Use a SLURM array job to process each XML file in parallel. First, count files and submit:

```bash
# Count XMLs
N=$(ls -1 data/*.xml | wc -l)
# Submit array 0..N-1; adjust partition/account as needed inside the sbatch file
sbatch --array=0-$((N-1)) scripts/slurm_array.sbatch
```

The job will:
- Activate .venv
- Select the XML by SLURM_ARRAY_TASK_ID
- Call the CLI with your extra arguments

Edit scripts/slurm_array.sbatch to set:
- DATA_DIR, OUT_DIR
- EXTRA_ARGS for provider/model (e.g., --provider openai-compatible --model meta-llama-3-70b-instruct)
- The SBATCH resources (partition, time, CPUs, GPUs if needed)
- Export API key env vars in the script or via your scheduler’s secrets mechanism

## Package structure

```
toxicity_extraction/
  __init__.py
  models.py
  extractor.py
  pipeline.py
  parsers/
    __init__.py
    pmc_xml.py
  llm/
    __init__.py
    base.py
    providers.py
scripts/
  cli.py
```

## Extending

- Add new LLM providers by implementing `LLMProvider` in `toxicity_extraction/llm`.
- Add more parsers under `toxicity_extraction/parsers`.
- Improve prompts or parsing logic in `extractor.py`.
