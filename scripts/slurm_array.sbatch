#!/usr/bin/env bash
#SBATCH --job-name=tox-extract
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
# Uncomment if you need GPU for local LLMs
##SBATCH --gres=gpu:1
# Set with: sbatch --array=0-(N-1) scripts/slurm_array.sbatch
#SBATCH --array=0-0

set -euo pipefail

# User-editable settings
DATA_DIR="${DATA_DIR:-data}"
OUT_DIR="${OUT_DIR:-toxicity_output}"
EXTRA_ARGS="${EXTRA_ARGS:---provider openai-compatible --model meta-llama-3-70b-instruct}"

# Export API credentials here or in your job submission environment
# export OPENAI_API_KEY=...
# export OPENAI_BASE_URL=...

# Module/venv setup (adjust to your HPC)
# module load python/3.10
if [[ -d "$SLURM_SUBMIT_DIR/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

mkdir -p "$OUT_DIR" logs

# Resolve XML by array index
mapfile -t FILES < <(ls -1 "$DATA_DIR"/*.xml)
COUNT=${#FILES[@]}
if (( COUNT == 0 )); then
  echo "No XML files in $DATA_DIR"
  exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= COUNT )); then
  echo "TASK_ID ${SLURM_ARRAY_TASK_ID} out of range [0,$((COUNT-1))]"
  exit 1
fi

XML="${FILES[$SLURM_ARRAY_TASK_ID]}"
echo "[$SLURM_JOB_ID:$SLURM_ARRAY_TASK_ID] Processing: $XML"

python -m scripts.cli --xml "$XML" --out "$OUT_DIR" $EXTRA_ARGS
