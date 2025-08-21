#!/bin/bash
#SBATCH --job-name=extract_remaining
#SBATCH --account=PCON0020
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# -------- User configuration --------
XML_ROOT="/fs/ess/PCON0020/cond0103/animal_xmls"   # Root folder containing 180k XML files
SHARDS=6                                           # Must match --array size above (0..SHARDS-1)
OUT_BASE="$SLURM_SUBMIT_DIR/toxicity_output_meta" # Base output directory

# LLM provider config: export these in your shell or define here.
# Avoid committing secrets to version control.
: "${LLAMA_API_KEY:?Set LLAMA_API_KEY in your environment}"
: "${LLAMA_BASE_URL:?Set LLAMA_BASE_URL in your environment}"
: "${MODEL_NAME:?Set MODEL_NAME in your environment}"

PROVIDER="meta-llama"
MAX_TOKENS=3072
TEMPERATURE=0.0

# Python environment
# Activate your pre-created virtual environment with required dependencies
# Example:
# source /path/to/venv/bin/activate

# Optional: limit thread usage per task
export OMP_NUM_THREADS=1

# -------- Derived settings --------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
JOB_ID=${SLURM_JOB_ID:-manual}
RUN_TAG="job${JOB_ID}_shard${TASK_ID}"

mkdir -p "${OUT_BASE}" "${OUT_BASE}_enhanced" slurm_logs manifests shards

# Build a stable manifest of all XML files once, then shard
MANIFEST="manifests/all_xmls.txt"
if [[ ! -s "${MANIFEST}" ]]; then
  echo "[$(date)] Building manifest at ${MANIFEST} from ${XML_ROOT} ..."
  # Use a temp file and atomic move to avoid races between array tasks
  TMP_MANIFEST="${MANIFEST}.tmp.$$"
  find "${XML_ROOT}" -type f -name "*.xml" | LC_ALL=C sort > "${TMP_MANIFEST}"
  mv "${TMP_MANIFEST}" "${MANIFEST}"
  echo "[$(date)] Manifest created: $(wc -l < "${MANIFEST}") files"
else
  echo "[$(date)] Using existing manifest: ${MANIFEST} (lines=$(wc -l < "${MANIFEST}"))"
fi

# Create this task's shard list using modulo partitioning (zero-based)
SHARD_LIST="shards/xml_shard_${TASK_ID}_of_${SHARDS}.txt"
if [[ ! -s "${SHARD_LIST}" ]]; then
  echo "[$(date)] Creating shard ${TASK_ID}/${SHARDS} -> ${SHARD_LIST}"
  awk -v N=${SHARDS} -v K=${TASK_ID} '((NR-1) % N) == K {print}' "${MANIFEST}" > "${SHARD_LIST}"
  echo "[$(date)] Shard size: $(wc -l < "${SHARD_LIST}") files"
else
  echo "[$(date)] Using existing shard file: ${SHARD_LIST} (lines=$(wc -l < "${SHARD_LIST}"))"
fi

# Per-shard output directory and log file
OUT_DIR="${OUT_BASE}_enhanced/shard_${TASK_ID}"
mkdir -p "${OUT_DIR}"

echo "[$(date)] Starting extraction for shard ${TASK_ID}/${SHARDS} -> ${OUT_DIR}"

# Run the enhanced pipeline with Meta Llama provider using the file list
python -m scripts.enhanced_cli \
  --enhanced \
  --provider "${PROVIDER}" \
  --model "${MODEL_NAME}" \
  --api-key "${LLAMA_API_KEY}" \
  --base-url "${LLAMA_BASE_URL}" \
  --xml-list "${SHARD_LIST}" \
  --out "${OUT_DIR}" \
  --max-tokens ${MAX_TOKENS} \
  --temperature ${TEMPERATURE} \
  --save-individual-json \
  --save-intermediate

status=$?
echo "[$(date)] Extraction finished for shard ${TASK_ID} with status ${status}"
exit ${status}
