#!/bin/bash
#SBATCH --job-name=toxicity_extraction
#SBATCH --output=slurm_logs/toxicity_%A_%a.out
#SBATCH --error=slurm_logs/toxicity_%A_%a.err
#SBATCH --array=1-1000%50
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# SLURM Array Job Example for Animal Toxicity Extraction
# 
# This script processes XML files in parallel using SLURM array jobs.
# Each job processes one XML file and saves individual JSON results.
# 
# Usage:
#   1. Create a file list: find /path/to/xml/files -name "*.xml" > xml_file_list.txt
#   2. Update SBATCH array parameter: --array=1-N where N is the number of XML files
#   3. Submit: sbatch SLURM_EXAMPLE.sbatch

# Setup environment
module load python/3.9  # Adjust for your system
source /path/to/your/venv/bin/activate  # Adjust path

# Set working directory
cd /path/to/animal-toxi-extraction  # Adjust path

# Create output directories
mkdir -p slurm_logs
mkdir -p slurm_results

# Get the XML file for this array job
XML_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" xml_file_list.txt)

if [ -z "$XML_FILE" ]; then
    echo "No XML file found for array task ID ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "Processing file: $XML_FILE"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Set output directory for this job
OUTPUT_DIR="slurm_results/job_${SLURM_ARRAY_TASK_ID}"

# Set API key (use environment variable or uncomment and set directly)
# export LLAMA_API_KEY="your_api_key_here"

# Run the toxicity extraction with individual JSON output
PYTHONPATH=/path/to/animal-toxi-extraction python scripts/enhanced_cli.py \
    --enhanced \
    --api-key "$LLAMA_API_KEY" \
    --xml "$XML_FILE" \
    --save-individual-json \
    --out "$OUTPUT_DIR" \
    --log-level INFO

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "Successfully processed $XML_FILE"
    echo "Results saved to $OUTPUT_DIR"
    
    # Optional: Move individual JSON to a central location
    # PMID=$(basename "$XML_FILE" .xml)
    # cp "$OUTPUT_DIR/${PMID}_results.json" slurm_results/individual_results/
else
    echo "Failed to process $XML_FILE"
    exit 1
fi

echo "End time: $(date)"