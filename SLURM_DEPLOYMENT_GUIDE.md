# SLURM Deployment Guide for Animal Toxicity Extraction

This guide explains how to use the enhanced toxicity extraction system with SLURM for processing large numbers of XML files (180,000+) in parallel.

## Key Features for SLURM

The system now supports individual JSON file output per paper, perfect for SLURM array jobs:

- `--save-individual-json` flag generates individual JSON files for each processed paper
- Each JSON file contains complete extraction results, metadata, and processing statistics
- Independent processing means failed jobs don't affect other files
- Results can be easily combined or analyzed separately

## Individual JSON Output Format

Each processed XML file generates a JSON file named `{PMID}_results.json` containing:

```json
{
  "pmid": "PMC11131026",
  "file_path": "data/PMC11131026.xml", 
  "processing_timestamp": "2025-08-15T15:02:26.589108",
  "file_stats": {
    "table_extractions": 2,
    "text_extractions": 2
  },
  "lab_tests": [...],           // Complete lab test results
  "organ_injuries": [...],      // Complete organ injury results  
  "summary": {
    "total_lab_tests": 9,
    "total_organ_injuries": 20,
    "unique_drugs": ["ITO", "ITO NPs"],
    "tables_processed": 2,
    "text_sections_processed": 2,
    "extraction_success": true
  }
}
```

## SLURM Setup Instructions

### 1. Prepare File List

Create a list of all XML files to process:

```bash
find /path/to/xml/files -name "*.xml" -type f > xml_file_list.txt
wc -l xml_file_list.txt  # Count total files
```

### 2. Setup Environment

```bash
# Load required modules (adjust for your system)
module load python/3.9

# Create and activate virtual environment
python -m venv toxicity_env
source toxicity_env/bin/activate

# Install dependencies
cd /path/to/animal-toxi-extraction
pip install -r requirements.txt
```

### 3. Configure API Access

Set your Meta Llama API key:

```bash
export LLAMA_API_KEY="LLM|your_api_key_here"
# Or add to your job script directly
```

### 4. Customize SLURM Script

Edit `SLURM_EXAMPLE.sbatch`:

```bash
# Update these lines:
#SBATCH --array=1-180000%50    # Process 180K files, max 50 concurrent
#SBATCH --time=00:30:00        # Adjust based on file complexity
#SBATCH --mem=4G               # Adjust memory requirements

# Update paths:
cd /your/path/to/animal-toxi-extraction
source /your/path/to/toxicity_env/bin/activate
```

### 5. Submit Jobs

```bash
# Create output directories
mkdir -p slurm_logs slurm_results

# Submit the array job
sbatch SLURM_EXAMPLE.sbatch

# Monitor progress
squeue -u $USER
```

## Managing Large-Scale Processing

### Array Job Best Practices

1. **Batch Size**: Use `%50` to limit concurrent jobs (adjust based on API limits)
2. **Time Limits**: Set appropriate `--time` based on file complexity
3. **Memory**: 4GB should be sufficient for most XML files
4. **Error Handling**: Individual failures won't affect other jobs

### Result Collection

After jobs complete, collect individual results:

```bash
# Find all individual JSON files
find slurm_results -name "*_results.json" > individual_results_list.txt

# Optional: Combine results for analysis
python -c "
import json
import glob

all_results = []
for file in glob.glob('slurm_results/*/PMC*_results.json'):
    with open(file) as f:
        all_results.append(json.load(f))
        
with open('combined_all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
"
```

### Error Analysis

Check for failed extractions:

```bash
# Find failed jobs
grep -l "extraction_success.*false" slurm_results/*/PMC*_results.json

# Analyze error patterns
grep "error" slurm_results/*/PMC*_results.json | sort | uniq -c
```

## Resource Estimation

For 180,000 XML files:

- **Processing Time**: ~5-15 minutes per file (depending on complexity)
- **Storage**: ~2-10KB per JSON result file (~1-2GB total)
- **API Calls**: ~5-15 calls per file (monitor rate limits)
- **Parallelization**: 50-100 concurrent jobs recommended

## Example Workflow

```bash
# 1. Prepare environment
cd /path/to/animal-toxi-extraction
find /data/xml_files -name "*.xml" > xml_file_list.txt
echo "Total files: $(wc -l xml_file_list.txt)"

# 2. Update SLURM script
# Edit SLURM_EXAMPLE.sbatch with correct paths and array size

# 3. Submit jobs  
sbatch SLURM_EXAMPLE.sbatch

# 4. Monitor progress
watch 'squeue -u $USER | head -20'

# 5. Check completion
find slurm_results -name "*_results.json" | wc -l

# 6. Analyze results
python scripts/analyze_slurm_results.py slurm_results/
```

## Testing Individual Files

Before running large batches, test individual files:

```bash
# Test single file
PYTHONPATH=/path/to/animal-toxi-extraction python scripts/enhanced_cli.py \
    --enhanced \
    --api-key "$LLAMA_API_KEY" \
    --xml /path/to/test.xml \
    --save-individual-json \
    --out test_output

# Check result
cat test_output/PMC*_results.json | jq '.summary'
```

## Troubleshooting

Common issues and solutions:

1. **API Rate Limits**: Reduce concurrent jobs (`%25` instead of `%50`)
2. **Memory Issues**: Increase `--mem` for large XML files
3. **Time Limits**: Increase `--time` for complex files
4. **Failed Parsing**: Check individual error logs in JSON files

## Output Structure

```
slurm_results/
├── job_1/
│   ├── PMC123456_results.json
│   ├── combined_lab_results.csv
│   ├── combined_organ_results.csv
│   └── run.log
├── job_2/
│   └── ...
└── individual_results/  # Optional collected results
    ├── PMC123456_results.json
    ├── PMC123457_results.json
    └── ...
```

This approach enables efficient parallel processing of large XML collections while maintaining detailed per-file results and error tracking.