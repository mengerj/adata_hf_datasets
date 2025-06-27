#!/usr/bin/env bash
#SBATCH --job-name=embed_array
#SBATCH --mem=60G
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup and Logging
# ─────────────────────────────────────────────────────────────────────────────

# Required environment variables (set by embed_launcher.py)
INPUT_DIR="${INPUT_DIR:?INPUT_DIR environment variable is required}"
DATASET_CONFIG="${DATASET_CONFIG:?DATASET_CONFIG environment variable is required}"
MODE="${MODE:-cpu}"
METHODS="${METHODS:-pca hvg}"
BATCH_KEY="${BATCH_KEY:-batch}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"
WORKFLOW_DIR="${WORKFLOW_DIR:-}"

# Get array job information
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?This script must be run as a SLURM array job}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# Set up logging directory
if [[ -n "$WORKFLOW_DIR" ]]; then
    if [[ "$PREPARE_ONLY" == "true" ]]; then
        LOG_DIR="${WORKFLOW_DIR}/embedding_prepare/array_${ARRAY_JOB_ID}"
        OUTPUT_DIR="${WORKFLOW_DIR}/embedding_prepare/array_${ARRAY_JOB_ID}/${TASK_ID}"
    else
        LOG_DIR="${WORKFLOW_DIR}/embedding/array_${ARRAY_JOB_ID}"
        OUTPUT_DIR="${WORKFLOW_DIR}/embedding/array_${ARRAY_JOB_ID}/${TASK_ID}"
    fi
else
    # Fallback to default structure
    if [[ "$PREPARE_ONLY" == "true" ]]; then
        LOG_DIR="logs/embed_prepare/${ARRAY_JOB_ID}"
        OUTPUT_DIR="outputs/$(date +%Y-%m-%d)/embedding_prepare/${ARRAY_JOB_ID}/${TASK_ID}"
    else
        LOG_DIR="logs/embed/${ARRAY_JOB_ID}"
        OUTPUT_DIR="outputs/$(date +%Y-%m-%d)/embed/${ARRAY_JOB_ID}/${TASK_ID}"
    fi
fi

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# Redirect logs
exec 1>"${LOG_DIR}/${TASK_ID}.out"
exec 2>"${LOG_DIR}/${TASK_ID}.err"

# Record start time
start_ts=$(date +%s)

echo "=== SLURM Array Task Started ==="
echo "Array Job ID: $ARRAY_JOB_ID"
echo "Task ID: $TASK_ID"
echo "Input Directory: $INPUT_DIR"
echo "Dataset Config: $DATASET_CONFIG"
echo "Mode: $MODE"
echo "Methods: $METHODS"
echo "Prepare Only: $PREPARE_ONLY"
echo "Output Directory: $OUTPUT_DIR"
echo "Start Time: $(date)"
echo "================================="

# ─────────────────────────────────────────────────────────────────────────────
# File Processing
# ─────────────────────────────────────────────────────────────────────────────

# Get the specific file for this array task
files=( "${INPUT_DIR}"/*.zarr )
if [[ ${#files[@]} -eq 0 ]]; then
    echo "ERROR: No .zarr files found in $INPUT_DIR"
    exit 1
fi

if [[ $TASK_ID -ge ${#files[@]} ]]; then
    echo "ERROR: Task ID $TASK_ID exceeds number of files ${#files[@]}"
    exit 1
fi

target_file="${files[$TASK_ID]}"
if [[ ! -d "$target_file" ]]; then
    echo "ERROR: Target file does not exist: $target_file"
    exit 1
fi

echo "Processing file: $target_file"

# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────────────────────

# Change to project directory
cd /home/menger/git/adata_hf_datasets

# Activate environment
source .venv/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Build Configuration Parameters
# ─────────────────────────────────────────────────────────────────────────────

# Convert space-separated methods to JSON array format for Hydra
METHODS_JSON="[$(printf '"%s",' $METHODS | sed 's/,$//')]"

# Determine embedding section based on MODE and PREPARE_ONLY
if [[ "$PREPARE_ONLY" == "true" ]]; then
    # For preparation, try embedding_preparation first, then fall back to CPU config
    EMBEDDING_SECTION="embedding_preparation"
elif [[ "$MODE" == "cpu" ]]; then
    EMBEDDING_SECTION="embedding_cpu"
elif [[ "$MODE" == "gpu" ]]; then
    EMBEDDING_SECTION="embedding_gpu"
else
    # Fallback to legacy embedding section
    EMBEDDING_SECTION="embedding"
fi

echo "Using embedding section: $EMBEDDING_SECTION"
echo "Methods JSON: $METHODS_JSON"

# ─────────────────────────────────────────────────────────────────────────────
# Run Embedding Core
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting embedding processing..."

python3 scripts/embed/embed_core.py \
    --config-name="$DATASET_CONFIG" \
    ++prepare_only="$PREPARE_ONLY" \
    ++"$EMBEDDING_SECTION".methods="$METHODS_JSON" \
    ++batch_key="$BATCH_KEY" \
    ++batch_size="$BATCH_SIZE" \
    ++"$EMBEDDING_SECTION".input_files="[\"$target_file\"]" \
    ++hydra.run.dir="$OUTPUT_DIR"

embedding_exit_code=$?

# ─────────────────────────────────────────────────────────────────────────────
# Completion and Cleanup
# ─────────────────────────────────────────────────────────────────────────────

# Record end time and compute elapsed time
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))
hours=$(( elapsed/3600 ))
mins=$(( (elapsed%3600)/60 ))
secs=$(( elapsed%60 ))

echo "================================="
echo "=== SLURM Array Task Complete ==="
echo "Array Job ID: $ARRAY_JOB_ID"
echo "Task ID: $TASK_ID"
echo "File: $(basename "$target_file")"
echo "Exit Code: $embedding_exit_code"
echo "Elapsed Time: ${hours}h${mins}m${secs}s"
echo "End Time: $(date)"
echo "================================="

# Exit with the same code as the embedding script
exit $embedding_exit_code
