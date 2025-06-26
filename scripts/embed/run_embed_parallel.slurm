#!/bin/bash
#SBATCH --job-name=embed_parallel
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Exit on any error
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_JOB_ID}"
else
  # local run: use date+seconds to make it unique
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
# Check if this is prepare-only mode
PREPARE_ONLY="${PREPARE_ONLY:-false}"

# If WORKFLOW_DIR is set (from master job), use it; otherwise use old structure
if [[ -n "${WORKFLOW_DIR:-}" ]]; then
  if [[ "$PREPARE_ONLY" == "true" ]]; then
    BASE_OUT="${WORKFLOW_DIR}/embedding_prepare/job_${RUN_ID}"
  else
    BASE_OUT="${WORKFLOW_DIR}/embedding/job_${RUN_ID}"
  fi
else
  if [[ "$PREPARE_ONLY" == "true" ]]; then
    BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding_prepare/${RUN_ID}"
  else
    BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding/${RUN_ID}"
  fi
fi

mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/embedding.out
exec 2>"$BASE_OUT"/embedding.err

# Load environment variables from orchestrator
# These will be set by the orchestrator via --export
echo "=== Embedding Job Started ==="
echo "Dataset config: $DATASET_CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $BASE_OUT"
echo "Prepare only mode: $PREPARE_ONLY"

# Load the dataset config to get embedding parameters
cd /home/menger/git/adata_hf_datasets

# Activate conda environment (adjust path as needed)
source .venv/bin/activate

# Run the embedding script with parameters from the dataset config
# We'll use a Python script to extract parameters and run the bash script

python scripts/embed/run_embed_with_config.py --config-name "$DATASET_CONFIG"

# Wait for array jobs to complete if we're running under SLURM
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Waiting for array jobs to complete..."

    # Read array job IDs from the temporary file
    job_file="/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"
    if [[ -f "$job_file" ]]; then
        while IFS= read -r array_job_id; do
            if [[ -n "$array_job_id" ]]; then
                echo "Waiting for array job $array_job_id to complete..."
                while squeue -j "$array_job_id" &>/dev/null; do
                    sleep 30
                done
                echo "Array job $array_job_id completed"
            fi
        done < "$job_file"

        # Clean up the temporary file
        rm -f "$job_file"
    else
        echo "No array job file found, assuming no array jobs were submitted"
    fi
fi

echo "=== Embedding Job Completed ==="
