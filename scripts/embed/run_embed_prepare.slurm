#!/bin/bash
#SBATCH --job-name=embed_prepare
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Exit on any error
set -euo pipefail

# === Signal handling to cancel spawned array jobs ===
cleanup_and_cancel() {
    local signal="$1"
    local job_file="/tmp/embedding_array_jobs_${SLURM_JOB_ID:-$$}.txt"

    # If we received a termination signal (not EXIT), cancel spawned array jobs
    if [[ "$signal" != "EXIT" ]]; then
        echo "Received $signal signal, cancelling spawned array jobs..."
        if [[ -f "$job_file" ]]; then
            while IFS= read -r array_job_id; do
                if [[ -n "$array_job_id" && "$array_job_id" =~ ^[0-9]+$ ]]; then
                    echo "Cancelling array job $array_job_id..."
                    scancel "$array_job_id" 2>/dev/null && \
                        echo "Cancelled array job $array_job_id" || \
                        echo "Failed to cancel array job $array_job_id"
                fi
            done < "$job_file"
        fi
    fi

    # Clean up temp file
    if [[ -f "$job_file" ]]; then
        rm -f "$job_file"
    fi

    # If terminated by signal, exit with appropriate code
    if [[ "$signal" == "TERM" || "$signal" == "INT" ]]; then
        exit 130
    fi
}

trap 'cleanup_and_cancel EXIT' EXIT
trap 'cleanup_and_cancel TERM' TERM
trap 'cleanup_and_cancel INT' INT

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
# If WORKFLOW_DIR is set (from master job), use it; otherwise use old structure
if [[ -n "${WORKFLOW_DIR:-}" ]]; then
  BASE_OUT="${WORKFLOW_DIR}/embedding_prepare/job_${RUN_ID}"
else
  BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding_prepare/${RUN_ID}"
fi

mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/embedding_prepare.out
exec 2>"$BASE_OUT"/embedding_prepare.err

# Load environment variables from orchestrator
# These will be set by the orchestrator via --export
echo "=== Embedding Preparation Job Started ==="
echo "Dataset config: $DATASET_CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $BASE_OUT"

# Load the dataset config to get embedding parameters
cd /home/menger/git/adata_hf_datasets

# Activate conda environment
source .venv/bin/activate

# Run the embedding preparation script with parameters from the dataset config
# Force CPU mode and prepare_only=true for this step
python scripts/embed/embed_configure_submit_parallel_sh.py --config-name "$DATASET_CONFIG" --prepare-only --cpu-only

# Wait for array jobs to complete if we're running under SLURM
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Waiting for preparation array jobs to complete..."

    # Read array job IDs from the temporary file
    job_file="/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"
    if [[ -f "$job_file" ]]; then
        while IFS= read -r array_job_id; do
            if [[ -n "$array_job_id" ]]; then
                echo "Waiting for preparation array job $array_job_id to complete..."
                while squeue -j "$array_job_id" &>/dev/null; do
                    sleep 30
                done
                echo "Preparation array job $array_job_id completed"
            fi
        done < "$job_file"

        # Clean up the temporary file
        rm -f "$job_file"
    else
        echo "No preparation array job file found, assuming no array jobs were submitted"
    fi
fi

echo "=== Embedding Preparation Job Completed ==="
