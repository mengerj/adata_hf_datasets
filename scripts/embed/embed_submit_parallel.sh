#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATION NOTICE
# ═══════════════════════════════════════════════════════════════════════════════
# This script is DEPRECATED and kept only for standalone/legacy usage.
#
# For the new unified workflow, use:
#   python scripts/workflow/submit_workflow.py --config-name <dataset_config>
#
# The new workflow uses EmbeddingArraySubmitter to directly submit array jobs
# via SSH without needing this intermediate shell script.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail


# === Array job tracking for cleanup ===
SPAWNED_JOBS=()

# Trap function to cancel all spawned array jobs on termination
cleanup_jobs() {
    echo "Received termination signal, cancelling spawned array jobs..."
    for jid in "${SPAWNED_JOBS[@]}"; do
        if [[ -n "$jid" ]]; then
            scancel "$jid" 2>/dev/null && echo "Cancelled job $jid" || true
        fi
    done
    # Clean up temp file
    rm -f "/tmp/embedding_array_jobs_${SLURM_JOB_ID:-$$}.txt" 2>/dev/null || true
    exit 130
}
trap cleanup_jobs SIGTERM SIGINT SIGHUP

# === User‐configurable section ===
# Use environment variables if provided, otherwise use defaults
MODE="${MODE:-cpu}"         # "cpu" or "gpu"
GPU_COUNT="${GPU_COUNT:-1}"      # how many GPUs if MODE=gpu
DATANAME="${DATANAME:-cellxgene_pseudo_bulk_3_5k}"
BATCH_KEY="${BATCH_KEY:-dataset_title}"
BATCH_SIZE="${BATCH_SIZE:-128}"
METHODS="${METHODS:-geneformer scvi_fm pca hvg}" # space‐separated list - eg one string with spaces
SCRIPT="scripts/embed/embed_chunks_parallel.slurm"
MAX_PROCS="${MAX_PROCS:-2}"
# PREPARE_ONLY has been removed - embedding_preparation step is no longer supported
TRAIN_OR_TEST="${TRAIN_OR_TEST:-train}"
#DATA_BASE_DIR="/scratch/global/menger/data/RNA/processed"
DATA_BASE_DIR="${DATA_BASE_DIR:-data/RNA/processed/}"
# =================================
# build extra sbatch flags based on MODE
SBATCH_EXTRA=()
if [[ "$MODE" == "gpu" ]]; then
    # Use SLURM_PARTITION from environment if available, otherwise use default
    PARTITION="${SLURM_PARTITION:-gpu}"
    SBATCH_EXTRA+=( --partition="$PARTITION" --gres=gpu:"$GPU_COUNT" )
    JOB_SUFFIX="gpu"
else
    # Use SLURM_PARTITION from environment if available, otherwise use default
    PARTITION="${SLURM_PARTITION:-slurm}"
    SBATCH_EXTRA+=( --partition="$PARTITION" )
    JOB_SUFFIX="cpu"
fi

# function to submit one array job
submit_array() {
    local label=$1
    local dir=$2

    # count chunks
    local n
    n=$(ls -1d "${dir}"/*.zarr 2>/dev/null | wc -l)
    if (( n == 0 )); then
        echo "No chunks in $dir → skipping"
        return
    fi
    echo "Submitting $n chunks for '$label' in parallel"
    if command -v sbatch &>/dev/null; then
        # — under SLURM, submit an array job —
        # Note: No dependency on main job - array jobs run independently
        # Main job will wait for array jobs to complete

        # Submit the job and capture the job ID
        job_output=$(sbatch \
          --array=0-$((n-1))%1 \
          "${SBATCH_EXTRA[@]}" \
          --export=ALL,INPUT_DIR="${dir}",\
METHODS="${METHODS}",\
BATCH_KEY="${BATCH_KEY}",\
BATCH_SIZE="${BATCH_SIZE}",\
TRAIN_OR_TEST="${TRAIN_OR_TEST}",\
WORKFLOW_DIR="${WORKFLOW_DIR:-}",\
DATASET_CONFIG="${DATASET_CONFIG:-}" \
          "$SCRIPT")

        # Extract job ID from output
        job_id=$(echo "$job_output" | grep -o "Submitted batch job [0-9]*" | grep -o "[0-9]*")
        echo "Submitted batch job $job_id"

        # Store job ID in array for cleanup on termination
        if [[ -n "$job_id" ]]; then
            SPAWNED_JOBS+=("$job_id")
            # Also write to file for external tracking
            echo "$job_id" >> /tmp/embedding_array_jobs_${SLURM_JOB_ID:-$$}.txt
        fi
    else
        # — locally, just run the script (it will detect LOCAL MODE) —
        echo "[LOCAL] Running all $n chunks for '$label' in parallel"
        INPUT_DIR="$dir" METHODS="$METHODS" MAX_PROCS="$MAX_PROCS" \
BATCH_KEY="$BATCH_KEY" BATCH_SIZE="$BATCH_SIZE" \
TRAIN_OR_TEST="$TRAIN_OR_TEST" \
source "$SCRIPT"
    fi
}

if [[ "$TRAIN_OR_TEST" == "test" ]]; then
    DATA_DIR="${DATA_BASE_DIR}/test/${DATANAME}/all"
    echo "Processing test data: $DATA_DIR"
    submit_array test "$DATA_DIR"
else
    TRAIN_DIR="${DATA_BASE_DIR}/train/${DATANAME}/train"
    VAL_DIR="${DATA_BASE_DIR}/train/${DATANAME}/val"
    echo "Processing train data: $TRAIN_DIR"
    submit_array train "$TRAIN_DIR"
    echo "Processing val data: $VAL_DIR"
    submit_array val   "$VAL_DIR"
fi
