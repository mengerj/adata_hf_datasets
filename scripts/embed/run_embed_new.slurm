#!/bin/bash
#SBATCH --job-name=embed_master
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Exit on any error
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup and Logging
# ─────────────────────────────────────────────────────────────────────────────

# Get job information
SLURM_JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

# Required environment variables (should be set by workflow orchestrator)
DATASET_CONFIG="${DATASET_CONFIG:?DATASET_CONFIG environment variable is required}"
MODE="${MODE:-cpu}"
PREPARE_ONLY="${PREPARE_ONLY:-false}"
WORKFLOW_DIR="${WORKFLOW_DIR:-}"

echo "=== Embedding Master Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Config: $DATASET_CONFIG"
echo "Mode: $MODE"
echo "Prepare Only: $PREPARE_ONLY"
echo "Workflow Dir: $WORKFLOW_DIR"
echo "Start Time: $(date)"

# Set up output directory and logging
if [[ -n "$WORKFLOW_DIR" ]]; then
    if [[ "$PREPARE_ONLY" == "true" ]]; then
        BASE_OUT="${WORKFLOW_DIR}/embedding_prepare/job_${SLURM_JOB_ID}"
        JOB_TYPE="embedding_prepare"
    else
        BASE_OUT="${WORKFLOW_DIR}/embedding/job_${SLURM_JOB_ID}"
        JOB_TYPE="embedding"
    fi
else
    # Fallback to default structure
    if [[ "$PREPARE_ONLY" == "true" ]]; then
        BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding_prepare/${SLURM_JOB_ID}"
        JOB_TYPE="embedding_prepare"
    else
        BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding/${SLURM_JOB_ID}"
        JOB_TYPE="embedding"
    fi
fi

mkdir -p "$BASE_OUT"
echo "Output Directory: $BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/master.out
exec 2>"$BASE_OUT"/master.err

# Re-log after redirection
echo "=== Embedding Master Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Config: $DATASET_CONFIG"
echo "Mode: $MODE"
echo "Prepare Only: $PREPARE_ONLY"
echo "Job Type: $JOB_TYPE"
echo "Output Directory: $BASE_OUT"
echo "Start Time: $(date)"
echo "================================="

# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup
# ─────────────────────────────────────────────────────────────────────────────

# Change to project directory
cd /home/menger/git/adata_hf_datasets

# Activate environment
source .venv/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Launch Array Jobs
# ─────────────────────────────────────────────────────────────────────────────

echo "Launching embedding array jobs..."

# Build launcher command
launcher_cmd=(
    "python" "scripts/embed/embed_launcher.py"
    "--config-name" "$DATASET_CONFIG"
    "--mode" "$MODE"
)

if [[ "$PREPARE_ONLY" == "true" ]]; then
    launcher_cmd+=("--prepare-only")
fi

# Log the command
echo "Executing: ${launcher_cmd[*]}"

# Set environment variables for the launcher
export DATASET_CONFIG
export MODE
export PREPARE_ONLY
export WORKFLOW_DIR

# Run the launcher
if ! "${launcher_cmd[@]}"; then
    echo "ERROR: Failed to launch embedding array jobs"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Wait for Array Jobs to Complete
# ─────────────────────────────────────────────────────────────────────────────

echo "Waiting for array jobs to complete..."

# Check for array job IDs file
job_file="/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"
if [[ -f "$job_file" ]]; then
    echo "Found array job IDs file: $job_file"

    # Read and wait for each array job
    while IFS= read -r array_job_id; do
        if [[ -n "$array_job_id" && "$array_job_id" =~ ^[0-9]+$ ]]; then
            echo "Waiting for array job $array_job_id to complete..."

            # Wait for ALL array job tasks to complete
            # Use sacct to check for any running/pending array tasks
            while true; do
                # Get all array job states
                array_states=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | cut -d'|' -f2)

                if [[ -z "$array_states" ]]; then
                    # No array tasks found yet, wait a bit
                    echo "  No array tasks found yet for job $array_job_id, waiting..."
                    sleep 30
                    continue
                fi

                # Check if any tasks are still running or pending
                running_count=$(echo "$array_states" | grep -E "^(PENDING|RUNNING|COMPLETING)$" | wc -l)
                total_count=$(echo "$array_states" | wc -l)
                completed_count=$((total_count - running_count))

                if [[ $running_count -eq 0 ]]; then
                    echo "  All $total_count array tasks completed"
                    break
                else
                    echo "  Array job $array_job_id: $completed_count/$total_count tasks completed, $running_count still running..."
                    sleep 30
                fi
            done

            # Check final status of all array tasks
            failed_tasks=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | grep -E "FAILED|CANCELLED|TIMEOUT" | wc -l)
            total_tasks=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | wc -l)

            if [[ $failed_tasks -gt 0 ]]; then
                echo "ERROR: $failed_tasks out of $total_tasks array tasks failed for job $array_job_id"
                echo "Failed tasks:"
                sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | grep -E "FAILED|CANCELLED|TIMEOUT"
                echo "Check array job logs in: $BASE_OUT/../array_${array_job_id}/"
                exit 1
            else
                echo "✓ Array job $array_job_id: all $total_tasks tasks completed successfully"
            fi
        fi
    done < "$job_file"

    # Clean up the temporary file
    rm -f "$job_file"
    echo "All array jobs completed successfully"
else
    echo "WARNING: No array job file found at $job_file"
    echo "This might indicate that no array jobs were submitted"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Completion
# ─────────────────────────────────────────────────────────────────────────────

echo "================================="
echo "=== Embedding Master Job Complete ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Config: $DATASET_CONFIG"
echo "Mode: $MODE"
echo "Job Type: $JOB_TYPE"
echo "Output Directory: $BASE_OUT"
echo "End Time: $(date)"
echo "================================="

echo "Embedding job completed successfully!"
