#!/bin/bash
#SBATCH --job-name=embed_master
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ═══════════════════════════════════════════════════════════════════════════════
# DEPRECATION NOTICE
# ═══════════════════════════════════════════════════════════════════════════════
# This "master job" script is DEPRECATED and kept only for standalone/legacy usage.
#
# For the new unified workflow, use:
#   python scripts/workflow/submit_workflow.py --config-name <dataset_config>
#
# The new workflow uses EmbeddingArraySubmitter to directly submit array jobs
# via SSH without needing an intermediate master SLURM job. Benefits:
#   - No wasted resources (no idle master job)
#   - Direct array job tracking from Python orchestrator
#   - Better cancellation handling
#   - Cleaner architecture
#
# This script is only useful if you want to manually submit embedding jobs
# without using the workflow orchestrator.
# ═══════════════════════════════════════════════════════════════════════════════

# Exit on any error, but be more careful about pipefail
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
PROJECT_DIR=${PROJECT_DIR:-/home/menger/git/adata_hf_datasets}
cd "$PROJECT_DIR"

# Activate environment
VENV_PATH=${VENV_PATH:-.venv}
source "$VENV_PATH"/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Launch Array Jobs
# ─────────────────────────────────────────────────────────────────────────────

echo "Launching embedding array jobs..."

# Ensure temp directory exists
mkdir -p /scratch/global/menger/tmp

# Set up cleanup trap for temp files AND array job cancellation
cleanup_and_cancel() {
    local signal="$1"
    local job_file="/scratch/global/menger/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"

    # If we received a termination signal (not EXIT), cancel spawned array jobs
    if [[ "$signal" != "EXIT" ]]; then
        echo "Received $signal signal, cancelling spawned array jobs..."
        if [[ -f "$job_file" ]]; then
            while IFS= read -r job_line; do
                if [[ -n "$job_line" ]]; then
                    # Parse job line format: job_id:cluster_type:host
                    IFS=':' read -r array_job_id cluster_type cluster_host <<< "$job_line"
                    if [[ -n "$array_job_id" && "$array_job_id" =~ ^[0-9]+$ ]]; then
                        echo "Cancelling array job $array_job_id..."
                        if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                            ssh "$cluster_host" "scancel $array_job_id" 2>/dev/null && \
                                echo "Cancelled array job $array_job_id on $cluster_host" || \
                                echo "Failed to cancel array job $array_job_id on $cluster_host"
                        else
                            scancel "$array_job_id" 2>/dev/null && \
                                echo "Cancelled array job $array_job_id" || \
                                echo "Failed to cancel array job $array_job_id"
                        fi
                    fi
                fi
            done < "$job_file"
        fi
    fi

    # Clean up temp file
    if [[ -f "$job_file" ]]; then
        echo "Cleaning up temp file: $job_file"
        rm -f "$job_file"
    fi

    # Clean up any completion marker files for this job
    rm -f /scratch/global/menger/tmp/array_job_completed_*_*.txt 2>/dev/null || true

    # If terminated by signal, exit with appropriate code
    if [[ "$signal" == "TERM" || "$signal" == "INT" ]]; then
        exit 130
    fi
}

# Set up traps for different signals
trap 'cleanup_and_cancel EXIT' EXIT
trap 'cleanup_and_cancel TERM' TERM
trap 'cleanup_and_cancel INT' INT

# ─────────────────────────────────────────────────────────────────────────────
# Direct Array Job Submission
# ─────────────────────────────────────────────────────────────────────────────
# Note: This script now directly submits array jobs instead of relying on
# embed_launcher.py, which has been simplified to local-only execution.
# For unified workflow execution, use the orchestrator's RemoteExecutor.

# Determine input directories based on config
# Use Python to extract paths from the Hydra config
echo "Determining input directories from config..."

# Get base path and dataset name from config
INPUT_INFO=$(python -c "
import os
import sys
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from adata_hf_datasets.workflow import apply_all_transformations

GlobalHydra.instance().clear()
config_path = os.path.join(os.getcwd(), 'conf')

with initialize_config_dir(config_dir=config_path, version_base=None):
    cfg = compose(config_name='$DATASET_CONFIG')
cfg = apply_all_transformations(cfg)

# Determine base path
base_file_path = os.environ.get('BASE_FILE_PATH') or cfg.get('base_file_path', './data/RNA')

# Check if CPU embedding is enabled (for GPU mode input selection)
cpu_enabled = hasattr(cfg, 'embedding_cpu') and cfg.embedding_cpu is not None and cfg.embedding_cpu.get('enabled', True)
mode = '$MODE'
prepare_only = '$PREPARE_ONLY' == 'true'

# Determine input subdirectory
if mode == 'gpu' and not prepare_only and cpu_enabled:
    input_subdir = 'processed_with_emb'
else:
    input_subdir = 'processed'

# Determine train or test
split_dataset = cfg.preprocessing.get('split_dataset', True)
train_or_test = 'train' if split_dataset else 'test'
dataset_name = cfg.dataset.name

print(f'{base_file_path}|{input_subdir}|{train_or_test}|{dataset_name}|{split_dataset}')
" 2>/dev/null)

if [[ -z "$INPUT_INFO" ]]; then
    echo "ERROR: Failed to determine input directories from config"
    exit 1
fi

IFS='|' read -r BASE_PATH INPUT_SUBDIR TRAIN_OR_TEST DATASET_NAME SPLIT_DATASET <<< "$INPUT_INFO"
echo "Base path: $BASE_PATH"
echo "Input subdir: $INPUT_SUBDIR"
echo "Train/Test: $TRAIN_OR_TEST"
echo "Dataset name: $DATASET_NAME"
echo "Split dataset: $SPLIT_DATASET"

# Build list of input directories to process
declare -a INPUT_DIRS
declare -a LABELS

if [[ "$TRAIN_OR_TEST" == "test" ]]; then
    INPUT_DIRS=("$BASE_PATH/$INPUT_SUBDIR/test/$DATASET_NAME/all")
    LABELS=("test")
else
    INPUT_DIRS=(
        "$BASE_PATH/$INPUT_SUBDIR/train/$DATASET_NAME/train"
        "$BASE_PATH/$INPUT_SUBDIR/train/$DATASET_NAME/val"
    )
    LABELS=("train" "val")
fi

# Submit array jobs for each input directory
echo "Submitting array jobs..."
JOB_FILE="/scratch/global/menger/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"

for i in "${!INPUT_DIRS[@]}"; do
    INPUT_DIR="${INPUT_DIRS[$i]}"
    LABEL="${LABELS[$i]}"

    # Check if directory exists
    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "WARNING: Directory does not exist: $INPUT_DIR, skipping $LABEL"
        continue
    fi

    # Count zarr files
    FILE_COUNT=$(ls -1d "$INPUT_DIR"/*.zarr 2>/dev/null | wc -l)

    if [[ "$FILE_COUNT" -eq 0 ]]; then
        echo "WARNING: No .zarr files found in $INPUT_DIR, skipping $LABEL"
        continue
    fi

    echo "Found $FILE_COUNT zarr files in $INPUT_DIR for $LABEL"

    # Build sbatch command
    SBATCH_CMD=(
        "sbatch"
        "--job-name=embed_${LABEL}"
        "--array=0-$((FILE_COUNT - 1))%${FILE_COUNT}"
        "--time=6:00:00"
    )

    # Add resource requirements based on mode
    if [[ "$MODE" == "cpu" ]]; then
        SBATCH_CMD+=("--mem=60G" "--cpus-per-task=4" "--partition=slurm")
    else
        SBATCH_CMD+=("--mem=64G" "--cpus-per-task=4" "--gres=gpu:1" "--partition=gpu")
    fi

    # Determine embedding config section
    if [[ "$PREPARE_ONLY" == "true" ]]; then
        CONFIG_SECTION="embedding_preparation"
    elif [[ "$MODE" == "cpu" ]]; then
        CONFIG_SECTION="embedding_cpu"
    else
        CONFIG_SECTION="embedding_gpu"
    fi

    # Build environment variables
    ENV_VARS="INPUT_DIR=$INPUT_DIR"
    ENV_VARS+=",DATASET_CONFIG=$DATASET_CONFIG"
    ENV_VARS+=",PREPARE_ONLY=$PREPARE_ONLY"
    ENV_VARS+=",WORKFLOW_DIR=$WORKFLOW_DIR"
    ENV_VARS+=",MODE=$MODE"
    ENV_VARS+=",BASE_FILE_PATH=$BASE_PATH"
    ENV_VARS+=",EMBEDDING_CONFIG_SECTION=$CONFIG_SECTION"

    SBATCH_CMD+=("--export=ALL,$ENV_VARS")
    SBATCH_CMD+=("scripts/embed/embed_chunks_parallel.slurm")

    echo "Executing: ${SBATCH_CMD[*]}"

    # Submit the job
    SUBMIT_OUTPUT=$("${SBATCH_CMD[@]}" 2>&1)
    SUBMIT_RC=$?

    if [[ $SUBMIT_RC -ne 0 ]]; then
        echo "ERROR: Failed to submit array job for $LABEL: $SUBMIT_OUTPUT"
        continue
    fi

    # Parse job ID
    JOB_ID=$(echo "$SUBMIT_OUTPUT" | grep -o "Submitted batch job [0-9]*" | grep -o "[0-9]*")

    if [[ -n "$JOB_ID" ]]; then
        echo "Submitted array job $JOB_ID for $LABEL ($FILE_COUNT tasks)"
        # Write to tracking file
        echo "$JOB_ID:$MODE:local" >> "$JOB_FILE"
    else
        echo "ERROR: Could not parse job ID from: $SUBMIT_OUTPUT"
    fi
done

# Check if any jobs were submitted
if [[ ! -f "$JOB_FILE" ]] || [[ ! -s "$JOB_FILE" ]]; then
    echo "ERROR: No array jobs were submitted"
    exit 1
fi

echo "Successfully submitted array jobs. Tracking file: $JOB_FILE"
cat "$JOB_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# Wait for Array Jobs to Complete
# ─────────────────────────────────────────────────────────────────────────────

echo "Waiting for array jobs to complete..."

# Check for array job IDs file
job_file="/scratch/global/menger/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"
echo "🔍 DEBUG: Looking for array job IDs file: $job_file"

# Add a brief wait to allow the launcher to finish writing
echo "🔍 DEBUG: Waiting 10 seconds for launcher to finish writing temp file..."
sleep 10

# Check if temp directory exists
temp_dir="/scratch/global/menger/tmp"
if [[ -d "$temp_dir" ]]; then
    echo "🔍 DEBUG: Temp directory exists: $temp_dir"
    ls -la "$temp_dir"
else
    echo "🔍 DEBUG: ERROR - Temp directory does not exist: $temp_dir"
fi

# Check for the specific file
if [[ -f "$job_file" ]]; then
    echo "🔍 DEBUG: Found array job IDs file: $job_file"

    # Check file properties
    file_size=$(stat -c%s "$job_file" 2>/dev/null || echo "unknown")
    file_perms=$(stat -c%a "$job_file" 2>/dev/null || echo "unknown")
    echo "🔍 DEBUG: File size: $file_size bytes, permissions: $file_perms"

    # Show file contents for debugging
    echo "🔍 DEBUG: File contents:"
    cat "$job_file" | while IFS= read -r debug_line; do
        echo "🔍 DEBUG: Content line: '$debug_line'"
    done

    # Count lines
    line_count=$(wc -l < "$job_file")
    echo "🔍 DEBUG: File has $line_count lines"

    # Read and wait for each array job
    line_num=0
    while IFS= read -r job_line; do
        line_num=$((line_num + 1))
        echo "🔍 DEBUG: Processing line $line_num: '$job_line'"
        # Parse job line format: job_id:cluster_type:host
        if [[ -n "$job_line" ]]; then
            # Split the line by colons
            IFS=':' read -r array_job_id cluster_type cluster_host <<< "$job_line"

            # Validate job ID
            if [[ -n "$array_job_id" && "$array_job_id" =~ ^[0-9]+$ ]]; then
                echo "Waiting for array job $array_job_id to complete (cluster: $cluster_type, host: $cluster_host)..."
                echo "🔍 DEBUG: Parsed job_id=$array_job_id, cluster_type=$cluster_type, cluster_host=$cluster_host"

                # Wait for ALL array job tasks to complete
                # Use temp file approach - much more reliable than squeue/sacct
                wait_iterations=0
                echo "🔍 DEBUG: Using temp file approach to track array job completion"

                # First, we need to know how many array tasks to expect
                # This should match the file count that was used in the launcher
                echo "🔍 DEBUG: Checking expected number of array tasks for job $array_job_id"

                while true; do
                    wait_iterations=$((wait_iterations + 1))
                    echo "🔍 DEBUG: Wait iteration $wait_iterations for job $array_job_id"

                    # Count completion files for this array job
                    completion_files="/scratch/global/menger/tmp/array_job_completed_${array_job_id}_*.txt"
                    completed_count=0
                    if ls $completion_files 1> /dev/null 2>&1; then
                        completed_count=$(ls $completion_files | wc -l)
                    fi

                    echo "🔍 DEBUG: Found $completed_count completion files for job $array_job_id"

                    # Check if any array tasks are still running in squeue
                    running_count=0
                    if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                        running_tasks=$(ssh "$cluster_host" "squeue --noheader --format='%.10i' 2>/dev/null" | grep "${array_job_id}_" || echo "")
                    else
                        running_tasks=$(squeue --noheader --format='%.10i' 2>/dev/null | grep "${array_job_id}_" || echo "")
                    fi

                    if [[ -n "$running_tasks" ]]; then
                        running_count=$(echo "$running_tasks" | wc -l)
                    fi

                    echo "🔍 DEBUG: Array job $array_job_id status:"
                    echo "🔍 DEBUG:   - Completed tasks (temp files): $completed_count"
                    echo "🔍 DEBUG:   - Still running in squeue: $running_count"

                    # If no tasks are running AND we have completion files, we're done
                    if [[ $running_count -eq 0 && $completed_count -gt 0 ]]; then
                        echo "🔍 DEBUG: *** ALL ARRAY TASKS COMPLETED ***"
                        echo "  Array job $array_job_id: $completed_count tasks completed, none running"

                        # Show completion file details
                        echo "🔍 DEBUG: Completion file details:"
                        for file in $completion_files; do
                            if [[ -f "$file" ]]; then
                                echo "🔍 DEBUG:   $(basename "$file"): $(cat "$file")"
                            fi
                        done
                        break
                    elif [[ $wait_iterations -gt 7200 ]]; then
                        # 2 hour timeout
                        echo "🔍 DEBUG: *** TIMEOUT AFTER 120 HOURS ***"
                        echo "WARNING: Array job $array_job_id timeout - $completed_count completed, $running_count still running"
                        echo "Proceeding anyway..."
                        break
                    else
                        echo "🔍 DEBUG: Waiting for remaining tasks... (iteration $wait_iterations)"
                        if [[ -n "$running_tasks" ]]; then
                            echo "🔍 DEBUG: Still running: $running_tasks"
                        fi
                        sleep 120
                    fi
                done

                # Clean up completion files for this job
                echo "🔍 DEBUG: Cleaning up completion files for job $array_job_id"
                rm -f $completion_files
                echo "✓ Array job $array_job_id: monitoring complete"

                # Final status is determined by completion files - much more reliable than sacct!
            else
                echo "Warning: Invalid job line format: $job_line"
            fi
        fi
    done < "$job_file"

    # Clean up the temporary file
    rm -f "$job_file"
    echo "All array jobs completed successfully"

    # No temporary config files to clean up with the new environment variable approach!
else
    echo "🔍 DEBUG: ERROR - No array job file found at $job_file"
    echo "🔍 DEBUG: This might indicate that no array jobs were submitted or temp file creation failed"

    # Check if any files exist in the temp directory
    if [[ -d "$temp_dir" ]]; then
        echo "🔍 DEBUG: Contents of temp directory $temp_dir:"
        ls -la "$temp_dir" || echo "🔍 DEBUG: Failed to list temp directory contents"

        # Look for any files with similar names
        echo "🔍 DEBUG: Looking for any embedding_array_jobs files in temp directory:"
        find "$temp_dir" -name "*embedding_array_jobs*" -type f 2>/dev/null || echo "🔍 DEBUG: No embedding_array_jobs files found"
    else
        echo "🔍 DEBUG: Temp directory $temp_dir does not exist at all"
    fi

    # Check if there are any array jobs running under our job ID anyway
    echo "🔍 DEBUG: Checking for any array jobs with job ID pattern ${SLURM_JOB_ID}_*"
    squeue -u $USER --format="%.10i %.40j %.8t %.10M %.6D %R" | grep "${SLURM_JOB_ID}_" || echo "🔍 DEBUG: No array jobs found in squeue"

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

# Check if any of our array jobs failed by examining their exit codes
OVERALL_SUCCESS=true

if [[ -f "$job_file" ]]; then
    while IFS= read -r array_job_id; do
        if [[ -n "$array_job_id" ]]; then
            # Check if array job failed (this is a simple check - real status would need sacct)
            array_exit_code=$(sacct -j "$array_job_id" --format=ExitCode --noheader --parsable2 2>/dev/null | head -1)
            if [[ "$array_exit_code" != "0:0" && -n "$array_exit_code" ]]; then
                echo "WARNING: Array job $array_job_id may have failed with exit code: $array_exit_code"
                OVERALL_SUCCESS=false
            fi
        fi
    done < "$job_file"
fi

if [[ "$OVERALL_SUCCESS" == "true" ]]; then
    echo "Embedding job completed successfully!"
    echo "SUCCESS: Embedding finished successfully" >> "$BASE_OUT"/embedding.out
    exit 0
else
    echo "Embedding job completed with some failures!"
    echo "FAILED: Some embedding array jobs failed" >> "$BASE_OUT"/embedding.err
    exit 1
fi
