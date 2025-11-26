#!/bin/bash
#SBATCH --job-name=embed_master
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

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

# Set up cleanup trap for the temp file
temp_cleanup() {
    local job_file="/scratch/global/menger/tmp/embedding_array_jobs_${SLURM_JOB_ID}.txt"
    if [[ -f "$job_file" ]]; then
        echo "Cleaning up temp file: $job_file"
        rm -f "$job_file"
    fi
}

# Set up trap to clean up on exit (success, failure, or cancellation)
trap temp_cleanup EXIT INT TERM

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
