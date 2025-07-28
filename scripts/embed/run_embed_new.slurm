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
cd /home/menger/git/adata_hf_datasets

# Activate environment
source .venv/bin/activate

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
                # Use sacct to check for any running/pending array tasks
                wait_iterations=0
                while true; do
                    wait_iterations=$((wait_iterations + 1))
                    echo "🔍 DEBUG: Wait iteration $wait_iterations for job $array_job_id"

                    # Get all array job states - handle grep exit codes properly
                    # Choose the appropriate command based on cluster type
                    if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                        # GPU job - use SSH to check status on remote cluster
                        echo "🔍 DEBUG: Checking GPU job $array_job_id status via SSH to $cluster_host"
                        set +o pipefail
                        array_states=$(ssh "$cluster_host" "sacct -j $array_job_id --format=JobID,State --noheader --parsable2 2>/dev/null" | grep "${array_job_id}_" | cut -d'|' -f2 || true)
                        ssh_exit_code=$?
                        set -o pipefail
                        echo "🔍 DEBUG: SSH command exit code: $ssh_exit_code"
                    else
                        # CPU job - check locally
                        echo "🔍 DEBUG: Checking CPU job $array_job_id status locally"
                        set +o pipefail
                        array_states=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | cut -d'|' -f2 || true)
                        sacct_exit_code=$?
                        set -o pipefail
                        echo "🔍 DEBUG: sacct command exit code: $sacct_exit_code"
                    fi

                    echo "🔍 DEBUG: Raw array_states output: '$array_states'"

                    # Additional debugging - let's see what sacct actually shows for this job
                    echo "🔍 DEBUG: Full sacct output for job $array_job_id (no filtering):"
                    if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                        ssh "$cluster_host" "sacct -j $array_job_id --format=JobID,State,JobName --noheader --parsable2 2>/dev/null" || echo "🔍 DEBUG: sacct failed"
                    else
                        sacct -j "$array_job_id" --format=JobID,State,JobName --noheader --parsable2 2>/dev/null || echo "🔍 DEBUG: sacct failed"
                    fi

                    # Also check what squeue shows
                    echo "🔍 DEBUG: squeue output for job $array_job_id:"
                    if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                        ssh "$cluster_host" "squeue -j $array_job_id --format='%.10i %.40j %.8t %.10M %.6D %R' 2>/dev/null" || echo "🔍 DEBUG: squeue shows no job"
                    else
                        squeue -j "$array_job_id" --format="%.10i %.40j %.8t %.10M %.6D %R" 2>/dev/null || echo "🔍 DEBUG: squeue shows no job"
                    fi

                    if [[ -z "$array_states" ]]; then
                        # No array tasks found yet, wait a bit
                        echo "🔍 DEBUG: No array tasks found yet for job $array_job_id, waiting... (iteration $wait_iterations)"

                        # Check if the main job itself exists
                        if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                            main_job_check=$(ssh "$cluster_host" "sacct -j $array_job_id --format=JobID,State --noheader --parsable2 2>/dev/null | head -1" || echo "")
                        else
                            main_job_check=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | head -1 || echo "")
                        fi
                        echo "🔍 DEBUG: Main job check result: '$main_job_check'"

                        # IMPROVED: Check squeue to see if any array tasks are still running
                        # If no array tasks in squeue, they might have completed but not yet in sacct
                        echo "🔍 DEBUG: Checking squeue for running array tasks..."
                        if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                            # Look for any jobs with pattern like 7528531_*
                            squeue_output=$(ssh "$cluster_host" "squeue --noheader --format='%.10i' 2>/dev/null" || echo "")
                            echo "🔍 DEBUG: Full squeue output: '$squeue_output'"
                            # Handle leading whitespace by using word boundaries
                            running_array_tasks=$(echo "$squeue_output" | grep -E "(^|[[:space:]])${array_job_id}_" | wc -l || echo "0")
                        else
                            # Look for any jobs with pattern like 7528531_*
                            squeue_output=$(squeue --noheader --format='%.10i' 2>/dev/null || echo "")
                            echo "🔍 DEBUG: Full squeue output: '$squeue_output'"
                            # Handle leading whitespace by using word boundaries
                            running_array_tasks=$(echo "$squeue_output" | grep -E "(^|[[:space:]])${array_job_id}_" | wc -l || echo "0")
                        fi
                        echo "🔍 DEBUG: Looking for pattern '(^|[[:space:]])${array_job_id}_' in squeue output"
                        echo "🔍 DEBUG: Number of running array tasks in squeue: $running_array_tasks"

                        # Debug the break condition
                        echo "🔍 DEBUG: Break condition check:"
                        echo "🔍 DEBUG:   running_array_tasks=$running_array_tasks (should be 0)"
                        echo "🔍 DEBUG:   wait_iterations=$wait_iterations (should be > 5)"
                        echo "🔍 DEBUG:   Condition result: running_array_tasks -eq 0 = $([[ $running_array_tasks -eq 0 ]] && echo "true" || echo "false")"
                        echo "🔍 DEBUG:   Condition result: wait_iterations -gt 5 = $([[ $wait_iterations -gt 5 ]] && echo "true" || echo "false")"

                        # If no tasks are running in squeue, and we've waited a reasonable time, assume completion
                        if [[ $running_array_tasks -eq 0 && $wait_iterations -gt 5 ]]; then
                            echo "🔍 DEBUG: *** BREAK CONDITION TRIGGERED ***"
                            echo "🔍 DEBUG: No array tasks in squeue after $wait_iterations iterations - assuming completion"
                            echo "  Array job $array_job_id appears to have completed (not visible in squeue)"
                            break
                        else
                            echo "🔍 DEBUG: Break condition NOT met, continuing to wait..."
                        fi

                        sleep 30
                        continue
                    fi

                    # Count different states
                    total_count=$(echo "$array_states" | wc -l)
                    echo "🔍 DEBUG: Found $total_count array task states"

                    # Show individual states for debugging
                    echo "$array_states" | while IFS= read -r state; do
                        echo "🔍 DEBUG: Array task state: '$state'"
                    done

                    # Check if any tasks are still running or pending - handle grep exit codes
                    set +o pipefail
                    running_count=$(echo "$array_states" | grep -E "^(PENDING|RUNNING|COMPLETING)$" | wc -l || echo "0")
                    completed_count=$(echo "$array_states" | grep -E "^(COMPLETED|COMPLETED\\+)$" | wc -l || echo "0")
                    failed_count=$(echo "$array_states" | grep -E "^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL)$" | wc -l || echo "0")
                    set -o pipefail

                    echo "🔍 DEBUG: Array job $array_job_id status breakdown:"
                    echo "🔍 DEBUG:   - Total tasks: $total_count"
                    echo "🔍 DEBUG:   - Running/Pending: $running_count"
                    echo "🔍 DEBUG:   - Completed: $completed_count"
                    echo "🔍 DEBUG:   - Failed: $failed_count"

                    if [[ $running_count -eq 0 ]]; then
                        echo "🔍 DEBUG: No more running tasks, breaking out of wait loop"
                        echo "  All $total_count array tasks completed"
                        break
                    else
                        echo "  Array job $array_job_id: $completed_count/$total_count tasks completed, $running_count still running..."
                        sleep 30
                    fi
                done

                # Check final status of all array tasks - handle grep exit codes
                if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                    # GPU job - use SSH for final status check
                    set +o pipefail
                    failed_tasks=$(ssh "$cluster_host" "sacct -j $array_job_id --format=JobID,State --noheader --parsable2 2>/dev/null" | grep "${array_job_id}_" | grep -E "FAILED|CANCELLED|TIMEOUT" | wc -l || echo "0")
                    total_tasks=$(ssh "$cluster_host" "sacct -j $array_job_id --format=JobID,State --noheader --parsable2 2>/dev/null" | grep "${array_job_id}_" | wc -l || echo "0")
                    set -o pipefail
                else
                    # CPU job - check locally
                    set +o pipefail
                    failed_tasks=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | grep -E "FAILED|CANCELLED|TIMEOUT" | wc -l || echo "0")
                    total_tasks=$(sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | wc -l || echo "0")
                    set -o pipefail
                fi

                if [[ $failed_tasks -gt 0 ]]; then
                    echo "ERROR: $failed_tasks out of $total_tasks array tasks failed for job $array_job_id"
                    echo "Failed tasks:"
                    # Show failed tasks - disable pipefail for this command too
                    if [[ "$cluster_type" == "gpu" && "$cluster_host" != "local" ]]; then
                        set +o pipefail
                        ssh "$cluster_host" "sacct -j $array_job_id --format=JobID,State --noheader --parsable2 2>/dev/null" | grep "${array_job_id}_" | grep -E "FAILED|CANCELLED|TIMEOUT" || echo "Could not retrieve failed task details"
                        set -o pipefail
                    else
                        set +o pipefail
                        sacct -j "$array_job_id" --format=JobID,State --noheader --parsable2 2>/dev/null | grep "${array_job_id}_" | grep -E "FAILED|CANCELLED|TIMEOUT" || echo "Could not retrieve failed task details"
                        set -o pipefail
                    fi
                    echo "Check array job logs in: $BASE_OUT/../array_${array_job_id}/"
                    exit 1
                else
                    echo "✓ Array job $array_job_id: all $total_tasks tasks completed successfully"
                fi
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

echo "Embedding job completed successfully!"
