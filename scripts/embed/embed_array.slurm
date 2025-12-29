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
EMBEDDING_CONFIG_SECTION="${EMBEDDING_CONFIG_SECTION:?EMBEDDING_CONFIG_SECTION environment variable is required}"
WORKFLOW_DIR="${WORKFLOW_DIR:-}"

# Get array job information
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?This script must be run as a SLURM array job}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# Set up logging directory
if [[ -n "$WORKFLOW_DIR" ]]; then
    LOG_DIR="${WORKFLOW_DIR}/embedding/array_${ARRAY_JOB_ID}"
    OUTPUT_DIR="${WORKFLOW_DIR}/embedding/array_${ARRAY_JOB_ID}/${TASK_ID}"
else
    # Fallback to default structure
    LOG_DIR="logs/embed/${ARRAY_JOB_ID}"
    OUTPUT_DIR="outputs/$(date +%Y-%m-%d)/embed/${ARRAY_JOB_ID}/${TASK_ID}"
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
echo "Embedding Config Section: $EMBEDDING_CONFIG_SECTION"
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
PROJECT_DIR=${PROJECT_DIR:-/home/menger/git/adata_hf_datasets}
cd "$PROJECT_DIR"

# Activate environment
VENV_PATH=${VENV_PATH:-.venv}
source "$VENV_PATH"/bin/activate

# ─────────────────────────────────────────────────────────────────────────────
# Build Configuration Parameters
# ─────────────────────────────────────────────────────────────────────────────

echo "Using dataset config: $DATASET_CONFIG"
echo "Using embedding section: $EMBEDDING_CONFIG_SECTION"

# ─────────────────────────────────────────────────────────────────────────────
# Run Embedding Core
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting embedding processing..."

# Pass the config selection to embed_core.py via environment variables
# No temporary files needed!
export EMBEDDING_CONFIG_SECTION
#path always is read relative to the script by hydra
# If "gpu" is in the embedding config section, run GPU diagnostics and checks
if [[ "$EMBEDDING_CONFIG_SECTION" == *"gpu"* ]]; then
    echo "=== GPU Environment Validation ==="
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'not set'}"
    echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-'not set'}"
    echo "SLURM_GPUS: ${SLURM_GPUS:-'not set'}"

    echo "=== GPU diagnostics: nvidia-smi ==="
    if nvidia-smi; then
        echo "✓ nvidia-smi successful"
    else
        echo "✗ nvidia-smi failed - this is a critical issue"
        exit 1
    fi

    echo "=== GPU Memory Status ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

    echo "=== GPU diagnostics: nvidia-smi --query-compute-apps ==="
    # Minimal, machine-readable list of PIDs currently running CUDA kernels
    nvidia-smi --query-compute-apps=pid --format=csv,noheader

    echo "=== GPU diagnostics: nvidia-smi -q | grep -E 'Xid|Retired' ==="
    nvidia-smi -q | grep -E "Xid|Retired" || echo "No GPU errors found"

    echo "=== ECC page-retirement status ==="
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        nvidia-smi -i $CUDA_VISIBLE_DEVICES -q -d PAGE_RETIREMENT | \
        grep -E "Retired|Pending" || echo "No page retirement issues"
    fi

    echo "=== Device nodes visible in the cgroup ==="
    ls -l /dev/nvidia* | head

    echo "=== CUDA Availability Test ==="
    python3 -c "
import torch
import sys
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB')
        print(f'Memory cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB')
    # Test basic CUDA operations
    try:
        device = torch.device('cuda:0')
        test_tensor = torch.randn(10, 10).to(device)
        result = torch.mm(test_tensor, test_tensor.t())
        print('✓ Basic CUDA operations successful')
    except Exception as e:
        print(f'✗ Basic CUDA operations failed: {e}')
        sys.exit(1)
else:
    print('✗ CUDA not available')
    sys.exit(1)
"

    if [ $? -ne 0 ]; then
        echo "✗ CUDA availability test failed - GPU not accessible"
        exit 1
    fi
    echo "✓ GPU environment validation complete"
fi

echo "Overriding input_files for section: $EMBEDDING_CONFIG_SECTION"

CMD="python3 scripts/embed/embed_core.py --config-path=\"../../conf\" --config-name=\"$DATASET_CONFIG\" ++${EMBEDDING_CONFIG_SECTION}.input_files=\"[\"$target_file\"]\" ++hydra.run.dir=\"$OUTPUT_DIR\" ++embedding_config_section=\"$EMBEDDING_CONFIG_SECTION\""

if [[ -n "${BASE_FILE_PATH:-}" ]]; then
  echo "Using base_file_path override: $BASE_FILE_PATH"
  CMD="$CMD ++base_file_path=\"$BASE_FILE_PATH\""
fi

echo "Running: $CMD"
eval $CMD

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

# Write completion status to temp file for main job tracking
completion_file="/scratch/global/menger/tmp/array_job_completed_${ARRAY_JOB_ID}_${TASK_ID}.txt"
echo "Writing completion status to: $completion_file"
mkdir -p "/scratch/global/menger/tmp"
echo "COMPLETED:${ARRAY_JOB_ID}:${TASK_ID}:${embedding_exit_code}:$(date):${elapsed}" > "$completion_file"

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
