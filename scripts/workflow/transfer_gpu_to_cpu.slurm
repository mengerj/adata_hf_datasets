#!/bin/bash
#SBATCH --job-name=transfer_gpu_to_cpu
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Exit on any error
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Environment Setup and Logging
# ─────────────────────────────────────────────────────────────────────────────

# Get job information
SLURM_JOB_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

# Required environment variables (should be set by workflow orchestrator)
DATASET_CONFIG="${DATASET_CONFIG:?DATASET_CONFIG environment variable is required}"
BASE_FILE_PATH="${BASE_FILE_PATH:?BASE_FILE_PATH environment variable is required}"
DATASET_NAME="${DATASET_NAME:?DATASET_NAME environment variable is required}"
WORKFLOW_DIR="${WORKFLOW_DIR:-}"

echo "=== GPU to CPU Transfer Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Config: $DATASET_CONFIG"
echo "Base File Path: $BASE_FILE_PATH"
echo "Dataset Name: $DATASET_NAME"
echo "Workflow Dir: $WORKFLOW_DIR"
echo "Start Time: $(date)"

# Set up output directory and logging
if [[ -n "$WORKFLOW_DIR" ]]; then
    BASE_OUT="${WORKFLOW_DIR}/transfer_gpu_to_cpu/job_${SLURM_JOB_ID}"
else
    # Fallback to default structure
    BASE_OUT="outputs/$(date +%Y-%m-%d)/transfer_gpu_to_cpu/${SLURM_JOB_ID}"
fi

mkdir -p "$BASE_OUT"
echo "Output Directory: $BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/transfer.out
exec 2>"$BASE_OUT"/transfer.err

# Re-log after redirection
echo "=== GPU to CPU Transfer Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Config: $DATASET_CONFIG"
echo "Base File Path: $BASE_FILE_PATH"
echo "Dataset Name: $DATASET_NAME"
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

echo "Python environment activated"
echo "Python path: $(which python3)"
echo "Working directory: $(pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# Define Transfer Paths
# ─────────────────────────────────────────────────────────────────────────────

# Determine if this is a training or test dataset by checking the config
python3 -c "
import sys
sys.path.insert(0, '/home/menger/git/adata_hf_datasets/src')
from omegaconf import OmegaConf
from adata_hf_datasets.config_utils import apply_all_transformations

try:
    cfg = OmegaConf.load('/home/menger/git/adata_hf_datasets/conf/$DATASET_CONFIG.yaml')
    cfg = apply_all_transformations(cfg)
    is_training = cfg.preprocessing.get('split_dataset', True)
    train_or_test = 'train' if is_training else 'test'
    print(train_or_test)
except:
    print('train')  # Default fallback
" > /tmp/dataset_type_$$.txt

DATASET_TYPE=$(cat /tmp/dataset_type_$$.txt)
rm -f /tmp/dataset_type_$$.txt

echo "Detected dataset type: $DATASET_TYPE"

# For GPU to CPU transfer, we're transferring embeddings back
# The GPU embedding output should be in processed_with_emb directory under BASE_FILE_PATH
echo "Base file path: $BASE_FILE_PATH"
echo "Dataset type: $DATASET_TYPE"
echo "Dataset name: $DATASET_NAME"

# The GPU embeddings are stored in processed_with_emb directory
SOURCE_PATH="$BASE_FILE_PATH/processed_with_emb/$DATASET_TYPE/$DATASET_NAME"
DEST_PATH="$BASE_FILE_PATH/processed_with_emb/$DATASET_TYPE/$DATASET_NAME"
TEMP_ZIP_NAME="embeddings_${DATASET_NAME}_gpu_to_cpu.zip"
TEMP_ZIP_PATH="/tmp/$TEMP_ZIP_NAME"

echo "Source path (GPU): $SOURCE_PATH"
echo "Destination path (CPU): $DEST_PATH"
echo "Temporary zip file: $TEMP_ZIP_PATH"

# Validate source directory exists
if [ ! -d "$SOURCE_PATH" ]; then
    echo "ERROR: Source directory does not exist: $SOURCE_PATH"
    echo "This probably means the embedding step failed or hasn't completed yet."
    exit 1
fi

# Get directory size for logging
SOURCE_SIZE=$(du -sh "$SOURCE_PATH" | cut -f1)
echo "Source directory size: $SOURCE_SIZE"

# ─────────────────────────────────────────────────────────────────────────────
# Create Zip File Locally
# ─────────────────────────────────────────────────────────────────────────────

echo "Creating zip file locally on GPU cluster..."
ZIP_START=$(date +%s)

# Create zip file (balanced compression for speed, preserve directory structure)
cd "$(dirname "$SOURCE_PATH")"

# Try pigz (parallel gzip) with tar first, fallback to zip if not available
if command -v pigz >/dev/null 2>&1; then
    echo "Using pigz (parallel gzip) for faster compression..."
    tar -cf - "$(basename "$SOURCE_PATH")" | pigz -6 -p 4 > "$TEMP_ZIP_PATH.tar.gz"
    TEMP_ZIP_PATH="$TEMP_ZIP_PATH.tar.gz"
    TEMP_ZIP_NAME="$TEMP_ZIP_NAME.tar.gz"
else
    echo "Using zip with fast compression (pigz not available)..."
    # Use -1 for fastest compression instead of -9 for maximum compression
    zip -r -q -1 "$TEMP_ZIP_PATH" "$(basename "$SOURCE_PATH")"
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create zip file"
    exit 1
fi

ZIP_END=$(date +%s)
ZIP_TIME=$((ZIP_END - ZIP_START))

ZIP_SIZE=$(du -sh "$TEMP_ZIP_PATH" | cut -f1)
echo "Created zip file: $ZIP_SIZE in ${ZIP_TIME}s"

# ─────────────────────────────────────────────────────────────────────────────
# Transfer to CPU Cluster
# ─────────────────────────────────────────────────────────────────────────────

echo "Transferring zip file to CPU cluster..."
TRANSFER_START=$(date +%s)

# Store the project directory for later use
PROJECT_DIR="/home/menger/git/adata_hf_datasets"
CONFIG_FILE="$PROJECT_DIR/conf/workflow_orchestrator.yaml"

# Get CPU cluster connection info from config
echo "Loading CPU cluster configuration..."
echo "Project directory: $PROJECT_DIR"
echo "Config file: $CONFIG_FILE"

# First, let's debug the Python environment
echo "Debugging Python environment..."
python3 -c "
import sys
import os
print('Python version:', sys.version)
print('Python path:', sys.path)
print('Current working directory:', os.getcwd())
print('Files in current directory:', os.listdir('.'))
config_file = '$CONFIG_FILE'
if os.path.exists(config_file):
    print('Config file exists: YES')
else:
    print('Config file exists: NO')

try:
    from omegaconf import OmegaConf
    print('OmegaConf import: SUCCESS')
except Exception as e:
    print('OmegaConf import: FAILED -', str(e))
"

# Now get the CPU host info
echo "Getting CPU host information..."
CPU_HOST=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')

try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('$CONFIG_FILE')
    cpu_login = cfg.workflow.cpu_login
    print(f'{cpu_login.user}@{cpu_login.host}')
except Exception as e:
    print('ERROR')
")

echo "CPU_HOST result: '$CPU_HOST'"

if [[ "$CPU_HOST" == "ERROR" ]]; then
    echo "ERROR: Failed to get CPU host information"
    echo "Let's check the config file structure:"
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/src')
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('$CONFIG_FILE')
    print('Config structure:')
    print(OmegaConf.to_yaml(cfg))
except Exception as e:
    print('Error loading config:', str(e))
    import traceback
    traceback.print_exc()
"
    exit 1
fi

if [[ -z "$CPU_HOST" ]]; then
    echo "ERROR: CPU_HOST is empty"
    exit 1
fi

echo "CPU host: $CPU_HOST"

# Transfer zip file to CPU cluster
echo "Executing SCP command: scp '$TEMP_ZIP_PATH' '$CPU_HOST:/tmp/$TEMP_ZIP_NAME'"
echo "Source file exists: $(test -f "$TEMP_ZIP_PATH" && echo "YES" || echo "NO")"
echo "Source file size: $(ls -lh "$TEMP_ZIP_PATH" 2>/dev/null || echo "N/A")"

scp -v "$TEMP_ZIP_PATH" "$CPU_HOST:/tmp/$TEMP_ZIP_NAME"
SCP_EXIT_CODE=$?

echo "SCP exit code: $SCP_EXIT_CODE"

if [ $SCP_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to transfer zip file to CPU cluster"
    echo "SCP command failed with exit code: $SCP_EXIT_CODE"
    exit 1
fi

TRANSFER_END=$(date +%s)
TRANSFER_TIME=$((TRANSFER_END - TRANSFER_START))

echo "Zip file transferred in ${TRANSFER_TIME}s"

# ─────────────────────────────────────────────────────────────────────────────
# Unzip on CPU Cluster
# ─────────────────────────────────────────────────────────────────────────────

echo "Unzipping on CPU cluster..."
UNZIP_START=$(date +%s)

# Create destination directory on CPU cluster
ssh "$CPU_HOST" "mkdir -p '$(dirname "$DEST_PATH")'"

# Unzip on CPU cluster
if [[ "$TEMP_ZIP_NAME" == *.tar.gz ]]; then
    echo "Extracting tar.gz file..."
    ssh "$CPU_HOST" "cd '$(dirname "$DEST_PATH")' && tar -xzf '/tmp/$TEMP_ZIP_NAME'"
else
    echo "Extracting zip file..."
    ssh "$CPU_HOST" "cd '$(dirname "$DEST_PATH")' && unzip -q -o '/tmp/$TEMP_ZIP_NAME'"
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to unzip file on CPU cluster"
    exit 1
fi

# Handle case where extracted directory name doesn't match destination
ssh "$CPU_HOST" "
if [ -d '$(dirname "$DEST_PATH")/$(basename "$SOURCE_PATH")' ] && [ '$(dirname "$DEST_PATH")/$(basename "$SOURCE_PATH")' != '$DEST_PATH' ]; then
    mv '$(dirname "$DEST_PATH")/$(basename "$SOURCE_PATH")' '$DEST_PATH'
fi
"

UNZIP_END=$(date +%s)
UNZIP_TIME=$((UNZIP_END - UNZIP_START))

echo "Unzipped in ${UNZIP_TIME}s"

# ─────────────────────────────────────────────────────────────────────────────
# Verify Transfer and Cleanup
# ─────────────────────────────────────────────────────────────────────────────

echo "Verifying transfer..."

# Check if destination directory exists on CPU cluster
ssh "$CPU_HOST" "test -d '$DEST_PATH'"

if [ $? -ne 0 ]; then
    echo "ERROR: Transfer verification failed - destination directory not found on CPU cluster"
    exit 1
fi

# Get destination directory size for verification
DEST_SIZE=$(ssh "$CPU_HOST" "du -sh '$DEST_PATH' | cut -f1")
echo "Destination directory size: $DEST_SIZE"

# Cleanup temporary files
echo "Cleaning up temporary files..."
rm -f "$TEMP_ZIP_PATH"
ssh "$CPU_HOST" "rm -f '/tmp/$TEMP_ZIP_NAME'"

# ─────────────────────────────────────────────────────────────────────────────
# Calculate and Save Statistics
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - ZIP_START))

# Create transfer statistics file
STATS_FILE="$BASE_OUT/transfer_stats.json"

cat > "$STATS_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "dataset_name": "$DATASET_NAME",
  "source_path": "$SOURCE_PATH",
  "dest_path": "$DEST_PATH",
  "transfer_direction": "gpu_to_cpu",
  "transfer_type": "directory_zip_local",
  "source_size": "$SOURCE_SIZE",
  "dest_size": "$DEST_SIZE",
  "zip_size": "$ZIP_SIZE",
  "total_time_seconds": $TOTAL_TIME,
  "zip_time_seconds": $ZIP_TIME,
  "transfer_time_seconds": $TRANSFER_TIME,
  "unzip_time_seconds": $UNZIP_TIME,
  "temp_zip_name": "$TEMP_ZIP_NAME"
}
EOF

echo "Transfer statistics saved to: $STATS_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# Job Completion
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "=== Transfer Statistics ==="
echo "Total time: ${TOTAL_TIME}s"
echo "  Zip creation: ${ZIP_TIME}s"
echo "  File transfer: ${TRANSFER_TIME}s"
echo "  Unzip extraction: ${UNZIP_TIME}s"
echo "Source size: $SOURCE_SIZE"
echo "Zip size: $ZIP_SIZE"
echo "Destination size: $DEST_SIZE"
echo ""
echo "================================="
echo "=== GPU to CPU Transfer Completed Successfully ==="
echo "End Time: $(date)"
echo "✓ Embeddings are now available on CPU cluster at: $DEST_PATH"
echo "✓ Workflow can now proceed with further processing"

# Create success marker file
touch "$BASE_OUT/transfer_success"
