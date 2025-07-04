#!/bin/bash
#SBATCH --job-name=transfer_gpu_to_cpu
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=slurm

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

SOURCE_PATH="$BASE_FILE_PATH/processed_with_emb/$DATASET_NAME"
DEST_PATH="$BASE_FILE_PATH/processed_with_emb/$DATASET_NAME"

echo "Source path (GPU): $SOURCE_PATH"
echo "Destination path (CPU): $DEST_PATH"

echo "Checking if GPU embedding has completed..."

# Create transfer statistics file
STATS_FILE="$BASE_OUT/transfer_stats.json"

# ─────────────────────────────────────────────────────────────────────────────
# Perform Transfer
# ─────────────────────────────────────────────────────────────────────────────

echo "Starting zip-based directory transfer..."

python3 -c "
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add scripts/util to path
sys.path.insert(0, 'scripts/util')

from ssh_transfer_utils import SSHTransferUtils, SSHTransferError

def main():
    # Get configuration from environment
    dataset_config = os.environ.get('DATASET_CONFIG')
    base_file_path = os.environ.get('BASE_FILE_PATH')
    dataset_name = os.environ.get('DATASET_NAME')
    stats_file = os.environ.get('STATS_FILE')

    # Define paths
    source_path = f'{base_file_path}/processed_with_emb/{dataset_name}'
    dest_path = f'{base_file_path}/processed_with_emb/{dataset_name}'

    print(f'Transfer Details:')
    print(f'  Source: GPU:{source_path}')
    print(f'  Destination: CPU:{dest_path}')
    print(f'  Dataset: {dataset_name}')
    print('')

    # Initialize transfer utilities with workflow config
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.load('conf/workflow_orchestrator.yaml')
        cpu_login = OmegaConf.to_container(config.workflow.cpu_login)
        gpu_login = OmegaConf.to_container(config.workflow.gpu_login)

        transfer_utils = SSHTransferUtils(cpu_login, gpu_login)
        print('Transfer utilities initialized successfully')
    except Exception as e:
        print(f'Failed to initialize transfer utilities: {e}')
        return 1

    try:
        # Test connectivity first
        print('Testing SSH connectivity...')
        connectivity = transfer_utils.test_ssh_connectivity()
        if not all(connectivity.values()):
            print('ERROR: SSH connectivity test failed!')
            for cluster, status in connectivity.items():
                print(f'  {cluster}: {\"✓\" if status else \"✗\"}')
            return 1

        print('✓ SSH connectivity successful')

        # Wait for source directory to exist (GPU embedding might still be running)
        print('Waiting for GPU embedding to complete...')
        max_wait_time = 3600  # 1 hour
        wait_interval = 60    # 1 minute
        waited_time = 0

        while waited_time < max_wait_time:
            if transfer_utils.check_file_exists('gpu', source_path):
                print(f'✓ Source directory found after {waited_time} seconds')
                break

            print(f'  Waiting for source directory to appear... ({waited_time}/{max_wait_time}s)')
            time.sleep(wait_interval)
            waited_time += wait_interval
        else:
            print(f'ERROR: Source directory not found after {max_wait_time} seconds')
            print(f'GPU embedding may have failed or is taking longer than expected')
            return 1

        # Check disk space on CPU cluster
        try:
            print('Checking CPU cluster disk space...')
            cpu_disk = transfer_utils.get_disk_usage('cpu', base_file_path)
            available_gb = cpu_disk[\"available\"] // (1024**3)
            total_gb = cpu_disk[\"total\"] // (1024**3)
            used_gb = cpu_disk[\"used\"] // (1024**3)

            print(f'CPU cluster disk usage:')
            print(f'  Total: {total_gb} GB')
            print(f'  Used: {used_gb} GB')
            print(f'  Available: {available_gb} GB')

            # Get source directory size for space estimation
            source_size = transfer_utils.get_directory_size('gpu', source_path)
            source_gb = source_size // (1024**3)
            required_gb = source_gb * 2  # 2x for safety (zip + unzip)

            print(f'Source directory size: {source_gb} GB')
            print(f'Estimated space required: {required_gb} GB')

            if available_gb < required_gb:
                print(f'WARNING: Low disk space on CPU cluster!')
                print(f'  Available: {available_gb} GB')
                print(f'  Required: {required_gb} GB')
                print('  Proceeding anyway, but transfer may fail...')
            else:
                print(f'✓ Sufficient disk space available')

        except Exception as e:
            print(f'Warning: Could not check disk space: {e}')
            print('Proceeding with transfer...')

        # Perform zip-based directory transfer
        print('')
        print('=== Starting Directory Transfer ===')
        print(f'Transfer method: ZIP compression')
        print(f'Temporary zip name: processed_with_emb_{dataset_name}_gpu_to_cpu.zip')

        stats = transfer_utils.transfer_directory_as_zip(
            'gpu', source_path,
            'cpu', dest_path,
            temp_zip_name=f'processed_with_emb_{dataset_name}_gpu_to_cpu.zip'
        )

        # Log detailed transfer statistics
        print('')
        print('=== Transfer Statistics ===')
        print(f'Total time: {stats[\"total_time_seconds\"]:.2f} seconds')
        print(f'  Zip creation: {stats[\"zip_time_seconds\"]:.2f} seconds')
        print(f'  File transfer: {stats[\"download_time_seconds\"] + stats[\"upload_time_seconds\"]:.2f} seconds')
        print(f'  Unzip extraction: {stats[\"unzip_time_seconds\"]:.2f} seconds')
        print(f'Compressed size: {stats[\"zip_size_bytes\"] / (1024**2):.1f} MB')
        if \"compression_ratio\" in stats:
            print(f'Compression ratio: {stats[\"compression_ratio\"]:.2f}')
        if \"effective_throughput_mbps\" in stats:
            print(f'Effective throughput: {stats[\"effective_throughput_mbps\"]:.2f} MB/s')
        print('')

        # Verify transfer success
        print('Verifying transfer...')
        if not transfer_utils.check_file_exists('cpu', dest_path):
            print('ERROR: Transfer verification failed - destination directory not found')
            return 1

        # Save statistics to file
        if stats_file:
            stats['timestamp'] = datetime.now().isoformat()
            stats['dataset_name'] = dataset_name
            stats['source_path'] = source_path
            stats['dest_path'] = dest_path
            stats['transfer_direction'] = 'gpu_to_cpu'
            stats['wait_time_seconds'] = waited_time

            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f'Transfer statistics saved to: {stats_file}')

        # Optional: Clean up GPU data to save space
        try:
            print('Cleaning up GPU data to save space...')
            # Only remove the processed data (embeddings), keep the original processed folder
            transfer_utils.cleanup_files('gpu', [f'{base_file_path}/processed_with_emb/{dataset_name}'])
            print('✓ GPU cleanup completed')
        except Exception as e:
            print(f'Warning: GPU cleanup failed: {e}')
            print('  You may want to manually clean up GPU data later')

        print('✓ Transfer completed successfully!')
        print(f'✓ Embeddings are now available on CPU cluster at: {dest_path}')
        print('✓ CPU embedding step can now proceed (if enabled)')

        return 0

    except SSHTransferError as e:
        print(f'ERROR: Transfer failed: {e}')
        return 1
    except Exception as e:
        print(f'ERROR: Unexpected error during transfer: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
" STATS_FILE="$STATS_FILE"

TRANSFER_EXIT_CODE=$?

# ─────────────────────────────────────────────────────────────────────────────
# Job Completion
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "================================="
if [ $TRANSFER_EXIT_CODE -eq 0 ]; then
    echo "=== GPU to CPU Transfer Completed Successfully ==="
    echo "End Time: $(date)"
    echo "Next Step: CPU embedding/dataset creation can now proceed with data at $DEST_PATH"

    # Create success marker file
    touch "$BASE_OUT/transfer_success"
else
    echo "=== GPU to CPU Transfer Failed ==="
    echo "End Time: $(date)"
    echo "Exit code: $TRANSFER_EXIT_CODE"
    echo "Check logs in: $BASE_OUT"

    # Create failure marker file
    touch "$BASE_OUT/transfer_failed"
    exit $TRANSFER_EXIT_CODE
fi
