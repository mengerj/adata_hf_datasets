#!/bin/bash
#SBATCH --job-name=cross_dataset_pca
#SBATCH --output=cross_dataset_pca_%j.out
#SBATCH --error=cross_dataset_pca_%j.err
#SBATCH --time=24:00:00       # Max job time - adjust as needed
#SBATCH --mem=250G              # Memory request - adjust as needed
#SBATCH --cpus-per-task=8      # CPU cores - adjust as needed

echo "=========================================="
echo "Starting cross_dataset_pca job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Change to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Configure the config path here (edit as needed)
THIS_DIR="scripts/util"
CONFIG_PATH="${CONFIG_PATH:-$THIS_DIR/pca_config_example.yaml}"
LOG_DIR="${LOG_DIR:-logs}"

echo "Config path: $CONFIG_PATH"
echo "Log dir: $LOG_DIR"
echo "=========================================="

# Activate project virtual environment
source .venv/bin/activate
echo "Virtual environment activated"

# Build command
CMD="python $THIS_DIR/cross_dataset_pca.py --config $CONFIG_PATH --log-dir $LOG_DIR"

echo "Running command: $CMD"
echo "=========================================="

eval $CMD
EXIT_CODE=$?

echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
