#!/bin/bash
#SBATCH --job-name=embed_parallel
#SBATCH --output=logs/embed_parallel_%j.out
#SBATCH --error=logs/embed_parallel_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Exit on any error
set -euo pipefail

# Load environment variables from orchestrator
# These will be set by the orchestrator via --export
echo "=== Embedding Job Started ==="
echo "Dataset config: $DATASET_CONFIG"
echo "Job ID: $SLURM_JOB_ID"

# Load the dataset config to get embedding parameters
cd /home/menger/git/adata_hf_datasets

# Activate conda environment (adjust path as needed)
source .venv/bin/activate

# Run the embedding script with parameters from the dataset config
# We'll use a Python script to extract parameters and run the bash script
python scripts/embed/run_embed_with_config.py --config-name "$DATASET_CONFIG"

echo "=== Embedding Job Completed ==="
