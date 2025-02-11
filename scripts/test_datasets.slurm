#!/bin/bash
#SBATCH --job-name=test_datasets
#SBATCH --output=test_dataset.out
#SBATCH --error=test_dataset.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --time=02:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/test_datasets_create.py

