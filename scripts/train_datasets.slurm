#!/bin/bash
#SBATCH --job-name=train_dataset
#SBATCH --output=train_dataset.out
#SBATCH --error=train_dataset.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --time=02:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/train_datasets_create.py --geo_n 7k --cellxgene_n 3_5k

