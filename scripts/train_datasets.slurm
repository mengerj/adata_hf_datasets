#!/bin/bash
#SBATCH --job-name=train_dataset_350K
#SBATCH --output=train_dataset_350K.out
#SBATCH --error=train_dataset_350K.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=400G               # Request amount of memory
#SBATCH --time=24:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/train_datasets_create.py --geo_n None --cellxgene_n 350k
