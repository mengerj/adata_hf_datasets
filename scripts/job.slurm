#!/bin/bash
#SBATCH --job-name=intial_embeddings
#SBATCH --output=initial_emb_dataset.out
#SBATCH --error=initial_emb_dataset.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --time=02:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/create_datasets.py

