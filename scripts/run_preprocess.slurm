#!/bin/bash
#SBATCH --job-name=pp_adata
#SBATCH --output=pp_adata_3_5.out
#SBATCH --error=pp_adata_3_5.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=50G               # Request amount of memory
#SBATCH --time=2:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/preprocess_adata.py
