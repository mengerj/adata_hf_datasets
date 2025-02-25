#!/bin/bash
#SBATCH --job-name=split_adata
#SBATCH --output=split_adata.out
#SBATCH --error=split_adata.err
#SBATCH --mem=100G               # Request amount of memory
#SBATCH --time=12:00:00         # Max job time

echo "Starting job"
# Source the setup script to ensure the environment is ready
source split_venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/split_adata.py
