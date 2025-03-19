#!/bin/bash
#SBATCH --job-name=verify_ds
#SBATCH --output=verify_ds.out
#SBATCH --error=verify_ds.err
#SBATCH --time=4:00:00         # Max job time

echo "Starting job"
# Source the setup script to ensure the environment is ready
source cpu_venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/verify_datasets.py
