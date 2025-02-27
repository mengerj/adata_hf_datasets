#!/bin/bash
#SBATCH --job-name=create_ds
#SBATCH --output=create_ds.out
#SBATCH --error=create_ds.err
#SBATCH --time=12:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source cpu_venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/create_dataset.py
