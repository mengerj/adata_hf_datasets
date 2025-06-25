#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --output=logs/pp/pp_%j.out
#SBATCH --error=logs/pp/pp_%j.err
#SBATCH --mem=35G               # Request memory
#SBATCH --time=24:00:00         # Max job time

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real array job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_JOB_ID}"
else
  # local run: use date+seconds to make it unique
  RUN_ID="pp_$(date +%Y%m%d_%H%M%S)"
  # simulate an array task index if unset (so your redirects still work)
  SLURM_JOB_ID=0
fi

# Create a custom output directory name with date and SLURM job ID
CUSTOM_DIR="outputs/$(date +%Y-%m-%d)/preprocessing/${RUN_ID}"

# The config name is passed via environment variable DATASET_CONFIG
# If not set, use a default
DATASET_CONFIG="${DATASET_CONFIG:-dataset_cellxgene_pseudo_bulk_3_5k}"

echo "Starting job"
echo "Using dataset config: $DATASET_CONFIG"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script with the dataset config
python3 scripts/preprocessing/preprocess.py \
    --config-name=$DATASET_CONFIG \
    ++hydra.run.dir=$CUSTOM_DIR
