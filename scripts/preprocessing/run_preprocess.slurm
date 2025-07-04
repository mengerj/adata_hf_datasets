#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --mem=100G               # Request memory
#SBATCH --time=48:00:00         # Max job time
#SBATCH --partition=slurm

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

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
# If WORKFLOW_DIR is set (from master job), use it; otherwise use old structure
if [[ -n "${WORKFLOW_DIR:-}" ]]; then
  BASE_OUT="${WORKFLOW_DIR}/preprocessing/job_${RUN_ID}"
else
  BASE_OUT="outputs/$(date +%Y-%m-%d)/preprocessing/${RUN_ID}"
fi

mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/preprocessing.out
exec 2>"$BASE_OUT"/preprocessing.err

# The config name is passed via environment variable DATASET_CONFIG
# If not set, use a default
DATASET_CONFIG="${DATASET_CONFIG:-dataset_cellxgene_pseudo_bulk_3_5k}"

echo "Starting job"
echo "Using dataset config: $DATASET_CONFIG"
echo "Output directory: $BASE_OUT"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script with the dataset config
python3 scripts/preprocessing/preprocess.py \
    --config-name=$DATASET_CONFIG \
    ++hydra.run.dir="$BASE_OUT"
