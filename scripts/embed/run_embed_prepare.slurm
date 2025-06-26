#!/bin/bash
#SBATCH --job-name=embed_prepare
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Exit on any error
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_JOB_ID}"
else
  # local run: use date+seconds to make it unique
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
# If WORKFLOW_DIR is set (from master job), use it; otherwise use old structure
if [[ -n "${WORKFLOW_DIR:-}" ]]; then
  BASE_OUT="${WORKFLOW_DIR}/embedding_prepare/job_${RUN_ID}"
else
  BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding_prepare/${RUN_ID}"
fi

mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/embedding_prepare.out
exec 2>"$BASE_OUT"/embedding_prepare.err

# Load environment variables from orchestrator
# These will be set by the orchestrator via --export
echo "=== Embedding Preparation Job Started ==="
echo "Dataset config: $DATASET_CONFIG"
echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $BASE_OUT"

# Load the dataset config to get embedding parameters
cd /home/menger/git/adata_hf_datasets

# Activate conda environment
source .venv/bin/activate

# Run the embedding preparation script with parameters from the dataset config
# Force CPU mode and prepare_only=true for this step
python scripts/embed/run_embed_with_config.py --config-name "$DATASET_CONFIG" --prepare-only --cpu-only

echo "=== Embedding Preparation Job Completed ==="
