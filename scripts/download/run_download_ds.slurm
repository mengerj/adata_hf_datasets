#!/bin/bash
#SBATCH --job-name=download_ds
#SBATCH --time=6:00:00

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
BASE_OUT="outputs/$(date +%Y-%m-%d)/download_ds/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/download_ds.out
exec 2>"$BASE_OUT"/download_ds.err

###
# 2. Set dataset configuration:
#    Use a dataset config file that inherits from dataset_default.yaml
#    and overrides the necessary parameters for this specific dataset
###
# Example: Use dataset_cellxgene_pseudo_bulk_3_5k.yaml which should contain:
# - dataset metadata (name, download_url, file_path, etc.)
# - common keys (batch_key, annotation_key, etc.)
# - download overrides if needed (subset_size, stratify_keys, etc.)

DATASET_CONFIG="dataset_cellxgene_pseudo_bulk_3_5k"

###
# 3. Activate environment, etc.
###
echo "Starting job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
echo "Using dataset config: $DATASET_CONFIG"
source cpu_venv/bin/activate
echo "venv activated"

###
# 4. Run your Hydra-powered script with the dataset config
#    The script will automatically:
#    - Load the dataset config (which inherits from dataset_default.yaml)
#    - Apply path transformations to generate output_path from dataset metadata
#    - Extract all necessary parameters from the config
#    - Check if file already exists and skip download if it does
#    - Apply stratification if configured
###
python3 scripts/download/download_dataset_config.py \
    --config-name=$DATASET_CONFIG \
    ++hydra.run.dir="$BASE_OUT"
