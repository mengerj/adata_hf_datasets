#!/bin/bash
#SBATCH --job-name=create_ds
#SBATCH --time=12:00:00

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
BASE_OUT="outputs/$(date +%Y-%m-%d)/create_ds/${RUN_ID}"
mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/create_ds.out
exec 2>"$BASE_OUT"/create_ds.err

###
# 2. Set dataset configuration:
#    Use a dataset config file that inherits from dataset_default.yaml
#    and overrides the necessary parameters for this specific dataset
###
# Example: Use dataset_cellxgene_pseudo_bulk_3_5k.yaml which should contain:
# - dataset metadata (name, file_path, etc.)
# - common keys (batch_key, annotation_key, caption_key, etc.)
# - dataset_creation overrides if needed

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
#    - Apply path transformations to generate data_dir from dataset metadata
#    - Extract all necessary parameters from the config
###
python3 scripts/dataset_creation/create_dataset.py \
    --config-name=$DATASET_CONFIG \
    ++hydra.run.dir="$BASE_OUT"
