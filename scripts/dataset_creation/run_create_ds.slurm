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
# 2. Set default overrides:
#    These match the current YAML defaults in conf/create_dataset.yaml
###
# This is just a selection of settings. All settings are in conf/create_dataset.yaml
DATA_DIR="data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k"
BATCH_KEY="dataset_title"
CAPTION_KEY="natural_language_annotation"
DATASET_FORMAT="multiplets"

###
# 3. Activate environment, etc.
###
echo "Starting job with RUN_ID: $RUN_ID"
echo "Output directory: $BASE_OUT"
source cpu_venv/bin/activate
echo "venv activated"

###
# 4. Run your Hydra-powered script with overrides
#    Note: The syntax create_dataset.split_dataset=... etc. assumes
#          your config group or config file is "create_dataset".
###
python3 scripts/dataset_creation/create_dataset.py \
    ++data_dir=$DATA_DIR \
    ++caption_key=$CAPTION_KEY \
    ++dataset_format=$DATASET_FORMAT \
    ++batch_key=$BATCH_KEY \
    ++hydra.run.dir="$BASE_OUT"
