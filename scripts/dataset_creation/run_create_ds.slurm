#!/bin/bash
#SBATCH --job-name=create_ds
#SBATCH --time=72:00:00

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
  BASE_OUT="${WORKFLOW_DIR}/dataset_creation/job_${RUN_ID}"
else
  BASE_OUT="outputs/$(date +%Y-%m-%d)/create_ds/${RUN_ID}"
fi

mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/create_ds.out
exec 2>"$BASE_OUT"/create_ds.err

###
# 2. Set dataset configuration:
#    Use a dataset config file that inherits from dataset_default.yaml
#    and overrides the necessary parameters for this specific dataset
###
# The config name is passed via environment variable DATASET_CONFIG
# If not set, use a default
DATASET_CONFIG="${DATASET_CONFIG:-dataset_cellxgene_pseudo_bulk_3_5k}"

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

# Build the command with overrides
CMD="python3 scripts/dataset_creation/create_dataset.py --config-name=$DATASET_CONFIG ++hydra.run.dir=\"$BASE_OUT\""

# Check if CS_LENGTH_OVERRIDE is set and add it to the command
if [[ -n "${CS_LENGTH_OVERRIDE:-}" ]]; then
    echo "Using cs_length override: $CS_LENGTH_OVERRIDE"
    CMD="$CMD ++dataset_creation.cs_length=$CS_LENGTH_OVERRIDE"
else
    echo "Using cs_length from config file"
fi

# Check if CAPTION_KEY_OVERRIDE is set and add it to the command
if [[ -n "${CAPTION_KEY_OVERRIDE:-}" ]]; then
    echo "Using caption_key override: $CAPTION_KEY_OVERRIDE"
    CMD="$CMD ++caption_key=\"$CAPTION_KEY_OVERRIDE\""
else
    echo "Using caption_key from config file"
fi

# Execute the command
echo "Running command: $CMD"
eval $CMD

# Capture the exit code
EXIT_CODE=$?

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "Dataset creation completed successfully"
    echo "SUCCESS: Dataset creation finished successfully" >> "$BASE_OUT"/create_ds.out
else
    echo "Dataset creation failed with exit code: $EXIT_CODE"
    echo "FAILED: Dataset creation failed with exit code: $EXIT_CODE" >> "$BASE_OUT"/create_ds.err
fi

# Exit with the same code as the Python script
exit $EXIT_CODE
