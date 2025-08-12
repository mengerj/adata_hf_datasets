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
# If WORKFLOW_DIR is set (from master job), use it; otherwise use old structure
if [[ -n "${WORKFLOW_DIR:-}" ]]; then
  BASE_OUT="${WORKFLOW_DIR}/download/job_${RUN_ID}"
else
  BASE_OUT="outputs/$(date +%Y-%m-%d)/download_ds/${RUN_ID}"
fi

mkdir -p "$BASE_OUT"

# Redirect logs to the output directory
exec 1>"$BASE_OUT"/download_ds.out
exec 2>"$BASE_OUT"/download_ds.err

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
PROJECT_DIR=${PROJECT_DIR:-/home/menger/git/adata_hf_datasets}
cd "$PROJECT_DIR"
VENV_PATH=${VENV_PATH:-.venv}
source "$VENV_PATH"/bin/activate
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
echo "Starting download with config: $DATASET_CONFIG"

# Build command and include base_file_path override if provided via env
CMD="python3 scripts/download/download_dataset_config.py --config-name=$DATASET_CONFIG ++hydra.run.dir=\"$BASE_OUT\""
if [[ -n "${BASE_FILE_PATH:-}" ]]; then
  echo "Using base_file_path override: $BASE_FILE_PATH"
  CMD="$CMD ++base_file_path=\"$BASE_FILE_PATH\""
fi

echo "Running: $CMD"
eval $CMD

# Capture the exit code
EXIT_CODE=$?

# Log completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "Download completed successfully"
    echo "SUCCESS: Download finished successfully" >> "$BASE_OUT"/download.out
else
    echo "Download failed with exit code: $EXIT_CODE"
    echo "FAILED: Download failed with exit code: $EXIT_CODE" >> "$BASE_OUT"/download.err
fi

# Exit with the same code as the Python script
exit $EXIT_CODE
