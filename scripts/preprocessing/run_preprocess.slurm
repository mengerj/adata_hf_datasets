#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --mem=250G               # Request memory
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
# Ensure we're in the correct working directory
PROJECT_DIR=${PROJECT_DIR:-/home/menger/git/adata_hf_datasets}
cd "$PROJECT_DIR"

# Source the setup script to ensure the environment is ready
VENV_PATH=${VENV_PATH:-.venv}
source "$VENV_PATH"/bin/activate
echo "venv activated"

# Print working directory for debugging
echo "Current working directory: $(pwd)"
echo "Checking if data directory exists: $(ls -la data/ 2>/dev/null || echo 'data directory does not exist')"

# Now run your Python script with the dataset config
echo "Starting preprocessing with config: $DATASET_CONFIG"

CMD="python3 scripts/preprocessing/preprocess.py --config-name=$DATASET_CONFIG ++hydra.run.dir=\"$BASE_OUT\""
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
    echo "Preprocessing completed successfully"
    echo "SUCCESS: Preprocessing finished successfully" >> "$BASE_OUT"/preprocessing.out
else
    echo "Preprocessing failed with exit code: $EXIT_CODE"
    echo "FAILED: Preprocessing failed with exit code: $EXIT_CODE" >> "$BASE_OUT"/preprocessing.err
fi

# Exit with the same code as the Python script
exit $EXIT_CODE
