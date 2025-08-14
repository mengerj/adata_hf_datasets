#!/bin/bash
#SBATCH --job-name=workflow_master
#SBATCH --time=96:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

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
# 1) Load workflow config to get output directory and project dir
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR=${PROJECT_DIR:-/home/menger/git/adata_hf_datasets}
cd "$PROJECT_DIR"

# Activate venv (configurable)
VENV_PATH=${VENV_PATH:-.venv}
source "$VENV_PATH"/bin/activate

# Get output directory from config
OUTPUT_DIR=$(python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from hydra import compose, initialize_config_dir

config_path = str(Path.cwd() / 'conf')
with initialize_config_dir(config_dir=config_path, version_base=None):
    workflow_config = compose(config_name='workflow_orchestrator')
    output_dir = workflow_config.workflow.get('output_directory', '/home/menger/git/adata_hf_datasets/outputs')
    project_dir = workflow_config.workflow.get('project_directory', '/home/menger/git/adata_hf_datasets')
    print(output_dir)
")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
BASE_OUT="${OUTPUT_DIR}/$(date +%Y-%m-%d)/workflow_${RUN_ID}"
mkdir -p "$BASE_OUT/logs"

# Redirect logs to the output directory
exec 1>"$BASE_OUT/logs/workflow_master.out"
exec 2>"$BASE_OUT/logs/workflow_master.err"

echo "=== Workflow Master Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset config: $DATASET_CONFIG"
echo "Output directory: $BASE_OUT"
echo "Base output directory from config: $OUTPUT_DIR"

echo "Starting Python workflow execution..."
python scripts/workflow/run_workflow_master.py "$DATASET_CONFIG"

echo "=== Workflow Master Job Completed ==="
