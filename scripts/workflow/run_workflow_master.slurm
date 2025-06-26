#!/bin/bash
#SBATCH --job-name=workflow_master
#SBATCH --time=72:00:00
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
# 1) Prepare output directory and log redirects
# ─────────────────────────────────────────────────────────────────────────────
BASE_OUT="outputs/$(date +%Y-%m-%d)/workflow_${RUN_ID}"
mkdir -p "$BASE_OUT/logs"

# Redirect logs to the output directory
exec 1>"$BASE_OUT/logs/workflow_master.out"
exec 2>"$BASE_OUT/logs/workflow_master.err"

echo "=== Workflow Master Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset config: $DATASET_CONFIG"
echo "Output directory: $BASE_OUT"

# Load the dataset config to determine which steps to run
cd /home/menger/git/adata_hf_datasets

# Activate conda environment
source .venv/bin/activate
echo "Conda environment activated"

# Import the orchestrator and run the workflow
echo "Starting Python workflow execution..."
python -c "
import sys
import logging
from pathlib import Path

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print('Setting up Python environment...')
sys.path.insert(0, str(Path.cwd() / 'src'))

from hydra import compose, initialize_config_dir
from adata_hf_datasets.config_utils import apply_all_transformations
from scripts.workflow.orchestrate_workflow import WorkflowOrchestrator

print('Loading configs...')
# Use absolute path for config directory
config_path = str(Path.cwd() / 'conf')
print(f'Config path: {config_path}')

with initialize_config_dir(config_dir=config_path, version_base=None):
    # Load the workflow orchestrator config to get SSH parameters
    workflow_config = compose(config_name='workflow_orchestrator')
    print('Loaded workflow orchestrator config')
    print(f'Config keys: {list(workflow_config.keys())}')

    # Extract workflow section with proper error handling
    workflow_section = workflow_config.get('workflow', {})
    if not workflow_section:
        raise ValueError('No workflow section found in workflow_orchestrator config')

    cpu_login = workflow_section.get('cpu_login')
    gpu_login = workflow_section.get('gpu_login')

    if not cpu_login:
        raise ValueError('CPU login configuration required in workflow_orchestrator config')

    print(f'CPU login: {cpu_login}')
    print(f'GPU login: {gpu_login}')

print('Creating orchestrator...')
# Create orchestrator with SSH connections from config
orchestrator = WorkflowOrchestrator(
    cpu_login=cpu_login,
    gpu_login=gpu_login
)

print('Starting workflow execution...')
# Run the workflow in local mode (on the cluster)
orchestrator.run_workflow_local(
    dataset_config_name=sys.argv[1],
    workflow_config=workflow_section,
    force=False
)
print('Workflow execution completed')
" "$DATASET_CONFIG"

echo "=== Workflow Master Job Completed ==="
