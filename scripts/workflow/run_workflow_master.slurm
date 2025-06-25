#!/bin/bash
#SBATCH --job-name=workflow_master
#SBATCH --output=logs/workflow_master_%j.out
#SBATCH --error=logs/workflow_master_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Exit on any error
set -euo pipefail

echo "=== Workflow Master Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset config: $DATASET_CONFIG"

# Load the dataset config to determine which steps to run
cd /home/menger/git/adata_hf_datasets

# Activate conda environment
source .venv/bin/activate

# Import the orchestrator and run the workflow
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from omegaconf import OmegaConf
from adata_hf_datasets.config_utils import apply_all_transformations
from scripts.workflow.orchestrate_workflow import WorkflowOrchestrator

# Load the dataset config
config_path = Path('conf') / f'{sys.argv[1]}.yaml'
config = OmegaConf.load(config_path)
config = apply_all_transformations(config)

# Create orchestrator with SSH connections
orchestrator = WorkflowOrchestrator(
    cpu_login={'host': 'imbi13', 'user': 'menger'},
    gpu_login={'host': 'imbi_gpu_H100', 'user': 'menger'}
)

# Run the workflow in local mode (on the cluster)
orchestrator.run_workflow_local(
    dataset_config_name=sys.argv[1],
    workflow_config=config.workflow,
    force=False
)
" "$DATASET_CONFIG"

echo "=== Workflow Master Job Completed ==="
