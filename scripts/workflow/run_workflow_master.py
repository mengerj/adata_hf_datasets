#!/usr/bin/env python3
"""
Run the workflow master job from the SLURM script.
This script loads the workflow config, ensures login configs are dicts, and runs the orchestrator.
"""

import sys
import logging
from pathlib import Path
import os
from omegaconf import OmegaConf

from hydra import compose, initialize_config_dir
from adata_hf_datasets.workflow import (
    WorkflowOrchestrator,
    run_workflow_localhost,
    resolve_workflow_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("Setting up Python environment...")
# Add src to path for adata_hf_datasets imports
sys.path.insert(0, str(Path.cwd() / "src"))

if len(sys.argv) < 2:
    print("Usage: python run_workflow_master.py <DATASET_CONFIG_NAME_OR_PATH>")
    print("  DATASET_CONFIG_NAME_OR_PATH can be:")
    print("    - A config name (e.g., 'dataset_cellxgene_pseudo_bulk_10k')")
    print("    - A relative path (e.g., 'conf/my_dataset.yaml')")
    print("    - An absolute path (e.g., '/path/to/config.yaml')")
    sys.exit(1)

dataset_config_name_or_path = sys.argv[1]

print("Loading configs...")
config_path = str(Path.cwd() / "conf")
print(f"Config path: {config_path}")

# Load workflow config and exit Hydra context before calling orchestrator
# This allows the orchestrator to initialize Hydra for dataset config loading
with initialize_config_dir(config_dir=config_path, version_base=None):
    # Load the workflow orchestrator config to get parameters
    workflow_config = compose(config_name="workflow_orchestrator")
    print("Loaded workflow orchestrator config")
    print(f"Config keys: {list(workflow_config.keys())}")

    # Extract workflow section with proper error handling
    workflow_section = workflow_config.get("workflow", {})
    if not workflow_section:
        raise ValueError("No workflow section found in workflow_orchestrator config")

    execution_mode = os.environ.get(
        "EXECUTION_MODE", workflow_section.get("execution_mode", "slurm")
    )

    # Resolve workflow config based on execution mode
    resolved_config = resolve_workflow_config(workflow_section, execution_mode)

    # Set BASE_FILE_PATH early so config transformations use the correct backend path
    os.environ["BASE_FILE_PATH"] = str(resolved_config["base_file_path"])

    # Convert login configs to plain dicts if needed (before exiting Hydra context)
    if execution_mode != "local":
        cpu_login = workflow_section.get("cpu_login")
        gpu_login = workflow_section.get("gpu_login")

        # Convert to plain dicts if needed
        if not isinstance(cpu_login, dict):
            cpu_login = OmegaConf.to_container(cpu_login)
        if not isinstance(gpu_login, dict):
            gpu_login = OmegaConf.to_container(gpu_login)
    else:
        cpu_login = None
        gpu_login = None

# Exit Hydra context before calling orchestrator (orchestrator will initialize its own)
if execution_mode == "local":
    print("Execution mode: local - running localhost backend...")
    run_workflow_localhost(
        dataset_config_name_or_path=dataset_config_name_or_path,
        workflow_config=resolved_config,
        force=False,
    )
else:
    # For SLURM mode, we need login configs
    if not cpu_login:
        raise ValueError(
            "CPU login configuration required in workflow_orchestrator config"
        )

    print(f"CPU login: {cpu_login}")
    print(f"GPU login: {gpu_login}")
    print("Creating orchestrator...")
    orchestrator = WorkflowOrchestrator(cpu_login=cpu_login, gpu_login=gpu_login)
    print("Starting workflow execution (SLURM mode)...")
    orchestrator.run_workflow(
        dataset_config_name_or_path=dataset_config_name_or_path,
        workflow_config=resolved_config,
        force=False,
    )
print("Workflow execution completed")
