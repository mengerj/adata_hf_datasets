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
from adata_hf_datasets.workflow_orchestrator import (
    WorkflowOrchestrator,
    run_workflow_localhost,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("Setting up Python environment...")
# Add src to path for adata_hf_datasets imports
sys.path.insert(0, str(Path.cwd() / "src"))

if len(sys.argv) < 2:
    print("Usage: python run_workflow_master.py <DATASET_CONFIG>")
    sys.exit(1)

dataset_config_name = sys.argv[1]

print("Loading configs...")
config_path = str(Path.cwd() / "conf")
print(f"Config path: {config_path}")

with initialize_config_dir(config_dir=config_path, version_base=None):
    # Load the workflow orchestrator config to get parameters
    workflow_config = compose(config_name="workflow_orchestrator")
    print("Loaded workflow orchestrator config")
    print(f"Config keys: {list(workflow_config.keys())}")

    # Extract workflow section with proper error handling
    workflow_section = workflow_config.get("workflow", {})
    if not workflow_section:
        raise ValueError("No workflow section found in workflow_orchestrator config")

    cpu_login = workflow_section.get("cpu_login")
    gpu_login = workflow_section.get("gpu_login")
    execution_mode = os.environ.get(
        "EXECUTION_MODE", workflow_section.get("execution_mode", "slurm")
    )
    # Set BASE_FILE_PATH early so config transformations use the correct backend path
    if execution_mode == "local":
        os.environ["BASE_FILE_PATH"] = workflow_section.get(
            "local_base_file_path", str(Path.cwd() / "data")
        )
    else:
        os.environ["BASE_FILE_PATH"] = workflow_section["slurm_base_file_path"]

    # Convert to plain dicts if needed
    if not isinstance(cpu_login, dict):
        cpu_login = OmegaConf.to_container(cpu_login)
    if not isinstance(gpu_login, dict):
        gpu_login = OmegaConf.to_container(gpu_login)

    if not cpu_login:
        raise ValueError(
            "CPU login configuration required in workflow_orchestrator config"
        )

    print(f"CPU login: {cpu_login}")
    print(f"GPU login: {gpu_login}")

if execution_mode == "local":
    print("Execution mode: local - running localhost backend...")
    run_workflow_localhost(
        dataset_config_name=dataset_config_name,
        workflow_config=workflow_section,
        force=False,
    )
else:
    print("Creating orchestrator...")
    orchestrator = WorkflowOrchestrator(cpu_login=cpu_login, gpu_login=gpu_login)
    print("Starting workflow execution (SLURM, wait mode)...")
    orchestrator.run_workflow_local(
        dataset_config_name=dataset_config_name,
        workflow_config=workflow_section,
        force=False,
    )
print("Workflow execution completed")
