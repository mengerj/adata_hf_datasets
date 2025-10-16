#!/usr/bin/env python3
"""
Script to submit workflow jobs with pre-validation.

This script submits a master SLURM job that runs the workflow orchestrator.
The orchestrator uses the workflow_orchestrator.yaml config for SSH parameters
and the specified dataset config for processing parameters.

Usage:
    python scripts/workflow/submit_workflow.py --config-name dataset_cellxgene_pseudo_bulk_3_5k
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from omegaconf import DictConfig

from adata_hf_datasets.workflow import ensure_config_sync

logger = logging.getLogger(__name__)


def load_workflow_config() -> DictConfig:
    """Load the workflow orchestrator configuration."""
    from hydra import compose, initialize_config_dir

    config_path = Path(__file__).parent.parent.parent / "conf"

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="workflow_orchestrator")

    return cfg


def validate_config_sync_before_submission(
    dataset_config_name: str, workflow_config: DictConfig, force: bool = False
) -> None:
    """Validate config synchronization before submitting the master job."""
    if force:
        logger.warning("Skipping config synchronization check (force=True)")
        return

    logger.info(f"Validating config synchronization for {dataset_config_name}...")

    # Get CPU host from workflow config
    cpu_login = workflow_config.get("workflow", {}).get("cpu_login")
    if not cpu_login:
        raise ValueError(
            "CPU login configuration required in workflow_orchestrator config"
        )

    cpu_host = cpu_login.get("host")
    if not cpu_host:
        raise ValueError("CPU host not found in workflow config")

    project_dir = workflow_config.workflow.get("project_directory")
    logger.info(f"Project directory: {project_dir}")
    # Validate config sync
    ensure_config_sync(
        config_name=dataset_config_name,
        remote_host=cpu_host,
        remote_project_dir=project_dir,
        force=force,
    )

    logger.info("✓ Config synchronization validation passed")


def submit_master_job(
    dataset_config_name: str, workflow_config: DictConfig, force: bool = False
) -> None:
    """Submit the master SLURM job."""
    logger.info(f"Submitting master workflow job for dataset: {dataset_config_name}")

    # Get CPU host and partition from workflow config
    workflow_section = workflow_config.get("workflow", {})
    cpu_login = workflow_section.get("cpu_login")
    cpu_partition = workflow_section.get("cpu_partition", "slurm")

    if not cpu_login:
        raise ValueError(
            "CPU login configuration required in workflow_orchestrator config"
        )

    cpu_host = cpu_login.get("host")
    if not cpu_host:
        raise ValueError("CPU host not found in workflow config")

    # Build the sbatch command
    script_path = Path("scripts/workflow/run_workflow_master.slurm")
    project_dir = workflow_config.workflow.get("project_directory")

    cmd = ["ssh", cpu_host, f"cd {project_dir} && sbatch"]

    # Add partition
    cmd.extend(["--partition", cpu_partition])

    # Add environment variables
    env_vars = {
        "DATASET_CONFIG": dataset_config_name,
        "PROJECT_DIR": project_dir,
    }
    env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    cmd.extend(["--export", f"ALL,{env_str}"])

    # Add the script path
    cmd.append(str(script_path))

    logger.info(f"Submitting master job: {' '.join(cmd)}")

    # Execute the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        error_msg = (
            f"SLURM job submission timed out for master workflow job on {cpu_host}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        error_msg = "SSH command not found. Please ensure SSH is installed and available in PATH."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if result.returncode != 0:
        error_msg = f"SLURM job submission failed for master workflow job on {cpu_host}"
        logger.error(error_msg)
        logger.error(f"Command: {' '.join(cmd)}")
        logger.error(f"Exit code: {result.returncode}")
        logger.error(f"Error: {result.stderr}")
        raise RuntimeError(error_msg)

    # Parse job ID from output
    output = result.stdout.strip()
    import re

    job_id_match = re.search(r"Submitted batch job (\d+)", output)
    if not job_id_match:
        error_msg = f"Could not parse job ID from SLURM output: {output}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    job_id = int(job_id_match.group(1))
    logger.info(f"✓ Master workflow job submitted successfully (Job ID: {job_id})")
    logger.info(f"Job will run on {cpu_host} in partition {cpu_partition}")
    logger.info(f"You can monitor progress with: ssh {cpu_host} 'squeue -j {job_id}'")

    # Get output directory from config to show where logs will be
    try:
        workflow_config = load_workflow_config()
        output_dir = workflow_config.workflow.get(
            "output_directory", "/home/menger/git/adata_hf_datasets/outputs"
        )
        date_str = datetime.now().strftime("%Y-%m-%d")
        logger.info(
            f"Logs will be gathered at: {output_dir}/{date_str}/workflow_{job_id}/"
        )
    except Exception as e:
        logger.warning(f"Could not determine output directory from config: {e}")
        logger.info("Logs will be gathered in the outputs/ directory on the cluster")


def main():
    """Main function to submit workflow with pre-validation."""
    parser = argparse.ArgumentParser(
        description="Submit workflow with config validation"
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="Dataset config name (without .yaml extension)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip config synchronization check"
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        logger.info("=" * 80)
        logger.info("WORKFLOW SUBMISSION WITH PRE-VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Dataset config: {args.config_name}")
        logger.info(f"Force mode: {args.force}")

        # Load workflow configuration
        logger.info("Loading workflow configuration...")
        workflow_config = load_workflow_config()
        logger.info("✓ Workflow configuration loaded")

        # Validate config synchronization
        validate_config_sync_before_submission(
            args.config_name, workflow_config, force=args.force
        )

        # Submit the master job
        submit_master_job(args.config_name, workflow_config, force=args.force)

        logger.info("=" * 80)
        logger.info("WORKFLOW SUBMISSION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("WORKFLOW SUBMISSION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
