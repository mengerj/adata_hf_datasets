#!/usr/bin/env python3
"""
Unified workflow submission script.

This script automatically routes to local or SLURM execution based on the
execution_mode setting in workflow_orchestrator.yaml. You no longer need
to use different scripts for local vs SLURM execution.

Usage:
    # Local execution (when execution_mode: local in config)
    python scripts/workflow/submit_workflow.py --config my_dataset

    # SLURM execution (when execution_mode: slurm in config)
    python scripts/workflow/submit_workflow.py --config my_dataset

    # Override execution mode via environment variable
    EXECUTION_MODE=local python scripts/workflow/submit_workflow.py --config my_dataset
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import shutil

from adata_hf_datasets.workflow import ensure_config_sync, resolve_workflow_config

logger = logging.getLogger(__name__)


def load_workflow_config() -> DictConfig:
    """Load the workflow orchestrator configuration."""
    config_path = Path(__file__).parent.parent.parent / "conf"

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="workflow_orchestrator")

    return cfg


def validate_config_sync_before_submission(
    dataset_config_name: str, workflow_config: DictConfig, force: bool = False
) -> None:
    """Validate config synchronization before submitting the master job (SLURM only)."""
    if force:
        logger.warning("Skipping config synchronization check (force=True)")
        return

    logger.info(f"Validating config synchronization for {dataset_config_name}...")

    # Get CPU host from workflow config
    cpu_login = workflow_config.get("cpu_login")
    if not cpu_login:
        raise ValueError(
            "CPU login configuration required in workflow_orchestrator config"
        )

    cpu_host = cpu_login.get("host")
    if not cpu_host:
        raise ValueError("CPU host not found in workflow config")

    project_dir = workflow_config.get("slurm_project_directory")
    logger.info(f"Project directory: {project_dir}")
    # Validate config sync
    ensure_config_sync(
        config_name=dataset_config_name,
        remote_host=cpu_host,
        remote_project_dir=project_dir,
        force=force,
    )

    logger.info("✓ Config synchronization validation passed")


def submit_slurm_workflow(
    dataset_config_name: str,
    workflow_config: DictConfig,
    resolved_config: DictConfig,
    force: bool = False,
) -> None:
    """Submit the master SLURM job."""
    logger.info(f"Submitting master workflow job for dataset: {dataset_config_name}")

    # Get CPU host and partition from workflow config
    cpu_login = workflow_config.get("cpu_login")
    cpu_partition = workflow_config.get("cpu_partition", "slurm")

    if not cpu_login:
        raise ValueError(
            "CPU login configuration required in workflow_orchestrator config"
        )

    cpu_host = cpu_login.get("host")
    if not cpu_host:
        raise ValueError("CPU host not found in workflow config")

    # Build the sbatch command
    script_path = Path("scripts/workflow/run_workflow_master.slurm")
    project_dir = resolved_config["project_directory"]

    cmd = ["ssh", cpu_host, f"cd {project_dir} && sbatch"]

    # Add partition
    cmd.extend(["--partition", cpu_partition])

    # Add environment variables
    env_vars = {
        "DATASET_CONFIG": dataset_config_name,
        "PROJECT_DIR": project_dir,
        "EXECUTION_MODE": "slurm",  # Explicitly set execution mode
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

    # Get output directory from resolved config to show where logs will be
    output_dir = resolved_config["output_directory"]
    date_str = datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Logs will be gathered at: {output_dir}/{date_str}/workflow_{job_id}/")


def submit_local_workflow(
    dataset_config_name: str, resolved_config: DictConfig, foreground: bool = False
) -> None:
    """Submit the local workflow (runs on this machine)."""
    logger.info(f"Submitting local workflow for dataset: {dataset_config_name}")

    project_dir = Path(__file__).resolve().parents[2]

    # Compute output dir path for info printing
    output_dir = resolved_config["output_directory"]
    run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_out = Path(output_dir) / date_str / f"workflow_{run_id}"

    # Build command
    cmd = [
        sys.executable,
        "scripts/workflow/run_workflow_master.py",
        dataset_config_name,
    ]

    env = os.environ.copy()
    env["SLURM_JOB_ID"] = run_id
    env["EXECUTION_MODE"] = "local"

    if foreground:
        logging.info("Running local master in foreground...")
        logging.info("Press Ctrl+C to stop the workflow")
        subprocess.run(cmd, cwd=str(project_dir), env=env, check=False)
    else:
        logging.info(
            "Submitting local master in background (nohup + caffeinate if available)..."
        )
        log_dir = base_out / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "workflow_master.out"
        stderr_path = log_dir / "workflow_master.err"
        pid_file = log_dir / "workflow_master.pid"

        # Build background command with caffeinate if available (prevents sleep)
        bg_cmd = ["nohup"]
        if shutil.which("caffeinate"):
            bg_cmd += ["caffeinate", "-dimsu"]
        else:
            logging.warning(
                "caffeinate not found; background job will not prevent system sleep"
            )
        bg_cmd += cmd
        with open(stdout_path, "a") as out_f, open(stderr_path, "a") as err_f:
            process = subprocess.Popen(
                bg_cmd,
                cwd=str(project_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )

        # Save PID to file
        pid = process.pid
        with open(pid_file, "w") as f:
            f.write(f"{pid}\n")

        # Log kill command
        kill_cmd = f"kill {pid}"
        logging.info(f"Logs: {str(log_dir)}")
        logging.info(f"Outputs will be under: {str(base_out)}")
        logging.info("=" * 80)
        logging.info("WORKFLOW RUNNING IN BACKGROUND")
        logging.info("=" * 80)
        logging.info(f"Process ID (PID): {pid}")
        logging.info(f"To stop this workflow, run: {kill_cmd}")
        logging.info(f"Or use: kill $(cat {pid_file})")
        logging.info(f"PID file: {pid_file}")
        logging.info("=" * 80)


def main():
    """Main function to submit workflow with automatic routing."""
    parser = argparse.ArgumentParser(
        description="Submit workflow (automatically routes to local or SLURM based on config)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Dataset config name (without .yaml extension) or path to config file (e.g., 'conf/my_dataset.yaml' or '/absolute/path/to/config.yaml')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip config synchronization check (SLURM only)",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground (local mode only, do not detach)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        logger.info("=" * 80)
        logger.info("WORKFLOW SUBMISSION")
        logger.info("=" * 80)
        logger.info(f"Dataset config: {args.config}")
        logger.info(f"Force mode: {args.force}")

        # Load workflow configuration
        logger.info("Loading workflow configuration...")
        workflow_config_full = load_workflow_config()
        workflow_section = workflow_config_full.get("workflow", {})
        logger.info("✓ Workflow configuration loaded")

        # Determine execution mode (can be overridden via environment variable)
        execution_mode = os.environ.get(
            "EXECUTION_MODE", workflow_section.get("execution_mode", "slurm")
        )
        logger.info(f"Execution mode: {execution_mode}")

        # Resolve workflow config based on execution mode
        resolved_config = resolve_workflow_config(workflow_section, execution_mode)
        logger.info(f"Resolved output directory: {resolved_config['output_directory']}")
        logger.info(
            f"Resolved project directory: {resolved_config['project_directory']}"
        )
        logger.info(f"Resolved base file path: {resolved_config['base_file_path']}")

        if execution_mode == "local":
            # Local execution
            logger.info("Routing to local execution...")
            submit_local_workflow(
                dataset_config_name=args.config,
                resolved_config=resolved_config,
                foreground=args.foreground,
            )
        else:
            # SLURM execution
            logger.info("Routing to SLURM execution...")
            # Validate config synchronization (SLURM only)
            validate_config_sync_before_submission(
                args.config, workflow_section, force=args.force
            )
            # Submit the master job
            submit_slurm_workflow(
                dataset_config_name=args.config,
                workflow_config=workflow_section,
                resolved_config=resolved_config,
                force=args.force,
            )

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
