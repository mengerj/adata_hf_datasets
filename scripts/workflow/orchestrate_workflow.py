#!/usr/bin/env python3
"""
Workflow Orchestrator for Multi-Server SLURM Pipeline

This script orchestrates the execution of dataset processing steps across different
SSH servers (CPU and GPU clusters) that share a filesystem. It manages job dependencies
and provides both SLURM and local execution modes.

Usage:
    python orchestrate_workflow.py --config-name workflow_orchestrator dataset_config_name=dataset_cellxgene_pseudo_bulk_3_5k
"""

import logging
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import re
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from adata_hf_datasets.config_utils import (
    apply_all_transformations,
    ensure_config_sync,
)

logger = logging.getLogger(__name__)


class WorkflowLogger:
    """Manages comprehensive logging for the entire workflow."""

    def __init__(self, base_dir: Path, master_job_id: str, dataset_config_name: str):
        """
        Initialize the workflow logger.

        Parameters
        ----------
        base_dir : Path
            Base directory for all outputs
        master_job_id : str
            The master SLURM job ID
        dataset_config_name : str
            Name of the dataset config being used
        """
        self.base_dir = base_dir
        self.master_job_id = master_job_id
        self.dataset_config_name = dataset_config_name

        # Create the workflow directory structure
        self.workflow_dir = self._create_workflow_directory()

        # Set up logging
        self._setup_logging()

        # Copy the dataset config
        self._copy_dataset_config()

    def _create_workflow_directory(self) -> Path:
        """Create the unified workflow directory structure."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        workflow_dir = self.base_dir / date_str / f"workflow_{self.master_job_id}"

        # Create directory structure
        (workflow_dir / "config").mkdir(parents=True, exist_ok=True)
        (workflow_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Create step directories
        for step in [
            "download",
            "preprocessing",
            "embedding_preparation",
            "embedding",
            "dataset_creation",
        ]:
            (workflow_dir / step).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created workflow directory: {workflow_dir}")
        return workflow_dir

    def _setup_logging(self):
        """Set up comprehensive logging."""
        # Create a custom formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Set up file handlers
        log_dir = self.workflow_dir / "logs"

        # Main workflow log
        workflow_handler = logging.FileHandler(log_dir / "workflow_summary.log")
        workflow_handler.setFormatter(formatter)
        workflow_handler.setLevel(logging.INFO)

        # Error consolidation log
        error_handler = logging.FileHandler(log_dir / "errors_consolidated.log")
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)

        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(workflow_handler)
        root_logger.addHandler(error_handler)

        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        logger.info(f"Logging setup complete. Logs in: {log_dir}")

    def _copy_dataset_config(self):
        """Copy the dataset config to the workflow directory."""
        config_path = Path(__file__).parent.parent.parent / "conf"
        config_file = config_path / f"{self.dataset_config_name}.yaml"

        if config_file.exists():
            dest_file = (
                self.workflow_dir / "config" / f"{self.dataset_config_name}.yaml"
            )
            shutil.copy2(config_file, dest_file)
            logger.info(f"Copied dataset config to: {dest_file}")
        else:
            logger.warning(f"Dataset config file not found: {config_file}")

    def get_step_log_dir(self, step_name: str, job_id: str) -> Path:
        """Get the log directory for a specific step and job."""
        return self.workflow_dir / step_name / f"job_{job_id}"

    def log_workflow_start(self, dataset_config: DictConfig):
        """Log workflow start information."""
        logger.info("=" * 80)
        logger.info("WORKFLOW STARTED")
        logger.info("=" * 80)
        logger.info(f"Master Job ID: {self.master_job_id}")
        logger.info(f"Dataset Config: {self.dataset_config_name}")
        logger.info(f"Dataset Name: {dataset_config.dataset.name}")
        logger.info(f"Workflow Directory: {self.workflow_dir}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 80)

    def log_workflow_complete(self):
        """Log workflow completion."""
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Final logs available in: {self.workflow_dir}")
        logger.info("=" * 80)

    def log_step_start(self, step_name: str, job_id: str, host: str):
        """Log the start of a workflow step."""
        logger.info(f"Starting {step_name} step (Job ID: {job_id}, Host: {host})")

    def log_step_complete(self, step_name: str, job_id: str):
        """Log the completion of a workflow step."""
        logger.info(f"✓ {step_name} step completed (Job ID: {job_id})")

    def log_step_skipped(self, step_name: str, reason: str):
        """Log when a step is skipped."""
        logger.info(f"⏭ {step_name} step skipped: {reason}")


class WorkflowOrchestrator:
    """
    Orchestrates workflow execution across multiple SSH servers.
    """

    def __init__(
        self,
        cpu_login: Optional[Tuple[str, str]] = None,
        gpu_login: Optional[Tuple[str, str]] = None,
    ):
        """
        Initialize the orchestrator.

        Parameters
        ----------
        cpu_login : Optional[Tuple[str, str]]
            (host, user) for CPU cluster SSH connection
        gpu_login : Optional[Tuple[str, str]]
            (host, user) for GPU cluster SSH connection
        """
        self.cpu_login = cpu_login
        self.gpu_login = gpu_login
        self.workflow_logger = None

        # Validate that SSH command is available
        try:
            subprocess.run(["ssh", "-V"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("SSH command not available")

    def validate_config_sync(
        self, dataset_config_name: str, force: bool = False
    ) -> None:
        """Validate that the remote config matches the local one."""
        if not force:
            logger.info(
                f"Validating config synchronization for {dataset_config_name}..."
            )

            ensure_config_sync(
                config_name=dataset_config_name,
                remote_host=self.cpu_login["host"],
                remote_project_dir="/home/menger/git/adata_hf_datasets",
                force=force,
            )

    def _submit_slurm_job(
        self,
        host: str,
        script_path: Path,
        partition: str = "cpu",
        dependencies: Optional[List[int]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        step_name: str = "unknown",
    ) -> int:
        """Submit a SLURM job using ssh command."""
        project_dir = "/home/menger/git/adata_hf_datasets"

        # Build the sbatch command
        cmd = ["ssh", host, f"cd {project_dir} && sbatch"]

        # Add partition
        cmd.extend(["--partition", partition])

        # Add dependencies if specified
        if dependencies:
            deps_str = ":".join(map(str, dependencies))
            cmd.extend(["--dependency", f"afterok:{deps_str}"])

        # Add environment variables if specified
        if env_vars:
            env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
            cmd.extend(["--export", f"ALL,{env_str}"])

        # Add the script path
        cmd.append(str(script_path))

        logger.info(f"Submitting {script_path.name} ➜ {' '.join(cmd)}")

        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to submit job: {result.stderr}")

        # Parse job ID from output
        output = result.stdout.strip()
        job_id_match = re.search(r"Submitted batch job (\d+)", output)
        if not job_id_match:
            raise RuntimeError(f"Could not parse job ID from output: {output}")

        job_id = int(job_id_match.group(1))

        # Log the job submission
        if self.workflow_logger:
            self.workflow_logger.log_step_start(step_name, job_id, host)

        return job_id

    def run_download_step(
        self, dataset_config_name: str, workflow_config: DictConfig
    ) -> Optional[int]:
        """Run the download step and return job ID."""
        logger.info("=== Starting Download Step ===")
        script_path = Path("scripts/download/run_download_ds.slurm")

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Pass the dataset config name and workflow directory as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for download
            script_path,
            partition=workflow_config.cpu_partition,  # Use partition from config
            env_vars=env_vars,
            step_name="Download",
        )
        return job_id

    def run_preprocessing_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the preprocessing step and return job ID."""
        logger.info("=== Starting Preprocessing Step ===")
        script_path = Path("scripts/preprocessing/run_preprocess.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Pass the dataset config name and workflow directory as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for preprocessing
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Preprocessing",
        )
        return job_id

    def run_embedding_prepare_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the embedding preparation step on CPU and return job ID."""
        logger.info("=== Starting Embedding Preparation Step ===")
        script_path = Path("scripts/embed/run_embed_parallel.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Pass the dataset config name, workflow directory, and prepare_only flag as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "PREPARE_ONLY": "true",  # Force prepare_only mode
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for preparation
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Embedding Preparation",
        )
        return job_id

    def run_embedding_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the embedding step and return job ID."""
        logger.info("=== Starting Embedding Step ===")
        script_path = Path("scripts/embed/run_embed_parallel.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Pass the dataset config name and workflow directory as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
        }

        job_id = self._submit_slurm_job(
            self.gpu_login["host"],  # Use GPU cluster for embedding
            script_path,
            partition=workflow_config.gpu_partition,  # Use GPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Embedding",
        )
        return job_id

    def run_dataset_creation_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the dataset creation step and return job ID."""
        logger.info("=== Starting Dataset Creation Step ===")
        script_path = Path("scripts/dataset_creation/run_create_ds.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Pass the dataset config name and workflow directory as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for dataset creation
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Dataset Creation",
        )
        return job_id

    def run_workflow(
        self, dataset_config_name: str, workflow_config: DictConfig, force: bool = False
    ) -> None:
        """Run the complete workflow."""
        logger.info(f"Starting workflow for dataset config: {dataset_config_name}")

        # Validate config synchronization unless forced
        self.validate_config_sync(dataset_config_name, force=force)

        # Load the dataset config to check enabled flags
        # We'll load it here to get the configuration for each step
        dataset_config = self._load_dataset_config(dataset_config_name)
        logger.info(f"Dataset name: {dataset_config.dataset.name}")

        # Step 1: Download (if enabled)
        download_job_id = None
        download_enabled = getattr(dataset_config.download, "enabled", True)
        if download_enabled:
            download_job_id = self.run_download_step(
                dataset_config_name, workflow_config
            )
            logger.info(
                f"✓ Download job {download_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        # Step 2: Preprocessing (depends on download if download was enabled)
        preprocessing_job_id = None
        preprocessing_enabled = getattr(dataset_config.preprocessing, "enabled", True)
        if preprocessing_enabled:
            preprocessing_job_id = self.run_preprocessing_step(
                dataset_config_name, workflow_config, dependency_job_id=download_job_id
            )
            logger.info(
                f"✓ Preprocessing job {preprocessing_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        # Step 3: Embedding Preparation (depends on preprocessing)
        embedding_prepare_job_id = None
        embedding_prepare_enabled = getattr(dataset_config.embedding, "enabled", True)
        if embedding_prepare_enabled:
            embedding_prepare_job_id = self.run_embedding_prepare_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=preprocessing_job_id,
            )
            logger.info(
                f"✓ Embedding preparation job {embedding_prepare_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        # Step 4: Embedding (depends on embedding preparation)
        embedding_job_id = None
        embedding_enabled = getattr(dataset_config.embedding, "enabled", True)
        if embedding_enabled:
            embedding_job_id = self.run_embedding_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=embedding_prepare_job_id,
            )
            logger.info(
                f"✓ Embedding job {embedding_job_id} submitted to cluster ({self.gpu_login['host']})"
            )

        # Step 5: Dataset Creation (depends on embedding)
        dataset_creation_enabled = getattr(
            dataset_config.dataset_creation, "enabled", True
        )
        if dataset_creation_enabled:
            dataset_job_id = self.run_dataset_creation_step(
                dataset_config_name, workflow_config, dependency_job_id=embedding_job_id
            )
            logger.info(
                f"✓ Dataset creation job {dataset_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        logger.info("=== Workflow Complete ===")
        logger.info("All jobs have been submitted to the clusters.")
        logger.info("Jobs will run with dependencies managed by SLURM.")
        logger.info(
            "You can monitor progress using 'squeue' on the respective clusters."
        )

    def run_workflow_local(
        self, dataset_config_name: str, workflow_config: DictConfig, force: bool = False
    ) -> None:
        """Run the complete workflow locally on the cluster, waiting for each step to complete."""
        # Initialize the workflow logger
        master_job_id = os.environ.get(
            "SLURM_JOB_ID", f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        base_dir = Path("/home/menger/git/adata_hf_datasets/outputs")
        self.workflow_logger = WorkflowLogger(
            base_dir, master_job_id, dataset_config_name
        )

        logger.info(
            f"Starting local workflow for dataset config: {dataset_config_name}"
        )

        # Validate config synchronization unless forced
        self.validate_config_sync(dataset_config_name, force=force)

        # Load the dataset config to check enabled flags
        dataset_config = self._load_dataset_config(dataset_config_name)

        # Log workflow start
        self.workflow_logger.log_workflow_start(dataset_config)

        logger.info(f"Dataset name: {dataset_config.dataset.name}")

        # Step 1: Download (if enabled)
        download_job_id = None
        download_enabled = getattr(dataset_config.download, "enabled", True)
        if download_enabled:
            logger.info("=== Starting Download Step ===")
            download_job_id = self.run_download_step(
                dataset_config_name, workflow_config
            )
            logger.info(
                f"✓ Download job {download_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

            # Wait for download job to complete
            self._wait_for_job_completion(
                self.cpu_login["host"], download_job_id, "Download"
            )
            self.workflow_logger.log_step_complete("Download", download_job_id)
        else:
            logger.info("=== Download Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped("Download", "disabled in config")

        # Step 2: Preprocessing (depends on download if download was enabled)
        preprocessing_job_id = None
        preprocessing_enabled = getattr(dataset_config.preprocessing, "enabled", True)
        if preprocessing_enabled:
            logger.info("=== Starting Preprocessing Step ===")
            preprocessing_job_id = self.run_preprocessing_step(
                dataset_config_name, workflow_config, dependency_job_id=download_job_id
            )
            logger.info(
                f"✓ Preprocessing job {preprocessing_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

            # Wait for preprocessing job to complete
            self._wait_for_job_completion(
                self.cpu_login["host"], preprocessing_job_id, "Preprocessing"
            )
            self.workflow_logger.log_step_complete(
                "Preprocessing", preprocessing_job_id
            )
        else:
            logger.info("=== Preprocessing Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped("Preprocessing", "disabled in config")

        # Step 3: Embedding Preparation (depends on preprocessing)
        embedding_prepare_job_id = None
        embedding_prepare_enabled = getattr(dataset_config.embedding, "enabled", True)
        if embedding_prepare_enabled:
            logger.info("=== Starting Embedding Preparation Step ===")
            embedding_prepare_job_id = self.run_embedding_prepare_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=preprocessing_job_id,
            )
            logger.info(
                f"✓ Embedding preparation job {embedding_prepare_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

            # Wait for embedding preparation job to complete
            self._wait_for_job_completion(
                self.cpu_login["host"],
                embedding_prepare_job_id,
                "Embedding Preparation",
            )
            self.workflow_logger.log_step_complete(
                "Embedding Preparation", embedding_prepare_job_id
            )
        else:
            logger.info("=== Embedding Preparation Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped(
                "Embedding Preparation", "disabled in config"
            )

        # Step 4: Embedding (depends on embedding preparation)
        embedding_job_id = None
        embedding_enabled = getattr(dataset_config.embedding, "enabled", True)
        if embedding_enabled:
            logger.info("=== Starting Embedding Step ===")
            embedding_job_id = self.run_embedding_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=embedding_prepare_job_id,
            )
            logger.info(
                f"✓ Embedding job {embedding_job_id} submitted to cluster ({self.gpu_login['host']})"
            )

            # Wait for embedding job to complete
            self._wait_for_job_completion(
                self.gpu_login["host"], embedding_job_id, "Embedding"
            )
            self.workflow_logger.log_step_complete("Embedding", embedding_job_id)
        else:
            logger.info("=== Embedding Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped("Embedding", "disabled in config")

        # Step 5: Dataset Creation (depends on embedding)
        dataset_creation_enabled = getattr(
            dataset_config.dataset_creation, "enabled", True
        )
        if dataset_creation_enabled:
            logger.info("=== Starting Dataset Creation Step ===")
            dataset_job_id = self.run_dataset_creation_step(
                dataset_config_name, workflow_config, dependency_job_id=embedding_job_id
            )
            logger.info(
                f"✓ Dataset creation job {dataset_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

            # Wait for dataset creation job to complete
            self._wait_for_job_completion(
                self.cpu_login["host"], dataset_job_id, "Dataset Creation"
            )
            self.workflow_logger.log_step_complete("Dataset Creation", dataset_job_id)
        else:
            logger.info("=== Dataset Creation Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped(
                "Dataset Creation", "disabled in config"
            )

        # Log workflow completion
        self.workflow_logger.log_workflow_complete()

        logger.info("=== Workflow Complete ===")
        logger.info("All steps have been completed successfully.")

    def _wait_for_job_completion(self, host: str, job_id: int, step_name: str) -> None:
        """Wait for a SLURM job to complete and check for errors."""
        logger.info(f"Waiting for {step_name} job {job_id} to complete...")

        while True:
            # Check job status
            cmd = ["ssh", host, f"squeue -j {job_id} --noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0 or not result.stdout.strip():
                # Job is no longer in queue (completed, failed, or cancelled)
                # Check the exit status
                exit_cmd = [
                    "ssh",
                    host,
                    f"sacct -j {job_id} --format=JobID,State,ExitCode --noheader",
                ]
                exit_result = subprocess.run(
                    exit_cmd, capture_output=True, text=True, timeout=30
                )

                if exit_result.returncode == 0 and exit_result.stdout.strip():
                    # Parse the job state
                    lines = exit_result.stdout.strip().split("\n")
                    for line in lines:
                        if str(job_id) in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                state = parts[1]
                                if state in ["COMPLETED", "COMPLETED+"]:
                                    logger.info(
                                        f"✓ {step_name} job {job_id} completed successfully"
                                    )
                                    return
                                elif state in ["FAILED", "CANCELLED", "TIMEOUT"]:
                                    error_msg = f"✗ {step_name} job {job_id} failed with state: {state}"
                                    logger.error(error_msg)
                                    # Log to consolidated error log
                                    if self.workflow_logger:
                                        error_log_path = (
                                            self.workflow_logger.workflow_dir
                                            / "logs"
                                            / "errors_consolidated.log"
                                        )
                                        with open(error_log_path, "a") as f:
                                            f.write(
                                                f"{datetime.now().isoformat()} - {error_msg}\n"
                                            )
                                    raise RuntimeError(error_msg)
                                else:
                                    logger.warning(
                                        f"? {step_name} job {job_id} ended with unknown state: {state}"
                                    )
                                    return

                # If we can't get detailed status, assume it completed
                logger.info(f"✓ {step_name} job {job_id} completed")
                break

            # Job is still running, wait a bit
            logger.info(f"  {step_name} job {job_id} still running...")
            import time

            time.sleep(60)  # Wait 1 minute before checking again

    def _load_dataset_config(self, dataset_config_name: str) -> DictConfig:
        """Load the dataset configuration."""
        # Load the dataset config using Hydra's config store
        # This is a simplified approach - in practice, you might want to use
        # Hydra's config store or load the config file directly
        config_path = Path(__file__).parent.parent.parent / "conf"
        config_file = config_path / f"{dataset_config_name}.yaml"

        if not config_file.exists():
            raise ValueError(f"Dataset config file not found: {config_file}")

        # Load the config using OmegaConf
        config = OmegaConf.load(config_file)

        # Apply transformations
        config = apply_all_transformations(config)

        return config


def create_orchestrator_from_config(config: DictConfig) -> WorkflowOrchestrator:
    workflow_config = config.get("workflow", {})
    cpu_login = workflow_config.get("cpu_login")
    gpu_login = workflow_config.get("gpu_login")
    if not cpu_login:
        raise ValueError("CPU login configuration required for SSH mode")
    logger.info("Running in SSH/SLURM mode")
    return WorkflowOrchestrator(
        cpu_login=cpu_login,
        gpu_login=gpu_login,
    )


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="workflow_orchestrator",  # Use workflow_orchestrator as the base config
)
def main(cfg: DictConfig):
    logger.info("Validating workflow config...")
    # Don't apply transformations to the workflow config - it doesn't have dataset fields
    # validate_config(cfg)  # This would fail since workflow config doesn't have dataset section

    # Create orchestrator from workflow config
    orchestrator = create_orchestrator_from_config(cfg)

    # Get the dataset config name from the config
    dataset_config_name = cfg.get("dataset_config_name", "dataset_test_workflow")
    logger.info(f"Using dataset config: {dataset_config_name}")

    # Extract workflow config from the loaded config
    workflow_config = cfg.workflow

    # Check for force flag
    force = getattr(cfg, "force", False)
    if force:
        logger.warning("Running with --force flag (skipping config sync validation)")

    orchestrator.run_workflow(dataset_config_name, workflow_config, force=force)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        main()
    except Exception:
        logger.exception("Workflow orchestration failed")
        sys.exit(1)
