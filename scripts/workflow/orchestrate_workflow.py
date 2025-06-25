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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import re

import hydra
from omegaconf import DictConfig, OmegaConf

from adata_hf_datasets.config_utils import (
    apply_all_transformations,
    ensure_config_sync,
)

logger = logging.getLogger(__name__)


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
        return job_id

    def run_download_step(
        self, dataset_config_name: str, workflow_config: DictConfig
    ) -> Optional[int]:
        """Run the download step and return job ID."""
        logger.info("=== Starting Download Step ===")
        script_path = Path("scripts/download/run_download_ds.slurm")

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Pass the dataset config name as environment variable
        env_vars = {"DATASET_CONFIG": dataset_config_name}

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use hostname from dict
            script_path,
            partition=workflow_config.cpu_partition,  # Use partition from config
            env_vars=env_vars,
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

        # Pass the dataset config name as environment variable
        env_vars = {"DATASET_CONFIG": dataset_config_name}

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use hostname from dict
            script_path,
            partition=workflow_config.cpu_partition,  # Use partition from config
            dependencies=dependencies,
            env_vars=env_vars,
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

        # Pass the dataset config name as environment variable
        env_vars = {"DATASET_CONFIG": dataset_config_name}

        job_id = self._submit_slurm_job(
            self.gpu_login["host"],  # Use GPU cluster for embedding
            script_path,
            partition=workflow_config.gpu_partition,  # Use GPU partition
            dependencies=dependencies,
            env_vars=env_vars,
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

        # Pass the dataset config name as environment variable
        env_vars = {"DATASET_CONFIG": dataset_config_name}

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for dataset creation
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
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

        # Step 3: Embedding (depends on preprocessing)
        embedding_job_id = None
        embedding_enabled = getattr(dataset_config.embedding, "enabled", True)
        if embedding_enabled:
            embedding_job_id = self.run_embedding_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=preprocessing_job_id,
            )
            logger.info(
                f"✓ Embedding job {embedding_job_id} submitted to cluster ({self.gpu_login['host']})"
            )

        # Step 4: Dataset Creation (depends on embedding)
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
