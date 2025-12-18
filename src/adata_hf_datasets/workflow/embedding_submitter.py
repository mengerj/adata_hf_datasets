#!/usr/bin/env python3
"""
Embedding Array Job Submitter

This module provides direct submission of SLURM array jobs for embedding steps,
bypassing the master job wrapper pattern. The orchestrator can directly track
the array jobs and wait for their completion.
"""

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import DictConfig

from .data_transfer import LocationConfig

logger = logging.getLogger(__name__)


@dataclass
class ArrayJobInfo:
    """Information about a submitted array job."""

    job_id: str
    label: str  # e.g., "train", "val", "test"
    input_dir: str
    task_count: int
    mode: str  # "cpu" or "gpu"
    prepare_only: bool


class EmbeddingArraySubmitter:
    """
    Submits SLURM array jobs for embedding steps directly via SSH.

    This class handles:
    - Determining input directories from config
    - Counting zarr files per directory
    - Generating SLURM array job scripts
    - Submitting array jobs directly via SSH
    - Returning job IDs for tracking
    """

    # Path to the array task SLURM script (relative to project directory)
    ARRAY_SCRIPT_PATH = "scripts/embed/embed_chunks_parallel.slurm"

    def __init__(
        self,
        location_config: LocationConfig,
        dataset_config: DictConfig,
        workflow_dir: Path,
        dataset_config_name: str,
    ):
        """
        Initialize the embedding array submitter.

        Parameters
        ----------
        location_config : LocationConfig
            Configuration for the execution location (cpu or gpu cluster)
        dataset_config : DictConfig
            The resolved dataset configuration
        workflow_dir : Path
            Local workflow directory for logs
        dataset_config_name : str
            Name of the dataset config (for passing to array tasks)
        """
        self.location_config = location_config
        self.dataset_config = dataset_config
        self.workflow_dir = workflow_dir
        self.dataset_config_name = dataset_config_name

        # Build SSH target
        if location_config.is_remote:
            self.ssh_target = location_config.ssh_target
        else:
            self.ssh_target = None

    def _run_ssh_command(
        self, command: str, timeout: int = 300
    ) -> Tuple[int, str, str]:
        """Run a command on the remote host via SSH."""
        if not self.ssh_target:
            # Local execution
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                return result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return -1, "", f"Command timed out after {timeout}s"

        ssh_cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "BatchMode=yes",
            self.ssh_target,
            command,
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"SSH command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    def get_input_directories(
        self, mode: str, prepare_only: bool = False
    ) -> List[Tuple[str, str, int]]:
        """
        Get list of input directories to process with file counts.

        Parameters
        ----------
        mode : str
            Processing mode: "cpu" or "gpu"
        prepare_only : bool
            Whether this is a prepare-only run

        Returns
        -------
        List[Tuple[str, str, int]]
            List of (label, directory_path, file_count) tuples
        """
        base_file_path = self.location_config.base_file_path

        # Determine input subdirectory based on config and mode
        cpu_embedding_enabled = (
            hasattr(self.dataset_config, "embedding_cpu")
            and self.dataset_config.embedding_cpu is not None
            and self.dataset_config.embedding_cpu.get("enabled", True)
        )

        if mode == "gpu" and not prepare_only and cpu_embedding_enabled:
            input_subdir = "processed_with_emb"
            logger.info(
                "GPU mode with CPU embedding enabled: looking for input in processed_with_emb/"
            )
        else:
            input_subdir = "processed"
            logger.info("Looking for input in processed/")

        base_dir = f"{base_file_path}/{input_subdir}"
        dataset_name = self.dataset_config.dataset.name

        # Determine if we're processing train or test data
        split_dataset = self.dataset_config.preprocessing.get("split_dataset", True)
        train_or_test = "train" if split_dataset else "test"

        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Split dataset: {split_dataset} -> Looking for: {train_or_test}")
        logger.info(f"Base directory: {base_dir}")

        directories = []

        if train_or_test == "test":
            test_dir = f"{base_dir}/test/{dataset_name}/all"
            file_count = self._count_zarr_files(test_dir)
            if file_count > 0:
                directories.append(("test", test_dir, file_count))
                logger.info(f"Found test directory with {file_count} files: {test_dir}")
        else:
            # Process train and validation directories
            train_dir = f"{base_dir}/train/{dataset_name}/train"
            val_dir = f"{base_dir}/train/{dataset_name}/val"

            train_count = self._count_zarr_files(train_dir)
            if train_count > 0:
                directories.append(("train", train_dir, train_count))
                logger.info(
                    f"Found train directory with {train_count} files: {train_dir}"
                )

            val_count = self._count_zarr_files(val_dir)
            if val_count > 0:
                directories.append(("val", val_dir, val_count))
                logger.info(
                    f"Found validation directory with {val_count} files: {val_dir}"
                )

        if not directories:
            raise ValueError(
                f"No input directories found for dataset {dataset_name} "
                f"in {base_dir}/{train_or_test}"
            )

        return directories

    def _count_zarr_files(self, directory: str) -> int:
        """Count the number of .zarr files in a directory on the remote."""
        cmd = f"ls -1d '{directory}'/*.zarr 2>/dev/null | wc -l"
        code, stdout, stderr = self._run_ssh_command(cmd, timeout=60)

        if code != 0:
            logger.debug(f"Could not list zarr files in {directory}: {stderr}")
            return 0

        try:
            return int(stdout.strip())
        except ValueError:
            return 0

    def _get_embedding_methods(self, mode: str, prepare_only: bool) -> List[str]:
        """Get the list of embedding methods to run based on mode and config."""
        if prepare_only:
            # For preparation, use embedding_preparation config if available
            if (
                hasattr(self.dataset_config, "embedding_preparation")
                and self.dataset_config.embedding_preparation is not None
            ):
                return list(
                    self.dataset_config.embedding_preparation.get("methods", [])
                )
            elif (
                hasattr(self.dataset_config, "embedding_cpu")
                and self.dataset_config.embedding_cpu is not None
            ):
                return list(self.dataset_config.embedding_cpu.get("methods", []))

        if mode == "cpu":
            if (
                hasattr(self.dataset_config, "embedding_cpu")
                and self.dataset_config.embedding_cpu is not None
            ):
                return list(self.dataset_config.embedding_cpu.get("methods", []))

        if mode == "gpu":
            if (
                hasattr(self.dataset_config, "embedding_gpu")
                and self.dataset_config.embedding_gpu is not None
            ):
                return list(self.dataset_config.embedding_gpu.get("methods", []))

        return ["pca", "hvg"]  # Default fallback

    def _get_embedding_config_section(self, mode: str, prepare_only: bool) -> str:
        """Get the embedding config section name to use."""
        if prepare_only:
            if (
                hasattr(self.dataset_config, "embedding_preparation")
                and self.dataset_config.embedding_preparation is not None
            ):
                return "embedding_preparation"
            return "embedding_cpu"

        if mode == "cpu":
            return "embedding_cpu"
        elif mode == "gpu":
            return "embedding_gpu"
        return "embedding"

    def submit_array_jobs(
        self,
        mode: str,
        prepare_only: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> List[ArrayJobInfo]:
        """
        Submit SLURM array jobs directly via SSH.

        Parameters
        ----------
        mode : str
            Processing mode: "cpu" or "gpu"
        prepare_only : bool
            Whether to run only the prepare step
        env : Dict[str, str], optional
            Additional environment variables to pass to jobs

        Returns
        -------
        List[ArrayJobInfo]
            Information about submitted jobs for tracking
        """
        directories = self.get_input_directories(mode, prepare_only)

        if not directories:
            logger.warning("No input directories found, no jobs to submit")
            return []

        methods = self._get_embedding_methods(mode, prepare_only)
        config_section = self._get_embedding_config_section(mode, prepare_only)

        logger.info(f"Embedding methods: {methods}")
        logger.info(f"Config section: {config_section}")

        submitted_jobs = []

        for i, (label, input_dir, file_count) in enumerate(directories):
            # Add staggered delay for GPU jobs to prevent resource conflicts
            if i > 0 and mode == "gpu":
                delay_seconds = 30
                logger.info(
                    f"Adding {delay_seconds}s delay before submitting {label} job"
                )
                time.sleep(delay_seconds)

            job_info = self._submit_single_array_job(
                label=label,
                input_dir=input_dir,
                file_count=file_count,
                mode=mode,
                prepare_only=prepare_only,
                methods=methods,
                config_section=config_section,
                env=env,
            )

            if job_info:
                submitted_jobs.append(job_info)
                logger.info(
                    f"Submitted array job {job_info.job_id} for {label} "
                    f"({file_count} tasks)"
                )

        return submitted_jobs

    def _submit_single_array_job(
        self,
        label: str,
        input_dir: str,
        file_count: int,
        mode: str,
        prepare_only: bool,
        methods: List[str],
        config_section: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[ArrayJobInfo]:
        """Submit a single SLURM array job."""
        if file_count == 0:
            logger.warning(f"No files in {input_dir}, skipping {label}")
            return None

        # Determine partition and resources based on mode
        if mode == "gpu":
            partition = self.location_config.slurm_partition or "gpu"
            gres = "--gres=gpu:1"
            mem = "64G"
        else:
            partition = self.location_config.slurm_partition or "slurm"
            gres = ""
            mem = "60G"

        # Build job name
        job_name = f"embed_{label}"
        if prepare_only:
            job_name = f"prep_{label}"

        # Build environment variables string
        methods_str = " ".join(methods)
        batch_key = self.dataset_config.get("batch_key", "dataset_title")
        batch_size = 128
        if hasattr(self.dataset_config, config_section):
            section = getattr(self.dataset_config, config_section)
            if section:
                batch_size = section.get("batch_size", 128)

        # Remote workflow directory
        remote_workflow_dir = env.get("WORKFLOW_DIR", "") if env else ""

        env_vars = {
            "INPUT_DIR": input_dir,
            "METHODS": methods_str,
            "BATCH_KEY": batch_key,
            "BATCH_SIZE": str(batch_size),
            "PREPARE_ONLY": "true" if prepare_only else "false",
            "MODE": mode,
            "DATASET_CONFIG": self.dataset_config_name,
            "WORKFLOW_DIR": remote_workflow_dir,
            "EMBEDDING_CONFIG_SECTION": config_section,
        }

        # Add any additional environment variables
        if env:
            for key in ["BASE_FILE_PATH", "PROJECT_DIR", "VENV_PATH"]:
                if key in env:
                    env_vars[key] = env[key]

        # Build export string
        export_str = ",".join([f"{k}={v}" for k, v in env_vars.items() if v])

        # Build sbatch command
        project_dir = self.location_config.project_directory
        script_path = f"{project_dir}/{self.ARRAY_SCRIPT_PATH}"

        sbatch_cmd = (
            f"cd {project_dir} && "
            f"sbatch "
            f"--job-name={job_name} "
            f"--array=0-{file_count - 1}%{min(file_count, 10)} "
            f"--partition={partition} "
            f"--mem={mem} "
            f"--time=6:00:00 "
        )

        if gres:
            sbatch_cmd += f"{gres} "

        # Add node constraint if specified
        if self.location_config.node:
            sbatch_cmd += f"--nodelist={self.location_config.node} "

        sbatch_cmd += f"--export=ALL,{export_str} {script_path}"

        logger.info(f"Submitting array job: {sbatch_cmd[:200]}...")

        code, stdout, stderr = self._run_ssh_command(sbatch_cmd, timeout=120)

        if code != 0:
            logger.error(f"Failed to submit array job for {label}: {stderr}")
            return None

        # Parse job ID from output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", stdout)
        if not match:
            logger.error(f"Could not parse job ID from sbatch output: {stdout}")
            return None

        job_id = match.group(1)

        return ArrayJobInfo(
            job_id=job_id,
            label=label,
            input_dir=input_dir,
            task_count=file_count,
            mode=mode,
            prepare_only=prepare_only,
        )
