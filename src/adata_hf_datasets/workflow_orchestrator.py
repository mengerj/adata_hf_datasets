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
import signal
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

        # Initialize timing tracking
        self.workflow_start_time = datetime.now()
        self.step_timings = {}  # Store start/end times for each step
        self.step_durations = {}  # Store calculated durations

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
            "embedding_prepare",
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
        # First log the timing summary
        self.log_workflow_timing_summary()

        # Then log completion message
        logger.info("=" * 80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Final logs available in: {self.workflow_dir}")
        logger.info("=" * 80)

    def log_step_start(self, step_name: str, job_id: str, host: str):
        """Log the start of a workflow step."""
        start_time = datetime.now()
        self.step_timings[step_name] = {
            "start_time": start_time,
            "job_id": job_id,
            "host": host,
        }
        logger.info(
            f"Starting {step_name} step (Job ID: {job_id}, Host: {host}) at {start_time.strftime('%H:%M:%S')}"
        )

    def log_step_complete(self, step_name: str, job_id: str):
        """Log the completion of a workflow step."""
        end_time = datetime.now()
        if step_name in self.step_timings:
            self.step_timings[step_name]["end_time"] = end_time
            duration = end_time - self.step_timings[step_name]["start_time"]
            self.step_durations[step_name] = duration

            duration_str = self._format_duration(duration)
            logger.info(
                f"✓ {step_name} step completed (Job ID: {job_id}) - Duration: {duration_str}"
            )
        else:
            logger.info(f"✓ {step_name} step completed (Job ID: {job_id})")

    def log_step_skipped(self, step_name: str, reason: str):
        """Log when a step is skipped."""
        logger.info(f"⏭ {step_name} step skipped: {reason}")

    def _format_duration(self, duration):
        """Format a duration timedelta into a readable string."""
        total_seconds = int(duration.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def log_workflow_timing_summary(self):
        """Log a comprehensive timing summary of the entire workflow."""
        workflow_end_time = datetime.now()
        total_workflow_duration = workflow_end_time - self.workflow_start_time

        logger.info("=" * 80)
        logger.info("WORKFLOW TIMING SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Workflow started: {self.workflow_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(
            f"Workflow ended: {workflow_end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(
            f"Total workflow duration: {self._format_duration(total_workflow_duration)}"
        )
        logger.info("")

        if self.step_durations:
            logger.info("Step-by-step timing breakdown:")
            logger.info("-" * 50)

            total_step_time = sum(self.step_durations.values(), datetime.timedelta())

            for step_name, duration in self.step_durations.items():
                percentage = (
                    duration.total_seconds() / total_step_time.total_seconds()
                ) * 100
                start_time = self.step_timings[step_name]["start_time"]
                end_time = self.step_timings[step_name]["end_time"]
                host = self.step_timings[step_name]["host"]

                logger.info(
                    f"{step_name:20} | {self._format_duration(duration):>10} | {percentage:5.1f}% | {host}"
                )
                logger.info(
                    f"{'':20} | {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')}"
                )
                logger.info("")

            logger.info("-" * 50)
            logger.info(
                f"{'Total step time':20} | {self._format_duration(total_step_time):>10} | 100.0%"
            )

            # Calculate overhead (time not spent in active processing)
            overhead = total_workflow_duration - total_step_time
            if overhead.total_seconds() > 0:
                overhead_percentage = (
                    overhead.total_seconds() / total_workflow_duration.total_seconds()
                ) * 100
                logger.info(
                    f"{'Workflow overhead':20} | {self._format_duration(overhead):>10} | {overhead_percentage:5.1f}%"
                )

        logger.info("=" * 80)


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
        # Validate SSH connection parameters
        if cpu_login is None:
            raise ValueError("CPU login configuration is required")

        if not isinstance(cpu_login, dict):
            raise ValueError(
                "CPU login must be a dictionary with 'host' and 'user' keys"
            )

        if "host" not in cpu_login or "user" not in cpu_login:
            raise ValueError("CPU login must contain 'host' and 'user' keys")

        if gpu_login is not None:
            if not isinstance(gpu_login, dict):
                raise ValueError(
                    "GPU login must be a dictionary with 'host' and 'user' keys"
                )

            if "host" not in gpu_login or "user" not in gpu_login:
                raise ValueError("GPU login must contain 'host' and 'user' keys")

        self.cpu_login = cpu_login
        self.gpu_login = gpu_login
        self.workflow_logger = None
        self.submitted_jobs = []  # Track jobs for cancellation

        # Validate that SSH command is available
        try:
            subprocess.run(["ssh", "-V"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "SSH command not available. Please ensure SSH is installed and available in PATH. "
                "On Windows, you may need to install OpenSSH or use WSL."
            )

        # Test basic SSH connectivity to CPU host
        try:
            test_cmd = [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                f"{cpu_login['user']}@{cpu_login['host']}",
                "echo 'SSH test successful'",
            ]
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 seconds for initial SSH test
            )
            if result.returncode != 0:
                logger.warning(
                    f"SSH connection test to CPU host {cpu_login['host']} failed: {result.stderr}"
                )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.warning(
                f"SSH connection test to CPU host {cpu_login['host']} failed: {e}"
            )
        except Exception as e:
            logger.warning(f"Unexpected error during SSH connection test: {e}")

    def validate_config_sync(
        self, dataset_config_name: str, force: bool = False
    ) -> None:
        """Validate that the remote config matches the local one."""
        logger.info(f"Validating config synchronization for {dataset_config_name}...")

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
        partition: str = "slurm",
        dependencies: Optional[List[int]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        step_name: str = "unknown",
        memory_gb: Optional[int] = None,
    ) -> int:
        """Submit a SLURM job using ssh command.

        Parameters
        ----------
        host : str
            SSH host to submit the job to
        script_path : Path
            Path to the SLURM script to execute
        partition : str
            SLURM partition to submit to
        dependencies : Optional[List[int]]
            List of job IDs this job depends on
        env_vars : Optional[Dict[str, str]]
            Environment variables to pass to the job
        step_name : str
            Human-readable name for the step (for logging)
        memory_gb : Optional[int]
            Memory allocation in GB (e.g., 60 for 60GB)

        Returns
        -------
        int
            SLURM job ID of the submitted job
        """
        project_dir = "/home/menger/git/adata_hf_datasets"

        # Build the sbatch command with proper environment setup
        # Start with the base command
        sbatch_cmd = ["sbatch"]

        # Add partition
        sbatch_cmd.extend(["--partition", partition])

        # Add memory allocation if specified
        if memory_gb is not None:
            sbatch_cmd.extend(["--mem", f"{memory_gb}G"])

        # Add dependencies if specified
        if dependencies:
            deps_str = ":".join(map(str, dependencies))
            sbatch_cmd.extend(["--dependency", f"afterok:{deps_str}"])

        # Add environment variables if specified
        if env_vars:
            env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
            sbatch_cmd.extend(["--export", f"ALL,{env_str}"])

        # Add the script path
        sbatch_cmd.append(str(script_path))

        # Construct the full SSH command
        # Use bash -l to load login shell environment (includes PATH)
        ssh_cmd = f"bash -l -c 'cd {project_dir} && {' '.join(sbatch_cmd)}'"
        cmd = ["ssh", host, ssh_cmd]

        # Log submission with memory info if specified
        memory_info = f" (Memory: {memory_gb}GB)" if memory_gb else ""
        logger.info(f"Submitting {script_path.name}{memory_info} ➜ {' '.join(cmd)}")

        # Execute the command
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )  # 5 minutes for job submission
        except subprocess.TimeoutExpired:
            error_msg = f"SLURM job submission timed out for {step_name} step on {host}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            error_msg = "SSH command not found. Please ensure SSH is installed and available in PATH."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f"SLURM job submission failed for {step_name} step on {host} (subprocess error)"
            logger.error(error_msg)
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"Error: {e.stderr}")
            if e.stderr and "Invalid partition" in e.stderr:
                error_msg += f"\n\nPartition '{partition}' does not exist on {host}. "
                error_msg += (
                    "Please check available partitions with 'sinfo' on the target host."
                )
            elif e.stderr and "Connection refused" in e.stderr:
                error_msg += f"\n\nSSH connection to {host} failed. "
                error_msg += (
                    "Please check your SSH configuration and network connectivity."
                )
            elif e.stderr and "Permission denied" in e.stderr:
                error_msg += "\n\nPermission denied. Please check your SSH key configuration and user permissions."
            raise RuntimeError(error_msg)

        if result.returncode != 0:
            error_msg = f"SLURM job submission failed for {step_name} step on {host}"
            logger.error(error_msg)
            logger.error(f"Command: {' '.join(cmd)}")
            logger.error(f"Exit code: {result.returncode}")
            logger.error(f"Error: {result.stderr}")

            # Provide helpful error messages based on common issues
            if "Invalid partition" in result.stderr:
                error_msg += f"\n\nPartition '{partition}' does not exist on {host}. "
                error_msg += (
                    "Please check available partitions with 'sinfo' on the target host."
                )
            elif "Connection refused" in result.stderr:
                error_msg += f"\n\nSSH connection to {host} failed. "
                error_msg += (
                    "Please check your SSH configuration and network connectivity."
                )
            elif "Permission denied" in result.stderr:
                error_msg += "\n\nPermission denied. Please check your SSH key configuration and user permissions."

            raise RuntimeError(error_msg)

        # Parse job ID from output
        output = result.stdout.strip()
        job_id_match = re.search(r"Submitted batch job (\d+)", output)
        if not job_id_match:
            error_msg = f"Could not parse job ID from SLURM output: {output}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        job_id = int(job_id_match.group(1))

        # Track this job for potential cancellation
        self.submitted_jobs.append((host, job_id, step_name))

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

    def run_transfer_cpu_to_gpu_step(
        self,
        dataset_config_name: str,
        dataset_config: DictConfig,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the CPU to GPU transfer step and return job ID."""
        logger.info("=== Starting CPU to GPU Transfer Step ===")
        script_path = Path("scripts/workflow/transfer_cpu_to_gpu.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Extract required information from dataset config
        base_file_path = dataset_config.get("base_file_path", "/scratch/local")
        dataset_name = dataset_config.dataset.name

        # Pass the required environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "BASE_FILE_PATH": base_file_path,
            "DATASET_NAME": dataset_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for transfer coordination
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Transfer CPU→GPU",
        )
        return job_id

    def run_transfer_gpu_to_cpu_step(
        self,
        dataset_config_name: str,
        dataset_config: DictConfig,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the GPU to CPU transfer step and return job ID."""
        logger.info("=== Starting GPU to CPU Transfer Step ===")
        script_path = Path("scripts/workflow/transfer_gpu_to_cpu.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Extract required information from dataset config
        base_file_path = dataset_config.get("base_file_path", "/scratch/local")
        dataset_name = dataset_config.dataset.name

        # Pass the required environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "BASE_FILE_PATH": base_file_path,
            "DATASET_NAME": dataset_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for transfer coordination
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Transfer GPU→CPU",
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
        """Run the embedding preparation step using the new simplified structure."""
        logger.info("=== Starting Embedding Preparation Step (New) ===")
        script_path = Path("scripts/embed/run_embed_new.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Load dataset config to extract base_file_path and memory settings
        dataset_config = self._load_dataset_config(dataset_config_name)
        base_file_path = dataset_config.get("base_file_path", "/scratch/local")

        # Extract memory setting from embedding_preparation config (default: 60GB)
        memory_gb = getattr(dataset_config.embedding_preparation, "memory_gb", 60)
        logger.info(f"Using {memory_gb}GB memory for embedding preparation")

        # Pass the dataset config name, workflow directory, and mode settings as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "BASE_FILE_PATH": base_file_path,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "MODE": "cpu",  # Preparation typically runs on CPU
            "PREPARE_ONLY": "true",  # This is preparation mode
            "SLURM_PARTITION": workflow_config.cpu_partition,
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for preparation
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Embedding Preparation",
            memory_gb=memory_gb,
        )
        return job_id

    def run_embedding_cpu_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the CPU embedding step using the new simplified structure."""
        logger.info("=== Starting CPU Embedding Step (New) ===")
        script_path = Path("scripts/embed/run_embed_new.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Load dataset config to extract base_file_path and memory settings
        dataset_config = self._load_dataset_config(dataset_config_name)
        base_file_path = dataset_config.get("base_file_path", "/scratch/local")

        # Extract memory setting from embedding_cpu config (default: 60GB)
        memory_gb = getattr(dataset_config.embedding_cpu, "memory_gb", 60)
        logger.info(f"Using {memory_gb}GB memory for CPU embedding")

        # Pass the dataset config name, workflow directory, and mode settings as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "BASE_FILE_PATH": base_file_path,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "MODE": "cpu",  # Force CPU mode
            "PREPARE_ONLY": "false",  # This is full embedding mode
            "SLURM_PARTITION": workflow_config.cpu_partition,
        }

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for CPU embedding
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="CPU Embedding",
            memory_gb=memory_gb,
        )
        return job_id

    def run_embedding_gpu_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the GPU embedding step using the new simplified structure."""
        logger.info("=== Starting GPU Embedding Step (New) ===")
        script_path = Path("scripts/embed/run_embed_new.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Load dataset config to extract base_file_path and memory settings
        dataset_config = self._load_dataset_config(dataset_config_name)
        base_file_path = dataset_config.get("base_file_path", "/scratch/local")

        # Extract memory setting from embedding_gpu config (default: 60GB)
        memory_gb = getattr(dataset_config.embedding_gpu, "memory_gb", 60)
        logger.info(f"Using {memory_gb}GB memory for GPU embedding")

        # Pass the dataset config name, workflow directory, and mode settings as environment variables
        # IMPORTANT: Master job runs on CPU cluster to avoid consuming GPU resources
        # Only the array jobs will use GPU resources
        env_vars = {
            "DATASET_CONFIG": dataset_config_name,
            "BASE_FILE_PATH": base_file_path,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "MODE": "gpu",  # Force GPU mode for array jobs
            "PREPARE_ONLY": "false",  # This is full embedding mode
            "SLURM_PARTITION": workflow_config.gpu_partition,  # Pass GPU partition for array jobs
            "GPU_HOST": f"{self.gpu_login['user']}@{self.gpu_login['host']}"
            if self.gpu_login
            else "",  # GPU cluster info for array job submission
        }

        job_id = self._submit_slurm_job(
            self.cpu_login[
                "host"
            ],  # Use CPU cluster for master job (coordination only)
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition for master job
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="GPU Embedding",
            memory_gb=memory_gb,
        )
        return job_id

    def run_dataset_creation_step(
        self,
        dataset_config_name: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> List[int]:
        """Run the dataset creation step(s) and return list of job IDs."""
        logger.info("=== Starting Dataset Creation Step ===")
        script_path = Path("scripts/dataset_creation/run_create_ds.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name}")

        # Load dataset config to check for multiple cs_length and caption_keys values
        dataset_config = self._load_dataset_config(dataset_config_name)

        # Extract cs_length configuration
        cs_length_config = dataset_config.dataset_creation.cs_length

        # Handle both single values and lists (including OmegaConf lists)
        from omegaconf import ListConfig

        if isinstance(cs_length_config, (list, tuple, ListConfig)):
            cs_length_values = list(cs_length_config)
        else:
            cs_length_values = [cs_length_config]

        # Extract caption_keys configuration
        caption_keys_config = dataset_config.dataset_creation.get("caption_keys", None)

        if caption_keys_config is not None:
            # Handle both single values and lists for caption_keys
            if isinstance(caption_keys_config, (list, tuple, ListConfig)):
                caption_key_names = list(caption_keys_config)
            else:
                caption_key_names = [caption_keys_config]

            # Resolve config parameter names to their actual values
            caption_key_values = []
            for key_name in caption_key_names:
                if hasattr(dataset_config, key_name):
                    resolved_value = getattr(dataset_config, key_name)
                    if resolved_value is not None:  # Only add non-null values
                        caption_key_values.append(resolved_value)
                        logger.info(f"Resolved {key_name} to '{resolved_value}'")
                    else:
                        logger.warning(f"Skipping {key_name} as it resolves to null")
                else:
                    logger.warning(f"Config parameter '{key_name}' not found, skipping")

            if not caption_key_values:
                logger.warning(
                    "No valid caption keys found, falling back to default caption_key"
                )
                caption_key_values = [dataset_config.get("caption_key", None)]
        else:
            # No caption_keys specified, use default caption_key
            caption_key_values = [dataset_config.get("caption_key", None)]

        logger.info(
            f"Dataset creation will run with cs_length values: {cs_length_values}"
        )
        logger.info(
            f"Dataset creation will run with caption_key values: {caption_key_values}"
        )

        # Create combinations of cs_length and caption_keys
        job_ids = []
        job_counter = 0
        total_jobs = len(cs_length_values) * len(caption_key_values)

        for cs_length in cs_length_values:
            for caption_key_value in caption_key_values:
                job_counter += 1

                # Create a unique step name for each combination
                step_name = f"Dataset Creation (cs_length={cs_length}, caption_key={caption_key_value or 'none'})"
                if total_jobs > 1:
                    step_name += f" [{job_counter}/{total_jobs}]"

                logger.info(
                    f"Submitting dataset creation job with cs_length={cs_length}, caption_key={caption_key_value}"
                )

                # Pass the dataset config name, workflow directory, and specific overrides
                env_vars = {
                    "DATASET_CONFIG": dataset_config_name,
                    "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
                    if self.workflow_logger
                    else "",
                    "CS_LENGTH_OVERRIDE": str(cs_length),
                    "CAPTION_KEY_OVERRIDE": str(caption_key_value)
                    if caption_key_value is not None
                    else "",
                }

                job_id = self._submit_slurm_job(
                    self.cpu_login["host"],  # Use CPU cluster for dataset creation
                    script_path,
                    partition=workflow_config.cpu_partition,  # Use CPU partition
                    dependencies=dependencies,
                    env_vars=env_vars,
                    step_name=step_name,
                )
                job_ids.append(job_id)

        return job_ids

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

        # Check if transfers are enabled - if so, stop execution
        # Handle both cases: workflow_config could be the full config or just the workflow section
        if hasattr(workflow_config, "workflow"):
            transfers_enabled = getattr(
                workflow_config.workflow, "enable_transfers", True
            )
        else:
            transfers_enabled = getattr(workflow_config, "enable_transfers", True)

        if transfers_enabled:
            logger.error("=" * 80)
            logger.error("WORKFLOW STOPPED: Transfer mode is enabled but not supported")
            logger.error("=" * 80)
            logger.error(
                "The file transfer implementation between CPU and GPU clusters is not working correctly."
            )
            logger.error(
                "Please disable transfers in your workflow configuration and use a shared filesystem instead."
            )
            logger.error("")
            logger.error(
                "To fix this, set 'enable_transfers: false' in your workflow configuration."
            )
            logger.error(
                "This will use the shared filesystem for data access across CPU and GPU clusters."
            )
            logger.error("=" * 80)
            raise RuntimeError(
                "Transfer mode is enabled but not supported. Please use shared filesystem mode instead."
            )

        logger.info("Transfer mode: Disabled (using shared filesystem)")

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
        embedding_prepare_enabled = getattr(
            dataset_config.embedding_preparation, "enabled", True
        )
        if embedding_prepare_enabled:
            embedding_prepare_job_id = self.run_embedding_prepare_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=preprocessing_job_id,
            )
            logger.info(
                f"✓ Embedding preparation job {embedding_prepare_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        # Check if GPU embedding is enabled to determine step dependencies
        embedding_gpu_enabled = getattr(dataset_config.embedding_gpu, "enabled", True)
        embedding_cpu_enabled = getattr(dataset_config.embedding_cpu, "enabled", True)
        dataset_creation_enabled = getattr(
            dataset_config.dataset_creation, "enabled", True
        )

        # Step 4a: CPU Embedding (depends on embedding preparation)
        embedding_cpu_job_id = None
        if embedding_cpu_enabled:
            embedding_cpu_job_id = self.run_embedding_cpu_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=embedding_prepare_job_id,
            )
            logger.info(
                f"✓ CPU embedding job {embedding_cpu_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        # Step 4b: GPU Embedding (depends on CPU embedding if enabled, otherwise embedding preparation)
        embedding_gpu_job_id = None
        if embedding_gpu_enabled:
            # Dependency logic: depend on CPU embedding if enabled, otherwise embedding preparation
            gpu_embedding_dependency = (
                embedding_cpu_job_id
                if embedding_cpu_enabled
                else embedding_prepare_job_id
            )

            embedding_gpu_job_id = self.run_embedding_gpu_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=gpu_embedding_dependency,
            )
            logger.info(
                f"✓ GPU embedding job {embedding_gpu_job_id} submitted to cluster ({self.gpu_login['host']})"
            )

        # Step 5: Dataset Creation (depends on final embedding results)
        # Use the last completed embedding job for dependency
        embedding_dependency = None
        if embedding_gpu_job_id:
            embedding_dependency = embedding_gpu_job_id
        elif embedding_cpu_job_id:
            embedding_dependency = embedding_cpu_job_id
        elif embedding_prepare_job_id:
            embedding_dependency = embedding_prepare_job_id

        if dataset_creation_enabled:
            dataset_job_ids = self.run_dataset_creation_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=embedding_dependency,
            )
            if len(dataset_job_ids) == 1:
                logger.info(
                    f"✓ Dataset creation job {dataset_job_ids[0]} submitted to cluster ({self.cpu_login['host']})"
                )
            else:
                logger.info(
                    f"✓ Dataset creation jobs {dataset_job_ids} submitted to cluster ({self.cpu_login['host']})"
                )

        logger.info("=== Workflow Complete ===")
        logger.info("All jobs have been submitted to the clusters.")
        logger.info("Jobs will run with dependencies managed by SLURM.")
        logger.info(
            "You can monitor progress using 'squeue' on the respective clusters."
        )
        logger.info(
            "Check the logs in outputs/ (on the cluster) to see the progress of the jobs and intermediate results."
        )

    def run_workflow_local(
        self, dataset_config_name: str, workflow_config: DictConfig, force: bool = False
    ) -> None:
        """Run the complete workflow locally on the cluster, waiting for each step to complete."""

        # Set up signal handlers for graceful cancellation
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, cancelling all jobs...")
            self.cancel_all_jobs()
            sys.exit(1)

        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

        # Initialize the workflow logger
        master_job_id = os.environ.get(
            "SLURM_JOB_ID", f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Get output directory from config, with fallback to default
        base_dir = Path(
            workflow_config.get(
                "output_directory", "/home/menger/git/adata_hf_datasets/outputs"
            )
        )

        self.workflow_logger = WorkflowLogger(
            base_dir, master_job_id, dataset_config_name
        )

        logger.info(
            f"Starting local workflow for dataset config: {dataset_config_name}"
        )
        logger.info(f"Output directory: {base_dir}")

        # Validate config synchronization unless forced
        self.validate_config_sync(dataset_config_name, force=force)

        # Load the dataset config to check enabled flags
        dataset_config = self._load_dataset_config(dataset_config_name)

        # Log workflow start
        self.workflow_logger.log_workflow_start(dataset_config)

        logger.info(f"Dataset name: {dataset_config.dataset.name}")

        # Check if transfers are enabled - if so, stop execution
        # Handle both cases: workflow_config could be the full config or just the workflow section
        if hasattr(workflow_config, "workflow"):
            transfers_enabled = getattr(
                workflow_config.workflow, "enable_transfers", True
            )
        else:
            transfers_enabled = getattr(workflow_config, "enable_transfers", True)

        if transfers_enabled:
            logger.error("=" * 80)
            logger.error("WORKFLOW STOPPED: Transfer mode is enabled but not supported")
            logger.error("=" * 80)
            logger.error(
                "The file transfer implementation between CPU and GPU clusters is not working correctly."
            )
            logger.error(
                "Please disable transfers in your workflow configuration and use a shared filesystem instead."
            )
            logger.error("")
            logger.error(
                "To fix this, set 'enable_transfers: false' in your workflow configuration."
            )
            logger.error(
                "This will use the shared filesystem for data access across CPU and GPU clusters."
            )
            logger.error("=" * 80)
            raise RuntimeError(
                "Transfer mode is enabled but not supported. Please use shared filesystem mode instead."
            )

        logger.info("Transfer mode: Disabled (using shared filesystem)")

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
            try:
                self._wait_for_job_completion(
                    self.cpu_login["host"], download_job_id, "Download"
                )
                self.workflow_logger.log_step_complete("Download", download_job_id)
            except Exception as e:
                error_msg = f"Download step failed: {e}"
                logger.error(error_msg)
                self._log_error_to_consolidated_log(error_msg)
                raise RuntimeError(error_msg) from e
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
            try:
                self._wait_for_job_completion(
                    self.cpu_login["host"], preprocessing_job_id, "Preprocessing"
                )
                self.workflow_logger.log_step_complete(
                    "Preprocessing", preprocessing_job_id
                )
            except Exception as e:
                error_msg = f"Preprocessing step failed: {e}"
                logger.error(error_msg)
                self._log_error_to_consolidated_log(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            logger.info("=== Preprocessing Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped("Preprocessing", "disabled in config")

        # Step 3: Embedding Preparation (depends on preprocessing)
        embedding_prepare_job_id = None
        embedding_prepare_enabled = getattr(
            dataset_config.embedding_preparation, "enabled", True
        )
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
            try:
                self._wait_for_job_completion(
                    self.cpu_login["host"],
                    embedding_prepare_job_id,
                    "Embedding Preparation",
                )
                self.workflow_logger.log_step_complete(
                    "Embedding Preparation", embedding_prepare_job_id
                )
            except Exception as e:
                error_msg = f"Embedding preparation step failed: {e}"
                logger.error(error_msg)
                self._log_error_to_consolidated_log(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            logger.info("=== Embedding Preparation Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped(
                "Embedding Preparation", "disabled in config"
            )

        # Check if GPU embedding is enabled to determine step dependencies
        embedding_gpu_enabled = getattr(dataset_config.embedding_gpu, "enabled", True)
        embedding_cpu_enabled = getattr(dataset_config.embedding_cpu, "enabled", True)
        dataset_creation_enabled = getattr(
            dataset_config.dataset_creation, "enabled", True
        )

        # Step 4a: CPU Embedding (depends on embedding preparation)
        embedding_cpu_job_id = None
        if embedding_cpu_enabled:
            logger.info("=== Starting CPU Embedding Step ===")
            embedding_cpu_job_id = self.run_embedding_cpu_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=embedding_prepare_job_id,
            )
            logger.info(
                f"✓ CPU embedding job {embedding_cpu_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

            # Wait for CPU embedding job to complete
            try:
                self._wait_for_job_completion(
                    self.cpu_login["host"], embedding_cpu_job_id, "CPU Embedding"
                )
                self.workflow_logger.log_step_complete(
                    "CPU Embedding", embedding_cpu_job_id
                )
            except Exception as e:
                error_msg = f"CPU embedding step failed: {e}"
                logger.error(error_msg)
                self._log_error_to_consolidated_log(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            logger.info("=== CPU Embedding Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped("CPU Embedding", "disabled in config")

        # Step 4b: GPU Embedding (depends on CPU embedding if enabled, otherwise embedding preparation)
        embedding_gpu_job_id = None
        if embedding_gpu_enabled:
            logger.info("=== Starting GPU Embedding Step ===")
            # Dependency logic: depend on CPU embedding if enabled, otherwise embedding preparation
            gpu_embedding_dependency = (
                embedding_cpu_job_id
                if embedding_cpu_enabled
                else embedding_prepare_job_id
            )

            embedding_gpu_job_id = self.run_embedding_gpu_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=gpu_embedding_dependency,
            )
            logger.info(
                f"✓ GPU embedding job {embedding_gpu_job_id} submitted to cluster ({self.gpu_login['host']})"
            )

            # Wait for GPU embedding job to complete
            try:
                self._wait_for_job_completion(
                    self.cpu_login["host"], embedding_gpu_job_id, "GPU Embedding Master"
                )
                self.workflow_logger.log_step_complete(
                    "GPU Embedding Master", embedding_gpu_job_id
                )
            except Exception as e:
                error_msg = f"GPU embedding step failed: {e}"
                logger.error(error_msg)
                self._log_error_to_consolidated_log(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            logger.info("=== GPU Embedding Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped("GPU Embedding", "disabled in config")

        # Step 5: Dataset Creation (depends on final embedding results)
        # Use the last completed embedding job for dependency
        embedding_dependency = None
        if embedding_gpu_job_id:
            embedding_dependency = embedding_gpu_job_id
        elif embedding_cpu_job_id:
            embedding_dependency = embedding_cpu_job_id
        elif embedding_prepare_job_id:
            embedding_dependency = embedding_prepare_job_id

        if dataset_creation_enabled:
            logger.info("=== Starting Dataset Creation Step ===")
            dataset_job_ids = self.run_dataset_creation_step(
                dataset_config_name,
                workflow_config,
                dependency_job_id=embedding_dependency,
            )
            if len(dataset_job_ids) == 1:
                logger.info(
                    f"✓ Dataset creation job {dataset_job_ids[0]} submitted to cluster ({self.cpu_login['host']})"
                )
            else:
                logger.info(
                    f"✓ Dataset creation jobs {dataset_job_ids} submitted to cluster ({self.cpu_login['host']})"
                )

            # Wait for all dataset creation jobs to complete
            for i, dataset_job_id in enumerate(dataset_job_ids):
                step_name = "Dataset Creation"
                if len(dataset_job_ids) > 1:
                    step_name += f" [{i + 1}/{len(dataset_job_ids)}]"

                try:
                    self._wait_for_job_completion(
                        self.cpu_login["host"], dataset_job_id, step_name
                    )
                    self.workflow_logger.log_step_complete(step_name, dataset_job_id)
                except Exception as e:
                    error_msg = f"{step_name} failed: {e}"
                    logger.error(error_msg)
                    self._log_error_to_consolidated_log(error_msg)
                    raise RuntimeError(error_msg) from e
        else:
            logger.info("=== Dataset Creation Step Skipped (disabled) ===")
            self.workflow_logger.log_step_skipped(
                "Dataset Creation", "disabled in config"
            )

        # Log workflow completion
        self.workflow_logger.log_workflow_complete()

        logger.info("=== Workflow Complete ===")
        logger.info("All steps have been completed successfully.")

    def cancel_all_jobs(self) -> None:
        """Cancel all submitted jobs."""
        if not self.submitted_jobs:
            logger.info("No jobs to cancel")
            return

        logger.info(f"Cancelling {len(self.submitted_jobs)} submitted jobs...")

        for host, job_id, step_name in self.submitted_jobs:
            try:
                # Cancel the job (this will also cancel any array jobs it spawned)
                cmd = ["ssh", host, f"scancel {job_id}"]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )  # 1 minute for cancellation

                if result.returncode == 0:
                    logger.info(f"✓ Cancelled {step_name} job {job_id} on {host}")
                else:
                    logger.warning(
                        f"Failed to cancel job {job_id} on {host}: {result.stderr}"
                    )

            except Exception as e:
                logger.warning(f"Error cancelling job {job_id} on {host}: {e}")

        logger.info("Job cancellation complete")

    def _wait_for_job_completion(
        self, host: str, job_id: int, step_name: str, timeout_hours: int = 144
    ) -> None:
        """Wait for a SLURM job to complete and check for errors."""
        logger.info(
            f"Waiting for {step_name} job {job_id} to complete (timeout: {timeout_hours}h)..."
        )

        import time

        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        check_interval = 600  # check every 10 minutes

        while True:
            # Check if we've exceeded the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                error_msg = (
                    f"✗ {step_name} job {job_id} timed out after {timeout_hours} hours"
                )
                logger.error(error_msg)
                self._log_error_to_consolidated_log(error_msg)

                # Try to cancel the job
                try:
                    cancel_cmd = ["ssh", host, f"scancel {job_id}"]
                    subprocess.run(
                        cancel_cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,  # 1 minute for cancellation
                    )
                    logger.info(f"Cancelled timed-out job {job_id}")
                except Exception as cancel_e:
                    logger.warning(
                        f"Failed to cancel timed-out job {job_id}: {cancel_e}"
                    )

                raise RuntimeError(error_msg)
            # Check job status
            cmd = ["ssh", host, f"squeue -j {job_id} --noheader"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0 or not result.stdout.strip():
                # Job is no longer in queue (completed, failed, or cancelled)
                # Check the exit status using more comprehensive sacct query
                exit_cmd = [
                    "ssh",
                    host,
                    f"sacct -j {job_id} --format=JobID,State,ExitCode --noheader --parsable2",
                ]
                exit_result = subprocess.run(
                    exit_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 5 minutes for monitoring
                )

                if exit_result.returncode == 0 and exit_result.stdout.strip():
                    # Parse all job states (main job and array tasks)
                    lines = exit_result.stdout.strip().split("\n")
                    main_job_state = None
                    main_job_exit_code = None
                    failed_array_jobs = []

                    for line in lines:
                        if not line.strip():
                            continue

                        parts = line.split("|")
                        if len(parts) >= 3:
                            job_id_part = parts[0]
                            state = parts[1]
                            exit_code = parts[2]

                            # Check if this is the main job (no underscore or .batch/.extern suffix)
                            if (
                                job_id_part == str(job_id)
                                or job_id_part == f"{job_id}.batch"
                            ):
                                main_job_state = state
                                main_job_exit_code = exit_code
                            # Check for failed array jobs
                            elif f"{job_id}_" in job_id_part and state not in [
                                "COMPLETED",
                                "COMPLETED+",
                            ]:
                                failed_array_jobs.append(
                                    (job_id_part, state, exit_code)
                                )

                    # Log detailed information about job completion
                    self._log_job_completion_details(host, job_id, step_name)

                    # Check main job state
                    success_states = ["COMPLETED", "COMPLETED+"]
                    failure_states = [
                        "FAILED",
                        "CANCELLED",
                        "TIMEOUT",
                        "OUT_OF_MEMORY",
                        "NODE_FAIL",
                        "PREEMPTED",
                    ]

                    if main_job_state in success_states:
                        # Check if main job succeeded but array jobs failed
                        if failed_array_jobs:
                            error_msg = f"✗ {step_name} job {job_id} main task completed but {len(failed_array_jobs)} array task(s) failed:"
                            for (
                                array_job_id,
                                array_state,
                                array_exit_code,
                            ) in failed_array_jobs:
                                error_msg += f"\n  - Array job {array_job_id}: {array_state} (exit code: {array_exit_code})"

                            logger.error(error_msg)
                            self._log_error_to_consolidated_log(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            logger.info(
                                f"✓ {step_name} job {job_id} completed successfully"
                            )
                            return

                    elif main_job_state in failure_states:
                        error_msg = f"✗ {step_name} job {job_id} failed with state: {main_job_state}"
                        if main_job_exit_code and main_job_exit_code != "0:0":
                            error_msg += f" (exit code: {main_job_exit_code})"

                        logger.error(error_msg)
                        self._log_error_to_consolidated_log(error_msg)
                        raise RuntimeError(error_msg)

                    elif main_job_state:
                        # Unknown state - treat as potential failure
                        error_msg = f"? {step_name} job {job_id} ended with unknown state: {main_job_state}"
                        if main_job_exit_code and main_job_exit_code != "0:0":
                            error_msg += f" (exit code: {main_job_exit_code})"
                            # If exit code is non-zero, treat as failure
                            logger.error(error_msg)
                            self._log_error_to_consolidated_log(error_msg)
                            raise RuntimeError(error_msg)
                        else:
                            logger.warning(error_msg)
                            self._log_error_to_consolidated_log(f"WARNING: {error_msg}")
                            return
                    else:
                        # No main job state found - this is suspicious
                        error_msg = f"✗ {step_name} job {job_id} - could not determine job state from sacct output"
                        logger.error(error_msg)
                        logger.error(f"Raw sacct output: {exit_result.stdout}")
                        self._log_error_to_consolidated_log(error_msg)
                        raise RuntimeError(error_msg)
                else:
                    # Could not get job status via sacct - try fallback methods
                    logger.warning(
                        f"sacct failed for job {job_id}: {exit_result.stderr}"
                    )
                    logger.info(
                        f"Attempting fallback job status check for {step_name} job {job_id}"
                    )

                    # Try to determine job status using fallback methods
                    job_success = self._check_job_status_fallback(
                        host, job_id, step_name
                    )

                    if job_success:
                        logger.info(
                            f"✓ {step_name} job {job_id} completed successfully (via fallback check)"
                        )
                        return
                    else:
                        error_msg = f"✗ {step_name} job {job_id} failed (determined via fallback check)"
                        logger.error(error_msg)
                        self._log_error_to_consolidated_log(error_msg)
                        raise RuntimeError(error_msg)

            # Job is still running, wait a bit
            elapsed_minutes = int(elapsed_time / 60)
            remaining_minutes = int((timeout_seconds - elapsed_time) / 60)
            logger.info(
                f"  {step_name} job {job_id} still running... (elapsed: {elapsed_minutes}m, remaining: {remaining_minutes}m)"
            )
            time.sleep(check_interval)

    def _check_job_status_fallback(
        self, host: str, job_id: int, step_name: str
    ) -> bool:
        """Check job status using fallback methods when sacct fails."""
        try:
            # Check for job output files
            output_file_path = (
                f"/path/to/job/output/{job_id}.out"  # Adjust path as necessary
            )
            if os.path.exists(output_file_path):
                with open(output_file_path, "r") as f:
                    content = f.read()
                    if "SUCCESS" in content:
                        logger.info(f"Job {job_id} completed successfully.")
                        return True
                    elif "FAILED" in content:
                        logger.warning(f"Job {job_id} failed based on output.")
                        return False
                    else:
                        logger.warning(
                            f"Job {job_id} status is unknown based on output."
                        )
                        return True  # Or return None based on your preference
            else:
                logger.warning(
                    f"No output file found for job {job_id}. Status is unknown."
                )
                return True  # Or return None based on your preference
        except Exception as e:
            logger.warning(f"Error checking job status for {job_id}: {e}")
            return True  # Or return None based on your preference

    def _check_remote_job_output(
        self, host: str, file_path: Path, step_name: str, job_id: int
    ) -> Tuple[Optional[bool], bool]:
        """
        Check a remote job output file for success/error indicators.

        Returns:
            tuple: (success_indicator_found, error_found)
                - success_indicator_found: True if file indicates success, False if failure, None if indeterminate
                - error_found: True if Python errors were found and logged
        """
        try:
            # Check if file exists and get its content
            check_cmd = [
                "ssh",
                host,
                f"if [ -f {file_path} ]; then cat {file_path}; else echo 'FILE_NOT_FOUND'; fi",
            ]

            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for monitoring
            )

            if result.returncode != 0 or "FILE_NOT_FOUND" in result.stdout:
                return None, False  # File not found

            content = result.stdout.strip()
            if not content:
                return None, False  # Empty file

            logger.info(f"Found job output file: {file_path}")
            logger.info(f"Content length: {len(content)} characters")

            # Look for success indicators
            success_indicators = [
                "finished successfully",
                "completed successfully",
                "SUCCESS",
                "All done",
                "Processing complete",
            ]

            # Look for error indicators and Python errors
            error_indicators = [
                "ERROR",
                "Error:",
                "Exception:",
                "Traceback",
                "FileNotFoundError",
                "ImportError",
                "ValueError",
                "KeyError",
                "RuntimeError",
                "FAILED",
                "FATAL",
            ]

            content_lower = content.lower()
            found_success = any(
                indicator.lower() in content_lower for indicator in success_indicators
            )
            found_error = any(
                indicator.lower() in content_lower for indicator in error_indicators
            )

            # Extract and log Python errors to consolidated log
            python_errors_found = self._extract_python_errors_from_output(
                content, step_name, job_id
            )

            if python_errors_found:
                logger.error(f"Found Python errors in {step_name} job {job_id} output")
                return False, True  # Definite failure due to Python errors
            elif found_error and not found_success:
                logger.error(
                    f"Found error indicators in {step_name} job {job_id} output"
                )
                # Log last 20 lines to consolidated log for context
                lines = content.split("\n")
                last_lines = "\n".join(lines[-20:]) if len(lines) > 20 else content
                error_msg = (
                    f"Job output errors for {step_name} job {job_id}:\n{last_lines}"
                )
                self._log_error_to_consolidated_log(error_msg)
                return False, True
            elif found_success and not found_error:
                logger.info(
                    f"Found success indicators in {step_name} job {job_id} output"
                )
                return True, False
            else:
                # Ambiguous - both or neither found
                logger.warning(
                    f"Ambiguous status in {step_name} job {job_id} output (success: {found_success}, error: {found_error})"
                )
                return None, found_error

        except Exception as e:
            logger.warning(f"Failed to check remote job output {file_path}: {e}")
            return None, False

    def _extract_python_errors_from_output(
        self, content: str, step_name: str, job_id: int
    ) -> bool:
        """
        Extract Python errors/tracebacks from job output and log them to consolidated log.

        Returns:
            bool: True if Python errors were found and logged
        """
        lines = content.split("\n")
        python_errors = []
        current_error = []
        in_traceback = False

        for line in lines:
            line_stripped = line.strip()

            # Look for traceback start
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                current_error = [line]
                continue

            # Look for error lines
            if any(
                error_type in line
                for error_type in [
                    "Error:",
                    "Exception:",
                    "FileNotFoundError:",
                    "ImportError:",
                    "ValueError:",
                    "KeyError:",
                    "RuntimeError:",
                ]
            ):
                if not in_traceback:
                    current_error = []
                current_error.append(line)

                # If we have collected an error, save it
                if current_error:
                    python_errors.append("\n".join(current_error))
                    current_error = []
                    in_traceback = False
                continue

            # If in traceback, collect all lines
            if in_traceback:
                current_error.append(line)

            # Look for specific error patterns
            elif any(
                error_pattern in line_stripped
                for error_pattern in [
                    "FileNotFoundError:",
                    "No such file or directory",
                    "ImportError:",
                    "ModuleNotFoundError:",
                    "ValueError:",
                    "KeyError:",
                    "RuntimeError:",
                    "Exception:",
                    "Error type:",
                    "Error message:",
                ]
            ):
                if not current_error:  # Start new error collection
                    current_error = [line]
                else:
                    current_error.append(line)

        # Save any remaining error
        if current_error:
            python_errors.append("\n".join(current_error))

        # Log all found errors to consolidated log
        if python_errors:
            error_msg = f"Python errors found in {step_name} job {job_id}:"
            for i, error in enumerate(python_errors, 1):
                error_msg += f"\n\n--- Error {i} ---\n{error}"

            logger.error(
                f"Extracted {len(python_errors)} Python error(s) from {step_name} job {job_id}"
            )
            self._log_error_to_consolidated_log(error_msg)
            return True

        return False

    def _log_error_to_consolidated_log(self, error_msg: str) -> None:
        """Log error message to the consolidated error log."""
        if self.workflow_logger:
            error_log_path = (
                self.workflow_logger.workflow_dir / "logs" / "errors_consolidated.log"
            )
            try:
                with open(error_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{datetime.now().isoformat()} - {error_msg}\n")
            except Exception as e:
                logger.warning(f"Failed to write to consolidated error log: {e}")

    def _log_job_completion_details(
        self, host: str, job_id: int, step_name: str
    ) -> None:
        """Log detailed information about job completion including SLURM logs."""
        try:
            # Get more detailed job information
            detail_cmd = [
                "ssh",
                host,
                f"sacct -j {job_id} --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,MaxVMSize --noheader",
            ]
            detail_result = subprocess.run(
                detail_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for monitoring
            )

            if detail_result.returncode == 0 and detail_result.stdout.strip():
                logger.info(f"Detailed job information for {step_name} job {job_id}:")
                lines = detail_result.stdout.strip().split("\n")
                for line in lines:
                    if line.strip() and str(job_id) in line:
                        logger.info(f"  {line.strip()}")

                # Also log to consolidated log for record keeping
                if self.workflow_logger:
                    log_msg = f"Job completion details for {step_name} job {job_id}:\n"
                    for line in lines:
                        if line.strip() and str(job_id) in line:
                            log_msg += f"  {line.strip()}\n"

                    detail_log_path = (
                        self.workflow_logger.workflow_dir / "logs" / "job_details.log"
                    )
                    with open(detail_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now().isoformat()} - {log_msg}\n")

            # Try to get the SLURM output files if they exist
            self._log_slurm_output_files(host, job_id, step_name)

        except Exception as e:
            logger.warning(
                f"Failed to get detailed job information for job {job_id}: {e}"
            )

    def _log_slurm_output_files(self, host: str, job_id: int, step_name: str) -> None:
        """Try to log the contents of SLURM output files for debugging."""
        try:
            # Look for common SLURM output file patterns
            output_patterns = [
                f"slurm-{job_id}.out",
                f"slurm-{job_id}.err",
                f"{job_id}.out",
                f"{job_id}.err",
            ]

            for pattern in output_patterns:
                # Check if file exists and get last few lines
                check_cmd = [
                    "ssh",
                    host,
                    f"if [ -f {pattern} ]; then echo 'Found {pattern}'; tail -20 {pattern}; else echo 'File {pattern} not found'; fi",
                ]

                result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes for monitoring
                )

                if result.returncode == 0 and result.stdout.strip():
                    output = result.stdout.strip()
                    if "Found" in output and "not found" not in output:
                        logger.info(
                            f"Last 20 lines of {pattern} for {step_name} job {job_id}:"
                        )
                        logger.info(output)

                        # Log to file as well
                        if self.workflow_logger:
                            output_log_path = (
                                self.workflow_logger.workflow_dir
                                / "logs"
                                / "slurm_outputs.log"
                            )
                            with open(output_log_path, "a", encoding="utf-8") as f:
                                f.write(
                                    f"\n{datetime.now().isoformat()} - {step_name} job {job_id} - {pattern}:\n"
                                )
                                f.write(output)
                                f.write("\n" + "=" * 80 + "\n")
                        break  # Only log the first file we find

        except Exception as e:
            logger.warning(
                f"Failed to retrieve SLURM output files for job {job_id}: {e}"
            )

    def _load_dataset_config(self, dataset_config_name: str) -> DictConfig:
        """Load the dataset configuration using proper Hydra composition."""
        from hydra import compose, initialize_config_dir

        config_path = Path(__file__).parent.parent.parent / "conf"
        config_file = config_path / f"{dataset_config_name}.yaml"

        if not config_file.exists():
            raise ValueError(f"Dataset config file not found: {config_file}")

        # Use Hydra's proper composition to handle defaults inheritance
        try:
            with initialize_config_dir(config_dir=str(config_path), version_base=None):
                config = compose(config_name=dataset_config_name)
        except Exception as e:
            logger.error(f"Failed to load config using Hydra composition: {e}")
            logger.info("Falling back to OmegaConf.load() without defaults")
            # Fallback to direct loading if Hydra fails
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

    # Set up signal handlers for graceful cancellation
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, cancelling all jobs...")
        orchestrator.cancel_all_jobs()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    # Get the dataset config name from the config
    dataset_config_name = cfg.get("dataset_config_name", "dataset_test_workflow")
    logger.info(f"Using dataset config: {dataset_config_name}")

    # Extract workflow config from the loaded config
    workflow_config = cfg.workflow

    # Check for force flag
    force = getattr(cfg, "force", False)
    if force:
        logger.warning("Running with --force flag (skipping config sync validation)")

    try:
        orchestrator.run_workflow(dataset_config_name, workflow_config, force=force)
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user, cancelling jobs...")
        orchestrator.cancel_all_jobs()
        sys.exit(1)
    except Exception as e:
        error_msg = f"Workflow failed with error: {e}"
        logger.error(error_msg)
        logger.exception("Full traceback:")

        # Log to consolidated error log if possible
        if hasattr(orchestrator, "_log_error_to_consolidated_log"):
            orchestrator._log_error_to_consolidated_log(
                f"WORKFLOW FAILURE: {error_msg}"
            )

        logger.info("Cancelling any remaining jobs...")
        orchestrator.cancel_all_jobs()
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        main()
    except Exception:
        logger.exception("Workflow orchestration failed")
        sys.exit(1)


'''
    def _check_job_status_fallback(
        self, host: str, job_id: int, step_name: str
    ) -> bool:
        """
        Fallback method to check job status when sacct is unavailable.

        Returns:
            bool: True if job appears to have succeeded, False if it failed
        """
        logger.info(f"Using fallback status check for {step_name} job {job_id}")

        try:
            # Get the job output directory from workflow logger
            if self.workflow_logger:
                job_output_dir = self.workflow_logger.get_step_log_dir(
                    step_name.lower().replace(" ", "_"), str(job_id)
                )
            else:
                # Fallback to default output location
                job_output_dir = Path(
                    f"outputs/{datetime.now().strftime('%Y-%m-%d')}/{step_name.lower().replace(' ', '_')}/{job_id}"
                )

            # Look for job output files in multiple possible locations
            output_file_locations = [
                # Workflow logger location
                job_output_dir / f"{step_name.lower().replace(' ', '_')}.out",
                job_output_dir / f"{step_name.lower().replace(' ', '_')}.err",
                # Standard SLURM locations
                Path(f"slurm-{job_id}.out"),
                Path(f"slurm-{job_id}.err"),
                Path(f"{job_id}.out"),
                Path(f"{job_id}.err"),
            ]

            # Check remote files via SSH
            for file_path in output_file_locations:
                success, error_found = self._check_remote_job_output(
                    host, file_path, step_name, job_id
                )
                if success is not None:  # Found definitive answer
                    return success and not error_found

            # If no output files found, try to get basic job info
            logger.warning(f"No job output files found for {step_name} job {job_id}")

            # As a last resort, check if the job process is still running
            check_cmd = ["ssh", host, f"ps aux | grep {job_id} | grep -v grep"]
            result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for monitoring
            )

            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"Job {job_id} process still appears to be running")
                return False  # Still running, shouldn't be considered complete
            else:
                logger.warning(
                    f"Cannot determine status for job {job_id} - assuming failure for safety"
                )
                return False

        except Exception as e:
            logger.warning(f"Fallback status check failed for job {job_id}: {e}")
            return False  # Assume failure if we can't determine status
'''
