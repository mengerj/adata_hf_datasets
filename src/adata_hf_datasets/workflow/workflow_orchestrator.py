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

# Check for hydra-core at module import
try:
    import hydra
except ImportError:
    raise ImportError(
        "hydra-core is required to use the workflow module. "
        "Please install it with: pip install 'adata-hf-datasets[workflow]' "
        "or: pip install hydra-core"
    )

from omegaconf import DictConfig, OmegaConf

from .config_utils import (
    apply_all_transformations,
    ensure_config_sync,
)
from .data_transfer import (
    DataTransfer,
    LocationConfig,
    DataTransferError,
    create_data_transfer_from_config,
)

# Import unified workflow runner for per-step execution locations
from .workflow_runner import (
    run_unified_workflow,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STEP LOCATION HELPERS
# =============================================================================

# Mapping of step names to their config section names
STEP_TO_CONFIG_SECTION = {
    "download": "download",
    "preprocessing": "preprocessing",
    "embedding_preparation": "embedding_preparation",
    "embedding_cpu": "embedding_cpu",
    "embedding_gpu": "embedding_gpu",
    "dataset_creation": "dataset_creation",
}

# Workflow step order for determining data flow
WORKFLOW_STEPS = [
    "download",
    "preprocessing",
    "embedding_preparation",
    "embedding_cpu",
    "embedding_gpu",
    "dataset_creation",
]


def get_step_execution_location(
    step_name: str,
    dataset_config: DictConfig,
    default_location: str = "local",
) -> str:
    """
    Get the execution location for a workflow step.

    Priority:
    1. Step-specific execution_location from dataset config
    2. Default execution location

    Parameters
    ----------
    step_name : str
        Name of the workflow step
    dataset_config : DictConfig
        The dataset configuration
    default_location : str
        Default location if not specified

    Returns
    -------
    str
        Execution location: "local", "cpu", or "gpu"
    """
    section_name = STEP_TO_CONFIG_SECTION.get(step_name)
    if section_name and hasattr(dataset_config, section_name):
        section = getattr(dataset_config, section_name)
        if hasattr(section, "execution_location") and section.execution_location:
            return str(section.execution_location)
    return default_location


def get_step_output_paths(
    step_name: str,
    dataset_config: DictConfig,
    location_config: LocationConfig,
) -> List[str]:
    """
    Get the output paths for a workflow step.

    These are the paths that need to be transferred to the next step
    if it runs on a different location.

    Parameters
    ----------
    step_name : str
        Name of the workflow step
    dataset_config : DictConfig
        The dataset configuration
    location_config : LocationConfig
        Configuration for the step's execution location

    Returns
    -------
    List[str]
        List of output paths
    """
    base_path = location_config.base_file_path
    dataset_name = dataset_config.dataset.name

    # Determine split from preprocessing config (consistent with config_utils.py)
    split_dataset = dataset_config.preprocessing.get(
        "split_dataset", dataset_config.get("split_dataset", True)
    )
    split_subdir = "train" if split_dataset else "test"

    if step_name == "download":
        # Download outputs raw h5ad file
        return [f"{base_path}/raw/{split_subdir}/{dataset_name}.h5ad"]

    elif step_name == "preprocessing":
        # Preprocessing outputs to processed directory
        return [f"{base_path}/processed/{split_subdir}/{dataset_name}"]

    elif step_name in ["embedding_preparation", "embedding_cpu", "embedding_gpu"]:
        # Embedding outputs to processed_with_emb directory
        return [f"{base_path}/processed_with_emb/{split_subdir}/{dataset_name}"]

    elif step_name == "dataset_creation":
        # Dataset creation outputs to hf_datasets directory
        return [f"{location_config.output_directory}/data/hf_datasets/{dataset_name}"]

    return []


def get_step_input_paths(
    step_name: str,
    dataset_config: DictConfig,
    location_config: LocationConfig,
) -> List[str]:
    """
    Get the input paths required for a workflow step.

    Parameters
    ----------
    step_name : str
        Name of the workflow step
    dataset_config : DictConfig
        The dataset configuration
    location_config : LocationConfig
        Configuration for the step's execution location

    Returns
    -------
    List[str]
        List of input paths
    """
    base_path = location_config.base_file_path
    dataset_name = dataset_config.dataset.name

    # Determine split from preprocessing config (consistent with config_utils.py)
    split_dataset = dataset_config.preprocessing.get(
        "split_dataset", dataset_config.get("split_dataset", True)
    )
    split_subdir = "train" if split_dataset else "test"

    if step_name == "download":
        # Download has no input dependencies (downloads from URL)
        return []

    elif step_name == "preprocessing":
        # Preprocessing needs raw data
        return [f"{base_path}/raw/{split_subdir}/{dataset_name}.h5ad"]

    elif step_name == "embedding_preparation":
        # Embedding prep needs processed data
        return [f"{base_path}/processed/{split_subdir}/{dataset_name}"]

    elif step_name in ["embedding_cpu", "embedding_gpu"]:
        # Embedding needs processed_with_emb (where prep step put data)
        return [f"{base_path}/processed_with_emb/{split_subdir}/{dataset_name}"]

    elif step_name == "dataset_creation":
        # Dataset creation needs embedded data
        return [f"{base_path}/processed_with_emb/{split_subdir}/{dataset_name}"]

    return []


def _extract_config_name(dataset_config_name_or_path: str) -> str:
    """
    Extract a clean config name from either a path or a name.

    Parameters
    ----------
    dataset_config_name_or_path : str
        Either a config name or a file path

    Returns
    -------
    str
        A clean config name suitable for logging/identification
    """
    # Check if it's a path
    if "/" in dataset_config_name_or_path or "\\" in dataset_config_name_or_path:
        # It's a path - extract the stem (filename without extension)
        return Path(dataset_config_name_or_path).stem
    else:
        # It's already a name
        return dataset_config_name_or_path


class WorkflowLogger:
    """Manages comprehensive logging for the entire workflow."""

    def __init__(
        self,
        base_dir: Path,
        master_job_id: str,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
    ):
        """
        Initialize the workflow logger.

        Parameters
        ----------
        base_dir : Path
            Base directory for all outputs
        master_job_id : str
            The master SLURM job ID
        dataset_config_name_or_path : str
            Name or path of the dataset config being used
        workflow_config : DictConfig
            The workflow config
        """
        self.base_dir = base_dir
        self.master_job_id = master_job_id
        # Extract clean name for logging/identification
        self.dataset_config_name = _extract_config_name(dataset_config_name_or_path)

        # Create the workflow directory structure
        self.workflow_dir = self._create_workflow_directory()

        # Set up logging
        self._setup_logging()

        project_dir = Path(workflow_config.get("project_directory"))
        # Copy the dataset config
        self._copy_dataset_config(project_dir)

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

    def _copy_dataset_config(self, project_path: Path):
        """Copy the dataset config to the workflow directory."""
        config_path = project_path / "conf"
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
        self,
        dataset_config_name: str,
        force: bool = False,
        project_directory: Optional[str] = None,
    ) -> None:
        """Validate that the remote config matches the local one."""
        logger.info(f"Validating config synchronization for {dataset_config_name}...")
        logger.info(f"Project directory: {project_directory}")
        ensure_config_sync(
            config_name=dataset_config_name,
            remote_host=self.cpu_login["host"],
            remote_project_dir=project_directory,
            force=force,
        )

    def _get_venv_path(
        self,
        dataset_config: DictConfig,
        step_name: str,
        workflow_config: DictConfig,
    ) -> str:
        """Get the virtual environment path for a workflow step.

        Priority order:
        1. Step-specific venv_path from dataset config (e.g., dataset_config.download.venv_path)
        2. Workflow-level venv_path from workflow_config
        3. Environment variable VENV_PATH
        4. Default ".venv"

        Parameters
        ----------
        dataset_config : DictConfig
            The loaded dataset configuration
        step_name : str
            Name of the workflow step (e.g., "download", "preprocessing", "embedding_preparation")
        workflow_config : DictConfig
            The workflow configuration

        Returns
        -------
        str
            Path to the virtual environment (relative to project_directory)
        """
        # Map step names to config section names
        step_to_section = {
            "download": "download",
            "preprocessing": "preprocessing",
            "embedding_preparation": "embedding_preparation",
            "embedding_cpu": "embedding_cpu",
            "embedding_gpu": "embedding_gpu",
            "dataset_creation": "dataset_creation",
        }

        section_name = step_to_section.get(step_name)

        # Try step-specific venv_path from dataset config
        if section_name and hasattr(dataset_config, section_name):
            section = getattr(dataset_config, section_name)
            if hasattr(section, "venv_path") and section.venv_path is not None:
                venv_path = str(section.venv_path)
                logger.info(f"Using venv_path from {section_name} config: {venv_path}")
                return venv_path

        # Fall back to workflow-level venv_path
        if hasattr(workflow_config, "venv_path") and workflow_config.venv_path:
            venv_path = str(workflow_config.venv_path)
            logger.info(f"Using venv_path from workflow config: {venv_path}")
            return venv_path

        # Fall back to environment variable
        env_venv_path = os.environ.get("VENV_PATH")
        if env_venv_path:
            logger.info(f"Using venv_path from environment variable: {env_venv_path}")
            return env_venv_path

        # Default fallback
        default_venv = ".venv"
        logger.info(f"Using default venv_path: {default_venv}")
        return default_venv

    def _submit_slurm_job(
        self,
        host: str,
        script_path: Path,
        partition: str = "slurm",
        dependencies: Optional[List[int]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        step_name: str = "unknown",
        memory_gb: Optional[int] = None,
        node: Optional[str] = None,
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
        node : Optional[str]
            Specific node to run the job on (adds --nodelist constraint)

        Returns
        -------
        int
            SLURM job ID of the submitted job
        """
        # Note: project_dir is now passed via env_vars in each step method
        # This is kept for backward compatibility but should use workflow_config["project_directory"]
        project_dir = os.environ.get(
            "PROJECT_DIR", "/home/menger/git/adata_hf_datasets"
        )

        # Build the sbatch command with proper environment setup
        # Start with the base command
        sbatch_cmd = ["sbatch"]

        # Add partition
        sbatch_cmd.extend(["--partition", partition])

        # Add node constraint if specified
        if node:
            sbatch_cmd.extend(["--nodelist", node])

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
        self, dataset_config_name_or_path: str, workflow_config: DictConfig
    ) -> Optional[int]:
        """Run the download step and return job ID."""
        logger.info("=== Starting Download Step ===")
        script_path = Path("scripts/download/run_download_ds.slurm")

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Load dataset config to get venv_path
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)
        venv_path = self._get_venv_path(dataset_config, "download", workflow_config)

        # Pass the dataset config name and workflow directory as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
        }

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for download
            script_path,
            partition=workflow_config.cpu_partition,  # Use partition from config
            env_vars=env_vars,
            step_name="Download",
            node=cpu_node,
        )
        return job_id

    def run_transfer_cpu_to_gpu_step(
        self,
        dataset_config_name_or_path: str,
        dataset_config: DictConfig,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the CPU to GPU transfer step and return job ID."""
        logger.info("=== Starting CPU to GPU Transfer Step ===")
        script_path = Path("scripts/workflow/transfer_cpu_to_gpu.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Extract required information from dataset config
        # Use resolved base_file_path from workflow config
        # base_file_path = dataset_config.get(
        #    "base_file_path", workflow_config["base_file_path"]
        # )
        dataset_name = dataset_config.dataset.name

        # Get venv_path (transfer steps don't have dataset config sections, so use workflow-level)
        venv_path = self._get_venv_path(
            dataset_config, "transfer_cpu_to_gpu", workflow_config
        )

        # Pass the required environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
            "DATASET_NAME": dataset_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
        }

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for transfer coordination
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Transfer CPU→GPU",
            node=cpu_node,
        )
        return job_id

    def run_transfer_gpu_to_cpu_step(
        self,
        dataset_config_name_or_path: str,
        dataset_config: DictConfig,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the GPU to CPU transfer step and return job ID."""
        logger.info("=== Starting GPU to CPU Transfer Step ===")
        script_path = Path("scripts/workflow/transfer_gpu_to_cpu.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Extract required information from dataset config
        # Use resolved base_file_path from workflow config
        # base_file_path = dataset_config.get(
        #    "base_file_path", workflow_config["base_file_path"]
        # )
        dataset_name = dataset_config.dataset.name

        # Get venv_path (transfer steps don't have dataset config sections, so use workflow-level)
        venv_path = self._get_venv_path(
            dataset_config, "transfer_gpu_to_cpu", workflow_config
        )

        # Pass the required environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
            "DATASET_NAME": dataset_name,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
        }

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for transfer coordination
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Transfer GPU→CPU",
            node=cpu_node,
        )
        return job_id

    def run_preprocessing_step(
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the preprocessing step and return job ID."""
        logger.info("=== Starting Preprocessing Step ===")
        script_path = Path("scripts/preprocessing/run_preprocess.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Load dataset config to get venv_path
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)
        venv_path = self._get_venv_path(
            dataset_config, "preprocessing", workflow_config
        )

        # Pass the dataset config name and workflow directory as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
        }

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for preprocessing
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Preprocessing",
            node=cpu_node,
        )
        return job_id

    def run_embedding_prepare_step(
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the embedding preparation step using the new simplified structure."""
        logger.info("=== Starting Embedding Preparation Step (New) ===")
        script_path = Path("scripts/embed/run_embed.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Load dataset config to extract base_file_path and memory settings
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)
        # Use resolved base_file_path from workflow config
        # base_file_path = dataset_config.get(
        #    "base_file_path", workflow_config["base_file_path"]
        # )

        # Extract memory setting from embedding_preparation config (default: 60GB)
        memory_gb = getattr(dataset_config.embedding_preparation, "memory_gb", 60)
        logger.info(f"Using {memory_gb}GB memory for embedding preparation")

        # Get venv_path for this step
        venv_path = self._get_venv_path(
            dataset_config, "embedding_preparation", workflow_config
        )

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        # Pass the dataset config name, workflow directory, and mode settings as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "MODE": "cpu",  # Preparation typically runs on CPU
            "PREPARE_ONLY": "true",  # This is preparation mode
            "SLURM_PARTITION": workflow_config.cpu_partition,
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
        }
        # Pass cpu_node to embed_launcher if set
        if cpu_node:
            env_vars["CPU_NODE"] = cpu_node

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for preparation
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="Embedding Preparation",
            memory_gb=memory_gb,
            node=cpu_node,
        )
        return job_id

    def run_embedding_cpu_step(
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the CPU embedding step using the new simplified structure."""
        logger.info("=== Starting CPU Embedding Step (New) ===")
        script_path = Path("scripts/embed/run_embed.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Load dataset config to extract base_file_path and memory settings
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)
        # Use resolved base_file_path from workflow config
        # base_file_path = dataset_config.get(
        #    "base_file_path", workflow_config["base_file_path"]
        # )

        # Extract memory setting from embedding_cpu config (default: 60GB)
        memory_gb = getattr(dataset_config.embedding_cpu, "memory_gb", 60)
        logger.info(f"Using {memory_gb}GB memory for CPU embedding")

        # Get venv_path for this step
        venv_path = self._get_venv_path(
            dataset_config, "embedding_cpu", workflow_config
        )

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        # Pass the dataset config name, workflow directory, and mode settings as environment variables
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "MODE": "cpu",  # Force CPU mode
            "PREPARE_ONLY": "false",  # This is full embedding mode
            "SLURM_PARTITION": workflow_config.cpu_partition,
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
        }
        # Pass cpu_node to embed_launcher if set
        if cpu_node:
            env_vars["CPU_NODE"] = cpu_node

        # Get cpu_node from config (if set, all CPU jobs will run on this node)
        cpu_node = workflow_config.get("cpu_node")

        job_id = self._submit_slurm_job(
            self.cpu_login["host"],  # Use CPU cluster for CPU embedding
            script_path,
            partition=workflow_config.cpu_partition,  # Use CPU partition
            dependencies=dependencies,
            env_vars=env_vars,
            step_name="CPU Embedding",
            memory_gb=memory_gb,
            node=cpu_node,
        )
        return job_id

    def run_embedding_gpu_step(
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> Optional[int]:
        """Run the GPU embedding step using the new simplified structure."""
        logger.info("=== Starting GPU Embedding Step (New) ===")
        script_path = Path("scripts/embed/run_embed.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Load dataset config to extract base_file_path and memory settings
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)
        # base_file_path = dataset_config.get(
        #    "base_file_path", "/scratch/global/menger/data/RNA"
        # )

        # Extract memory setting from embedding_gpu config (default: 60GB)
        memory_gb = getattr(dataset_config.embedding_gpu, "memory_gb", 60)
        logger.info(f"Using {memory_gb}GB memory for GPU embedding")

        # Get venv_path for this step
        venv_path = self._get_venv_path(
            dataset_config, "embedding_gpu", workflow_config
        )

        # Pass the dataset config name, workflow directory, and mode settings as environment variables
        # IMPORTANT: Master job runs on CPU cluster to avoid consuming GPU resources
        # Only the array jobs will use GPU resources
        env_vars = {
            "DATASET_CONFIG": dataset_config_name_or_path,
            # Enforce base path from orchestrator (already resolved)
            "BASE_FILE_PATH": workflow_config["base_file_path"],
            "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
            if self.workflow_logger
            else "",
            "MODE": "gpu",  # Force GPU mode for array jobs
            "PREPARE_ONLY": "false",  # This is full embedding mode
            "SLURM_PARTITION": workflow_config.gpu_partition,  # Pass GPU partition for array jobs
            "GPU_HOST": f"{self.gpu_login['user']}@{self.gpu_login['host']}"
            if self.gpu_login
            else "",  # GPU cluster info for array job submission
            "PROJECT_DIR": workflow_config.get(
                "project_directory", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": venv_path,
        }

        # GPU embedding master job: do NOT apply cpu_node constraint
        # (this is a coordination job that submits GPU array jobs)
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
            node=None,  # Explicitly exclude from cpu_node constraint
        )
        return job_id

    def run_dataset_creation_step(
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        dependency_job_id: Optional[int] = None,
    ) -> List[int]:
        """Run the dataset creation step(s) and return list of job IDs."""
        logger.info("=== Starting Dataset Creation Step ===")
        script_path = Path("scripts/dataset_creation/run_create_ds.slurm")
        dependencies = [dependency_job_id] if dependency_job_id else None

        logger.info(f"Using dataset config: {dataset_config_name_or_path}")

        # Load dataset config to check for multiple cs_length and caption_keys values
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)

        # Get venv_path for this step
        venv_path = self._get_venv_path(
            dataset_config, "dataset_creation", workflow_config
        )

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

                # Get cpu_node from config (if set, all CPU jobs will run on this node)
                cpu_node = workflow_config.get("cpu_node")

                # Pass the dataset config name, workflow directory, and specific overrides
                env_vars = {
                    "DATASET_CONFIG": dataset_config_name_or_path,
                    "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
                    if self.workflow_logger
                    else "",
                    "CS_LENGTH_OVERRIDE": str(cs_length),
                    "CAPTION_KEY_OVERRIDE": str(caption_key_value)
                    if caption_key_value is not None
                    else "",
                    # Enforce base path from orchestrator (already resolved)
                    "BASE_FILE_PATH": workflow_config["base_file_path"],
                    "PROJECT_DIR": workflow_config.get(
                        "project_directory", "/home/menger/git/adata_hf_datasets"
                    ),
                    "VENV_PATH": venv_path,
                }

                job_id = self._submit_slurm_job(
                    self.cpu_login["host"],  # Use CPU cluster for dataset creation
                    script_path,
                    partition=workflow_config.cpu_partition,  # Use CPU partition
                    dependencies=dependencies,
                    env_vars=env_vars,
                    step_name=step_name,
                    node=cpu_node,
                )
                job_ids.append(job_id)

        return job_ids

    def run_workflow(
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        force: bool = False,
    ) -> None:
        """Run the complete workflow."""
        logger.info(
            f"Starting workflow for dataset config: {dataset_config_name_or_path}"
        )

        # Ensure base path is available for any config transformations in this process
        try:
            os.environ["BASE_FILE_PATH"] = workflow_config["base_file_path"]
        except Exception:
            pass

        # Validate config synchronization unless forced
        # project_dir = workflow_config.get("project_directory")
        # self.validate_config_sync(
        #    dataset_config_name, force=force, project_directory=project_dir
        # )

        # Load the dataset config to check enabled flags
        # We'll load it here to get the configuration for each step
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)
        logger.info(f"Dataset name: {dataset_config.dataset.name}")

        # Check if transfers are enabled - if so, stop execution
        transfers_enabled = workflow_config.get("enable_transfers", True)

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
                dataset_config_name_or_path, workflow_config
            )
            logger.info(
                f"✓ Download job {download_job_id} submitted to cluster ({self.cpu_login['host']})"
            )

        # Step 2: Preprocessing (depends on download if download was enabled)
        preprocessing_job_id = None
        preprocessing_enabled = getattr(dataset_config.preprocessing, "enabled", True)
        if preprocessing_enabled:
            preprocessing_job_id = self.run_preprocessing_step(
                dataset_config_name_or_path,
                workflow_config,
                dependency_job_id=download_job_id,
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
                dataset_config_name_or_path,
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
                dataset_config_name_or_path,
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
                dataset_config_name_or_path,
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
                dataset_config_name_or_path,
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
        self,
        dataset_config_name_or_path: str,
        workflow_config: DictConfig,
        force: bool = False,
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

        # Get output directory from resolved config
        base_dir = Path(workflow_config["output_directory"])

        self.workflow_logger = WorkflowLogger(
            base_dir, master_job_id, dataset_config_name_or_path, workflow_config
        )

        logger.info(
            f"Starting local workflow for dataset config: {dataset_config_name_or_path}"
        )
        logger.info(f"Output directory: {base_dir}")

        # Validate config synchronization unless forced
        # project_dir = workflow_config.get("project_directory")
        # self.validate_config_sync(
        #    dataset_config_name, force=force, project_directory=project_dir
        # )

        # Load the dataset config to check enabled flags
        dataset_config = self._load_dataset_config(dataset_config_name_or_path)

        # Log workflow start
        self.workflow_logger.log_workflow_start(dataset_config)

        logger.info(f"Dataset name: {dataset_config.dataset.name}")

        # Check if transfers are enabled - if so, stop execution
        transfers_enabled = workflow_config.get("enable_transfers", True)

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
                dataset_config_name_or_path, workflow_config
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
                dataset_config_name_or_path,
                workflow_config,
                dependency_job_id=download_job_id,
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
                dataset_config_name_or_path,
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
                dataset_config_name_or_path,
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
                dataset_config_name_or_path,
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
                dataset_config_name_or_path,
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

    def _load_dataset_config(self, dataset_config_name_or_path: str) -> DictConfig:
        """Load the dataset configuration using proper Hydra composition.

        Accepts either a config name (e.g., "dataset_cellxgene_pseudo_bulk_10k")
        or a file path (e.g., "conf/my_dataset.yaml" or "/absolute/path/to/config.yaml").

        Parameters
        ----------
        dataset_config_name_or_path : str
            Either a config name (without .yaml) or a path to a config file

        Returns
        -------
        DictConfig
            The loaded and transformed dataset configuration
        """
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        # Determine if input is a path or a name
        config_path_obj = Path(dataset_config_name_or_path)

        # Check if it's a path (has directory separators or is absolute)
        is_path = (
            "/" in dataset_config_name_or_path
            or "\\" in dataset_config_name_or_path
            or config_path_obj.is_absolute()
            or dataset_config_name_or_path.endswith(".yaml")
            or dataset_config_name_or_path.endswith(".yml")
        )

        if is_path:
            # It's a path - resolve it
            if config_path_obj.is_absolute():
                config_file = config_path_obj
            else:
                # Relative path - resolve relative to project root
                project_root = Path(__file__).resolve().parents[3]
                config_file = (project_root / config_path_obj).resolve()

            if not config_file.exists():
                raise ValueError(f"Dataset config file not found: {config_file}")

            # Extract config name from path for Hydra (without extension)
            config_name = config_file.stem
            config_dir = str(config_file.parent)

            logger.info(f"Loading dataset config from path: {config_file}")

            # Use Hydra's proper composition to handle defaults inheritance
            try:
                # Clear Hydra if already initialized (e.g., from previous config loading)
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    config = compose(config_name=config_name)
            except Exception as e:
                logger.error(f"Failed to load config using Hydra composition: {e}")
                logger.info("Falling back to OmegaConf.load() without defaults")
                # Fallback to direct loading if Hydra fails
                config = OmegaConf.load(config_file)
        else:
            # It's a name - use existing logic
            config_path = Path(__file__).resolve().parents[3] / "conf"
            config_file = config_path / f"{dataset_config_name_or_path}.yaml"

            if not config_file.exists():
                raise ValueError(f"Dataset config file not found: {config_file}")

            logger.info(
                f"Loading dataset config by name: {dataset_config_name_or_path}"
            )

            # Use Hydra's proper composition to handle defaults inheritance
            try:
                # Clear Hydra if already initialized (e.g., from previous config loading)
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                with initialize_config_dir(
                    config_dir=str(config_path), version_base=None
                ):
                    config = compose(config_name=dataset_config_name_or_path)
            except Exception as e:
                logger.error(f"Failed to load config using Hydra composition: {e}")
                logger.info("Falling back to OmegaConf.load() without defaults")
                # Fallback to direct loading if Hydra fails
                config = OmegaConf.load(config_file)

        # Apply transformations
        config = apply_all_transformations(config)

        return config

    def _load_dataset_config_with_base(
        self, dataset_config_name_or_path: str, base_file_path: str
    ) -> DictConfig:
        """Load dataset config and force-set base_file_path before transformations.

        Accepts either a config name or a file path (see _load_dataset_config for details).
        """
        # Use the main loader first
        config = self._load_dataset_config(dataset_config_name_or_path)

        # Force base_file_path into config prior to path transformations
        from omegaconf import OmegaConf as _OC

        cfg_container = _OC.to_container(config, resolve=True)
        cfg_container["base_file_path"] = str(base_file_path)
        config = _OC.create(cfg_container)
        # Apply transformations again with the new base_file_path
        config = apply_all_transformations(config)
        return config


def resolve_workflow_config(
    workflow_config: DictConfig, execution_mode: Optional[str] = None
) -> DictConfig:
    """
    Resolve workflow configuration values based on the new location-based format.

    This function processes the locations and transfer configuration to create
    a resolved config dict that can be passed to orchestrator methods.

    Parameters
    ----------
    workflow_config : DictConfig
        The workflow configuration section from the config
    execution_mode : Optional[str]
        Deprecated. For backward compatibility only.
        Use default_execution_location in config instead.

    Returns
    -------
    DictConfig
        Resolved configuration with location-specific values
    """
    from omegaconf import OmegaConf

    # Check for new location-based config format
    locations = workflow_config.get("locations", None)

    if locations is not None:
        # New location-based config format
        default_location = workflow_config.get("default_execution_location", "local")

        # Extract local location config for backward compatibility
        local_config = locations.get("local", {})
        cpu_config = locations.get("cpu", {})
        gpu_config = locations.get("gpu", {})

        # Build resolved config
        resolved = {
            "default_execution_location": default_location,
            # Locations config (pass through for DataTransfer)
            "locations": OmegaConf.to_container(locations, resolve=True),
            # Transfer config
            "transfer": OmegaConf.to_container(
                workflow_config.get("transfer", {}), resolve=True
            ),
            # Backward compatibility: use local location as default paths
            "output_directory": local_config.get("output_directory", "./outputs"),
            "project_directory": local_config.get("project_directory", "."),
            "base_file_path": local_config.get("base_file_path", "./data/RNA"),
            # CPU/GPU settings from location configs
            "cpu_partition": cpu_config.get("slurm_partition", "slurm"),
            "gpu_partition": gpu_config.get("slurm_partition", "gpu"),
            "cpu_node": cpu_config.get("node"),
            "gpu_node": gpu_config.get("node"),
            # Local execution settings
            "local_max_workers": local_config.get("max_workers", 4),
            "local_enable_gpu": local_config.get("enable_gpu", False),
            # SSH login info (reconstructed from location configs)
            "cpu_login": {
                "host": cpu_config.get("ssh_host"),
                "user": cpu_config.get("ssh_user"),
            }
            if cpu_config.get("ssh_host")
            else None,
            "gpu_login": {
                "host": gpu_config.get("ssh_host"),
                "user": gpu_config.get("ssh_user"),
            }
            if gpu_config.get("ssh_host")
            else None,
            # Poll and timeout settings
            "poll_interval": workflow_config.get("poll_interval", 120),
            "job_timeout": workflow_config.get("job_timeout", 0),
        }
    else:
        # Legacy config format - backward compatibility
        if execution_mode is None:
            execution_mode = workflow_config.get("execution_mode", "local")

        if execution_mode == "local":
            output_directory = workflow_config.get(
                "local_output_directory",
                str(Path(__file__).resolve().parents[3] / "outputs"),
            )
            project_directory = workflow_config.get("local_project_directory", ".")
            base_file_path = workflow_config.get("local_base_file_path", "./data/RNA")
        else:  # slurm
            output_directory = workflow_config.get(
                "slurm_output_directory", "/home/menger/git/adata_hf_datasets/outputs"
            )
            project_directory = workflow_config.get(
                "slurm_project_directory", "/home/menger/git/adata_hf_datasets"
            )
            base_file_path = workflow_config.get(
                "slurm_base_file_path", "/scratch/global/menger/data/RNA"
            )

        # Create legacy resolved config
        resolved = {
            "output_directory": output_directory,
            "project_directory": project_directory,
            "base_file_path": base_file_path,
            "execution_mode": execution_mode,
            "default_execution_location": "local"
            if execution_mode == "local"
            else "cpu",
            "cpu_partition": workflow_config.get("cpu_partition", "slurm"),
            "gpu_partition": workflow_config.get("gpu_partition", "gpu"),
            "cpu_node": workflow_config.get("cpu_node"),
            "gpu_node": workflow_config.get("gpu_node"),
            "venv_path": workflow_config.get("venv_path", ".venv"),
            "enable_transfers": workflow_config.get("enable_transfers", False),
            "local_max_workers": workflow_config.get("local_max_workers", 4),
            "local_enable_gpu": workflow_config.get("local_enable_gpu", False),
            "cpu_login": workflow_config.get("cpu_login"),
            "gpu_login": workflow_config.get("gpu_login"),
            "poll_interval": workflow_config.get("poll_interval", 120),
            "job_timeout": workflow_config.get("job_timeout", 0),
            # Empty locations/transfer for legacy mode
            "locations": None,
            "transfer": {"enabled": False},
        }

    return OmegaConf.create(resolved)


def create_orchestrator_from_config(config: DictConfig) -> WorkflowOrchestrator:
    workflow_config = config.get("workflow", {})
    # Prefer explicit execution_mode when present
    execution_mode = workflow_config.get("execution_mode", None)
    if execution_mode == "local":
        logger.info("Execution mode is 'local' - orchestrator (SSH/SLURM) not created")
        raise RuntimeError(
            "create_orchestrator_from_config called in local mode. Use run_workflow_localhost via run_workflow_master.py"
        )
    cpu_login = workflow_config.get("cpu_login")
    gpu_login = workflow_config.get("gpu_login")
    if not cpu_login:
        raise ValueError("CPU login configuration required for SSH mode")
    logger.info("Running in SSH/SLURM mode")
    return WorkflowOrchestrator(
        cpu_login=cpu_login,
        gpu_login=gpu_login,
    )


def run_workflow_localhost(
    dataset_config_name_or_path: str, workflow_config: DictConfig, force: bool = False
) -> None:
    """Run the complete workflow on the local machine (no SSH/SLURM).

    This mirrors the step ordering and logging of run_workflow_local, but executes
    each step via local subprocesses and bounded parallelism inside the step scripts.
    """
    import signal

    logger.info(
        f"Starting localhost workflow for dataset config: {dataset_config_name_or_path}"
    )

    # Initialize the workflow logger
    master_job_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Get output directory from resolved config
    base_dir = Path(workflow_config["output_directory"])
    # Resolve relative paths relative to project directory
    if not base_dir.is_absolute():
        project_dir = Path(workflow_config.get("project_directory", "."))
        if not project_dir.is_absolute():
            project_dir = Path(__file__).resolve().parents[3]
        base_dir = (project_dir / base_dir).resolve()

    workflow_logger = WorkflowLogger(
        base_dir, master_job_id, dataset_config_name_or_path, workflow_config
    )

    def project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    # Track active subprocesses for cleanup
    active_processes: List[subprocess.Popen] = []
    current_step_name: Optional[str] = None

    def signal_handler(signum, frame):
        """Handle termination signals by killing all subprocesses."""
        logger.warning(
            f"Received signal {signum}, terminating workflow and subprocesses..."
        )
        for proc in active_processes:
            if proc.poll() is None:  # Process is still running
                logger.info(
                    f"Terminating subprocess {proc.pid} (step: {current_step_name})"
                )
                try:
                    proc.terminate()
                    # Wait a bit for graceful termination
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing subprocess {proc.pid}")
                        proc.kill()
                except Exception as e:
                    logger.error(f"Error terminating subprocess {proc.pid}: {e}")
        # Re-raise the signal to exit
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    def run_logged(
        cmd: List[str],
        step: str,
        out_name: str,
        err_name: str,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        nonlocal current_step_name
        current_step_name = step
        job_id = master_job_id
        step_dir = workflow_logger.get_step_log_dir(step, job_id)
        step_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = step_dir / out_name
        stderr_path = step_dir / err_name
        logger.info(f"Executing local step '{step}': {' '.join(cmd)}")

        with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
            # Use Popen instead of run to track the process
            proc = subprocess.Popen(
                cmd,
                cwd=str(project_root()),
                env=env,
                text=True,
                stdout=out_f,
                stderr=err_f,
                start_new_session=False,  # Keep in same process group for signal propagation
            )
            active_processes.append(proc)
            try:
                result = proc.wait()
                active_processes.remove(proc)
            except KeyboardInterrupt:
                logger.warning(
                    f"Interrupted during step '{step}', terminating subprocess..."
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                active_processes.remove(proc)
                raise

        if result != 0:
            raise RuntimeError(f"Step {step} failed with exit code {result}")

    # Load dataset config to check flags
    # Reuse internal config loader
    try:
        # Ensure base path is available to config transformations
        os.environ["BASE_FILE_PATH"] = str(workflow_config["base_file_path"])
        # Create a minimal orchestrator just to access _load_dataset_config
        # We'll create a dummy instance to use its method
        temp_orch = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
        dataset_config = temp_orch._load_dataset_config(dataset_config_name_or_path)
    except Exception as e:
        logger.error(
            f"Failed to load dataset config {dataset_config_name_or_path}: {e}"
        )
        raise

    # Log workflow start
    workflow_logger.log_workflow_start(dataset_config)

    # Initialize data transfer if locations are configured
    data_transfer: Optional[DataTransfer] = None
    transfer_enabled = False
    locations_config = workflow_config.get("locations")
    transfer_config = workflow_config.get("transfer", {})

    if locations_config is not None and transfer_config.get("enabled", False):
        try:
            data_transfer = create_data_transfer_from_config(workflow_config)
            transfer_enabled = True
            logger.info(
                "Data transfer enabled - will transfer data between locations as needed"
            )

            # Test connectivity to configured locations
            connectivity = data_transfer.test_connectivity()
            for loc, connected in connectivity.items():
                if not connected:
                    logger.warning(
                        f"Cannot connect to {loc} - steps on this location may fail"
                    )
        except Exception as e:
            logger.warning(f"Failed to initialize data transfer: {e}")
            logger.warning("Continuing without automatic transfers")

    # Track execution location for transfer decisions
    default_location = workflow_config.get("default_execution_location", "local")
    previous_step_location: Optional[str] = None
    previous_step_outputs: List[str] = []

    def maybe_transfer_data(
        step_name: str,
        step_location: str,
    ) -> None:
        """Transfer data if the step runs on a different location than the previous step."""
        nonlocal previous_step_location, previous_step_outputs

        if not transfer_enabled or data_transfer is None:
            return

        if previous_step_location is None or previous_step_location == step_location:
            # No transfer needed
            return

        if not previous_step_outputs:
            logger.info(
                f"No data to transfer from {previous_step_location} to {step_location}"
            )
            return

        logger.info(
            f"Location changed: {previous_step_location} -> {step_location}, "
            f"transferring {len(previous_step_outputs)} path(s)"
        )

        for source_path in previous_step_outputs:
            try:
                target_path = data_transfer.translate_path(
                    source_path, previous_step_location, step_location
                )
                logger.info(f"Transferring: {source_path} -> {target_path}")

                stats = data_transfer.transfer(
                    source_location=previous_step_location,
                    target_location=step_location,
                    source_path=source_path,
                    target_path=target_path,
                )

                logger.info(
                    f"Transfer completed in {stats.total_time_seconds:.2f}s "
                    f"({stats.throughput_mbps:.2f} MB/s)"
                )
                if stats.errors:
                    for err in stats.errors:
                        logger.warning(f"Transfer warning: {err}")

            except DataTransferError as e:
                logger.error(f"Transfer failed for {source_path}: {e}")
                raise RuntimeError(f"Data transfer failed: {e}") from e

    def get_location_config(location_name: str) -> Optional[LocationConfig]:
        """Get LocationConfig for a location name."""
        if data_transfer is not None:
            return data_transfer.locations.get(location_name)
        return None

    env_base = os.environ.copy()
    env_base["DATASET_CONFIG"] = dataset_config_name_or_path
    env_base["WORKFLOW_DIR"] = str(workflow_logger.workflow_dir)
    # Allow local parallelism override
    local_max = str(workflow_config.get("local_max_workers", 4))
    env_base["LOCAL_MAX_WORKERS"] = local_max

    # Resolve default base_file_path if dataset config didn't specify
    if not dataset_config.get("base_file_path", None):
        dataset_config["base_file_path"] = workflow_config["base_file_path"]

    # Step: Download
    download_enabled = getattr(dataset_config, "download", None) is None or getattr(
        dataset_config.download, "enabled", True
    )
    if download_enabled:
        step_location = get_step_execution_location(
            "download", dataset_config, default_location
        )
        logger.info(f"Download step will run on: {step_location}")

        # Transfer data if location changed (download typically has no inputs)
        maybe_transfer_data("download", step_location)

        cmd = [
            sys.executable,
            "scripts/download/download_dataset.py",
            "--config-name",
            dataset_config_name_or_path,
            f"++hydra.run.dir={workflow_logger.get_step_log_dir('download', master_job_id)}",
        ]
        run_logged(cmd, "download", "download.out", "download.err", env=env_base)
        workflow_logger.log_step_complete("Download", master_job_id)

        # Update location tracking
        loc_config = get_location_config(step_location)
        if loc_config:
            previous_step_outputs = get_step_output_paths(
                "download", dataset_config, loc_config
            )
        previous_step_location = step_location
    else:
        workflow_logger.log_step_skipped("Download", "disabled in config")

    # Step: Preprocessing
    if getattr(dataset_config.preprocessing, "enabled", True):
        step_location = get_step_execution_location(
            "preprocessing", dataset_config, default_location
        )
        logger.info(f"Preprocessing step will run on: {step_location}")

        # Transfer data if location changed
        maybe_transfer_data("preprocessing", step_location)

        cmd = [
            sys.executable,
            "scripts/preprocessing/preprocess.py",
            "--config-name",
            dataset_config_name_or_path,
            f"++hydra.run.dir={workflow_logger.get_step_log_dir('preprocessing', master_job_id)}",
        ]
        run_logged(
            cmd, "preprocessing", "preprocessing.out", "preprocessing.err", env=env_base
        )
        workflow_logger.log_step_complete("Preprocessing", master_job_id)

        # Update location tracking
        loc_config = get_location_config(step_location)
        if loc_config:
            previous_step_outputs = get_step_output_paths(
                "preprocessing", dataset_config, loc_config
            )
        previous_step_location = step_location
    else:
        workflow_logger.log_step_skipped("Preprocessing", "disabled in config")

    # Step: Embedding Preparation
    if getattr(dataset_config.embedding_preparation, "enabled", True):
        step_location = get_step_execution_location(
            "embedding_preparation", dataset_config, default_location
        )
        logger.info(f"Embedding Preparation step will run on: {step_location}")

        # Transfer data if location changed
        maybe_transfer_data("embedding_preparation", step_location)

        env_prep = env_base.copy()
        env_prep["MODE"] = "cpu"
        cmd = [
            sys.executable,
            "scripts/embed/embed_launcher.py",
            "--config-name",
            dataset_config_name_or_path,
            "--mode",
            "cpu",
            "--backend",
            "local",
            "--prepare-only",
        ]
        run_logged(cmd, "embedding_prepare", "master.out", "master.err", env=env_prep)
        workflow_logger.log_step_complete("Embedding Preparation", master_job_id)

        # Update location tracking
        loc_config = get_location_config(step_location)
        if loc_config:
            previous_step_outputs = get_step_output_paths(
                "embedding_preparation", dataset_config, loc_config
            )
        previous_step_location = step_location
    else:
        workflow_logger.log_step_skipped("Embedding Preparation", "disabled in config")

    # Step: CPU Embedding
    if getattr(dataset_config.embedding_cpu, "enabled", True):
        step_location = get_step_execution_location(
            "embedding_cpu", dataset_config, default_location
        )
        logger.info(f"CPU Embedding step will run on: {step_location}")

        # Transfer data if location changed
        maybe_transfer_data("embedding_cpu", step_location)

        env_cpu = env_base.copy()
        env_cpu["MODE"] = "cpu"
        cmd = [
            sys.executable,
            "scripts/embed/embed_launcher.py",
            "--config-name",
            dataset_config_name_or_path,
            "--mode",
            "cpu",
            "--backend",
            "local",
        ]
        run_logged(cmd, "embedding", "cpu_master.out", "cpu_master.err", env=env_cpu)
        workflow_logger.log_step_complete("CPU Embedding", master_job_id)

        # Update location tracking
        loc_config = get_location_config(step_location)
        if loc_config:
            previous_step_outputs = get_step_output_paths(
                "embedding_cpu", dataset_config, loc_config
            )
        previous_step_location = step_location
    else:
        workflow_logger.log_step_skipped("CPU Embedding", "disabled in config")

    # Step: GPU Embedding (local backend)
    if getattr(dataset_config.embedding_gpu, "enabled", True):
        step_location = get_step_execution_location(
            "embedding_gpu", dataset_config, default_location
        )
        logger.info(f"GPU Embedding step will run on: {step_location}")

        # Check if GPU embedding is enabled for local backend
        local_gpu_enabled = workflow_config.get("local_enable_gpu", False)
        # If location is "local", respect the local_enable_gpu setting
        # If location is "gpu" (remote), allow it to run on the GPU cluster
        if step_location == "local" and not local_gpu_enabled:
            logger.info(
                "GPU embedding enabled in config but disabled for local backend (local_enable_gpu=false)"
            )
            workflow_logger.log_step_skipped(
                "GPU Embedding", "disabled for local backend"
            )
        else:
            # Transfer data if location changed
            maybe_transfer_data("embedding_gpu", step_location)

            env_gpu = env_base.copy()
            env_gpu["MODE"] = "gpu"
            cmd = [
                sys.executable,
                "scripts/embed/embed_launcher.py",
                "--config-name",
                dataset_config_name_or_path,
                "--mode",
                "gpu",
                "--backend",
                "local",
            ]
            run_logged(
                cmd, "embedding", "gpu_master.out", "gpu_master.err", env=env_gpu
            )
            workflow_logger.log_step_complete("GPU Embedding", master_job_id)

            # Update location tracking
            loc_config = get_location_config(step_location)
            if loc_config:
                previous_step_outputs = get_step_output_paths(
                    "embedding_gpu", dataset_config, loc_config
                )
            previous_step_location = step_location
    else:
        workflow_logger.log_step_skipped("GPU Embedding", "disabled in config")

    # Step: Dataset Creation
    if getattr(dataset_config.dataset_creation, "enabled", True):
        from omegaconf import ListConfig

        step_location = get_step_execution_location(
            "dataset_creation", dataset_config, default_location
        )
        logger.info(f"Dataset Creation step will run on: {step_location}")

        # Transfer data if location changed
        maybe_transfer_data("dataset_creation", step_location)

        cs_lengths = dataset_config.dataset_creation.cs_length
        cs_values = (
            list(cs_lengths)
            if isinstance(cs_lengths, (list, tuple, ListConfig))
            else [cs_lengths]
        )
        caption_keys = dataset_config.dataset_creation.get("caption_keys", None)
        if caption_keys is None:
            caption_values = [dataset_config.get("caption_key", None)]
        else:
            caption_values = (
                list(caption_keys)
                if isinstance(caption_keys, (list, tuple, ListConfig))
                else [caption_keys]
            )

        for idx, (cs_len, cap_key_name) in enumerate(
            [(c, k) for c in cs_values for k in caption_values]
        ):
            # Resolve cap_key_name to value from top-level config if it's a param name
            cap_value = (
                getattr(dataset_config, cap_key_name)
                if cap_key_name and hasattr(dataset_config, cap_key_name)
                else cap_key_name
            )
            env_dc = env_base.copy()
            if cs_len is not None:
                env_dc["CS_LENGTH_OVERRIDE"] = str(cs_len)
            if cap_value is not None:
                env_dc["CAPTION_KEY_OVERRIDE"] = str(cap_value)
            cmd = [
                sys.executable,
                "scripts/dataset_creation/create_dataset.py",
                "--config-name",
                dataset_config_name_or_path,
                f"++hydra.run.dir={workflow_logger.get_step_log_dir('dataset_creation', master_job_id) / f'job_{idx}'}",
            ]
            # Also pass as CLI overrides for transparency
            if cs_len is not None:
                cmd.append(f"++dataset_creation.cs_length={cs_len}")
            if cap_value is not None:
                cmd.append(f"++caption_key={cap_value}")
            run_logged(
                cmd,
                "dataset_creation",
                f"create_ds_{idx}.out",
                f"create_ds_{idx}.err",
                env=env_dc,
            )

        workflow_logger.log_step_complete("Dataset Creation", master_job_id)

        # Update location tracking
        loc_config = get_location_config(step_location)
        if loc_config:
            previous_step_outputs = get_step_output_paths(
                "dataset_creation", dataset_config, loc_config
            )
        previous_step_location = step_location
    else:
        workflow_logger.log_step_skipped("Dataset Creation", "disabled in config")

    workflow_logger.log_workflow_complete()


def run_workflow_unified(
    dataset_config_name_or_path: str,
    workflow_config: DictConfig,
    force: bool = False,
    workflow_id: Optional[str] = None,
) -> bool:
    """
    Run a unified workflow that respects per-step execution locations.

    This is a wrapper around the new WorkflowRunner that provides backward
    compatibility while enabling per-step execution location support.

    Unlike run_workflow_localhost (which runs all steps locally) or
    WorkflowOrchestrator.run_workflow (which submits all steps to SLURM),
    this function respects the execution_location setting in each step's
    configuration and can mix local and remote execution.

    Parameters
    ----------
    dataset_config_name_or_path : str
        Name or path of the dataset configuration
    workflow_config : DictConfig
        The resolved workflow configuration (from resolve_workflow_config)
    force : bool
        Whether to skip certain validations
    workflow_id : Optional[str]
        Optional workflow ID for the output directory (if None, auto-generated)

    Returns
    -------
    bool
        True if workflow completed successfully

    Examples
    --------
    >>> # Load and resolve config
    >>> workflow_cfg = OmegaConf.load("conf/workflow_orchestrator.yaml")
    >>> resolved = resolve_workflow_config(workflow_cfg.workflow)
    >>>
    >>> # Run workflow with per-step execution locations
    >>> success = run_workflow_unified("my_dataset", resolved)
    """
    return run_unified_workflow(
        dataset_config_name_or_path,
        workflow_config,
        force=force,
        workflow_id=workflow_id,
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
