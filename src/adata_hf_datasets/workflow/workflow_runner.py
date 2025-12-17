#!/usr/bin/env python3
"""
Unified Workflow Runner for Per-Step Execution Locations

This module provides a unified workflow runner that can execute steps
on different locations (local, cpu cluster, gpu cluster) based on
per-step configuration.

Key Features:
- Per-step execution location (respects execution_location in dataset config)
- Automatic data transfer between locations when steps run on different machines
- Centralized log collection (all logs end up locally regardless of where steps ran)
- Unified interface for local subprocess and remote SLURM execution
"""

import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

from .data_transfer import (
    DataTransfer,
    DataTransferError,
    LocationConfig,
)
from .executors import (
    ExecutionResult,
    StepExecutor,
    create_executor,
)
from .log_retrieval import (
    ConsolidatedLogWriter,
    LogParser,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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
    # Fall back to top-level split_dataset if preprocessing doesn't have it
    split_dataset = dataset_config.preprocessing.get(
        "split_dataset", dataset_config.get("split_dataset", True)
    )
    # split_dataset=True means training data (uses train/), False means test data (uses test/)
    split_subdir = "train" if split_dataset else "test"

    if step_name == "download":
        return [f"{base_path}/raw/{split_subdir}/{dataset_name}.h5ad"]

    elif step_name == "preprocessing":
        return [f"{base_path}/processed/{split_subdir}/{dataset_name}"]

    elif step_name in ["embedding_preparation", "embedding_cpu", "embedding_gpu"]:
        return [f"{base_path}/processed_with_emb/{split_subdir}/{dataset_name}"]

    elif step_name == "dataset_creation":
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
        return []

    elif step_name == "preprocessing":
        return [f"{base_path}/raw/{split_subdir}/{dataset_name}.h5ad"]

    elif step_name == "embedding_preparation":
        return [f"{base_path}/processed/{split_subdir}/{dataset_name}"]

    elif step_name in ["embedding_cpu", "embedding_gpu"]:
        return [f"{base_path}/processed_with_emb/{split_subdir}/{dataset_name}"]

    elif step_name == "dataset_creation":
        return [f"{base_path}/processed_with_emb/{split_subdir}/{dataset_name}"]

    return []


def _is_step_enabled(step_name: str, dataset_config: DictConfig) -> bool:
    """Check if a step is enabled in the dataset config."""
    section_name = STEP_TO_CONFIG_SECTION.get(step_name)
    if section_name and hasattr(dataset_config, section_name):
        section = getattr(dataset_config, section_name)
        return getattr(section, "enabled", True)
    return True


# =============================================================================
# WORKFLOW RUNNER
# =============================================================================


class WorkflowRunner:
    """
    Unified workflow runner supporting per-step execution locations.

    This class handles:
    - Executing steps on local, CPU cluster, or GPU cluster based on config
    - Automatic data transfer between locations
    - Centralized log collection
    """

    def __init__(
        self,
        workflow_config: DictConfig,
        dataset_config_name_or_path: str,
        workflow_id: Optional[str] = None,
    ):
        """
        Initialize the workflow runner.

        Parameters
        ----------
        workflow_config : DictConfig
            The resolved workflow configuration
        dataset_config_name_or_path : str
            Name or path of the dataset configuration
        workflow_id : Optional[str]
            Optional workflow ID (if None, auto-generated)
        """
        self.workflow_config = workflow_config
        self.dataset_config_name = dataset_config_name_or_path
        self.default_location = workflow_config.get(
            "default_execution_location", "local"
        )

        # Create workflow directory
        self.workflow_id = (
            workflow_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._setup_workflow_directory()

        # Initialize locations and executors
        self.locations: Dict[str, LocationConfig] = {}
        self.executors: Dict[str, Optional[StepExecutor]] = {}
        self._build_locations()
        self._build_executors()

        # Initialize data transfer if enabled
        self.data_transfer: Optional[DataTransfer] = None
        self._setup_data_transfer()

        # Initialize log handling
        self.log_writer = ConsolidatedLogWriter(self.workflow_dir)
        self.log_parser = LogParser()

        # Track execution state
        self.previous_location: Optional[str] = None
        self.previous_outputs: List[str] = []
        self.step_results: Dict[str, ExecutionResult] = {}

        # Track active processes for cleanup
        self._active_processes: List[subprocess.Popen] = []
        self._setup_signal_handlers()

    def _setup_workflow_directory(self) -> None:
        """Create the workflow directory structure."""
        output_dir = self.workflow_config.get("output_directory", "./outputs")
        if not Path(output_dir).is_absolute():
            project_dir = self.workflow_config.get("project_directory", ".")
            if not Path(project_dir).is_absolute():
                project_dir = Path(__file__).resolve().parents[3]
            output_dir = Path(project_dir) / output_dir

        date_str = datetime.now().strftime("%Y-%m-%d")
        self.workflow_dir = Path(output_dir) / date_str / self.workflow_id
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

        # Create step directories
        for step in WORKFLOW_STEPS:
            (self.workflow_dir / step).mkdir(parents=True, exist_ok=True)

        logger.info(f"Created workflow directory: {self.workflow_dir}")

    def _build_locations(self) -> None:
        """Build LocationConfig objects from workflow config."""
        locations_config = self.workflow_config.get("locations")

        if locations_config is None:
            # Legacy mode - create minimal local config
            self.locations["local"] = LocationConfig(
                name="local",
                base_file_path=self.workflow_config.get("base_file_path", "./data/RNA"),
                project_directory=self.workflow_config.get("project_directory", "."),
                venv_path=self.workflow_config.get("venv_path", ".venv"),
                output_directory=self.workflow_config.get(
                    "output_directory", "./outputs"
                ),
            )
            return

        # Build from new locations config
        for name, loc_config in locations_config.items():
            if isinstance(loc_config, dict):
                loc_config = OmegaConf.create(loc_config)
            self.locations[name] = LocationConfig.from_config(name, loc_config)

    def _build_executors(self) -> None:
        """Create executors for each configured location."""
        poll_interval = self.workflow_config.get("poll_interval", 60)
        job_timeout = self.workflow_config.get("job_timeout", 0)

        for name, loc_config in self.locations.items():
            try:
                self.executors[name] = create_executor(
                    location_name=name,
                    location_config=loc_config,
                    workflow_dir=self.workflow_dir,
                    poll_interval=poll_interval,
                    job_timeout=job_timeout,
                )
                logger.info(f"Created executor for location: {name}")
            except Exception as e:
                logger.warning(f"Failed to create executor for {name}: {e}")
                self.executors[name] = None

    def _setup_data_transfer(self) -> None:
        """Initialize data transfer if enabled (connectivity check deferred)."""
        transfer_config = self.workflow_config.get("transfer", {})

        if not transfer_config.get("enabled", False):
            logger.info("Data transfer disabled")
            return

        if not self.locations:
            logger.warning("No locations configured - data transfer disabled")
            return

        try:
            self.data_transfer = DataTransfer(self.locations, transfer_config)
            # Note: Connectivity check is deferred to run_workflow() when we know
            # which locations are actually used
        except Exception as e:
            logger.warning(f"Failed to initialize data transfer: {e}")
            self.data_transfer = None

    def _get_used_locations(self, dataset_config: DictConfig) -> set:
        """
        Determine which locations are actually used by enabled steps.

        Parameters
        ----------
        dataset_config : DictConfig
            The dataset configuration

        Returns
        -------
        set
            Set of location names that are used by enabled steps
        """
        used_locations = set()

        for step in WORKFLOW_STEPS:
            step_config = getattr(dataset_config, step, None)
            if step_config is None:
                continue

            # Check if step is enabled
            enabled = getattr(step_config, "enabled", True)
            if not enabled:
                continue

            # Get execution location
            location = get_step_execution_location(
                step, dataset_config, self.default_location
            )
            used_locations.add(location)

        return used_locations

    def _check_connectivity(self, used_locations: set) -> None:
        """
        Check connectivity only to locations that are actually used.

        Parameters
        ----------
        used_locations : set
            Set of location names to check connectivity for
        """
        if self.data_transfer is None:
            return

        # Only check remote locations that are actually used
        remote_locations_to_check = {
            loc
            for loc in used_locations
            if loc in self.locations and self.locations[loc].is_remote
        }

        if not remote_locations_to_check:
            logger.info("All steps run locally - skipping remote connectivity check")
            return

        logger.info(f"Checking connectivity to: {', '.join(remote_locations_to_check)}")

        # Test connectivity only to used remote locations
        connectivity = self.data_transfer.test_connectivity(remote_locations_to_check)
        for loc, connected in connectivity.items():
            if not connected:
                logger.warning(
                    f"Cannot connect to {loc} - steps on this location may fail"
                )
            else:
                logger.info(f"Connectivity to {loc}: OK")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful termination."""

        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, terminating workflow...")
            self._cleanup_all()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _cleanup_all(self) -> None:
        """Terminate all active processes and jobs (local and remote)."""
        logger.info("Cleaning up all running processes and jobs...")

        # Terminate all executors (this handles both local processes and remote SLURM jobs)
        for name, executor in self.executors.items():
            if executor is not None:
                try:
                    logger.info(f"Terminating executor for {name}...")
                    executor.terminate()
                except Exception as e:
                    logger.error(f"Error terminating executor {name}: {e}")

        # Also terminate any tracked processes (legacy)
        for proc in self._active_processes:
            if proc.poll() is None:
                logger.info(f"Terminating subprocess {proc.pid}")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing subprocess {proc.pid}")
                    proc.kill()
                except Exception as e:
                    logger.error(f"Error terminating subprocess: {e}")

        logger.info("Cleanup complete")

    def _transfer_data_if_needed(
        self,
        step_name: str,
        step_location: str,
    ) -> None:
        """
        Transfer data if the step runs on a different location than the previous step.

        Parameters
        ----------
        step_name : str
            Name of the step about to run
        step_location : str
            Location where the step will run
        """
        if self.data_transfer is None:
            return

        if self.previous_location is None or self.previous_location == step_location:
            return

        if not self.previous_outputs:
            logger.info(
                f"No data to transfer from {self.previous_location} to {step_location}"
            )
            return

        logger.info(
            f"Location changed: {self.previous_location} -> {step_location}, "
            f"transferring {len(self.previous_outputs)} path(s)"
        )

        for source_path in self.previous_outputs:
            try:
                target_path = self.data_transfer.translate_path(
                    source_path, self.previous_location, step_location
                )
                logger.info(f"Transferring: {source_path} -> {target_path}")

                stats = self.data_transfer.transfer(
                    source_location=self.previous_location,
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

    def _build_step_command(
        self,
        step_name: str,
        dataset_config: DictConfig,
        step_location: str,
        **kwargs,
    ) -> List[str]:
        """
        Build the command to execute a workflow step.

        Parameters
        ----------
        step_name : str
            Name of the workflow step
        dataset_config : DictConfig
            The dataset configuration
        step_location : str
            Where the step will run
        **kwargs
            Additional arguments for specific steps

        Returns
        -------
        List[str]
            Command and arguments
        """
        loc_config = self.locations.get(step_location)
        if loc_config is None:
            raise ValueError(f"Unknown location: {step_location}")

        # For remote execution, use the remote python path
        if loc_config.is_remote:
            python_path = f"{loc_config.venv_path}/bin/python"
        else:
            python_path = sys.executable

        step_log_dir = self.workflow_dir / step_name

        if step_name == "download":
            return [
                python_path,
                "scripts/download/download_dataset.py",
                "--config-name",
                self.dataset_config_name,
                f"++hydra.run.dir={step_log_dir}",
            ]

        elif step_name == "preprocessing":
            return [
                python_path,
                "scripts/preprocessing/preprocess.py",
                "--config-name",
                self.dataset_config_name,
                f"++hydra.run.dir={step_log_dir}",
            ]

        elif step_name == "embedding_preparation":
            return [
                python_path,
                "scripts/embed/embed_launcher.py",
                "--config-name",
                self.dataset_config_name,
                "--mode",
                "cpu",
                "--backend",
                "local" if not loc_config.is_remote else "slurm",
                "--prepare-only",
            ]

        elif step_name == "embedding_cpu":
            return [
                python_path,
                "scripts/embed/embed_launcher.py",
                "--config-name",
                self.dataset_config_name,
                "--mode",
                "cpu",
                "--backend",
                "local" if not loc_config.is_remote else "slurm",
            ]

        elif step_name == "embedding_gpu":
            return [
                python_path,
                "scripts/embed/embed_launcher.py",
                "--config-name",
                self.dataset_config_name,
                "--mode",
                "gpu",
                "--backend",
                "local" if not loc_config.is_remote else "slurm",
            ]

        elif step_name == "dataset_creation":
            cs_length = kwargs.get("cs_length")
            caption_key = kwargs.get("caption_key")
            job_idx = kwargs.get("job_idx", 0)

            cmd = [
                python_path,
                "scripts/dataset_creation/create_dataset.py",
                "--config-name",
                self.dataset_config_name,
                f"++hydra.run.dir={step_log_dir / f'job_{job_idx}'}",
            ]

            if cs_length is not None:
                cmd.append(f"++dataset_creation.cs_length={cs_length}")
            if caption_key is not None:
                cmd.append(f"++caption_key={caption_key}")

            return cmd

        else:
            raise ValueError(f"Unknown step: {step_name}")

    def _build_step_env(self, step_name: str, step_location: str) -> Dict[str, str]:
        """Build environment variables for a step."""
        env = os.environ.copy()
        env["DATASET_CONFIG"] = self.dataset_config_name
        env["WORKFLOW_DIR"] = str(self.workflow_dir)

        loc_config = self.locations.get(step_location)
        if loc_config:
            env["BASE_FILE_PATH"] = loc_config.base_file_path
            env["PROJECT_DIR"] = loc_config.project_directory

        # Local parallelism setting
        local_max = str(self.workflow_config.get("local_max_workers", 4))
        env["LOCAL_MAX_WORKERS"] = local_max

        return env

    def run_step(
        self,
        step_name: str,
        dataset_config: DictConfig,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute a single workflow step.

        Parameters
        ----------
        step_name : str
            Name of the step to execute
        dataset_config : DictConfig
            The dataset configuration
        **kwargs
            Additional arguments for specific steps

        Returns
        -------
        ExecutionResult
            Result of the execution
        """
        # Determine execution location
        step_location = get_step_execution_location(
            step_name, dataset_config, self.default_location
        )
        logger.info(f"Executing {step_name} on {step_location}")

        # Get executor
        executor = self.executors.get(step_location)
        if executor is None:
            raise RuntimeError(f"No executor available for location: {step_location}")

        # Transfer data if location changed
        self._transfer_data_if_needed(step_name, step_location)

        # Build command and environment
        cmd = self._build_step_command(
            step_name, dataset_config, step_location, **kwargs
        )
        env = self._build_step_env(step_name, step_location)

        # Execute
        result = executor.execute(
            cmd=cmd,
            step_name=step_name,
            env=env,
            cwd=self.locations[step_location].project_directory,
        )

        # Store result and update tracking
        self.step_results[step_name] = result

        loc_config = self.locations.get(step_location)
        if loc_config:
            self.previous_outputs = get_step_output_paths(
                step_name, dataset_config, loc_config
            )
        self.previous_location = step_location

        # Write log summary
        summary = self.log_parser.create_summary(
            step_name=step_name,
            location=step_location,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            success=result.success,
        )
        self.log_writer.write_step_summary(summary)

        if not result.success:
            logger.error(f"Step {step_name} failed: {result.error_message}")

        return result

    def run_workflow(self, dataset_config: DictConfig, force: bool = False) -> bool:
        """
        Run the complete workflow.

        Parameters
        ----------
        dataset_config : DictConfig
            The loaded dataset configuration
        force : bool
            Whether to skip certain validations

        Returns
        -------
        bool
            True if workflow completed successfully
        """
        import time

        start_time = time.time()
        workflow_success = True

        # Determine which locations are actually used and check connectivity
        used_locations = self._get_used_locations(dataset_config)
        logger.info(
            f"Locations used in this workflow: {', '.join(sorted(used_locations))}"
        )
        self._check_connectivity(used_locations)

        # Log workflow start
        self.log_writer.write_workflow_start(
            dataset_name=dataset_config.dataset.name,
            workflow_id=self.workflow_id,
        )

        logger.info("=" * 80)
        logger.info("WORKFLOW STARTED")
        logger.info(f"Workflow ID: {self.workflow_id}")
        logger.info(f"Dataset: {dataset_config.dataset.name}")
        logger.info(f"Workflow Directory: {self.workflow_dir}")
        logger.info("=" * 80)

        try:
            # Download
            if _is_step_enabled("download", dataset_config):
                result = self.run_step("download", dataset_config)
                if not result.success:
                    raise RuntimeError(f"Download step failed: {result.error_message}")
            else:
                logger.info("Download step skipped (disabled in config)")

            # Preprocessing
            if _is_step_enabled("preprocessing", dataset_config):
                result = self.run_step("preprocessing", dataset_config)
                if not result.success:
                    raise RuntimeError(
                        f"Preprocessing step failed: {result.error_message}"
                    )
            else:
                logger.info("Preprocessing step skipped (disabled in config)")

            # Embedding Preparation
            if _is_step_enabled("embedding_preparation", dataset_config):
                result = self.run_step("embedding_preparation", dataset_config)
                if not result.success:
                    raise RuntimeError(
                        f"Embedding preparation failed: {result.error_message}"
                    )
            else:
                logger.info("Embedding preparation skipped (disabled in config)")

            # CPU Embedding
            if _is_step_enabled("embedding_cpu", dataset_config):
                result = self.run_step("embedding_cpu", dataset_config)
                if not result.success:
                    raise RuntimeError(f"CPU embedding failed: {result.error_message}")
            else:
                logger.info("CPU embedding skipped (disabled in config)")

            # GPU Embedding
            if _is_step_enabled("embedding_gpu", dataset_config):
                step_location = get_step_execution_location(
                    "embedding_gpu", dataset_config, self.default_location
                )
                local_gpu_enabled = self.workflow_config.get("local_enable_gpu", False)

                if step_location == "local" and not local_gpu_enabled:
                    logger.info("GPU embedding skipped (disabled for local backend)")
                else:
                    result = self.run_step("embedding_gpu", dataset_config)
                    if not result.success:
                        raise RuntimeError(
                            f"GPU embedding failed: {result.error_message}"
                        )
            else:
                logger.info("GPU embedding skipped (disabled in config)")

            # Dataset Creation (may have multiple combinations)
            if _is_step_enabled("dataset_creation", dataset_config):
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

                job_idx = 0
                for cs_len in cs_values:
                    for cap_key in caption_values:
                        # Resolve caption key if it's a reference
                        cap_value = (
                            getattr(dataset_config, cap_key)
                            if cap_key and hasattr(dataset_config, cap_key)
                            else cap_key
                        )

                        result = self.run_step(
                            "dataset_creation",
                            dataset_config,
                            cs_length=cs_len,
                            caption_key=cap_value,
                            job_idx=job_idx,
                        )

                        if not result.success:
                            raise RuntimeError(
                                f"Dataset creation failed (cs_length={cs_len}, caption={cap_value}): "
                                f"{result.error_message}"
                            )

                        job_idx += 1
            else:
                logger.info("Dataset creation skipped (disabled in config)")

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            workflow_success = False

        # Log workflow completion
        total_time = time.time() - start_time
        self.log_writer.write_workflow_complete(workflow_success, total_time)

        logger.info("=" * 80)
        if workflow_success:
            logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        else:
            logger.info("WORKFLOW COMPLETED WITH ERRORS")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Logs available at: {self.workflow_dir}")
        logger.info("=" * 80)

        return workflow_success


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_workflow_runner(
    workflow_config: DictConfig,
    dataset_config_name_or_path: str,
    workflow_id: Optional[str] = None,
) -> WorkflowRunner:
    """
    Create a WorkflowRunner from configuration.

    Parameters
    ----------
    workflow_config : DictConfig
        The resolved workflow configuration
    dataset_config_name_or_path : str
        Name or path of the dataset configuration
    workflow_id : Optional[str]
        Optional workflow ID (if None, auto-generated)

    Returns
    -------
    WorkflowRunner
        Configured workflow runner
    """
    return WorkflowRunner(
        workflow_config, dataset_config_name_or_path, workflow_id=workflow_id
    )


def run_unified_workflow(
    dataset_config_name_or_path: str,
    workflow_config: DictConfig,
    force: bool = False,
    workflow_id: Optional[str] = None,
) -> bool:
    """
    Run a unified workflow that respects per-step execution locations.

    This is the main entry point for the new unified workflow execution.

    Parameters
    ----------
    dataset_config_name_or_path : str
        Name or path of the dataset configuration
    workflow_config : DictConfig
        The resolved workflow configuration
    workflow_id : Optional[str]
        Optional workflow ID for the output directory (if None, auto-generated)
    force : bool
        Whether to skip certain validations

    Returns
    -------
    bool
        True if workflow completed successfully
    """
    from .config_utils import apply_all_transformations

    # Load dataset config
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Determine if input is a path or a name
    config_path_obj = Path(dataset_config_name_or_path)
    is_path = (
        "/" in dataset_config_name_or_path
        or "\\" in dataset_config_name_or_path
        or config_path_obj.is_absolute()
        or dataset_config_name_or_path.endswith(".yaml")
        or dataset_config_name_or_path.endswith(".yml")
    )

    if is_path:
        if config_path_obj.is_absolute():
            config_file = config_path_obj
        else:
            project_root = Path(__file__).resolve().parents[3]
            config_file = (project_root / config_path_obj).resolve()

        if not config_file.exists():
            raise ValueError(f"Dataset config file not found: {config_file}")

        config_name = config_file.stem
        config_dir = str(config_file.parent)
    else:
        config_path = Path(__file__).resolve().parents[3] / "conf"
        config_file = config_path / f"{dataset_config_name_or_path}.yaml"

        if not config_file.exists():
            raise ValueError(f"Dataset config file not found: {config_file}")

        config_name = dataset_config_name_or_path
        config_dir = str(config_path)

    # Load config with Hydra
    try:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            dataset_config = compose(config_name=config_name)
    except Exception as e:
        logger.error(f"Failed to load config with Hydra: {e}")
        dataset_config = OmegaConf.load(config_file)

    # Apply transformations
    dataset_config = apply_all_transformations(dataset_config)

    # Set base_file_path if not specified
    if not dataset_config.get("base_file_path"):
        dataset_config["base_file_path"] = workflow_config.get(
            "base_file_path", "./data/RNA"
        )

    # Create and run workflow
    runner = create_workflow_runner(
        workflow_config, dataset_config_name_or_path, workflow_id=workflow_id
    )
    return runner.run_workflow(dataset_config, force=force)
