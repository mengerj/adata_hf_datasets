#!/usr/bin/env python3
"""
Embedding Launcher - Simplified Configuration & Job Submission

This script consolidates the configuration loading and SLURM array job submission
for embedding tasks. It replaces the complex chain of Python->Bash->Python calls
with a clean Python-only approach.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig

from adata_hf_datasets.config_utils import apply_all_transformations

# Add src directory to Python path for imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)

# Add error handler for central error log if WORKFLOW_DIR is set
WORKFLOW_DIR = os.environ.get("WORKFLOW_DIR")
if WORKFLOW_DIR:
    error_log_path = os.path.join(WORKFLOW_DIR, "logs", "errors_consolidated.log")
    try:
        # Ensure the directory exists before creating the log file
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        error_handler = logging.FileHandler(error_log_path, mode="a")
        error_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        error_handler.setFormatter(formatter)
        logging.getLogger().addHandler(error_handler)
    except Exception as e:
        # Don't fail if we can't set up the error log - just continue without it
        print(f"Warning: Could not set up error logging to {error_log_path}: {e}")
        print("Continuing without centralized error logging...")


class EmbeddingLauncher:
    """Handles configuration loading and SLURM array job submission for embedding tasks."""

    def __init__(
        self, config_name: str, mode: str = "auto", prepare_only: bool = False
    ):
        """
        Initialize the embedding launcher.

        Parameters
        ----------
        config_name : str
            Name of the dataset configuration to use
        mode : str
            Processing mode: "cpu", "gpu", or "auto"
        prepare_only : bool
            If True, run only the preparation step
        """
        self.config_name = config_name
        self.mode = self._determine_mode(mode)
        self.prepare_only = prepare_only
        self.config = self._load_config()
        self.job_ids: List[int] = []
        # No longer need to track temp files since we use environment variables

    def _determine_mode(self, mode: str) -> str:
        """Determine the processing mode based on input and environment."""
        if mode != "auto":
            return mode

        # Check environment variable
        env_mode = os.environ.get("MODE", "cpu")
        logger.info(f"Auto-detected mode from environment: {env_mode}")
        return env_mode

    def _load_config(self) -> DictConfig:
        """Load and validate the dataset configuration using proper Hydra composition."""
        from hydra import compose, initialize_config_dir

        config_path = project_root / "conf"
        logger.info(f"Loading config '{self.config_name}' from {config_path}")

        try:
            with initialize_config_dir(config_dir=str(config_path), version_base=None):
                cfg = compose(config_name=self.config_name)

            # Apply transformations
            cfg = apply_all_transformations(cfg)
            logger.info(f"Successfully loaded config for dataset: {cfg.dataset.name}")

            # Debug: Log the base_file_path to verify it's loaded correctly
            base_path = cfg.get("base_file_path", "NOT_FOUND")
            logger.info(f"Loaded base_file_path from config: {base_path}")

            return cfg

        except Exception as e:
            logger.error(f"Failed to load config '{self.config_name}': {e}")
            raise

    def _get_embedding_config(self) -> DictConfig:
        """Get the appropriate embedding configuration based on mode and prepare_only."""
        if self.prepare_only:
            # For embedding preparation, use embedding_preparation config if available
            if (
                hasattr(self.config, "embedding_preparation")
                and self.config.embedding_preparation is not None
            ):
                logger.info("Using embedding_preparation configuration")
                return self.config.embedding_preparation
            else:
                # Fallback to CPU config for preparation
                logger.info(
                    "Using CPU embedding configuration for preparation (fallback)"
                )
                return self.config.embedding_cpu

        # For actual embedding, use mode-specific config
        if self.mode == "cpu":
            logger.info("Using CPU embedding configuration")
            return self.config.embedding_cpu
        elif self.mode == "gpu":
            logger.info("Using GPU embedding configuration")
            return self.config.embedding_gpu
        else:
            # Fallback to legacy embedding config
            if hasattr(self.config, "embedding"):
                logger.info("Using legacy embedding configuration")
                return self.config.embedding
            else:
                raise ValueError(
                    f"Unknown mode: {self.mode} and no legacy embedding config found"
                )

    def _get_input_directories(self) -> List[Tuple[str, Path]]:
        """Get list of input directories to process."""
        # Use base_file_path from config with processed subdirectory
        # Environment variable takes precedence (passed from workflow orchestrator)
        env_base_path = os.environ.get("BASE_FILE_PATH")
        config_base_path = self.config.get("base_file_path")
        base_file_path = env_base_path or config_base_path

        # Debug logging
        logger.info(f"Environment BASE_FILE_PATH: {env_base_path}")
        logger.info(f"Config base_file_path: {config_base_path}")
        logger.info(f"Final base_file_path: {base_file_path}")

        # Persist the resolved base path for other methods (e.g., sbatch/env)
        self.resolved_base_file_path = str(base_file_path)
        base_dir = Path(base_file_path) / "processed"
        dataset_name = self.config.dataset.name

        # Determine if we're processing train or test data
        split_dataset = self.config.preprocessing.get("split_dataset", True)
        train_or_test = "train" if split_dataset else "test"

        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Split dataset: {split_dataset} -> Looking for: {train_or_test}")
        logger.info(f"Base directory: {base_dir}")

        directories = []

        if train_or_test == "test":
            test_dir = base_dir / "test" / dataset_name / "all"
            if test_dir.exists():
                directories.append(("test", test_dir))
                logger.info(f"Found test directory: {test_dir}")
        else:
            # Process train and validation directories
            train_dir = base_dir / "train" / dataset_name / "train"
            val_dir = base_dir / "train" / dataset_name / "val"

            if train_dir.exists():
                directories.append(("train", train_dir))
                logger.info(f"Found train directory: {train_dir}")

            if val_dir.exists():
                directories.append(("val", val_dir))
                logger.info(f"Found validation directory: {val_dir}")

        if not directories:
            raise ValueError(f"No input directories found for dataset {dataset_name}")

        return directories

    def _count_zarr_files(self, directory: Path) -> int:
        """Count the number of .zarr files in a directory."""
        try:
            zarr_files = list(directory.glob("*.zarr"))
            return len(zarr_files)
        except Exception as e:
            logger.warning(f"Could not count zarr files in {directory}: {e}")
            return 0

    def _submit_array_job(
        self, label: str, input_dir: Path, file_count: int
    ) -> Optional[int]:
        """Submit a SLURM array job for a specific input directory."""
        if file_count == 0:
            logger.warning(f"No files found in {input_dir}, skipping {label}")
            return None

        logger.info(
            f"Submitting array job for {label}: {input_dir} ({file_count} tasks)"
        )

        # Determine partition based on mode
        partition = None
        if self.mode == "cpu":
            partition = os.environ.get("SLURM_PARTITION", "slurm")
        elif self.mode == "gpu":
            partition = os.environ.get("SLURM_PARTITION", "gpu")

        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--job-name=embed_{label}",
            f"--array=0-{file_count - 1}",
            "--time=24:00:00",
        ]

        # Add GPU-specific settings for better resource management
        if self.mode == "gpu":
            # Add exclusive node access for GPU jobs to avoid resource contention
            sbatch_cmd.extend(["--exclusive"])
            logger.info(
                "Adding --exclusive flag for GPU jobs to prevent resource contention"
            )

        logger.info(f"🔍 DEBUG: Building sbatch command for {label}")
        logger.info(
            f"🔍 DEBUG: File count: {file_count}, so array will be 0-{file_count - 1}"
        )

        # Add resource requirements based on mode
        if self.mode == "cpu":
            sbatch_cmd.extend(
                [
                    "--mem=32G",
                    "--cpus-per-task=4",
                ]
            )
        elif self.mode == "gpu":
            sbatch_cmd.extend(
                [
                    "--mem=64G",
                    "--cpus-per-task=8",
                    "--gres=gpu:1",
                ]
            )

        # Add partition if specified
        if partition:
            sbatch_cmd.extend([f"--partition={partition}"])

        logger.info(f"🔍 DEBUG: Complete sbatch command: {' '.join(sbatch_cmd)}")

        # Instead of creating temporary config files, pass the config selection via environment
        # This eliminates the race condition entirely!

        # Determine which config section to use
        if self.prepare_only:
            # For embedding preparation, use embedding_preparation config if available
            if (
                hasattr(self.config, "embedding_preparation")
                and self.config.embedding_preparation is not None
            ):
                config_section = "embedding_preparation"
            else:
                # Fallback to CPU config for preparation
                config_section = "embedding_cpu"
        else:
            # For actual embedding, use mode-specific config
            if self.mode == "cpu":
                config_section = "embedding_cpu"
            elif self.mode == "gpu":
                config_section = "embedding_gpu"
            else:
                # Fallback to legacy embedding config
                config_section = "embedding"

        logger.info(f"Using config section: {config_section}")

        # Build environment variables - NO TEMPORARY FILES!
        resolved_base = os.environ.get("BASE_FILE_PATH") or getattr(
            self, "resolved_base_file_path", ""
        )

        env_vars = {
            "INPUT_DIR": str(input_dir),
            "DATASET_CONFIG": self.config_name,  # Pass original config name
            "EMBEDDING_CONFIG_SECTION": config_section,  # Tell array job which section to use
            "PREPARE_ONLY": str(self.prepare_only).lower(),
            "WORKFLOW_DIR": os.environ.get("WORKFLOW_DIR", ""),
            # Propagate project and venv so array script can cd/activate correctly
            "PROJECT_DIR": os.environ.get(
                "PROJECT_DIR", "/home/menger/git/adata_hf_datasets"
            ),
            "VENV_PATH": os.environ.get("VENV_PATH", ".venv"),
        }
        if resolved_base:
            env_vars["BASE_FILE_PATH"] = str(resolved_base)

        # Add environment variables to sbatch command
        env_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])

        sbatch_cmd.extend(["--export", f"ALL,{env_str}"])

        # Add the script path
        script_path = "scripts/embed/embed_array.slurm"
        sbatch_cmd.append(script_path)

        # Execute the command
        # For GPU jobs, we might need to submit to a different cluster
        # Check if we need to SSH to submit the job
        try:
            gpu_host = os.environ.get("GPU_HOST")
            logger.info(f"🔍 DEBUG: Mode: {self.mode}")
            logger.info(f"🔍 DEBUG: GPU_HOST environment variable: '{gpu_host}'")

            if self.mode == "gpu" and gpu_host:
                # Submit via SSH to GPU cluster
                logger.info(f"🔍 DEBUG: Will submit via SSH to GPU cluster: {gpu_host}")

                # Test SSH connection first
                logger.info("🔍 DEBUG: Testing SSH connection...")
                ssh_test_cmd = [
                    "ssh",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "BatchMode=yes",
                    gpu_host,
                    "echo 'SSH connection test successful'",
                ]
                test_result = subprocess.run(
                    ssh_test_cmd, capture_output=True, text=True, timeout=30
                )

                logger.info(f"🔍 DEBUG: SSH test return code: {test_result.returncode}")
                logger.info(
                    f"🔍 DEBUG: SSH test stdout: '{test_result.stdout.strip()}'"
                )
                logger.info(
                    f"🔍 DEBUG: SSH test stderr: '{test_result.stderr.strip()}'"
                )

                if test_result.returncode != 0:
                    logger.error(
                        f"SSH connection test failed to {gpu_host}: {test_result.stderr}"
                    )
                    logger.warning(
                        "SSH connection failed, attempting local submission as fallback..."
                    )
                    # Fall back to local submission
                    logger.info(
                        f"🔍 DEBUG: Falling back to local submission: {' '.join(sbatch_cmd)}"
                    )
                    result = subprocess.run(
                        sbatch_cmd, capture_output=True, text=True, timeout=60
                    )
                else:
                    logger.info("✓ SSH connection test successful")

                    # Change to project directory before running sbatch
                    project_dir = os.environ.get(
                        "PROJECT_DIR", "/home/menger/git/adata_hf_datasets"
                    )
                    remote_cmd = f"cd {project_dir} && {' '.join(sbatch_cmd)}"
                    ssh_cmd = ["ssh", "-o", "ConnectTimeout=30", gpu_host, remote_cmd]
                    logger.info(
                        f"🔍 DEBUG: Executing via SSH to {gpu_host}: {' '.join(ssh_cmd)}"
                    )
                    result = subprocess.run(
                        ssh_cmd, capture_output=True, text=True, timeout=120
                    )
            else:
                # Submit locally (same cluster)
                if self.mode == "gpu":
                    logger.warning(
                        "GPU mode requested but GPU_HOST not set, submitting locally"
                    )
                logger.info(f"🔍 DEBUG: Executing locally: {' '.join(sbatch_cmd)}")
                result = subprocess.run(
                    sbatch_cmd, capture_output=True, text=True, timeout=60
                )

            logger.info("🔍 DEBUG: sbatch execution completed")
            logger.info(f"🔍 DEBUG: Return code: {result.returncode}")
            logger.info(f"🔍 DEBUG: stdout: '{result.stdout.strip()}'")
            logger.info(f"🔍 DEBUG: stderr: '{result.stderr.strip()}'")

            if result.returncode != 0:
                logger.error(f"Failed to submit array job for {label}")
                logger.error(f"Command: {' '.join(sbatch_cmd)}")
                logger.error(f"Error: {result.stderr}")
                raise RuntimeError(f"SLURM job submission failed: {result.stderr}")

            # Parse job ID from output
            import re

            job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
            logger.info(
                f"🔍 DEBUG: Looking for job ID in stdout: '{result.stdout.strip()}'"
            )
            logger.info(f"🔍 DEBUG: Regex match result: {job_id_match}")

            if not job_id_match:
                logger.error(f"Could not parse job ID from output: {result.stdout}")
                raise RuntimeError("Could not parse job ID from SLURM output")

            job_id = int(job_id_match.group(1))
            logger.info(f"🔍 DEBUG: Extracted job ID: {job_id}")
            logger.info(
                f"🔍 DEBUG: This should be an ARRAY JOB with tasks {job_id}_0 through {job_id}_{file_count - 1}"
            )
            logger.info(
                f"✓ Submitted array job {job_id} for {label} ({file_count} tasks)"
            )

            # IMMEDIATELY write job ID to temp file for tracking
            job_file = f"/scratch/global/menger/tmp/embedding_array_jobs_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
            logger.info(
                f"🔍 DEBUG: Attempting to write job ID {job_id} to temp file: {job_file}"
            )

            try:
                # Ensure the directory exists
                temp_dir = "/scratch/global/menger/tmp"
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"🔍 DEBUG: Created/verified temp directory: {temp_dir}")

                # Check directory permissions
                dir_stat = os.stat(temp_dir)
                logger.info(
                    f"🔍 DEBUG: Temp directory permissions: {oct(dir_stat.st_mode)[-3:]}"
                )

                with open(job_file, "a") as f:  # Use append mode
                    # Include cluster information for cross-cluster monitoring
                    if self.mode == "gpu" and os.environ.get("GPU_HOST"):
                        # GPU job - include cluster info
                        gpu_host = os.environ.get("GPU_HOST")
                        line_to_write = f"{job_id}:gpu:{gpu_host}\n"
                        f.write(line_to_write)
                        f.flush()  # Force write to disk
                        logger.info(
                            f"🔍 DEBUG: Wrote line to temp file: '{line_to_write.strip()}'"
                        )
                        logger.info(
                            f"✓ Job ID {job_id} written to tracking file (GPU cluster: {gpu_host}): {job_file}"
                        )
                    else:
                        # CPU job - local cluster
                        line_to_write = f"{job_id}:cpu:local\n"
                        f.write(line_to_write)
                        f.flush()  # Force write to disk
                        logger.info(
                            f"🔍 DEBUG: Wrote line to temp file: '{line_to_write.strip()}'"
                        )
                        logger.info(
                            f"✓ Job ID {job_id} written to tracking file (CPU cluster): {job_file}"
                        )

                # Verify the file was written correctly
                if os.path.exists(job_file):
                    file_stat = os.stat(job_file)
                    logger.info(
                        f"🔍 DEBUG: Temp file exists, size: {file_stat.st_size} bytes, permissions: {oct(file_stat.st_mode)[-3:]}"
                    )

                    # Read back the file contents to verify
                    with open(job_file, "r") as f:
                        contents = f.read()
                        logger.info(
                            f"🔍 DEBUG: Temp file contents after write: '{contents.strip()}'"
                        )
                else:
                    logger.error(
                        "🔍 DEBUG: ERROR - Temp file does not exist after write attempt!"
                    )

            except Exception as write_error:
                logger.error(
                    f"🔍 DEBUG: Exception during temp file write: {write_error}"
                )
                logger.warning(
                    f"Failed to write job ID to tracking file: {write_error}"
                )
                # Don't fail the submission if we can't write to temp file

            return job_id

        except subprocess.TimeoutExpired:
            logger.error(f"SLURM job submission timed out for {label}")
            raise RuntimeError("SLURM job submission timed out")
        except Exception as e:
            logger.error(f"Failed to submit array job for {label}: {e}")
            raise

    def _run_local_tasks(
        self,
        label: str,
        input_dir: Path,
        max_workers: int,
        prepare_only: bool,
    ) -> None:
        """Run embedding tasks locally in parallel without SLURM.

        Each .zarr file in input_dir is processed by invoking embed_core.py directly.
        """
        zarr_files = sorted(input_dir.glob("*.zarr"))
        if not zarr_files:
            logger.warning(f"No .zarr files found in {input_dir}, skipping {label}")
            return

        # Determine output base structure consistent with WORKFLOW_DIR, if provided
        workflow_dir = os.environ.get("WORKFLOW_DIR", "")
        job_id = f"local_{label}_{int(__import__('time').time())}"

        if workflow_dir:
            if prepare_only:
                base_out = Path(workflow_dir) / "embedding_prepare" / f"array_{job_id}"
            else:
                base_out = Path(workflow_dir) / "embedding" / f"array_{job_id}"
        else:
            # Fallback to outputs structure
            date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
            if prepare_only:
                base_out = Path("outputs") / date_str / "embedding_prepare" / job_id
            else:
                base_out = Path("outputs") / date_str / "embedding" / job_id

        log_dir = base_out
        base_out.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Local backend: processing {len(zarr_files)} files for {label} with up to {max_workers} workers"
        )

        # Choose which embedding config section to use
        if prepare_only:
            if (
                hasattr(self.config, "embedding_preparation")
                and self.config.embedding_preparation is not None
            ):
                config_section = "embedding_preparation"
            else:
                config_section = "embedding_cpu"
        else:
            config_section = "embedding_cpu" if self.mode == "cpu" else "embedding_gpu"

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_one(task_idx: int, file_path: Path) -> Tuple[int, Path]:
            # Build per-task output dir
            task_out = base_out / str(task_idx)
            task_out.mkdir(parents=True, exist_ok=True)

            # Build command similar to embed_array.slurm
            cmd = [
                sys.executable,
                "scripts/embed/embed_core.py",
                "--config-path=../../conf",
                f"--config-name={self.config_name}",
                f'++{config_section}.input_files=["{str(file_path)}"]',
                f"++prepare_only={str(prepare_only).lower()}",
                f"++hydra.run.dir={str(task_out)}",
                f"++embedding_config_section={config_section}",
            ]

            # Ensure a base_file_path override is passed through (env + CLI)
            resolved_base_local = os.environ.get("BASE_FILE_PATH") or getattr(
                self, "resolved_base_file_path", ""
            )
            if resolved_base_local:
                cmd.append(f"++base_file_path={resolved_base_local}")

            env = os.environ.copy()
            env["EMBEDDING_CONFIG_SECTION"] = config_section
            env["PREPARE_ONLY"] = str(prepare_only).lower()
            if resolved_base_local:
                env["BASE_FILE_PATH"] = resolved_base_local

            # Redirect logs into the per-task directory
            stdout_path = log_dir / f"{task_idx}.out"
            stderr_path = log_dir / f"{task_idx}.err"

            with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(project_root),
                    env=env,
                    stdout=out_f,
                    stderr=err_f,
                    text=True,
                )
                return proc.returncode, file_path

        failures: List[Tuple[Path, int]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(run_one, idx, file_path): (idx, file_path)
                for idx, file_path in enumerate(zarr_files)
            }
            for fut in as_completed(futures):
                idx, file_path = futures[fut]
                try:
                    rc, _ = fut.result()
                    if rc != 0:
                        failures.append((file_path, rc))
                        logger.error(
                            f"Task {idx} failed for file {file_path} with code {rc}"
                        )
                    else:
                        logger.info(f"Task {idx} completed for file {file_path}")
                except Exception as e:
                    failures.append((file_path, -1))
                    logger.error(
                        f"Task {idx} raised exception for file {file_path}: {e}"
                    )

        if failures:
            raise RuntimeError(
                f"Local embedding encountered {len(failures)} failures for {label}: "
                + ", ".join([f"{p} (rc={rc})" for p, rc in failures])
            )

    def run(self) -> List[int]:
        """Run the embedding launcher and return list of submitted job IDs."""
        logger.info("=== Starting Embedding Launcher ===")
        logger.info(f"Dataset: {self.config.dataset.name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Prepare only: {self.prepare_only}")

        # Debug environment variables
        logger.info("=== Environment Variables Debug ===")
        relevant_env_vars = [
            "GPU_HOST",
            "SLURM_PARTITION",
            "MODE",
            "WORKFLOW_DIR",
            "BASE_FILE_PATH",
        ]
        for var in relevant_env_vars:
            value = os.environ.get(var, "NOT_SET")
            logger.info(f"{var}: '{value}'")
        logger.info("=== End Environment Debug ===")

        # Get embedding configuration
        embedding_config = self._get_embedding_config()
        logger.info(f"Methods: {getattr(embedding_config, 'methods', ['pca', 'hvg'])}")

        # Get input directories
        directories = self._get_input_directories()

        backend = os.environ.get("EMBED_BACKEND", "slurm")
        # Allow CLI override via args; we parse it below in main()
        self.backend = getattr(self, "backend", backend)

        job_ids = []
        if self.backend == "local":
            # Local execution path
            max_workers_env = os.environ.get("LOCAL_MAX_WORKERS")
            try:
                max_workers = (
                    int(max_workers_env) if max_workers_env else os.cpu_count() or 4
                )
            except Exception:
                max_workers = os.cpu_count() or 4

            for label, input_dir in directories:
                self._run_local_tasks(
                    label=label,
                    input_dir=input_dir,
                    max_workers=max_workers,
                    prepare_only=self.prepare_only,
                )
        else:
            # Submit array jobs for each directory via SLURM
            for i, (label, input_dir) in enumerate(directories):
                # Add staggered delay for GPU jobs to prevent resource conflicts
                if i > 0 and self.mode == "gpu":
                    delay_seconds = (
                        60  # 1 minute delay between GPU array job submissions
                    )
                    logger.info(
                        f"Adding {delay_seconds}s delay before submitting {label} job "
                        f"to prevent GPU resource conflicts with previous job"
                    )
                    import time

                    time.sleep(delay_seconds)

                file_count = self._count_zarr_files(input_dir)
                job_id = self._submit_array_job(label, input_dir, file_count)
                if job_id:
                    job_ids.append(job_id)

        if not job_ids:
            logger.warning("No array jobs were submitted")
        else:
            logger.info(
                f"✓ Successfully submitted {len(job_ids)} array jobs: {job_ids}"
            )

        self.job_ids = job_ids
        return job_ids

    def wait_for_completion(self) -> None:
        """Wait for all submitted jobs to complete."""
        if not self.job_ids:
            logger.info("No jobs to wait for")
            return

        logger.info(f"Waiting for {len(self.job_ids)} array jobs to complete...")

        for job_id in self.job_ids:
            logger.info(f"Waiting for job {job_id}...")

            # Determine which cluster this job is on
            if self.mode == "gpu" and os.environ.get("GPU_HOST"):
                # GPU job - use SSH to check status
                gpu_host = os.environ.get("GPU_HOST")
                logger.info(f"Monitoring GPU job {job_id} via SSH to {gpu_host}")

                while True:
                    try:
                        # Check if job is still in queue via SSH
                        ssh_cmd = ["ssh", gpu_host, f"squeue -j {job_id} --noheader"]
                        result = subprocess.run(
                            ssh_cmd, capture_output=True, text=True, timeout=30
                        )

                        if result.returncode != 0 or not result.stdout.strip():
                            # Job is no longer in queue
                            logger.info(f"✓ GPU job {job_id} completed")
                            break

                        # Job is still running
                        import time

                        time.sleep(30)

                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout checking status of GPU job {job_id}")
                        break
                    except Exception as e:
                        logger.warning(
                            f"Error checking status of GPU job {job_id}: {e}"
                        )
                        break
            else:
                # CPU job - check locally
                logger.info(f"Monitoring CPU job {job_id} locally")

                while True:
                    try:
                        # Check if job is still in queue
                        result = subprocess.run(
                            ["squeue", "-j", str(job_id), "--noheader"],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )

                        if result.returncode != 0 or not result.stdout.strip():
                            # Job is no longer in queue
                            logger.info(f"✓ CPU job {job_id} completed")
                            break

                        # Job is still running
                        import time

                        time.sleep(30)

                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout checking status of CPU job {job_id}")
                        break
                    except Exception as e:
                        logger.warning(
                            f"Error checking status of CPU job {job_id}: {e}"
                        )
                        break

                    logger.info("✓ All array jobs completed")

    def cleanup_temp_files(self) -> None:
        """Clean up temporary configuration files - NO LONGER NEEDED!"""
        # With the new environment variable approach, no temporary files are created
        logger.info(
            "No temporary files to clean up (using environment variable approach)"
        )

    # Cleanup methods removed - no longer needed with environment variable approach!


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Launch embedding array jobs")
    parser.add_argument(
        "--config-name", required=True, help="Dataset configuration name"
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Processing mode (default: auto-detect from environment)",
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Run only preparation step"
    )
    parser.add_argument(
        "--backend",
        choices=["slurm", "local"],
        default="slurm",
        help="Execution backend (slurm submits array jobs, local runs tasks in-process)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for all jobs to complete before exiting",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    job_ids = []
    launcher_success = False

    try:
        # Create launcher and run
        launcher = EmbeddingLauncher(
            config_name=args.config_name, mode=args.mode, prepare_only=args.prepare_only
        )
        launcher.backend = args.backend

        job_ids = launcher.run()
        launcher_success = True

        if args.wait and args.backend == "slurm":
            launcher.wait_for_completion()
            # Only clean up temp files if we waited for completion
            launcher.cleanup_temp_files()
        else:
            logger.info("Jobs submitted but not waiting for completion")
            logger.info(
                "No temporary files created (using environment variable approach)"
            )

        logger.info(
            f"✓ Embedding launcher completed successfully with {len(job_ids)} jobs"
        )

    except Exception as e:
        logger.error(f"Embedding launcher encountered an error: {e}")

        # Only clean up temp files if NO jobs were submitted successfully
        # This prevents deleting config files that submitted jobs still need
        if job_ids:
            logger.warning(
                f"Despite the error, {len(job_ids)} array jobs were successfully submitted: {job_ids}"
            )
            logger.warning("Master job will continue to track these jobs")
            launcher_success = True
        else:
            logger.error("No array jobs were submitted, failing master job")
            sys.exit(1)

    # Final attempt to write job IDs file (in case immediate writing failed)
    if job_ids:
        job_file = f"/scratch/global/menger/tmp/embedding_array_jobs_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
        logger.info(f"🔍 DEBUG: Final job ID write attempt to: {job_file}")
        logger.info(f"🔍 DEBUG: Job IDs to ensure are written: {job_ids}")

        try:
            # Ensure the directory exists
            temp_dir = "/scratch/global/menger/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            logger.info(f"🔍 DEBUG: Final write - verified temp directory: {temp_dir}")

            # Ensure all job IDs are in the file (in case some immediate writes failed)
            existing_entries = set()
            if os.path.exists(job_file):
                logger.info(
                    "🔍 DEBUG: Temp file exists for final check, reading existing entries..."
                )
                with open(job_file, "r") as f:
                    file_contents = f.read()
                    logger.info(
                        f"🔍 DEBUG: Current temp file contents: '{file_contents.strip()}'"
                    )

                    for line_num, line in enumerate(
                        file_contents.strip().split("\n"), 1
                    ):
                        if line.strip():
                            # Extract job ID from the line (format: job_id:cluster_type:host)
                            job_id_part = line.strip().split(":")[0]
                            existing_entries.add(job_id_part)
                            logger.info(
                                f"🔍 DEBUG: Found existing entry {line_num}: job_id={job_id_part}, full_line='{line.strip()}'"
                            )
            else:
                logger.warning(
                    f"🔍 DEBUG: Temp file does not exist for final check: {job_file}"
                )

            # Write any missing job IDs
            missing_ids = [jid for jid in job_ids if str(jid) not in existing_entries]
            logger.info(
                f"🔍 DEBUG: Missing job IDs that need to be written: {missing_ids}"
            )

            if missing_ids:
                logger.info(
                    f"🔍 DEBUG: Writing {len(missing_ids)} missing job IDs to temp file..."
                )
                with open(job_file, "a") as f:
                    for job_id in missing_ids:
                        # Determine cluster info for missing job IDs
                        if args.mode == "gpu" and os.environ.get("GPU_HOST"):
                            gpu_host = os.environ.get("GPU_HOST")
                            line_to_write = f"{job_id}:gpu:{gpu_host}\n"
                            f.write(line_to_write)
                            logger.info(
                                f"🔍 DEBUG: Wrote missing GPU job ID: '{line_to_write.strip()}'"
                            )
                        else:
                            line_to_write = f"{job_id}:cpu:local\n"
                            f.write(line_to_write)
                            logger.info(
                                f"🔍 DEBUG: Wrote missing CPU job ID: '{line_to_write.strip()}'"
                            )
                    f.flush()  # Force write to disk

                logger.info(f"✓ Added missing job IDs to tracking file: {missing_ids}")
            else:
                logger.info(
                    f"🔍 DEBUG: No missing job IDs - all {len(job_ids)} job IDs already in temp file"
                )

            # Final verification of temp file
            if os.path.exists(job_file):
                final_stat = os.stat(job_file)
                logger.info(
                    f"🔍 DEBUG: Final temp file status - size: {final_stat.st_size} bytes, permissions: {oct(final_stat.st_mode)[-3:]}"
                )

                with open(job_file, "r") as f:
                    final_contents = f.read()
                    final_lines = [
                        line.strip()
                        for line in final_contents.strip().split("\n")
                        if line.strip()
                    ]
                    logger.info(
                        f"🔍 DEBUG: Final temp file has {len(final_lines)} lines"
                    )
                    for i, line in enumerate(final_lines, 1):
                        logger.info(f"🔍 DEBUG: Final line {i}: '{line}'")

            logger.info(f"✓ Final job IDs tracking file: {job_file}")

        except Exception as file_error:
            logger.error(
                f"🔍 DEBUG: Exception during final job ID file write: {file_error}"
            )
            logger.error(f"Failed to finalize job IDs file: {file_error}")
            # Don't fail if we can't write the file - array jobs are already running
            logger.warning(
                "Array jobs are running independently, master job will continue"
            )

    if not launcher_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
