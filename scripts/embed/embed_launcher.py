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
        self.temp_config_files: List[str] = []  # Track temp files for cleanup

    def _determine_mode(self, mode: str) -> str:
        """Determine the processing mode based on input and environment."""
        if mode != "auto":
            return mode

        # Check environment variable
        env_mode = os.environ.get("MODE", "cpu")
        logger.info(f"Auto-detected mode from environment: {env_mode}")
        return env_mode

    def _load_config(self) -> DictConfig:
        """Load and validate the dataset configuration."""
        from hydra import compose, initialize_config_dir

        config_path = project_root / "conf"
        logger.info(f"Loading config '{self.config_name}' from {config_path}")

        try:
            with initialize_config_dir(config_dir=str(config_path), version_base=None):
                cfg = compose(config_name=self.config_name)

            # Apply transformations
            cfg = apply_all_transformations(cfg)
            logger.info(f"Successfully loaded config for dataset: {cfg.dataset.name}")
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
        base_dir = Path(self.config.get("data_base_dir", "data/RNA/processed"))
        dataset_name = self.config.dataset.name

        # Determine if we're processing train or test data
        train_or_test = (
            "train" if self.config.preprocessing.get("split_dataset", True) else "test"
        )

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

        # Create unified config file for this job
        unified_config_path = self._create_unified_config()

        # Build environment variables
        env_vars = {
            "INPUT_DIR": str(input_dir),
            "UNIFIED_CONFIG": unified_config_path,
            "PREPARE_ONLY": str(self.prepare_only).lower(),
            "WORKFLOW_DIR": os.environ.get("WORKFLOW_DIR", ""),
        }

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
            if self.mode == "gpu" and os.environ.get("GPU_HOST"):
                # Submit via SSH to GPU cluster
                gpu_host = os.environ.get("GPU_HOST")
                # Change to project directory before running sbatch
                project_dir = "/home/menger/git/adata_hf_datasets"
                remote_cmd = f"cd {project_dir} && {' '.join(sbatch_cmd)}"
                ssh_cmd = ["ssh", gpu_host, remote_cmd]
                logger.info(f"Executing via SSH to {gpu_host}: {' '.join(ssh_cmd)}")
                result = subprocess.run(
                    ssh_cmd, capture_output=True, text=True, timeout=60
                )
            else:
                # Submit locally (same cluster)
                logger.info(f"Executing locally: {' '.join(sbatch_cmd)}")
                result = subprocess.run(
                    sbatch_cmd, capture_output=True, text=True, timeout=60
                )

            if result.returncode != 0:
                logger.error(f"Failed to submit array job for {label}")
                logger.error(f"Command: {' '.join(sbatch_cmd)}")
                logger.error(f"Error: {result.stderr}")
                raise RuntimeError(f"SLURM job submission failed: {result.stderr}")

            # Parse job ID from output
            import re

            job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not job_id_match:
                logger.error(f"Could not parse job ID from output: {result.stdout}")
                raise RuntimeError("Could not parse job ID from SLURM output")

            job_id = int(job_id_match.group(1))
            logger.info(
                f"✓ Submitted array job {job_id} for {label} ({file_count} tasks)"
            )

            # IMMEDIATELY write job ID to temp file for tracking
            job_file = f"/tmp/embedding_array_jobs_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
            try:
                with open(job_file, "a") as f:  # Use append mode
                    # Include cluster information for cross-cluster monitoring
                    if self.mode == "gpu" and os.environ.get("GPU_HOST"):
                        # GPU job - include cluster info
                        gpu_host = os.environ.get("GPU_HOST")
                        f.write(f"{job_id}:gpu:{gpu_host}\n")
                        logger.info(
                            f"✓ Job ID {job_id} written to tracking file (GPU cluster: {gpu_host}): {job_file}"
                        )
                    else:
                        # CPU job - local cluster
                        f.write(f"{job_id}:cpu:local\n")
                        logger.info(
                            f"✓ Job ID {job_id} written to tracking file (CPU cluster): {job_file}"
                        )
            except Exception as write_error:
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

    def run(self) -> List[int]:
        """Run the embedding launcher and return list of submitted job IDs."""
        logger.info("=== Starting Embedding Launcher ===")
        logger.info(f"Dataset: {self.config.dataset.name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Prepare only: {self.prepare_only}")

        # Get embedding configuration
        embedding_config = self._get_embedding_config()
        logger.info(f"Methods: {getattr(embedding_config, 'methods', ['pca', 'hvg'])}")

        # Get input directories
        directories = self._get_input_directories()

        # Submit array jobs for each directory
        job_ids = []
        for label, input_dir in directories:
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
        """Clean up temporary configuration files."""
        for temp_file in self.temp_config_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary config file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")
        self.temp_config_files.clear()

    def _create_unified_config(self) -> str:
        """
        Create a unified configuration file that merges the selected embedding config
        into a standard structure for the core script to use.

        Returns
        -------
        str
            Path to the created unified config file
        """
        import tempfile
        import yaml
        from omegaconf import OmegaConf

        # Get the appropriate embedding config based on mode and prepare_only
        embedding_config = self._get_embedding_config()

        # Create a copy of the base config
        unified_config = OmegaConf.to_container(self.config, resolve=True)

        # Replace the embedding section with our selected config
        # Remove all mode-specific embedding sections
        for key in ["embedding_cpu", "embedding_gpu", "embedding_preparation"]:
            if key in unified_config:
                del unified_config[key]

        # Add the selected config as the standard 'embedding' section
        unified_config["embedding"] = OmegaConf.to_container(
            embedding_config, resolve=True
        )

        # Add runtime parameters
        unified_config["prepare_only"] = self.prepare_only

        # Create temporary config file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="embed_config_")
        try:
            with os.fdopen(temp_fd, "w") as f:
                yaml.dump(unified_config, f, default_flow_style=False)
            logger.info(f"Created unified config file: {temp_path}")
            self.temp_config_files.append(temp_path)  # Track for cleanup
            return temp_path
        except Exception:
            os.close(temp_fd)  # Make sure to close if yaml.dump fails
            raise


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

        job_ids = launcher.run()
        launcher_success = True

        if args.wait:
            launcher.wait_for_completion()

        logger.info(
            f"✓ Embedding launcher completed successfully with {len(job_ids)} jobs"
        )

        # Clean up temporary config files
        launcher.cleanup_temp_files()

    except Exception as e:
        logger.error(f"Embedding launcher encountered an error: {e}")

        # Clean up temporary config files
        if "launcher" in locals():
            launcher.cleanup_temp_files()

        # If we successfully submitted some jobs, don't fail the master job
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
        job_file = (
            f"/tmp/embedding_array_jobs_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
        )
        try:
            # Ensure all job IDs are in the file (in case some immediate writes failed)
            existing_entries = set()
            if os.path.exists(job_file):
                with open(job_file, "r") as f:
                    for line in f:
                        if line.strip():
                            # Extract job ID from the line (format: job_id:cluster_type:host)
                            job_id_part = line.strip().split(":")[0]
                            existing_entries.add(job_id_part)

            # Write any missing job IDs
            missing_ids = [jid for jid in job_ids if str(jid) not in existing_entries]
            if missing_ids:
                with open(job_file, "a") as f:
                    for job_id in missing_ids:
                        # Determine cluster info for missing job IDs
                        if args.mode == "gpu" and os.environ.get("GPU_HOST"):
                            gpu_host = os.environ.get("GPU_HOST")
                            f.write(f"{job_id}:gpu:{gpu_host}\n")
                        else:
                            f.write(f"{job_id}:cpu:local\n")
                logger.info(f"✓ Added missing job IDs to tracking file: {missing_ids}")

            logger.info(f"✓ Final job IDs tracking file: {job_file}")

        except Exception as file_error:
            logger.error(f"Failed to finalize job IDs file: {file_error}")
            # Don't fail if we can't write the file - array jobs are already running
            logger.warning(
                "Array jobs are running independently, master job will continue"
            )

    if not launcher_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
