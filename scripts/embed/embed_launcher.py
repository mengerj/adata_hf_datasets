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
    error_handler = logging.FileHandler(error_log_path, mode="a")
    error_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    error_handler.setFormatter(formatter)
    logging.getLogger().addHandler(error_handler)


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
        """Submit a SLURM array job for processing a directory."""
        if file_count == 0:
            logger.warning(f"No .zarr files found in {input_dir}, skipping {label}")
            return None

        logger.info(
            f"Submitting array job for {label}: {file_count} files in {input_dir}"
        )

        # Get embedding config for method extraction
        embedding_config = self._get_embedding_config()

        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--array=0-{file_count - 1}",
            f"--job-name=embed_{label}_{self.mode}",
        ]

        # Add partition and GPU settings based on mode
        if self.mode == "gpu":
            partition = os.environ.get("SLURM_PARTITION", "gpu")
            gpu_count = getattr(embedding_config, "gpu_count", 1)
            sbatch_cmd.extend(
                [
                    f"--partition={partition}",
                    f"--gres=gpu:{gpu_count}",
                ]
            )
        else:
            partition = os.environ.get("SLURM_PARTITION", "slurm")
            sbatch_cmd.extend([f"--partition={partition}"])

        # Build environment variables
        env_vars = {
            "INPUT_DIR": str(input_dir),
            "DATASET_CONFIG": self.config_name,
            "MODE": self.mode,
            "METHODS": " ".join(getattr(embedding_config, "methods", ["pca", "hvg"])),
            "BATCH_KEY": self.config.get("batch_key", "batch"),
            "BATCH_SIZE": str(getattr(embedding_config, "batch_size", 128)),
            "PREPARE_ONLY": str(self.prepare_only).lower(),
            "WORKFLOW_DIR": os.environ.get("WORKFLOW_DIR", ""),
        }

        # Add environment variables to sbatch command
        # Quote values that contain spaces or special characters
        env_pairs = []
        for k, v in env_vars.items():
            # Escape any quotes in the value and wrap in quotes if needed
            if " " in str(v) or "," in str(v) or "'" in str(v) or '"' in str(v):
                # Escape any existing quotes and wrap in quotes
                escaped_v = str(v).replace('"', '\\"')
                env_pairs.append(f'{k}="{escaped_v}"')
            else:
                env_pairs.append(f"{k}={v}")

        env_str = ",".join(env_pairs)
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
                ssh_cmd = ["ssh", gpu_host, " ".join(sbatch_cmd)]
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
                        logger.info(f"✓ Job {job_id} completed")
                        break

                    # Job is still running
                    import time

                    time.sleep(30)

                except subprocess.TimeoutExpired:
                    logger.warning(f"Timeout checking status of job {job_id}")
                    break
                except Exception as e:
                    logger.warning(f"Error checking status of job {job_id}: {e}")
                    break

        logger.info("✓ All array jobs completed")


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

    try:
        # Create launcher and run
        launcher = EmbeddingLauncher(
            config_name=args.config_name, mode=args.mode, prepare_only=args.prepare_only
        )

        job_ids = launcher.run()

        if args.wait:
            launcher.wait_for_completion()

        # Store job IDs for master script to read
        if job_ids:
            job_file = f"/tmp/embedding_array_jobs_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
            with open(job_file, "w") as f:
                for job_id in job_ids:
                    f.write(f"{job_id}\n")
            logger.info(f"Job IDs stored in: {job_file}")

    except Exception as e:
        logger.error(f"Embedding launcher failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
