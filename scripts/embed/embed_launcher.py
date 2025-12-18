#!/usr/bin/env python3
"""
Embedding Launcher - Local Execution for Embedding Tasks

This script handles local parallel execution of embedding tasks.
For SLURM array job submission, use the workflow orchestrator with
RemoteExecutor which submits array jobs directly via SSH.

Usage:
    # Local execution (parallel with multiple workers)
    python embed_launcher.py --config-name dataset_config --mode cpu --backend local

    # Local preparation only
    python embed_launcher.py --config-name dataset_config --mode cpu --backend local --prepare-only
"""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig

from adata_hf_datasets.workflow import apply_all_transformations

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
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        error_handler = logging.FileHandler(error_log_path, mode="a")
        error_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        error_handler.setFormatter(formatter)
        logging.getLogger().addHandler(error_handler)
    except Exception as e:
        print(f"Warning: Could not set up error logging to {error_log_path}: {e}")
        print("Continuing without centralized error logging...")


class EmbeddingLauncher:
    """Handles local parallel execution of embedding tasks."""

    def __init__(self, config_name: str, mode: str = "cpu", prepare_only: bool = False):
        """
        Initialize the embedding launcher.

        Parameters
        ----------
        config_name : str
            Name of the dataset configuration to use
        mode : str
            Processing mode: "cpu" or "gpu"
        prepare_only : bool
            If True, run only the preparation step
        """
        self.config_name = config_name
        self.mode = mode if mode != "auto" else os.environ.get("MODE", "cpu")
        self.prepare_only = prepare_only
        self.config = self._load_config()

    def _load_config(self) -> DictConfig:
        """Load and validate the dataset configuration using proper Hydra composition."""
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        config_path = project_root / "conf"
        logger.info(f"Loading config '{self.config_name}' from {config_path}")

        # Clear any existing Hydra state
        GlobalHydra.instance().clear()

        try:
            with initialize_config_dir(config_dir=str(config_path), version_base=None):
                cfg = compose(config_name=self.config_name)

            # Apply transformations
            cfg = apply_all_transformations(cfg)
            logger.info(f"Successfully loaded config for dataset: {cfg.dataset.name}")
            return cfg

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _get_input_directories(self) -> List[Tuple[str, Path]]:
        """Get list of input directories to process."""
        # Environment variable takes precedence
        env_base_path = os.environ.get("BASE_FILE_PATH")
        config_base_path = self.config.get("base_file_path")
        base_file_path = env_base_path or config_base_path

        logger.info(f"Environment BASE_FILE_PATH: {env_base_path}")
        logger.info(f"Config base_file_path: {config_base_path}")
        logger.info(f"Final base_file_path: {base_file_path}")

        # Persist the resolved base path for other methods
        self.resolved_base_file_path = str(base_file_path)

        # Determine input subdirectory based on config and mode
        cpu_embedding_enabled = (
            hasattr(self.config, "embedding_cpu")
            and self.config.embedding_cpu is not None
            and self.config.embedding_cpu.get("enabled", True)
        )

        if self.mode == "gpu" and not self.prepare_only and cpu_embedding_enabled:
            input_subdir = "processed_with_emb"
            logger.info(
                "GPU mode with CPU embedding enabled: looking for input in processed_with_emb/"
            )
        else:
            input_subdir = "processed"
            logger.info("Looking for input in processed/")

        base_dir = Path(base_file_path) / input_subdir
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

    def _get_embedding_config(self) -> Optional[DictConfig]:
        """Get the embedding configuration based on mode."""
        if self.prepare_only:
            if (
                hasattr(self.config, "embedding_preparation")
                and self.config.embedding_preparation is not None
            ):
                return self.config.embedding_preparation
            elif (
                hasattr(self.config, "embedding_cpu")
                and self.config.embedding_cpu is not None
            ):
                return self.config.embedding_cpu

        if self.mode == "cpu":
            if (
                hasattr(self.config, "embedding_cpu")
                and self.config.embedding_cpu is not None
            ):
                return self.config.embedding_cpu
        elif self.mode == "gpu":
            if (
                hasattr(self.config, "embedding_gpu")
                and self.config.embedding_gpu is not None
            ):
                return self.config.embedding_gpu

        return None

    def _run_local_tasks(
        self,
        label: str,
        input_dir: Path,
        max_workers: int,
        prepare_only: bool = False,
    ) -> None:
        """
        Run embedding tasks locally using parallel workers.

        Each .zarr file in input_dir is processed by invoking embed_core.py directly.
        """
        zarr_files = sorted(input_dir.glob("*.zarr"))
        if not zarr_files:
            logger.warning(f"No .zarr files found in {input_dir}, skipping {label}")
            return

        # Determine output base structure
        workflow_dir = os.environ.get("WORKFLOW_DIR", "")
        job_id = f"local_{label}_{int(__import__('time').time())}"

        if workflow_dir:
            if prepare_only:
                base_out = Path(workflow_dir) / "embedding_prepare" / f"array_{job_id}"
            else:
                base_out = Path(workflow_dir) / "embedding" / f"array_{job_id}"
        else:
            date_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
            if prepare_only:
                base_out = Path("outputs") / date_str / "embedding_prepare" / job_id
            else:
                base_out = Path("outputs") / date_str / "embedding" / job_id

        log_dir = base_out
        base_out.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Local backend: processing {len(zarr_files)} files for {label} "
            f"with up to {max_workers} workers"
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

        def run_one(task_idx: int, file_path: Path) -> Tuple[int, Path]:
            task_out = base_out / str(task_idx)
            task_out.mkdir(parents=True, exist_ok=True)

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

            resolved_base = os.environ.get("BASE_FILE_PATH") or getattr(
                self, "resolved_base_file_path", ""
            )
            if resolved_base:
                cmd.append(f"++base_file_path={resolved_base}")

            env = os.environ.copy()
            env["EMBEDDING_CONFIG_SECTION"] = config_section
            env["PREPARE_ONLY"] = str(prepare_only).lower()
            if resolved_base:
                env["BASE_FILE_PATH"] = resolved_base

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

    def run(self) -> None:
        """Run the embedding launcher for local execution."""
        logger.info("=== Starting Embedding Launcher (Local Mode) ===")
        logger.info(f"Dataset: {self.config.dataset.name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Prepare only: {self.prepare_only}")

        # Get embedding configuration
        embedding_config = self._get_embedding_config()
        if embedding_config:
            logger.info(
                f"Methods: {getattr(embedding_config, 'methods', ['pca', 'hvg'])}"
            )

        # Get input directories
        directories = self._get_input_directories()

        # Determine max workers
        max_workers_env = os.environ.get("LOCAL_MAX_WORKERS")
        try:
            max_workers = (
                int(max_workers_env) if max_workers_env else os.cpu_count() or 4
            )
        except Exception:
            max_workers = os.cpu_count() or 4

        # Process all directories
        for label, input_dir in directories:
            self._run_local_tasks(
                label=label,
                input_dir=input_dir,
                max_workers=max_workers,
                prepare_only=self.prepare_only,
            )

        logger.info("✓ Embedding launcher completed successfully")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Launch embedding tasks locally",
        epilog=(
            "Note: For SLURM array job submission, use the workflow orchestrator "
            "which submits jobs directly via RemoteExecutor."
        ),
    )
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
        choices=["local"],
        default="local",
        help="Execution backend (only 'local' is supported; use orchestrator for SLURM)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        launcher = EmbeddingLauncher(
            config_name=args.config_name,
            mode=args.mode,
            prepare_only=args.prepare_only,
        )
        launcher.run()

    except Exception as e:
        logger.error(f"Embedding launcher encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
