#!/usr/bin/env python3
"""
Script to run embedding with parameters from dataset config.

This script loads the dataset configuration, extracts embedding parameters,
and runs the bash script with the correct parameters.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
import os
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


def extract_embedding_params(
    config: DictConfig, force_prepare_only: bool = False, force_cpu_only: bool = False
) -> dict:
    """Extract embedding parameters from dataset config."""
    # Determine which embedding config to use based on command-line flags and MODE environment variable
    if force_cpu_only:
        mode = "cpu"
        logger.info("Forcing CPU mode due to --cpu-only flag")
    else:
        mode = os.environ.get("MODE", "gpu")
        logger.info(f"Using MODE from environment: {mode}")

    if mode == "cpu":
        embedding_config = config.embedding_cpu
        logger.info("Using CPU embedding configuration")
    elif mode == "gpu":
        embedding_config = config.embedding_gpu
        logger.info("Using GPU embedding configuration")
    else:
        # Fallback to old structure for backward compatibility
        if hasattr(config, "embedding"):
            embedding_config = config.embedding
            logger.info("Using legacy embedding configuration")
        else:
            raise ValueError(
                f"Unknown MODE: {mode} and no legacy embedding config found"
            )

    logger.info("Extracting embedding parameters from config:")
    logger.info(f"  Config methods: {embedding_config.methods}")
    logger.info(f"  Config mode: {mode}")
    logger.info(f"  Config batch_size: {getattr(embedding_config, 'batch_size', 128)}")

    # Check for command-line overrides first, then environment variables, then config
    if force_prepare_only:
        prepare_only = True
        logger.info(f"  Using PREPARE_ONLY from command line: {prepare_only}")
    elif force_cpu_only:
        prepare_only = True
        logger.info(f"  Using PREPARE_ONLY from CPU-only mode: {prepare_only}")
    else:
        prepare_only_env = os.environ.get("PREPARE_ONLY")
        if prepare_only_env is not None:
            prepare_only = prepare_only_env.lower() == "true"
            logger.info(f"  Using PREPARE_ONLY from environment: {prepare_only}")
        else:
            prepare_only = getattr(embedding_config, "prepare_only", False)
            logger.info(f"  Using PREPARE_ONLY from config: {prepare_only}")

    # Extract parameters with defaults
    params = {
        "MODE": mode,
        "GPU_COUNT": getattr(embedding_config, "gpu_count", "1"),
        "DATANAME": config.dataset.name,
        "BATCH_KEY": config.get("batch_key", "batch"),
        "BATCH_SIZE": getattr(embedding_config, "batch_size", 128),
        "METHODS": " ".join(getattr(embedding_config, "methods", ["pca", "hvg"])),
        "PREPARE_ONLY": str(prepare_only).lower(),
        "TRAIN_OR_TEST": "train"
        if config.preprocessing.get("split_dataset", True)
        else "test",
        "DATA_BASE_DIR": "data/RNA/processed/",
    }

    logger.info("Extracted parameters:")
    for key, value in params.items():
        logger.info(f"  {key}={value}")

    return params


def run_embedding_script(params: dict) -> None:
    """Run the embedding bash script with the given parameters."""
    script_path = Path("scripts/embed/run_embed_parallel.sh")

    if not script_path.exists():
        raise FileNotFoundError(f"Embedding script not found: {script_path}")

    # Build environment variables for the bash script
    # Start with the current environment and add our variables
    env = os.environ.copy()
    env.update(
        {
            "MODE": params["MODE"],
            "GPU_COUNT": params["GPU_COUNT"],
            "DATANAME": params["DATANAME"],
            "BATCH_KEY": params["BATCH_KEY"],
            "BATCH_SIZE": str(params["BATCH_SIZE"]),
            "METHODS": params["METHODS"],
            "PREPARE_ONLY": params["PREPARE_ONLY"],
            "TRAIN_OR_TEST": params["TRAIN_OR_TEST"],
            "DATA_BASE_DIR": params["DATA_BASE_DIR"],
        }
    )

    logger.info("Running embedding script with parameters:")
    for key, value in params.items():
        logger.info(f"  {key}={value}")

    # Run the bash script
    try:
        _result = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            check=True,
            capture_output=False,  # Let output go to console
            text=True,
        )
        logger.info("Embedding script completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Embedding script failed with exit code {e.returncode}")
        raise


def load_config(config_name: str) -> DictConfig:
    """Load the dataset configuration using Hydra without creating output directories."""
    # Initialize Hydra without the @hydra.main decorator
    from hydra import compose, initialize_config_dir

    config_path = Path(__file__).parent.parent.parent / "conf"

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name=config_name)

    # Apply transformations
    cfg = apply_all_transformations(cfg)

    return cfg


def main():
    """Main function to run embedding with config."""
    parser = argparse.ArgumentParser(description="Run embedding with dataset config")
    parser.add_argument(
        "--config-name",
        dest="config_name_flag",
        help="Dataset config name (alternative to positional argument)",
    )
    parser.add_argument(
        "config_name", nargs="?", help="Dataset config name (positional argument)"
    )
    parser.add_argument(
        "--prepare-only", action="store_true", help="Force prepare only mode"
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode")
    args = parser.parse_args()

    # Determine which config name to use
    config_name = args.config_name_flag or args.config_name
    if not config_name:
        parser.error(
            "Either --config-name or positional config_name argument is required"
        )

    # Load the config without Hydra's automatic output directory creation
    cfg = load_config(config_name)

    logger.info(f"Running embedding for dataset: {cfg.dataset.name}")

    # Extract embedding parameters
    params = extract_embedding_params(cfg, args.prepare_only, args.cpu_only)

    # Run the embedding script
    run_embedding_script(params)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        main()
    except Exception:
        logger.exception("Embedding failed")
        sys.exit(1)
