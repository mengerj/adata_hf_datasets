#!/usr/bin/env python3
"""
Dataset Download Script with Dataset-Centric Configuration

This script downloads datasets using the unified dataset configuration system.
It supports automatic path generation, stratification, and comprehensive logging.

Usage:
    python download_dataset_config.py --config-name=dataset_name [OPTIONS]

Examples:
    # Download using dataset config
    python download_dataset_config.py --config-name=dataset_cellxgene_pseudo_bulk_3_5k

    # Override subset size
    python download_dataset_config.py --config-name=dataset_cellxgene_pseudo_bulk_3_5k \
        ++download.subset_size=10000
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, List

import hydra
import anndata as ad
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from adata_hf_datasets.file_utils import download_from_link
from adata_hf_datasets.config_utils import apply_all_transformations
from adata_hf_datasets.utils import setup_logging

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


def ensure_directory_exists(file_path: str) -> None:
    """Ensure the parent directory of the file path exists."""
    parent_dir = Path(file_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {parent_dir}")


def check_file_exists(file_path: str) -> bool:
    """Check if the target file already exists and has reasonable size."""
    file_path_obj = Path(file_path)

    if not file_path_obj.exists():
        return False

    # Check if file has reasonable size (at least 1KB to avoid empty files)
    file_size = file_path_obj.stat().st_size
    if file_size < 1024:
        logger.warning(f"File exists but is too small ({file_size} bytes): {file_path}")
        return False

    logger.info(
        f"File already exists: {file_path} (size: {file_size / (1024 * 1024):.1f} MB)"
    )
    return True


def download_dataset(url: str, output_path: str) -> bool:
    """
    Download a dataset from a URL.

    Parameters
    ----------
    url : str
        URL to download from
    output_path : str
        Local path to save the file

    Returns
    -------
    bool
        True if download succeeded, False otherwise
    """
    logger.info(f"Starting download from: {url}")
    logger.info(f"Saving to: {output_path}")

    ensure_directory_exists(output_path)

    try:
        success = download_from_link(url, output_path)
        if success:
            logger.info("Download completed successfully")
            return True
        else:
            logger.error("Download failed")
            return False
    except Exception as e:
        logger.error(f"Download failed with exception: {e}")
        return False


def create_stratified_subset(
    adata: ad.AnnData, subset_size: int, stratify_keys: List[str]
) -> np.ndarray:
    """
    Create a stratified random subset preserving proportions of multiple categorical variables.

    This function creates a subset that preserves the joint distribution of the stratify keys.
    For example, if stratifying by both batch_key and annotation_key, it will preserve
    the proportions of each batch-annotation combination.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object
    subset_size : int
        Target subset size
    stratify_keys : List[str]
        List of column names in adata.obs to stratify by

    Returns
    -------
    np.ndarray
        Array of indices for the subset
    """
    # Create a combined stratification key
    combined_key = "_".join(stratify_keys)
    adata.obs[combined_key] = adata.obs[stratify_keys].apply(
        lambda x: "_".join(x.astype(str)), axis=1
    )

    category_counts = adata.obs[combined_key].value_counts()
    category_proportions = category_counts / adata.n_obs

    subset_indices = []
    for category, proportion in category_proportions.items():
        category_mask = adata.obs[combined_key] == category
        category_indices = np.where(category_mask)[0]

        # Calculate how many from this category to include
        target_count = max(1, int(np.round(proportion * subset_size)))
        actual_count = min(target_count, len(category_indices))

        # Randomly sample from this category
        selected = np.random.choice(category_indices, size=actual_count, replace=False)
        subset_indices.extend(selected)

    # If we have too many, randomly remove some
    if len(subset_indices) > subset_size:
        subset_indices = np.random.choice(
            subset_indices, size=subset_size, replace=False
        )

    # Clean up temporary column
    adata.obs.drop(columns=[combined_key], inplace=True)

    return np.array(subset_indices)


def create_random_subset(
    input_path: str,
    output_path: str,
    subset_size: int,
    seed: Optional[int] = None,
    stratify_keys: Optional[List[str]] = None,
    preserve_proportions: bool = True,
) -> bool:
    """
    Create a random subset of an AnnData object.

    Parameters
    ----------
    input_path : str
        Path to the input h5ad file
    output_path : str
        Path to save the subset
    subset_size : int
        Number of observations (cells) to include in subset
    seed : int, optional
        Random seed for reproducibility
    stratify_keys : List[str], optional
        List of column names in obs to stratify by
    preserve_proportions : bool, default True
        Whether to preserve proportions when using stratify_keys

    Returns
    -------
    bool
        True if subsetting succeeded, False otherwise
    """
    logger.info(f"Creating subset of size {subset_size} from {input_path}")

    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Using random seed: {seed}")

    try:
        # Read the data
        logger.info("Loading AnnData object...")
        adata = ad.read_h5ad(input_path)
        logger.info(f"Loaded data with shape: {adata.shape}")

        if subset_size >= adata.n_obs:
            logger.warning(
                f"Subset size ({subset_size}) >= total observations ({adata.n_obs}). Using all data."
            )
            subset_size = adata.n_obs

        # Create subset indices
        if stratify_keys and preserve_proportions:
            # Check if all stratify keys exist
            missing_keys = [
                key for key in stratify_keys if key not in adata.obs.columns
            ]
            if missing_keys:
                logger.warning(
                    f"Missing stratify keys: {missing_keys}. Using random subset instead."
                )
                subset_indices = np.random.choice(
                    adata.n_obs, size=subset_size, replace=False
                )
            else:
                logger.info(
                    f"Creating stratified subset preserving proportions of: {stratify_keys}"
                )
                subset_indices = create_stratified_subset(
                    adata, subset_size, stratify_keys
                )
        else:
            logger.info("Creating random subset")
            subset_indices = np.random.choice(
                adata.n_obs, size=subset_size, replace=False
            )

        # Create subset
        adata_subset = adata[subset_indices, :].copy()
        logger.info(f"Created subset with shape: {adata_subset.shape}")

        # Save subset
        ensure_directory_exists(output_path)
        adata_subset.write_h5ad(output_path)
        logger.info(f"Saved subset to: {output_path}")

        # Log some statistics
        log_subset_stats(adata, adata_subset, stratify_keys)

        return True

    except Exception as e:
        logger.error(f"Subsetting failed with exception: {e}")
        return False


def log_subset_stats(
    original: ad.AnnData, subset: ad.AnnData, stratify_keys: Optional[List[str]] = None
) -> None:
    """Log statistics about the original and subset data."""
    logger.info("=== Subset Statistics ===")
    logger.info(f"Original shape: {original.shape}")
    logger.info(f"Subset shape: {subset.shape}")
    logger.info(f"Subset ratio: {subset.n_obs / original.n_obs:.2%}")

    if stratify_keys:
        for key in stratify_keys:
            if key in original.obs.columns and key in subset.obs.columns:
                logger.info(f"\nProportions for '{key}':")
                orig_props = original.obs[key].value_counts(normalize=True).sort_index()
                subset_props = subset.obs[key].value_counts(normalize=True).sort_index()

                for category in orig_props.index:
                    orig_prop = orig_props.get(category, 0)
                    subset_prop = subset_props.get(category, 0)
                    logger.info(f"  {category}: {orig_prop:.2%} -> {subset_prop:.2%}")


def validate_file_format(file_path: str) -> bool:
    """Validate that the downloaded file has the expected format."""
    try:
        if file_path.endswith(".h5ad"):
            # Use a lighter validation approach - just check if we can open the file
            # without loading all data into memory
            import h5py

            with h5py.File(file_path, "r") as f:
                # Check if it has the basic AnnData structure
                if "obs" in f and "var" in f and "X" in f:
                    # Get basic info without loading data
                    obs_shape = f["obs"].shape if hasattr(f["obs"], "shape") else None
                    var_shape = f["var"].shape if hasattr(f["var"], "shape") else None
                    x_shape = f["X"].shape if hasattr(f["X"], "shape") else None
                    logger.info(
                        f"Validated h5ad file structure - obs: {obs_shape}, var: {var_shape}, X: {x_shape}"
                    )
                    return True
                else:
                    logger.error("File does not have expected AnnData structure")
                    return False
        elif file_path.endswith(".zarr"):
            # For zarr, we can check if the directory exists and has basic structure
            zarr_path = Path(file_path)
            if zarr_path.exists() and (zarr_path / ".zarray").exists():
                logger.info(f"Validated zarr directory: {file_path}")
                return True
            else:
                logger.error("Zarr directory does not have expected structure")
                return False
        else:
            logger.warning(f"Unknown file format: {file_path}")
            return True  # Assume it's valid
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        return False


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="dataset_tabula_sapiens_bone_marrow",
)
def main(cfg: DictConfig):
    """
    Download a dataset using the dataset-centric configuration system.
    """
    # Get Hydra run directory for logging
    hydra_run_dir = HydraConfig.get().run.dir
    setup_logging(log_dir=hydra_run_dir)

    # Apply all transformations to the config (paths, common keys, etc.)
    cfg = apply_all_transformations(cfg)

    # Extract download-specific config
    download_cfg = cfg.download
    dataset_cfg = cfg.dataset

    logger.info(f"Processing dataset: {dataset_cfg.name}")

    # Check if download is enabled
    if not download_cfg.enabled:
        logger.info("Download is disabled. Skipping download step.")
        return

    # Check if download URL is provided
    if not dataset_cfg.download_url:
        logger.warning(
            "Download is enabled but no download_url provided. Skipping download step."
        )
        return

    # Get paths
    full_file_path = download_cfg.full_file_path
    output_path = download_cfg.output_path
    url = dataset_cfg.download_url

    logger.info(f"Download URL: {url}")
    logger.info(f"Full file path: {full_file_path}")
    logger.info(f"Output path: {output_path}")

    # Determine if we need subsetting
    subset_size = download_cfg.subset_size
    stratify_keys = download_cfg.stratify_keys
    preserve_proportions = download_cfg.preserve_proportions
    seed = download_cfg.seed
    validate = download_cfg.validate
    keep_full_file = download_cfg.keep_full_file

    logger.info(f"Subset size: {subset_size}")
    logger.info(f"Stratify keys: {stratify_keys}")
    logger.info(f"Preserve proportions: {preserve_proportions}")
    logger.info(f"Random seed: {seed}")
    logger.info(f"Keep full file: {keep_full_file}")

    # Step 1: Check if full file already exists
    if check_file_exists(full_file_path):
        logger.info(f"Full file already exists: {full_file_path}")
        full_file_available = True
    else:
        logger.info(f"Full file does not exist: {full_file_path}")
        full_file_available = False

    # Step 2: Check if final output already exists
    if check_file_exists(output_path):
        logger.info(f"Final output already exists: {output_path}")
        logger.info("Skipping download and subsetting.")
        return

    # Step 3: Download full file if needed
    if not full_file_available:
        logger.info("Downloading full file...")
        success = download_dataset(url, full_file_path)
        if not success:
            logger.error("Download failed")
            sys.exit(1)

        # Validate if requested
        if validate:
            if not validate_file_format(full_file_path):
                logger.error("File validation failed")
                sys.exit(1)
    else:
        logger.info("Using existing full file for subsetting")

    # Step 4: Create subset if requested
    if subset_size is not None:
        logger.info("Creating subset from full file...")
        success = create_random_subset(
            input_path=full_file_path,
            output_path=output_path,
            subset_size=subset_size,
            seed=seed,
            stratify_keys=stratify_keys,
            preserve_proportions=preserve_proportions,
        )

        if not success:
            logger.error("Subsetting failed")
            sys.exit(1)

        # Clean up full file if not requested to keep
        if not keep_full_file:
            try:
                Path(full_file_path).unlink()
                logger.info(f"Removed full file: {full_file_path}")
            except Exception as e:
                logger.warning(f"Could not remove full file: {e}")
    else:
        # No subsetting requested, copy full file to output path
        logger.info("No subsetting requested, copying full file to output path...")

        # Check if paths are the same (avoid copying to itself)
        full_file_path_resolved = Path(full_file_path).resolve()
        output_path_resolved = Path(output_path).resolve()

        if full_file_path_resolved == output_path_resolved:
            logger.info(
                "Full file path and output path are the same. No copying needed."
            )
            logger.info(f"File already at target location: {output_path}")
        else:
            try:
                import shutil

                shutil.copy2(full_file_path, output_path)
                logger.info(f"Copied full file to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to copy file: {e}")
                sys.exit(1)

        # Clean up full file if not requested to keep
        if not keep_full_file:
            try:
                Path(full_file_path).unlink()
                logger.info(f"Removed full file: {full_file_path}")
            except Exception as e:
                logger.warning(f"Could not remove full file: {e}")

    logger.info("=== SUCCESS ===")
    logger.info(f"Final output saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Download failed.")
        sys.exit(1)
