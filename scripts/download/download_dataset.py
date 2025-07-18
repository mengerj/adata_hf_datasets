#!/usr/bin/env python3
"""
Dataset Download Script with Optional Subsetting

This script downloads datasets (primarily h5ad files for single-cell genomics)
with optional random subsetting capabilities. It supports various data formats
and provides comprehensive logging and error handling.

Usage:
    python download_dataset.py --url URL --output OUTPUT_PATH [OPTIONS]

Examples:
    # Basic download
    python download_dataset.py \
        --url "https://datasets.cellxgene.cziscience.com/f886c7d9-1392-4f09-9e10-31b953afa2da.h5ad" \
        --output "data/my_dataset.h5ad"

    # Download with subsetting
    python download_dataset.py \
        --url "https://allenimmunology.org/public/publication/download/84792154-cdfb-42d0-8e42-39e210e980b4/filesets/3a6afb68-0379-4afa-838a-c0b7f222b517/immune_health_atlas_full.h5ad" \
        --output "data/immune_health_atlas_subset.h5ad" \
        --subset-size 10000 \
        --seed 42
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
from adata_hf_datasets.file_utils import download_from_link

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("download_dataset.log"),
    ],
)
logger = logging.getLogger(__name__)


def ensure_directory_exists(file_path: str) -> None:
    """Ensure the parent directory of the file path exists."""
    parent_dir = Path(file_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {parent_dir}")


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


def create_random_subset(
    input_path: str,
    output_path: str,
    subset_size: int,
    seed: Optional[int] = None,
    obs_subset_key: Optional[str] = None,
    preserve_proportions: bool = False,
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
    obs_subset_key : str, optional
        If provided, subset will preserve proportions of this categorical variable
    preserve_proportions : bool, default False
        Whether to preserve proportions when using obs_subset_key

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
        if (
            obs_subset_key
            and preserve_proportions
            and obs_subset_key in adata.obs.columns
        ):
            logger.info(
                f"Creating stratified subset preserving proportions of '{obs_subset_key}'"
            )
            subset_indices = create_stratified_subset(
                adata, subset_size, obs_subset_key
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
        log_subset_stats(adata, adata_subset, obs_subset_key)

        return True

    except Exception as e:
        logger.error(f"Subsetting failed with exception: {e}")
        return False


def create_stratified_subset(
    adata: ad.AnnData, subset_size: int, stratify_key: str
) -> np.ndarray:
    """
    Create a stratified random subset preserving proportions of a categorical variable.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object
    subset_size : int
        Target subset size
    stratify_key : str
        Column in adata.obs to stratify by

    Returns
    -------
    np.ndarray
        Array of indices for the subset
    """
    category_counts = adata.obs[stratify_key].value_counts()
    category_proportions = category_counts / adata.n_obs

    subset_indices = []
    for category, proportion in category_proportions.items():
        category_mask = adata.obs[stratify_key] == category
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

    return np.array(subset_indices)


def log_subset_stats(
    original: ad.AnnData, subset: ad.AnnData, obs_key: Optional[str] = None
) -> None:
    """Log statistics about the original and subset data."""
    logger.info("=== Subset Statistics ===")
    logger.info(f"Original shape: {original.shape}")
    logger.info(f"Subset shape: {subset.shape}")
    logger.info(f"Subset ratio: {subset.n_obs / original.n_obs:.2%}")

    if obs_key and obs_key in original.obs.columns and obs_key in subset.obs.columns:
        logger.info(f"\nProportions for '{obs_key}':")
        orig_props = original.obs[obs_key].value_counts(normalize=True).sort_index()
        subset_props = subset.obs[obs_key].value_counts(normalize=True).sort_index()

        for category in orig_props.index:
            orig_prop = orig_props.get(category, 0)
            subset_prop = subset_props.get(category, 0)
            logger.info(f"  {category}: {orig_prop:.2%} -> {subset_prop:.2%}")


def validate_file_format(file_path: str) -> bool:
    """Validate that the downloaded file has the expected format."""
    try:
        if file_path.endswith(".h5ad"):
            # Try to read metadata without loading full data
            adata = ad.read_h5ad(file_path, backed="r")
            logger.info(f"Validated h5ad file with shape: {adata.shape}")
            return True
        elif file_path.endswith(".zarr"):
            adata = ad.read_zarr(file_path)
            logger.info(f"Validated zarr file with shape: {adata.shape}")
            return True
        else:
            logger.warning(f"Unknown file format: {file_path}")
            return True  # Assume it's valid
    except Exception as e:
        logger.error(f"File validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets with optional subsetting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--url", type=str, required=True, help="URL to download the dataset from"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Local path to save the downloaded file",
    )

    # Optional subsetting arguments
    parser.add_argument(
        "--subset-size",
        type=int,
        help="Number of observations to include in random subset (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsetting (default: 42)",
    )
    parser.add_argument(
        "--stratify-by",
        type=str,
        help="Column name in obs to stratify subset by (preserves proportions)",
    )
    parser.add_argument(
        "--preserve-proportions",
        action="store_true",
        help="When using --stratify-by, preserve original proportions",
    )

    # Other options
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Directory for temporary files (default: system temp)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate downloaded file format"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary downloaded file when subsetting",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.subset_size is not None and args.subset_size <= 0:
        logger.error("Subset size must be positive")
        sys.exit(1)

    # Determine paths
    if args.subset_size is not None:
        # We need to download to a temporary location first
        if args.temp_dir:
            temp_dir = Path(args.temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.gettempdir())

        # Create temporary file with same extension as output
        output_suffix = Path(args.output).suffix
        temp_file = temp_dir / f"temp_download{output_suffix}"
        download_path = str(temp_file)
        final_path = args.output
    else:
        # Direct download to final location
        download_path = args.output
        final_path = args.output

    try:
        # Step 1: Download the dataset
        success = download_dataset(args.url, download_path)
        if not success:
            logger.error("Download failed")
            sys.exit(1)

        # Step 2: Validate if requested
        if args.validate:
            if not validate_file_format(download_path):
                logger.error("File validation failed")
                sys.exit(1)

        # Step 3: Create subset if requested
        if args.subset_size is not None:
            success = create_random_subset(
                input_path=download_path,
                output_path=final_path,
                subset_size=args.subset_size,
                seed=args.seed,
                obs_subset_key=args.stratify_by,
                preserve_proportions=args.preserve_proportions,
            )

            if not success:
                logger.error("Subsetting failed")
                sys.exit(1)

            # Clean up temporary file unless requested to keep
            if not args.keep_temp and download_path != final_path:
                try:
                    os.remove(download_path)
                    logger.info(f"Removed temporary file: {download_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file: {e}")

        logger.info("=== SUCCESS ===")
        logger.info(f"Final output saved to: {final_path}")

    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
