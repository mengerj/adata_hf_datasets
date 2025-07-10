#!/usr/bin/env python3
"""
Script to clean layers and obsm entries from h5ad and zarr files with optional HVG selection.

This script can work with:
1. A single h5ad file
2. A directory containing zarr files

For each file, it will:
- Load the anndata object
- Remove specified layers and obsm keys
- Optionally perform robust data preprocessing using existing pipeline functions (quality control, normalization, filtering)
- Optionally perform highly variable gene selection (batch-aware or standard)
- Save the modified file to specified output location or original location

The script uses the existing `pp_quality_control` and `pp_adata_general` preprocessing functions
to ensure robust data processing and avoid common errors in HVG selection.

Usage:
    # Clean a single h5ad file
    python clean_layers_obsm.py --input path/to/file.h5ad --layers layer1,layer2 --obsm obsm1,obsm2

    # Clean all zarr files in a directory
    python clean_layers_obsm.py --input path/to/data/ --layers layer1,layer2 --obsm obsm1,obsm2

    # Perform HVG selection with custom output
    python clean_layers_obsm.py --input path/to/data/ --output processed/ --hvg --n-top-genes 3000

    # Batch-aware HVG selection with custom preprocessing
    python clean_layers_obsm.py --input file.h5ad --hvg --batch-key batch --n-top-genes 2000 --min-cells-per-gene 5

    # Clean and perform HVG selection with dry run
    python clean_layers_obsm.py --input path/to/data/ --layers layer1,layer2 --obsm obsm1,obsm2 --hvg --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
import anndata as ad
from typing import List, Optional, Set

# Add the src directory to the path to import utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.pp.qc import pp_quality_control
from adata_hf_datasets.pp.general import pp_adata_general

logger = setup_logging()


def get_zarr_files(directory: Path) -> List[Path]:
    """
    Get all zarr files in the given directory.

    Parameters
    ----------
    directory : Path
        Directory to search for zarr files

    Returns
    -------
    List[Path]
        List of zarr file paths
    """
    zarr_files = []
    for item in directory.iterdir():
        if item.is_dir() and item.suffix == ".zarr":
            zarr_files.append(item)
    return zarr_files


def clean_adata(
    adata: ad.AnnData,
    layers_to_remove: Optional[List[str]] = None,
    obsm_to_remove: Optional[List[str]] = None,
    dry_run: bool = False,
) -> tuple[Set[str], Set[str]]:
    """
    Clean layers and obsm entries from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to clean
    layers_to_remove : List[str], optional
        List of layer names to remove
    obsm_to_remove : List[str], optional
        List of obsm keys to remove
    dry_run : bool, default False
        If True, only show what would be removed without actually removing

    Returns
    -------
    tuple[Set[str], Set[str]]
        Sets of actually removed layers and obsm keys
    """
    layers_to_remove = layers_to_remove or []
    obsm_to_remove = obsm_to_remove or []

    removed_layers = set()
    removed_obsm = set()

    # Clean layers
    for layer_name in layers_to_remove:
        if layer_name in adata.layers:
            if dry_run:
                logger.info(f"Would remove layer: {layer_name}")
            else:
                del adata.layers[layer_name]
                logger.info(f"Removed layer: {layer_name}")
            removed_layers.add(layer_name)
        else:
            logger.warning(f"Layer '{layer_name}' not found in adata.layers")

    # Clean obsm
    for obsm_key in obsm_to_remove:
        if obsm_key in adata.obsm:
            if dry_run:
                logger.info(f"Would remove obsm key: {obsm_key}")
            else:
                del adata.obsm[obsm_key]
                logger.info(f"Removed obsm key: {obsm_key}")
            removed_obsm.add(obsm_key)
        else:
            logger.warning(f"Obsm key '{obsm_key}' not found in adata.obsm")

    return removed_layers, removed_obsm


def preprocess_for_hvg(
    adata: ad.AnnData,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    batch_key: Optional[str] = None,
    n_top_genes: int = 2000,
    dry_run: bool = False,
) -> bool:
    """
    Preprocess data using existing pipeline functions before HVG selection.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to preprocess
    min_genes_per_cell : int, default 200
        Minimum number of genes per cell
    min_cells_per_gene : int, default 3
        Minimum number of cells per gene
    batch_key : str, optional
        Key in adata.obs for batch-aware processing
    n_top_genes : int, default 2000
        Number of top highly variable genes to keep
    dry_run : bool, default False
        If True, only show what would be done without actually doing it

    Returns
    -------
    bool
        True if preprocessing was successful, False otherwise
    """
    try:
        original_n_obs = adata.n_obs
        original_n_vars = adata.n_vars

        if dry_run:
            logger.info(
                "Would perform preprocessing using existing pipeline functions:"
            )
            logger.info(
                "  - Quality control filtering (outlier detection, mitochondrial filtering)"
            )
            logger.info(f"  - Filter genes expressed in < {min_cells_per_gene} cells")
            logger.info(f"  - Filter cells with < {min_genes_per_cell} genes")
            logger.info("  - Normalization and log transformation")
            logger.info("  - Handle infinite/NaN values")
            if batch_key:
                logger.info(
                    f"  - Batch-aware HVG selection with batch_key='{batch_key}'"
                )
            else:
                logger.info("  - Standard HVG selection (no batch correction)")
            logger.info(f"  - Select top {n_top_genes} highly variable genes")
            return True

        logger.info("Running quality control preprocessing...")
        # Use existing QC function
        adata = pp_quality_control(adata)

        logger.info("Running general preprocessing...")
        # Use existing general preprocessing function
        adata = pp_adata_general(
            adata,
            min_cells=min_cells_per_gene,
            min_genes=min_genes_per_cell,
            batch_key=batch_key or "batch",  # fallback to "batch" if None
            n_top_genes=n_top_genes,
            categories=None,  # no category consolidation in this context
            category_threshold=1,
            remove=True,
        )

        logger.info(
            f"Preprocessing complete: {original_n_obs} -> {adata.n_obs} cells, {original_n_vars} -> {adata.n_vars} genes"
        )

        return True

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return False


def perform_hvg_selection(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    dry_run: bool = False,
) -> bool:
    """
    Perform preprocessing and highly variable gene selection using existing pipeline functions.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to perform HVG selection on
    n_top_genes : int, default 2000
        Number of top highly variable genes to keep
    batch_key : str, optional
        Key in adata.obs for batch-aware HVG selection
    min_genes_per_cell : int, default 200
        Minimum number of genes per cell for preprocessing
    min_cells_per_gene : int, default 3
        Minimum number of cells per gene for preprocessing
    dry_run : bool, default False
        If True, only show what would be done without actually doing it

    Returns
    -------
    bool
        True if HVG selection was performed successfully, False otherwise
    """
    try:
        original_n_genes = adata.n_vars
        logger.info(
            f"Starting preprocessing and HVG selection with {original_n_genes} genes"
        )

        # Use existing preprocessing pipeline which includes HVG selection
        preprocess_success = preprocess_for_hvg(
            adata,
            min_genes_per_cell,
            min_cells_per_gene,
            batch_key,
            n_top_genes,
            dry_run,
        )
        if not preprocess_success:
            logger.error("Preprocessing failed")
            return False

        if dry_run:
            logger.info(
                "Dry run complete - preprocessing and HVG selection would be performed"
            )
            return True

        # Check if HVG selection was successful
        if "highly_variable" not in adata.var.columns:
            logger.error(
                "HVG selection failed - 'highly_variable' column not found in adata.var"
            )
            return False

        n_hvg = adata.var["highly_variable"].sum()
        logger.info(f"Found {n_hvg} highly variable genes")

        if n_hvg == 0:
            logger.error("No highly variable genes found")
            return False

        # Subset to highly variable genes
        logger.info("Subsetting to highly variable genes...")
        adata._inplace_subset_var(adata.var["highly_variable"])

        logger.info(
            f"Successfully reduced from {original_n_genes} to {adata.n_vars} genes"
        )
        return True

    except Exception as e:
        logger.error(f"Error during HVG selection: {str(e)}")
        return False


def process_file(
    file_path: Path,
    output_path: Optional[Path] = None,
    layers_to_remove: Optional[List[str]] = None,
    obsm_to_remove: Optional[List[str]] = None,
    perform_hvg: bool = False,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    dry_run: bool = False,
) -> bool:
    """
    Process a single file (h5ad or zarr).

    Parameters
    ----------
    file_path : Path
        Path to the file to process
    output_path : Path, optional
        Path to save the processed file. If None, saves to same location as input
    layers_to_remove : List[str], optional
        List of layer names to remove
    obsm_to_remove : List[str], optional
        List of obsm keys to remove
    perform_hvg : bool, default False
        Whether to perform highly variable gene selection
    n_top_genes : int, default 2000
        Number of top highly variable genes to keep
    batch_key : str, optional
        Key in adata.obs for batch-aware HVG selection
    min_genes_per_cell : int, default 200
        Minimum number of genes per cell for preprocessing
    min_cells_per_gene : int, default 3
        Minimum number of cells per gene for preprocessing
    dry_run : bool, default False
        If True, only show what would be done without actually doing it

    Returns
    -------
    bool
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Processing file: {file_path}")

        # Load the anndata object
        if file_path.suffix == ".h5ad":
            adata = ad.read_h5ad(file_path)
        elif file_path.suffix == ".zarr":
            adata = ad.read_zarr(file_path)
        else:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return False

        # Show current state
        logger.info(
            f"Original file has {adata.n_obs} cells, {adata.n_vars} genes, {len(adata.layers)} layers and {len(adata.obsm)} obsm entries"
        )
        if adata.layers:
            logger.info(f"Available layers: {list(adata.layers.keys())}")
        if adata.obsm:
            logger.info(f"Available obsm keys: {list(adata.obsm.keys())}")

        # Clean the data
        removed_layers, removed_obsm = clean_adata(
            adata, layers_to_remove, obsm_to_remove, dry_run
        )

        # Perform HVG selection if requested
        hvg_success = True
        if perform_hvg:
            hvg_success = perform_hvg_selection(
                adata,
                n_top_genes,
                batch_key,
                min_genes_per_cell,
                min_cells_per_gene,
                dry_run,
            )
            if not hvg_success:
                logger.error("HVG selection failed")
                return False

        # Determine output path
        if output_path is None:
            save_path = file_path
        else:
            save_path = output_path
            # If output path is directory, create filename based on input
            if save_path.is_dir():
                save_path = save_path / file_path.name

        # Save if not dry run and something was done
        something_changed = (
            removed_layers or removed_obsm or (perform_hvg and hvg_success)
        )
        if not dry_run and something_changed:
            logger.info(f"Saving processed data to: {save_path}")

            # Create output directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if save_path.suffix == ".h5ad":
                adata.write_h5ad(save_path)
            elif save_path.suffix == ".zarr":
                adata.write_zarr(save_path)
            else:
                # Default to h5ad if no extension
                save_path = save_path.with_suffix(".h5ad")
                adata.write_h5ad(save_path)

            logger.info(f"Successfully saved processed data to: {save_path}")
        elif dry_run:
            logger.info(f"Dry run complete for: {file_path}")
            if output_path:
                logger.info(f"Would save to: {save_path}")
        else:
            logger.info(f"No changes needed for: {file_path}")

        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Main function to parse arguments and run the cleaning process."""
    parser = argparse.ArgumentParser(
        description="Clean layers and obsm entries from h5ad/zarr files with optional HVG selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Clean a single h5ad file
    python clean_layers_obsm.py --input file.h5ad --layers layer1,layer2 --obsm obsm1,obsm2

    # Clean all zarr files in a directory
    python clean_layers_obsm.py --input data/ --layers layer1,layer2 --obsm obsm1,obsm2

    # Perform HVG selection and save to different location
    python clean_layers_obsm.py --input data/ --output processed_data/ --hvg --n-top-genes 3000

    # Batch-aware HVG selection
    python clean_layers_obsm.py --input file.h5ad --hvg --batch-key batch --n-top-genes 2000

    # Clean layers and perform HVG selection
    python clean_layers_obsm.py --input data/ --layers counts --obsm X_pca --hvg --n-top-genes 2000

    # Dry run to see what would be done
    python clean_layers_obsm.py --input data/ --layers layer1 --obsm obsm1 --hvg --dry-run
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to h5ad file or directory containing zarr files",
    )

    parser.add_argument(
        "--layers", "-l", type=str, help="Comma-separated list of layer names to remove"
    )

    parser.add_argument(
        "--obsm", "-o", type=str, help="Comma-separated list of obsm keys to remove"
    )

    parser.add_argument(
        "--output",
        "-out",
        type=str,
        help="Output path for processed files. If not specified, files are saved in place. "
        + "Can be a file path or directory path.",
    )

    parser.add_argument(
        "--hvg", action="store_true", help="Perform highly variable gene selection"
    )

    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=2000,
        help="Number of top highly variable genes to keep (default: 2000)",
    )

    parser.add_argument(
        "--batch-key", type=str, help="Key in adata.obs for batch-aware HVG selection"
    )

    parser.add_argument(
        "--min-genes-per-cell",
        type=int,
        default=200,
        help="Minimum number of genes per cell for preprocessing (default: 200)",
    )

    parser.add_argument(
        "--min-cells-per-gene",
        type=int,
        default=3,
        help="Minimum number of cells per gene for preprocessing (default: 3)",
    )

    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    args = parser.parse_args()

    # Parse input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Parse output path
    output_path = None
    if args.output:
        output_path = Path(args.output)

    # Parse layers and obsm lists
    layers_to_remove = []
    if args.layers:
        layers_to_remove = [
            lay.strip() for lay in args.layers.split(",") if lay.strip()
        ]

    obsm_to_remove = []
    if args.obsm:
        obsm_to_remove = [o.strip() for o in args.obsm.split(",") if o.strip()]

    # Check if we have anything to do
    if not layers_to_remove and not obsm_to_remove and not args.hvg:
        logger.error("No layers, obsm keys, or HVG selection specified")
        sys.exit(1)

    logger.info("Starting processing with:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path or 'Same as input'}")
    logger.info(f"  Layers to remove: {layers_to_remove}")
    logger.info(f"  Obsm keys to remove: {obsm_to_remove}")
    logger.info(f"  HVG selection: {args.hvg}")
    if args.hvg:
        logger.info(f"  Top genes: {args.n_top_genes}")
        logger.info(f"  Batch key: {args.batch_key or 'None'}")
    logger.info(f"  Dry run: {args.dry_run}")

    # Process files
    files_to_process = []

    if input_path.is_file():
        # Single file
        if input_path.suffix not in [".h5ad", ".zarr"]:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            sys.exit(1)
        files_to_process.append(input_path)
    else:
        # Directory - find all zarr files
        zarr_files = get_zarr_files(input_path)
        if not zarr_files:
            logger.error(f"No zarr files found in directory: {input_path}")
            sys.exit(1)
        files_to_process.extend(zarr_files)

    logger.info(f"Found {len(files_to_process)} files to process")

    # Process each file
    success_count = 0
    for file_path in files_to_process:
        # Determine output path for this file
        if output_path is None:
            # Save in place
            file_output_path = None
        elif len(files_to_process) == 1:
            # Single file - output path can be specific file or directory
            file_output_path = output_path
        else:
            # Multiple files - output path should be directory
            if output_path.is_file():
                logger.error(
                    "Cannot specify a file as output path when processing multiple files"
                )
                sys.exit(1)
            file_output_path = output_path

        if process_file(
            file_path,
            file_output_path,
            layers_to_remove,
            obsm_to_remove,
            args.hvg,
            args.n_top_genes,
            args.batch_key,
            args.min_genes_per_cell,
            args.min_cells_per_gene,
            args.dry_run,
        ):
            success_count += 1

    logger.info(
        f"Processing complete: {success_count}/{len(files_to_process)} files processed successfully"
    )

    if success_count != len(files_to_process):
        sys.exit(1)


if __name__ == "__main__":
    main()
