#!/usr/bin/env python3
"""
Script to clean layers and obsm entries from h5ad and zarr files with optional HVG selection.

This script can work with:
1. A single h5ad file
2. A directory containing zarr files

For each file, it will:
- Load the anndata object
- Remove specified layers and obsm keys
- Optionally rename observation columns
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


def rename_obs_columns(
    adata: ad.AnnData,
    old_names: Optional[List[str]] = None,
    new_names: Optional[List[str]] = None,
) -> Set[str]:
    """
    Rename observation columns in an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to modify
    old_names : List[str], optional
        List of current column names to rename
    new_names : List[str], optional
        List of new column names (must match length of old_names)

    Returns
    -------
    Set[str]
        Set of successfully renamed columns
    """
    old_names = old_names or []
    new_names = new_names or []

    if len(old_names) != len(new_names):
        logger.error(
            f"Length mismatch: old_names has {len(old_names)} items, new_names has {len(new_names)} items"
        )
        return set()

    if not old_names:
        return set()

    renamed_columns = set()

    for old_name, new_name in zip(old_names, new_names):
        if old_name in adata.obs.columns:
            # Check if we're overwriting an existing column
            if new_name in adata.obs.columns and new_name != old_name:
                logger.warning(
                    f"Column '{new_name}' already exists in adata.obs. Overwriting with '{old_name}'"
                )

            # Rename the column (will overwrite if new_name already exists)
            adata.obs[new_name] = adata.obs[old_name]
            logger.info(f"Renamed obs column: '{old_name}' -> '{new_name}'")
            renamed_columns.add(old_name)
        else:
            logger.warning(f"Column '{old_name}' not found in adata.obs")

    return renamed_columns


def preprocess_for_hvg(
    adata: ad.AnnData,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    batch_key: Optional[str] = None,
    n_top_genes: int = 2000,
) -> Optional[ad.AnnData]:
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

    Returns
    -------
    Optional[anndata.AnnData]
        Preprocessed AnnData object, or None if preprocessing failed
    """
    try:
        original_n_obs = adata.n_obs
        original_n_vars = adata.n_vars

        logger.info("Running quality control preprocessing...")
        adata = pp_quality_control(adata)

        logger.info("Running general preprocessing...")

        # Validate and potentially adjust n_top_genes based on available genes
        max_possible_genes = min(adata.n_vars, 10000)  # reasonable upper limit
        adjusted_n_top_genes = min(n_top_genes, max_possible_genes)

        if adjusted_n_top_genes != n_top_genes:
            logger.warning(
                f"Adjusting n_top_genes from {n_top_genes} to {adjusted_n_top_genes} based on available genes ({adata.n_vars})"
            )

        adata = pp_adata_general(
            adata,
            min_cells=min_cells_per_gene,
            min_genes=min_genes_per_cell,
            batch_key=batch_key,
            n_top_genes=adjusted_n_top_genes,
            categories=None,  # no category consolidation in this context
            category_threshold=1,
            remove=True,
        )

        logger.info(
            f"Preprocessing complete: {original_n_obs} -> {adata.n_obs} cells, {original_n_vars} -> {adata.n_vars} genes"
        )
        return adata

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        return None


def perform_hvg_selection(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    batch_key: Optional[str] = None,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
) -> Optional[ad.AnnData]:
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

    Returns
    -------
    Optional[anndata.AnnData]
        Processed AnnData object with HVG selection, or None if failed
    """
    try:
        original_n_genes = adata.n_vars
        logger.info(
            f"Starting preprocessing and HVG selection with {original_n_genes} genes"
        )

        # Use existing preprocessing pipeline which includes HVG selection
        adata = preprocess_for_hvg(
            adata,
            min_genes_per_cell,
            min_cells_per_gene,
            batch_key,
            n_top_genes,
        )
        if adata is None:
            logger.error("Preprocessing failed")
            return None

        # Check if HVG selection was successful
        if "highly_variable" not in adata.var.columns:
            logger.error(
                "HVG selection failed - 'highly_variable' column not found in adata.var"
            )
            return None

        n_hvg = adata.var["highly_variable"].sum()
        logger.info(f"Found {n_hvg} highly variable genes")

        if n_hvg == 0:
            logger.error("No highly variable genes found")
            return None

        # Subset to highly variable genes
        logger.info("Subsetting to highly variable genes...")
        adata._inplace_subset_var(adata.var["highly_variable"])

        logger.info(
            f"Successfully reduced from {original_n_genes} to {adata.n_vars} genes"
        )
        return adata

    except Exception as e:
        logger.error(f"Error during HVG selection: {str(e)}")
        return None


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
    rename_obs_from: Optional[List[str]] = None,
    rename_obs_to: Optional[List[str]] = None,
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
    rename_obs_from : List[str], optional
        List of obs column names to rename
    rename_obs_to : List[str], optional
        List of new obs column names (must match length of rename_obs_from)

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

        # Rename obs columns if requested
        renamed_columns = rename_obs_columns(adata, rename_obs_from, rename_obs_to)

        # Clean the data
        removed_layers, removed_obsm = clean_adata(
            adata, layers_to_remove, obsm_to_remove, dry_run=False
        )

        if perform_hvg:
            adata = perform_hvg_selection(
                adata,
                n_top_genes,
                batch_key,
                min_genes_per_cell,
                min_cells_per_gene,
            )
            if adata is None:
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

        # Save if something was done
        something_changed = (
            removed_layers or removed_obsm or perform_hvg or renamed_columns
        )
        if something_changed:
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
        else:
            logger.info(f"No changes needed for: {file_path}")

        return True

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Main function to parse arguments and run the cleaning process."""
    # Default values for IDE debugging (when no command line arguments are provided)
    # These correspond to the shell script defaults:
    DEFAULT_INPUT_PATH = "data/RNA/raw/train/cellxgene_pseudo_bulk_3_5k.h5ad"
    DEFAULT_OUTPUT_PATH = "data/RNA/raw/train/cellxgene_pseudo_bulk_3_5k_cleaned.h5ad"
    DEFAULT_LAYERS = "replicate_1,replicate_2,replicate_3,replicate_4,replicate_5"
    DEFAULT_OBSM_KEYS = "natural_language_annotation_replicates"
    DEFAULT_HVG = True
    DEFAULT_N_TOP_GENES = 2000
    DEFAULT_BATCH_KEY = ""
    DEFAULT_MIN_GENES_PER_CELL = 200
    DEFAULT_MIN_CELLS_PER_GENE = 3
    DEFAULT_RENAME_OBS_FROM = "cell_type"
    DEFAULT_RENAME_OBS_TO = "celltype"

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

    # Rename observation columns
    python clean_layers_obsm.py --input file.h5ad --rename-obs-from old_col1,old_col2 --rename-obs-to new_col1,new_col2
        """,
    )

    # Check if we're running without command line arguments (IDE debugging)
    use_defaults = len(sys.argv) == 1

    if use_defaults:
        logger.info("No command line arguments provided - using IDE debugging defaults")

        # Create a mock args object with defaults
        class MockArgs:
            def __init__(self):
                self.input = DEFAULT_INPUT_PATH
                self.output = DEFAULT_OUTPUT_PATH
                self.layers = DEFAULT_LAYERS
                self.obsm = DEFAULT_OBSM_KEYS
                self.hvg = DEFAULT_HVG
                self.n_top_genes = DEFAULT_N_TOP_GENES
                self.batch_key = DEFAULT_BATCH_KEY if DEFAULT_BATCH_KEY else None
                self.min_genes_per_cell = DEFAULT_MIN_GENES_PER_CELL
                self.min_cells_per_gene = DEFAULT_MIN_CELLS_PER_GENE
                self.rename_obs_from = DEFAULT_RENAME_OBS_FROM
                self.rename_obs_to = DEFAULT_RENAME_OBS_TO

        args = MockArgs()
    else:
        # Normal command line argument parsing
        parser.add_argument(
            "--input",
            "-i",
            type=str,
            required=True,
            help="Path to h5ad file or directory containing zarr files",
        )

        parser.add_argument(
            "--layers",
            "-l",
            type=str,
            help="Comma-separated list of layer names to remove",
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
            "--batch-key",
            type=str,
            help="Key in adata.obs for batch-aware HVG selection",
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
            "--rename-obs-from",
            type=str,
            help="Comma-separated list of obs column names to rename",
        )

        parser.add_argument(
            "--rename-obs-to",
            type=str,
            help="Comma-separated list of new obs column names (must match length of --rename-obs-from)",
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

    # Parse rename obs columns
    rename_obs_from = []
    if args.rename_obs_from:
        rename_obs_from = [
            name.strip() for name in args.rename_obs_from.split(",") if name.strip()
        ]

    rename_obs_to = []
    if args.rename_obs_to:
        rename_obs_to = [
            name.strip() for name in args.rename_obs_to.split(",") if name.strip()
        ]

    # Validate rename arguments
    if rename_obs_from and not rename_obs_to:
        logger.error("--rename-obs-to must be provided when --rename-obs-from is used")
        sys.exit(1)
    if rename_obs_to and not rename_obs_from:
        logger.error("--rename-obs-from must be provided when --rename-obs-to is used")
        sys.exit(1)
    if len(rename_obs_from) != len(rename_obs_to):
        logger.error(
            f"Length mismatch: rename-obs-from has {len(rename_obs_from)} items, "
            f"rename-obs-to has {len(rename_obs_to)} items"
        )
        sys.exit(1)

    # Check if we have anything to do
    if (
        not layers_to_remove
        and not obsm_to_remove
        and not args.hvg
        and not rename_obs_from
    ):
        logger.error(
            "No layers, obsm keys, HVG selection, or column renaming specified"
        )
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
    if rename_obs_from:
        logger.info(
            f"  Rename obs columns: {dict(zip(rename_obs_from, rename_obs_to))}"
        )

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
            rename_obs_from,
            rename_obs_to,
        ):
            success_count += 1

    logger.info(
        f"Processing complete: {success_count}/{len(files_to_process)} files processed successfully"
    )

    if success_count != len(files_to_process):
        sys.exit(1)


if __name__ == "__main__":
    main()
