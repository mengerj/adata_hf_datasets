#!/usr/bin/env python3

import re
from pathlib import Path
from typing import List, Dict
from adata_hf_datasets.utils import setup_logging
import anndata as ad

logger = setup_logging()


def group_files_by_prefix(data_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Group h5ad files by prefix and split (train/val).

    Parameters
    ----------
    data_dir : Path
        Directory containing .h5ad files.

    Returns
    -------
    Dict[str, Dict[str, List[Path]]]
        Mapping of prefix -> {"train": [...], "val": [...]}
    """
    pattern = re.compile(r"^(.*)_(train|val).*\.h5ad$")
    grouped = {}
    for f in data_dir.glob("*.h5ad"):
        match = pattern.match(f.name)
        if not match:
            logger.warning("Skipping file not matching pattern: %s", f)
            continue
        prefix, split = match.groups()
        grouped.setdefault(prefix, {}).setdefault(split, []).append(f)
    return grouped


def merge_var_metadata(base_adata: ad.AnnData, adata: ad.AnnData) -> None:
    """Merge additional var columns from `adata` into `base_adata` if they match.

    Checks that the number of features is the same and that var_names
    are identical. New var columns (not already in base_adata.var) will be added.

    Parameters
    ----------
    base_adata : AnnData
        The base AnnData object whose var metadata will be extended.
    adata : AnnData
        The AnnData object providing additional var metadata.
    """
    if adata.shape[1] != base_adata.shape[1]:
        logger.warning(
            "Skipping var metadata merge: number of features differs (%d vs %d).",
            base_adata.shape[1],
            adata.shape[1],
        )
        return

    if not all(adata.var_names == base_adata.var_names):
        logger.warning("Skipping var metadata merge: var_names do not match.")
        return

    # For every column in adata.var that is not in base_adata.var, add it.
    for col in adata.var.columns:
        if col not in base_adata.var.columns:
            logger.info("Merging additional var column '%s'.", col)
            base_adata.var[col] = adata.var[col]
        else:
            logger.debug(
                "Skipping var column '%s' as it already exists in the base object.", col
            )


def merge_obsm_layers_preserve_data(file_paths: List[Path], output_file: Path) -> None:
    """Merge .obsm layers from multiple h5ad files into a base object,
    preserving .X, .var, .layers, and .uns from the first file.

    Additionally, if the additional files have matching var metadata, they are merged
    (any extra var columns that don't exist in the base are added).

    Parameters
    ----------
    file_paths : List[Path]
        List of file paths to merge.
    output_file : Path
        Path to write the merged AnnData object.
    """
    logger.info("Merging %d files -> %s", len(file_paths), output_file)

    # Use the first file as the base data (fully loaded to preserve .X, .var, .layers, .uns)
    base_adata = ad.read_h5ad(file_paths[0])
    obs_names = base_adata.obs_names.copy()

    # Iterate over the remaining files
    for file_path in file_paths[1:]:
        logger.info("Processing file: %s", file_path)
        adata = ad.read_h5ad(file_path, backed="r")

        # Check obs_names (features must align as well)
        if not adata.obs_names.equals(obs_names):
            raise ValueError(f"obs_names in {file_path} do not match the reference.")

        # Merge .var metadata if possible
        merge_var_metadata(base_adata, adata)

        # Merge obsm layers, skip if key already exists
        for key in adata.obsm_keys():
            if key in base_adata.obsm:
                logger.warning(
                    "Skipping obsm key '%s' from %s — already exists.", key, file_path
                )
                continue
            base_adata.obsm[key] = adata.obsm[key]

    logger.info("Saving merged AnnData to %s", output_file)
    base_adata.write_h5ad(output_file)


def main(data_dir: str, output_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_files = group_files_by_prefix(data_dir)

    for prefix, splits in grouped_files.items():
        for split, files in splits.items():
            output_file = output_dir / f"{prefix}_{split}.h5ad"
            logger.info("Merging files for prefix '%s', split '%s'.", prefix, split)
            merge_obsm_layers_preserve_data(files, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge h5ad files by split, combining obsm and extending var metadata where possible."
    )
    parser.add_argument(
        "--data-dir",
        default="data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k",
        help="Input directory with h5ad files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k/joined",
        help="Output directory",
    )

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
