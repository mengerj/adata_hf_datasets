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


def merge_obsm_layers(file_paths: List[Path], output_file: Path) -> None:
    """Merge multiple h5ad files by combining their .obsm layers.

    Skips duplicate .obsm keys.
    """
    logger.info("Merging %d files -> %s", len(file_paths), output_file)

    base_adata = ad.read_h5ad(file_paths[0], backed="r")
    obs_names = base_adata.obs_names.copy()

    merged_adata = ad.AnnData(obs=base_adata.obs.copy())

    for file_path in file_paths:
        logger.info("Reading obsm layers from %s", file_path)
        adata = ad.read_h5ad(file_path, backed="r")

        if not adata.obs_names.equals(obs_names):
            raise ValueError(f"obs_names in {file_path} do not match the reference.")

        for key in adata.obsm_keys():
            if key in merged_adata.obsm:
                logger.warning(
                    "Skipping obsm key '%s' from %s — already exists.", key, file_path
                )
                continue
            merged_adata.obsm[key] = adata.obsm[key]

    logger.info("Saving merged AnnData to %s", output_file)
    merged_adata.write_h5ad(output_file)


def main(data_dir: str, output_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_files = group_files_by_prefix(data_dir)

    for prefix, splits in grouped_files.items():
        for split, files in splits.items():
            output_file = output_dir / f"{prefix}_{split}.h5ad"
            merge_obsm_layers(files, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge h5ad files by split, combining obsm layers."
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
