#!/usr/bin/env python3

import os
import re
import logging
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


def concat_and_save(files: List[Path], output_file: Path) -> None:
    """Concatenate h5ad files on disk and save to output file."""
    logger.info("Concatenating %d files -> %s", len(files), output_file)
    adatas = [ad.read_h5ad(f, backed="r") for f in files]
    combined = ad.concat(adatas, join="outer", merge="unique", uns_merge="unique")
    combined.write(output_file)
    logger.info("Saved concatenated AnnData to %s", output_file)


def main(data_dir: str, output_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_files = group_files_by_prefix(data_dir)

    for prefix, splits in grouped_files.items():
        for split, files in splits.items():
            output_file = output_dir / f"{prefix}_{split}.h5ad"
            concat_and_save(files, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate h5ad files by split.")
    parser.add_argument("--data-dir", required=True, help="Input directory with h5ad files")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)