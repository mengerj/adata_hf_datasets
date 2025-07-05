#!.venv/bin/python3
# -*- coding: utf-8 -*-
#
# SBATCH --job-name=convert_h5ad_to_zarr
# SBATCH --output=convert_h5ad_to_zarr_%j.out
# SBATCH --error=convert_h5ad_to_zarr_%j.err
# SBATCH --time=06:00:00
# SBATCH --mem=64G
# SBATCH --cpus-per-task=4
"""Convert all *.h5ad files under a root directory (excluding directories named
"geneformer") to Zarr stores using :py:meth:`anndata.AnnData.write_zarr`.

This script can be submitted directly to Slurm via ::

    sbatch convert_h5ad_to_zarr.py --root /path/to/root

or executed interactively ::

    python convert_h5ad_to_zarr.py --root /path/to/root

Requirements
------------
- `anndata` >= 0.9
- Python >= 3.9

The output ``.zarr`` directories are written next to their corresponding
``.h5ad`` files using the same basename.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import anndata as ad

logger = logging.getLogger(__name__)


def find_h5ad_files(root: Path) -> Iterable[Path]:
    """Yield all ``*.h5ad`` files below *root*, skipping any path containing
    a directory named *geneformer*.

    Parameters
    ----------
    root
        Root directory to traverse.

    Yields
    ------
    pathlib.Path
        Path to each discovered ``.h5ad`` file.
    """
    for path in root.rglob("*.h5ad"):
        if "geneformer" in path.parts:
            logger.debug("Skipping %s (within geneformer directory)", path)
            continue
        yield path


def convert_to_zarr(h5ad_path: Path) -> None:
    """Convert a single ``.h5ad`` file to Zarr.

    The data originates from the given ``h5ad_path`` on the local filesystem.

    Parameters
    ----------
    h5ad_path
        Path to the input ``.h5ad`` file.

    Notes
    -----
    The output directory will have the same base name with a ``.zarr``
    extension, e.g. ``sample.h5ad`` -> ``sample.zarr``.
    """
    zarr_path = h5ad_path.with_suffix(".zarr")

    if zarr_path.exists():
        logger.info("Target %s already exists – skipping.", zarr_path)
        return

    logger.info("Reading %s", h5ad_path)
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as exc:
        logger.error("Failed to read %s: %s", h5ad_path, exc)
        return

    logger.info("Writing Zarr to %s", zarr_path)
    try:
        adata.write_zarr(zarr_path)
    except Exception as exc:
        logger.error("Failed to write %s: %s", zarr_path, exc)
        return


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Parameters
    ----------
    argv
        Command‑line arguments.  If *None*, :pydata:`sys.argv` is used.
    """
    parser = argparse.ArgumentParser(
        description="Recursively convert .h5ad files to .zarr stores, skipping geneformer directories.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to search recursively for .h5ad files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    root: Path = args.root.expanduser().resolve()

    if not root.is_dir():
        logger.error("Provided root path %s is not a directory.", root)
        sys.exit(1)

    logger.info("Starting conversion under %s", root)

    for h5ad in find_h5ad_files(root):
        convert_to_zarr(h5ad)

    logger.info("Conversion complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
