#!/usr/bin/env python
"""
Preprocess raw AnnData files by removing zero-variance features,
embedding them, and optionally splitting into train/val sets.

Data Sources
------------
- Possibly GEO or Cellxgene raw files: "xxx.h5ad"

References
----------
- Hydra: https://hydra.cc
- anndata: https://anndata.readthedocs.io
"""

import os
import sys
from pathlib import Path
import anndata

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig

from adata_hf_datasets.utils import (
    setup_logging,
    remove_zero_variance_cells,
    remove_zero_variance_genes,
    split_anndata,
)
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.sys_monitor import SystemMonitor

logger = setup_logging()


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_adata")
def main(cfg: DictConfig):
    """
    Main function for preprocessing raw AnnData files using Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object with fields:
        - files (list of str): Paths to raw .h5ad files
        - methods (list of str): Embedding methods to apply
        - batch_keys (list of str): Batch keys for each file
        - output_dir (str): Where processed .h5ad files are saved
        - train_split (float): Fraction of data in the train set
        - split_dataset (bool): Whether to split into train/val or keep single
    """
    load_dotenv(override=True)

    hydra_run_dir = HydraConfig.get().run.dir
    output_dir = cfg.output_dir
    monitor = SystemMonitor(logger=logger)
    monitor.start()

    files = cfg.files
    methods = cfg.methods
    batch_keys = cfg.batch_keys
    train_split = cfg.train_split
    split_dataset = cfg.split_dataset
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    # Validate length of batch_keys vs. files
    if len(batch_keys) != len(files):
        logger.warning(
            "Number of batch_keys != number of files. Reusing the first batch_key for all files."
        )
        if len(batch_keys) == 1:
            batch_keys = list(batch_keys) * len(files)

    for file_path_str, batch_key in zip(files, batch_keys):
        logger.info("Processing raw file: %s", file_path_str)
        preprocess_and_save_adata(
            raw_file_path=file_path_str,
            methods=methods,
            batch_key=batch_key,
            output_dir=output_dir,
            train_split=train_split,
            split_dataset=split_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    monitor.stop()
    monitor.save(hydra_run_dir)
    monitor.plot_metrics(hydra_run_dir)


def preprocess_and_save_adata(
    raw_file_path,
    methods,
    batch_key,
    output_dir,
    train_split=0.9,
    split_dataset=True,
    batch_size=32,
    num_workers=1,
):
    """
    Preprocess a single raw AnnData file: load, remove zero-variance features,
    embed, optionally split into train/val, and save to disk.

    Parameters
    ----------
    raw_file_path : str or Path
        Path to the raw .h5ad file.
    methods : list of str
        Embedding methods to apply (e.g. ['hvg','pca','scvi','geneformer']).
    batch_key : str
        Column in adata.obs used for batch correction (used by scvi, etc.).
    output_dir : str
        Path to save processed .h5ad files.
    train_split : float, optional
        Fraction of data in the train set (default is 0.9).
    split_dataset : bool, optional
        If True, split into train/val. If False, keep entire data in a single file named `all.h5ad`.

    Returns
    -------
    None
        Writes processed AnnData file(s) to disk.
    """
    file_stem = Path(raw_file_path).stem
    output_subdir = Path(output_dir) / file_stem

    if split_dataset:
        train_out_path = output_subdir / "train.h5ad"
        val_out_path = output_subdir / "val.h5ad"
        # If files exist, skip
        if train_out_path.is_file() and val_out_path.is_file():
            logger.info(
                "Processed train/val .h5ad already found for '%s'; skipping reprocessing.",
                file_stem,
            )
            return
    else:
        # Single dataset scenario
        all_out_path = output_subdir / "all.h5ad"
        if all_out_path.is_file():
            logger.info(
                "Processed single .h5ad already found for '%s'; skipping reprocessing.",
                file_stem,
            )
            return

    # Load raw AnnData
    logger.info("Loading raw AnnData from: %s", raw_file_path)
    adata = anndata.read_h5ad(raw_file_path)

    # Remove zero variance cells/genes
    adata = remove_zero_variance_cells(adata)
    adata = remove_zero_variance_genes(adata)

    # Embed using each method
    for method in methods:
        logger.info("Embedding with method: %s", method)
        embedder = InitialEmbedder(method=method)
        embedder.fit(
            adata, batch_key=batch_key, batch_size=batch_size, num_workers=num_workers
        )
        adata = embedder.embed(adata, batch_size=batch_size)

    # Either split or keep as single
    os.makedirs(output_subdir, exist_ok=True)

    if split_dataset:
        # Train/val split
        logger.info(
            "Splitting data: train=%.2f, val=%.2f", train_split, 1 - train_split
        )
        train_adata, val_adata = split_anndata(adata, train_size=train_split)
        del adata

        train_adata.write_h5ad(str(train_out_path))
        val_adata.write_h5ad(str(val_out_path))
        logger.info("Saved processed splits: %s, %s", train_out_path, val_out_path)
    else:
        # Keep as single dataset
        logger.info("Saving single dataset (no split).")
        all_out_path = output_subdir / "all.h5ad"
        adata.write_h5ad(str(all_out_path))
        logger.info("Saved processed single dataset: %s", all_out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred during preprocessing.")
        sys.exit(1)
