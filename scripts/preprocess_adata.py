#!/usr/bin/env python
"""
Preprocess raw AnnData files using pp_adata function.

This script:
1. Reads a raw AnnData file
2. Applies pp_adata preprocessing
3. Optionally splits into train/val sets

References
----------
- Hydra: https://hydra.cc
- anndata: https://anndata.readthedocs.io
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from adata_hf_datasets.utils import setup_logging, split_anndata
from adata_hf_datasets.pp import pp_adata
from adata_hf_datasets.sys_monitor import SystemMonitor
from adata_hf_datasets.plotting import qc_evaluation_plots
from hydra.core.hydra_config import HydraConfig
import anndata as ad


logger = setup_logging()


@hydra.main(version_base=None, config_path="../conf", config_name="preprocess_adata")
def main(cfg: DictConfig):
    """
    Main function for preprocessing raw AnnData files using Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object with fields:
        - input_file (str): Path to raw .h5ad file
        - output_dir (str): Where processed .h5ad files are saved
        - train_split (float): Fraction of data in the train set
        - split_dataset (bool): Whether to split into train/val or keep single
    """
    load_dotenv(override=True)
    hydra_run_dir = HydraConfig.get().run.dir
    monitor = SystemMonitor(logger=logger)
    monitor.start()
    input_file = cfg.input_file
    output_dir = cfg.output_dir
    train_split = cfg.train_split
    split_dataset = cfg.split_dataset

    # Create output directory structure
    file_stem = Path(input_file).stem
    output_subdir = Path(output_dir) / file_stem
    output_subdir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(input_file)

    if split_dataset:
        logger.info(
            "Splitting data: train=%.2f, val=%.2f", train_split, 1 - train_split
        )
        train_adata, val_adata = split_anndata(adata, train_size=train_split)
        del adata
        # processess each split
        for adata_split, split in zip([train_adata, val_adata], ["train", "val"]):
            out_path = output_subdir / f"{split}.h5ad"
            if out_path.is_file() and not cfg.overwrite:
                logger.info(
                    "Processed split .h5ad already found for '%s'; skipping reprocessing.",
                    file_stem,
                )
                continue
            logger.info("Processing %s split...", split)
            adata_split = pp_adata(
                adata=adata_split,
                category_threshold=cfg.category_threshold,
                categories=list(cfg.categories),
                tag=str(hydra_run_dir),
            )
            # Create some plots to check the data
            qc_evaluation_plots(
                adata_split,
                save_plots=True,
                save_dir=hydra_run_dir + "/" + split,
                metrics_of_interest=list(cfg.metrics_of_interest),
                categories_of_interest=list(cfg.categories_of_interest),
            )

            adata_split.write_h5ad(str(out_path))
            logger.info("Saved %s split to: %s", split, out_path)

    else:
        # Single dataset scenario
        all_out_path = output_subdir / "all.h5ad"
        if all_out_path.is_file():
            logger.info(
                "Processed single .h5ad already found for '%s'; skipping reprocessing.",
                file_stem,
            )
            return

        logger.info("Processing single dataset without splitting...")
        adata = pp_adata(adata=adata, tag=str(hydra_run_dir))
        logger.info("Saved processed dataset: %s", all_out_path)
        # Create some plots to check the data
        qc_evaluation_plots(
            adata,
            save_plots=True,
            save_dir=hydra_run_dir + "/all",
            metrics_of_interest=list(cfg.metrics_of_interest),
            categories_of_interest=list(cfg.categories_of_interest),
        )
        del adata

    monitor.stop()
    monitor.print_summary()
    monitor.save(hydra_run_dir)
    monitor.plot_metrics(hydra_run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred during preprocessing.")
        sys.exit(1)
