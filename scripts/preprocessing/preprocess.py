#!/usr/bin/env python
import logging
from pathlib import Path
import importlib
import numpy as np
import scanpy as sc
from anndata import AnnData
import anndata
import sys
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os

import adata_hf_datasets.pp as pp
from adata_hf_datasets.plotting import qc_evaluation_plots
from adata_hf_datasets.sys_monitor import SystemMonitor
from adata_hf_datasets.utils import subset_sra_and_plot
from adata_hf_datasets.config_utils import apply_all_transformations, validate_config

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


def default_split_fn(
    adata: AnnData, train_frac: float = 0.8, random_state: int = 0
) -> tuple[list[int], list[int]]:
    """
    Random train/val split returning lists of obs indices.
    """
    n = adata.n_obs
    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    cut = int(train_frac * n)
    return idx[:cut].tolist(), idx[cut:].tolist()


def import_callable(ref: str):
    module, fn = ref.split(":")
    mod = importlib.import_module(module)
    return getattr(mod, fn)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="dataset_cellxgene_pseudo_bulk_3_5k",
)
def main(cfg: DictConfig):
    """
    Hydra entrypoint to optionally split & preprocess H5AD.

    Now works with dataset-centric config structure where:
    - cfg.preprocessing contains all preprocessing parameters
    - cfg.embedding contains all embedding parameters
    - cfg.dataset_creation contains all dataset creation parameters
    - Common keys (batch_key, annotation_key, etc.) are at the top level

    The config is automatically transformed to include:
    - Generated paths (input_file, output_dir, etc.)
    - Auto-generated consolidation categories and categories of interest
    - Propagated common keys to workflow sections
    """
    # Apply all transformations to the resolved config
    cfg = apply_all_transformations(cfg)

    # Validate the transformed config
    validate_config(cfg)

    # Extract preprocessing config from the dataset-centric config
    preprocess_cfg = cfg.preprocessing

    # 1) Prepare paths & logger
    infile = Path(preprocess_cfg.input_file)
    logger.info("Input file: %s", infile)
    # Get the stem of the input file (filename without extension)
    # input_stem = infile.stem
    out_dir = Path(preprocess_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = HydraConfig.get().run.dir
    logger.info("Run dir: %s", run_dir)
    monitor = SystemMonitor(logger=logger)
    monitor.daemon = True  # to terminate the thread when the main thread exits
    monitor.start()

    try:
        ## Add a sample index without loading the whole object into memory
        # temp_infile = add_sample_index_to_h5ad(
        #    infile=infile, temp_out=out_dir / f"{input_stem}_temp_input.h5ad"
        # )
        ad_bk = sc.read_h5ad(infile, backed="r")

        # Plot some quality control plots prior to processing.
        subset_sra_and_plot(
            adata_bk=ad_bk, cfg=preprocess_cfg, run_dir=run_dir + "/before"
        )

        # 3) Decide split function
        split_fn = (
            import_callable(preprocess_cfg.split_fn)
            if preprocess_cfg.get("split_fn", None)
            else default_split_fn
        )
        logger.info("Split dataset? %s", preprocess_cfg.split_dataset)

        subsets = {}
        if preprocess_cfg.split_dataset:
            train_idx, val_idx = split_fn(
                ad_bk,
                float(preprocess_cfg.train_split),
                int(preprocess_cfg.random_seed),
            )
            subsets["train"] = train_idx
            subsets["val"] = val_idx
            logger.info("Train/val sizes: %d / %d", len(train_idx), len(val_idx))
        else:
            subsets["all"] = list(range(ad_bk.n_obs))
            logger.info("No split; processing full dataset of %d cells", ad_bk.n_obs)

        # Close the backed file; we'll re-open for each slice
        ad_bk.file.close()

        # Helper: write a slice of the backed file to disk without to_adata()
        def write_subset(indices: list[int], name: str) -> Path:
            out_path = out_dir / f"{name}_input.h5ad"
            logger.info(
                "Writing subset '%s' with %d cells to %s", name, len(indices), out_path
            )
            ad_view = anndata.read_h5ad(str(infile), backed="r")[indices]
            # view.obs still has sample_index from earlier
            ad_view.write_h5ad(out_path)
            ad_view.file.close()
            del ad_view
            return out_path

        # 4) Write each subset to its own input file
        subset_files = {name: write_subset(idx, name) for name, idx in subsets.items()}

        # 5) Now preprocess each subset on disk via your chunked pipeline
        for name, path_in in subset_files.items():
            out_dir_split = out_dir / name
            logger.info("Preprocessing %s → %s", path_in, out_dir)
            output_format = preprocess_cfg.get("output_format", "zarr")

            # Use batch_key from top level if available, otherwise from preprocessing
            batch_key = cfg.get("batch_key", preprocess_cfg.get("batch_key"))

            # Use instrument_key from top level if available, otherwise from preprocessing
            instrument_key = cfg.get(
                "instrument_key", preprocess_cfg.get("instrument_key")
            )

            # Use caption_key from top level if available, otherwise from preprocessing
            caption_key = cfg.get("caption_key", preprocess_cfg.get("description_key"))

            pp.preprocess_h5ad(
                path_in,
                out_dir_split,
                chunk_size=int(preprocess_cfg.chunk_size),
                min_cells=int(preprocess_cfg.min_cells),
                min_genes=int(preprocess_cfg.min_genes),
                batch_key=batch_key,
                count_layer_key=preprocess_cfg.count_layer_key,
                n_top_genes=int(preprocess_cfg.n_top_genes),
                consolidation_categories=list(preprocess_cfg.consolidation_categories)
                if preprocess_cfg.consolidation_categories
                else None,
                category_threshold=preprocess_cfg.get("category_threshold", 1),
                remove_low_frequency=preprocess_cfg.get("remove_low_frequency", False),
                geneformer_pp=bool(preprocess_cfg.geneformer_pp),
                sra_chunk_size=preprocess_cfg.get("sra_chunk_size", None),
                sra_extra_cols=preprocess_cfg.get("sra_extra_cols", None),
                skip_sra_fetch=preprocess_cfg.get("skip_sra_fetch", False),
                sra_max_retries=preprocess_cfg.get("sra_max_retries", 3),
                sra_continue_on_fail=preprocess_cfg.get("sra_continue_on_fail", False),
                instrument_key=instrument_key,
                description_key=caption_key,
                bimodal_col=preprocess_cfg.get("bimodal_col", None),
                split_bimodal=bool(preprocess_cfg.get("split_bimodal", False)),
                output_format=output_format,
            )
            logger.info("Preprocessing %s → %s", path_in, out_dir_split)
            if output_format == "h5ad":
                # If using h5ad, we need to close the file before reading it
                ad_bk = anndata.read_h5ad(
                    out_dir_split / f"chunk_0.{output_format}", backed="r"
                )
            else:
                ad_bk = anndata.read_zarr(out_dir_split / f"chunk_0.{output_format}")
            # Plot some quality control plots after processing
            qc_evaluation_plots(
                ad_bk,
                save_plots=True,
                save_dir=run_dir + "/after",
                metrics_of_interest=list(preprocess_cfg.metrics_of_interest),
                categories_of_interest=list(preprocess_cfg.categories_of_interest)
                if preprocess_cfg.categories_of_interest
                else None,
            )
            ad_bk.file.close()

        logger.info("Done. Outputs in %s", out_dir)

        # Clean up temporary input split files
        for path_in in subset_files.values():
            logger.info("Removing temporary file: %s", path_in)
            path_in.unlink()

        # Save system monitor metrics
    except Exception:
        logger.exception("Unhandled exception during preprocessing")
        raise  # optional: re-raise if you want Hydra/SLURM to register job as failed
    finally:
        ad_bk.file.close()
        monitor.stop()
        monitor.print_summary()
        monitor.save(run_dir)
        monitor.plot_metrics(run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
