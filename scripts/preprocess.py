#!/usr/bin/env python
import logging
from pathlib import Path
import importlib
import numpy as np
import scanpy as sc
from anndata import AnnData
import sys
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

import adata_hf_datasets.pp as pp
from adata_hf_datasets.plotting import qc_evaluation_plots
from adata_hf_datasets.sys_monitor import SystemMonitor
from adata_hf_datasets.utils import subset_sra_and_plot

logger = logging.getLogger(__name__)


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
    version_base=None, config_path="../conf", config_name="preprocess_adata_test"
)
def main(cfg: DictConfig):
    """
    Hydra entrypoint to optionally split & preprocess H5AD.

    cfg fields:
      input_file: str
      output_dir: str
      split_dataset: bool
      train_split: float
      random_seed: int
      split_fn: str (module:func)
      … plus preprocess_h5ad kwargs …
    """
    # 1) Prepare paths & logger
    infile = Path(cfg.input_file)
    logger.info("Input file: %s", infile)
    # Get the stem of the input file (filename without extension)
    input_stem = infile.stem
    out_dir = Path(cfg.output_dir) / input_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = HydraConfig.get().run.dir
    logger.info("Run dir: %s", run_dir)
    monitor = SystemMonitor(logger=logger)
    monitor.daemon = True  # to terminate the thread when the main thread exits
    monitor.start()

    # Create a temporary file path for the modified input
    temp_infile = out_dir / f"{input_stem}_temp_input.h5ad"

    try:
        # 2) Read in backed mode
        ad_bk = sc.read_h5ad(infile, backed="r")

        # Create a copy with sample_index added if needed
        if "sample_index" not in ad_bk.obs:
            logger.info("Adding sample_index to obs (0…%d)", ad_bk.n_obs - 1)
            # Add sample_index to the copy
            ad_bk.obs["sample_index"] = np.arange(ad_bk.n_obs)

            # Write the modified version to a temporary file
            logger.info(f"Writing temporary file with sample_index to {temp_infile}")
            ad_bk.write_h5ad(temp_infile)

            # Use the temporary file for further processing
            infile = temp_infile
            ad_bk.file.close()
            ad_bk = sc.read_h5ad(infile, backed="r")

        # Plot some quality control plots prior to processing.
        subset_sra_and_plot(adata_bk=ad_bk, cfg=cfg, run_dir=run_dir + "/before")

        # 3) Decide split function
        split_fn = (
            import_callable(cfg.split_fn)
            if cfg.get("split_fn", None)
            else default_split_fn
        )
        logger.info("Split dataset? %s", cfg.split_dataset)

        subsets = {}
        if cfg.split_dataset:
            train_idx, val_idx = split_fn(
                ad_bk, float(cfg.train_split), int(cfg.random_seed)
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
            ad_view = sc.read_h5ad(str(infile), backed="r")[indices]
            # view.obs still has sample_index from earlier
            ad_view.write_h5ad(out_path)
            ad_view.file.close()
            return out_path

        # 4) Write each subset to its own input file
        subset_files = {name: write_subset(idx, name) for name, idx in subsets.items()}

        # 5) Now preprocess each subset on disk via your chunked pipeline
        for name, path_in in subset_files.items():
            path_out = out_dir / f"{name}.h5ad"
            logger.info("Preprocessing %s → %s", path_in, path_out)
            pp.preprocess_h5ad(
                path_in,
                path_out,
                chunk_size=int(cfg.chunk_size),
                min_cells=int(cfg.min_cells),
                min_genes=int(cfg.min_genes),
                batch_key=cfg.batch_key,
                count_layer_key=cfg.count_layer_key,
                n_top_genes=int(cfg.n_top_genes),
                consolidation_categories=list(cfg.consolidation_categories)
                if cfg.consolidation_categories
                else None,
                category_threshold=cfg.get("category_threshold", 1),
                remove_low_frequency=cfg.get("remove_low_frequency", False),
                geneformer_pp=bool(cfg.geneformer_pp),
                sra_chunk_size=cfg.get("sra_chunk_size", None),
                sra_extra_cols=cfg.get("sra_extra_cols", None),
                instrument_key=cfg.get("instrument_key", None),
                description_key=cfg.get("description_key", None),
                bimodal_col=cfg.get("bimodal_col", None),
                split_bimodal=bool(cfg.get("split_bimodal", False)),
            )
            ad_bk = sc.read_h5ad(str(path_out), backed="r")
            # Plot some quality control plots after processing
            qc_evaluation_plots(
                ad_bk,
                save_plots=True,
                save_dir=run_dir + "/after",
                metrics_of_interest=list(cfg.metrics_of_interest),
                categories_of_interest=list(cfg.categories_of_interest),
            )
            ad_bk.file.close()

        logger.info("Done. Outputs in %s", out_dir)

        # Clean up temporary input split files
        for path_in in subset_files.values():
            logger.info("Removing temporary file: %s", path_in)
            path_in.unlink()

        # Clean up the temporary input file if it was created
        if temp_infile.exists():
            logger.info(f"Removing temporary input file: {temp_infile}")
            temp_infile.unlink()

        # Save system monitor metrics
    except Exception:
        logger.exception("Unhandled exception during preprocessing")
        raise  # optional: re-raise if you want Hydra/SLURM to register job as failed
    finally:
        monitor.stop()
        monitor.print_summary()
        monitor.save(run_dir)
        monitor.plot_metrics(run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
