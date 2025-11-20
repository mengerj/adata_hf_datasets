#!/usr/bin/env python
import logging
from pathlib import Path
import importlib
import numpy as np
from anndata import AnnData
import anndata
import sys
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os

from adata_hf_datasets.pp.plotting import qc_evaluation_plots
from adata_hf_datasets.pp.loader import BatchChunkLoader
from adata_hf_datasets.pp.orchestrator import preprocess_adata

# from adata_hf_datasets.sys_monitor import SystemMonitor
from adata_hf_datasets.utils import subset_sra_and_plot
from adata_hf_datasets.workflow import apply_all_transformations, validate_config
from adata_hf_datasets.file_utils import sanitize_zarr_keys

# Disable HDF5 file locking to prevent BlockingIOError on shared filesystems
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

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

    # Debug: Print current working directory and paths
    import os

    logger.info("Current working directory: %s", os.getcwd())
    logger.info("Output directory (before creation): %s", out_dir)
    logger.info("Output directory absolute path: %s", out_dir.absolute())

    # Create output directory with detailed logging
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Successfully created output directory: %s", out_dir)

        # Verify it exists
        if out_dir.exists():
            logger.info("✓ Verified: Output directory exists")
        else:
            logger.error("✗ Output directory was not created!")

    except Exception as e:
        logger.error("✗ Failed to create output directory: %s", e)
        raise

    run_dir = HydraConfig.get().run.dir
    logger.info("Run dir: %s", run_dir)
    # monitor = SystemMonitor(logger=logger)
    # monitor.daemon = True  # to terminate the thread when the main thread exits
    # monitor.start()

    # Extract configuration parameters
    enable_plotting = preprocess_cfg.get("enable_plotting", True)
    split_dataset = preprocess_cfg.get("split_dataset", False)
    output_format = preprocess_cfg.get("output_format", "zarr")
    n_chunks = preprocess_cfg.get("n_chunks", None)

    # Use batch_key from top level if available, otherwise from preprocessing
    batch_key = cfg.get("batch_key", preprocess_cfg.get("batch_key"))

    # Use instrument_key from top level if available, otherwise from preprocessing
    instrument_key = cfg.get("instrument_key", preprocess_cfg.get("instrument_key"))

    # Use caption_key from top level if available, otherwise from preprocessing
    caption_key = cfg.get("caption_key", preprocess_cfg.get("description_key"))

    # Decide split function
    split_fn = (
        import_callable(preprocess_cfg.split_fn)
        if preprocess_cfg.get("split_fn", None)
        else default_split_fn
    )

    logger.info("Split dataset? %s", split_dataset)
    if n_chunks is not None:
        logger.info(f"Limiting processing to {n_chunks} chunks")

    # Determine input file format
    if infile.suffix == ".zarr":
        file_format = "zarr"
    else:
        file_format = "h5ad"

    # Create BatchChunkLoader - this is the first step
    logger.info("Creating BatchChunkLoader for chunked processing")
    loader = BatchChunkLoader(
        path=infile,
        chunk_size=int(preprocess_cfg.chunk_size),
        batch_key=batch_key,
        file_format=file_format,
    )

    # Track chunk indices for each split
    chunk_counters = {"train": 0, "val": 0, "all": 0}
    chunks_processed = 0
    first_chunk_for_plotting = None

    # Process each chunk from the loader
    try:
        for chunk_idx, adata_chunk in enumerate(loader):
            # Check if we've reached the chunk limit
            if n_chunks is not None and chunks_processed >= n_chunks:
                logger.info(
                    f"Reached chunk limit ({n_chunks}). Stopping processing after {chunks_processed} chunks."
                )
                break

            logger.info(f"Processing chunk {chunk_idx} with {adata_chunk.n_obs} cells")

            # Store first chunk for plotting if enabled
            if enable_plotting and first_chunk_for_plotting is None:
                first_chunk_for_plotting = adata_chunk.copy()

            # Split chunk into train/val if requested
            if split_dataset:
                train_idx, val_idx = split_fn(
                    adata_chunk,
                    float(preprocess_cfg.train_split),
                    int(preprocess_cfg.random_seed),
                )
                logger.info(
                    f"Chunk {chunk_idx} split: train={len(train_idx)}, val={len(val_idx)}"
                )

                splits = {
                    "train": (adata_chunk[train_idx], train_idx),
                    "val": (adata_chunk[val_idx], val_idx),
                }
            else:
                splits = {"all": (adata_chunk, list(range(adata_chunk.n_obs)))}

            # Process each split
            for split_name, (split_adata, split_indices) in splits.items():
                if len(split_indices) == 0:
                    logger.info(
                        f"Skipping empty {split_name} split for chunk {chunk_idx}"
                    )
                    continue

                # Create output directory for this split
                out_dir_split = out_dir / split_name
                out_dir_split.mkdir(parents=True, exist_ok=True)

                # Get chunk index for this split
                chunk_counter = chunk_counters[split_name]
                chunk_path = out_dir_split / f"chunk_{chunk_counter}.{output_format}"

                try:
                    # Preprocess the split chunk
                    logger.info(
                        f"Preprocessing {split_name} split of chunk {chunk_idx} ({split_adata.n_obs} cells)"
                    )

                    processed_adata = preprocess_adata(
                        split_adata,
                        min_cells=int(preprocess_cfg.min_cells),
                        min_genes=int(preprocess_cfg.min_genes),
                        batch_key=batch_key,
                        count_layer_key=preprocess_cfg.count_layer_key,
                        n_top_genes=int(preprocess_cfg.n_top_genes),
                        consolidation_categories=list(
                            preprocess_cfg.consolidation_categories
                        )
                        if preprocess_cfg.consolidation_categories
                        else None,
                        category_threshold=preprocess_cfg.get("category_threshold", 1),
                        remove_low_frequency=preprocess_cfg.get(
                            "remove_low_frequency", False
                        ),
                        geneformer_pp=bool(preprocess_cfg.geneformer_pp),
                        sra_chunk_size=preprocess_cfg.get("sra_chunk_size", None),
                        sra_extra_cols=preprocess_cfg.get("sra_extra_cols", None),
                        skip_sra_fetch=preprocess_cfg.get("skip_sra_fetch", False),
                        sra_max_retries=preprocess_cfg.get("sra_max_retries", 3),
                        sra_continue_on_fail=preprocess_cfg.get(
                            "sra_continue_on_fail", False
                        ),
                        instrument_key=instrument_key,
                        description_key=caption_key,
                        bimodal_col=preprocess_cfg.get("bimodal_col", None),
                        split_bimodal=bool(preprocess_cfg.get("split_bimodal", False)),
                        layers_to_delete=preprocess_cfg.get("layers_to_delete", None),
                    )

                    # Write processed chunk
                    logger.info(
                        f"Writing {split_name} chunk {chunk_counter} to {chunk_path}"
                    )
                    if output_format == "zarr":
                        sanitize_zarr_keys(processed_adata)
                        processed_adata.write_zarr(chunk_path)
                    else:
                        processed_adata.write_h5ad(chunk_path)

                    chunk_counters[split_name] += 1

                except Exception as e:
                    logger.error(
                        f"Error processing {split_name} split of chunk {chunk_idx}: {e}"
                    )
                    continue

            chunks_processed += 1

        logger.info(f"Finished processing {chunks_processed} chunks")
        logger.info(f"Chunk counts: {chunk_counters}")

        # Plotting (if enabled)
        if enable_plotting and first_chunk_for_plotting is not None:
            # Pre-processing plots
            logger.info("Generating pre-processing plots")
            subset_sra_and_plot(
                adata_bk=first_chunk_for_plotting,
                cfg=preprocess_cfg,
                run_dir=run_dir + "/before",
            )

            # Post-processing plots (read first processed chunk)
            logger.info("Generating post-processing plots")
            # Determine which split to use for plotting
            plot_split = (
                "train" if split_dataset and (out_dir / "train").exists() else "all"
            )
            plot_dir = out_dir / plot_split

            if plot_dir.exists():
                # Find first chunk file
                chunk_files = sorted(plot_dir.glob(f"chunk_*.{output_format}"))
                if chunk_files:
                    if output_format == "h5ad":
                        plot_adata = anndata.read_h5ad(chunk_files[0], backed="r")
                    else:
                        plot_adata = anndata.read_zarr(chunk_files[0])

                    qc_evaluation_plots(
                        plot_adata,
                        save_plots=True,
                        save_dir=run_dir + "/after",
                        metrics_of_interest=list(preprocess_cfg.metrics_of_interest),
                        categories_of_interest=list(
                            preprocess_cfg.categories_of_interest
                        )
                        if preprocess_cfg.categories_of_interest
                        else None,
                    )

                    # Clean up
                    if hasattr(plot_adata, "file") and plot_adata.file is not None:
                        plot_adata.file.close()
        elif not enable_plotting:
            logger.info("Plotting is disabled. Skipping all plots.")

        logger.info("Done. Outputs in %s", out_dir)

        # Save system monitor metrics
    except Exception as e:
        logger.exception("Unhandled exception during preprocessing")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")

        # Force system exit with error code
        # monitor.stop()
        # monitor.print_summary()
        # monitor.save(run_dir)
        sys.exit(1)
    finally:
        # Cleanup is handled automatically by the BatchChunkLoader
        # which closes file handles when iteration completes
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        sys.exit(1)
