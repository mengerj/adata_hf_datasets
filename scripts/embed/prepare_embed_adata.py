#!/usr/bin/env python
"""
Prepare embeddings for preprocessed AnnData files without saving modified AnnData.

This script:
  1. Reads one or more preprocessed .h5ad or .zarr files (AnnData objects).
  2. For each file, loads it into memory once.
  3. Loops over cfg.methods, calling only `InitialEmbedder.prepare` to do CPU-intensive setup.
  4. Does not write out any AnnData; cached results live internally in your embedder.

Data source
----------
Input files are assumed to be preprocessed AnnData objects in .h5ad or .zarr format.

"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
import anndata as ad

from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.file_utils import safe_read_h5ad
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.sys_monitor import SystemMonitor

logger = setup_logging()


def load_adata_file(file_path: Path, input_format: str = "auto") -> ad.AnnData:
    """
    Load AnnData from file with format detection or explicit format specification.

    Parameters
    ----------
    file_path : Path
        Path to the AnnData file
    input_format : str
        Format specification: "auto", "h5ad", or "zarr"

    Returns
    -------
    ad.AnnData
        Loaded AnnData object
    """
    if input_format == "auto":
        if file_path.suffix == ".zarr":
            format_to_use = "zarr"
        elif file_path.suffix == ".h5ad":
            format_to_use = "h5ad"
        else:
            raise ValueError(
                f"Cannot auto-detect format for {file_path}. Please specify input_format."
            )
    else:
        format_to_use = input_format

    if format_to_use == "zarr":
        return ad.read_zarr(file_path, copy_local=False)
    elif format_to_use == "h5ad":
        return safe_read_h5ad(file_path, copy_local=False)
    else:
        raise ValueError(
            f"Unsupported format: {format_to_use}. Must be 'h5ad' or 'zarr'."
        )


@hydra.main(
    version_base=None, config_path="../../conf", config_name="prepare_embed_adata"
)
def main(cfg: DictConfig):
    """
    CPU-only preparation for embedding methods.

    Parameters
    ----------
    cfg.input_files : List[str]
        Paths to preprocessed files (.h5ad or .zarr).
    cfg.input_format : str, optional
        Input file format: "auto", "h5ad", or "zarr". Default is "auto".
    cfg.methods : List[str]
        Embedding methods to prepare (e.g., ['hvg', 'pca', 'scvi_fm']).
    cfg.batch_key : str
        Column in `adata.obs` storing batch labels.
    cfg.batch_size : int
        Batch size for embedders that use it.
    cfg.embedding_dim_map : Dict[str, int]
        Mapping from method name to embedding dimension.
    """
    load_dotenv(override=True)
    run_dir = HydraConfig.get().run.dir

    # Get format specifications with defaults
    input_format = getattr(cfg, "input_format", "auto")

    monitor = SystemMonitor(logger=logger)
    monitor.start()

    try:
        for input_path in cfg.input_files:
            infile = Path(input_path)
            logger.info("Preparing file: %s", infile)
            if not infile.exists():
                raise FileNotFoundError(f"Input file not found: {infile}")

            # Load AnnData with format detection
            adata = load_adata_file(infile, input_format)
            logger.info(
                "Loaded AnnData with %d cells, %d vars", adata.n_obs, adata.n_vars
            )

            # Run prepare for each method
            for method in cfg.methods:
                if method not in cfg.embedding_dim_map:
                    raise KeyError(f"No entry for '{method}' in embedding_dim_map")

                emb_dim = cfg.embedding_dim_map[method]
                monitor.log_event(f"Prepare {method}")

                embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)
                embedder.prepare(
                    adata=adata,
                    adata_path=str(infile),
                    batch_key=cfg.batch_key,
                )
                logger.info("Prepared embedding resources for '%s'", method)

        logger.info("All preparations complete; results cached internally.")
    except Exception as e:
        logger.exception("Preparation pipeline failed: %s", e)
        raise
    finally:
        monitor.stop()
        monitor.print_summary()
        monitor.save(run_dir)
        monitor.plot_metrics(run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in prepare_embed_adata.py")
        sys.exit(1)
