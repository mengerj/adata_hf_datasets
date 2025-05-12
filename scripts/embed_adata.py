#!/usr/bin/env python
"""
Apply initial embeddings to preprocessed AnnData files.

This script:
  1. Reads one or more preprocessed .h5ad files.
  2. For each file, loads it into memory once (or loads an existing combined output).
  3. Loops over cfg.methods, computing any missing embeddings.
  4. Stores each embedding in adata.obsm["X_<method>"].
  5. Writes out a single .h5ad with all embeddings attached.
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.file_utils import safe_read_h5ad, safe_write_zarr
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.sys_monitor import SystemMonitor
from hydra.core.hydra_config import HydraConfig

logger = setup_logging()


@hydra.main(version_base=None, config_path="../conf", config_name="embed_adata")
def main(cfg: DictConfig):
    """
    Apply multiple embedding methods to one or more AnnData files.

    Parameters
    ----------
    cfg.input_files : List[str]
        Paths to preprocessed .h5ad files.
    cfg.methods : List[str]
        Embedding methods to apply (e.g., ['hvg','pca','scvi_fm']).
    cfg.batch_key : str
        Column in adata.obs storing batch labels.
    cfg.batch_size : int
        Batch size for embedders that use it.
    cfg.embedding_dim_map : Dict[str,int]
        Mapping from method name to embedding dimension.
    cfg.overwrite : bool
        Whether to recompute and overwrite existing embeddings.
    """
    load_dotenv(override=True)
    hydra_run_dir = HydraConfig.get().run.dir

    monitor = SystemMonitor(logger=logger)
    monitor.start()

    try:
        for input_file in cfg.input_files:
            infile = Path(input_file)
            logger.info("Processing file: %s", infile)
            if not infile.exists():
                raise FileNotFoundError(f"Input file not found: {infile}")

            # Prepare output path
            out_dir = Path(
                str(infile.parent).replace("processed", "processed_with_emb")
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            outfile = out_dir / f"{infile.stem}.zarr"

            # Load existing combined file if present (and not overwrite), else raw
            if outfile.exists() and not cfg.overwrite:
                logger.info("Loading existing combined file %s", outfile)
                infile = outfile

            logger.info("Loading raw AnnData from %s", infile)
            adata = safe_read_h5ad(infile, copy_local=False)
            logger.info(
                "Loaded AnnData with %d cells and %d genes", adata.n_obs, adata.n_vars
            )

            # Determine which methods still need to run
            methods_to_run = []
            for method in cfg.methods:
                obsm_key = f"X_{method}"
                if obsm_key in adata.obsm and not cfg.overwrite:
                    logger.info("Skipping existing embedding '%s'", obsm_key)
                else:
                    methods_to_run.append(method)

            if not methods_to_run:
                logger.info("All embeddings present for %s; skipping write.", infile)
                continue

            # Compute missing embeddings
            for method in methods_to_run:
                if method not in cfg.embedding_dim_map:
                    raise KeyError(f"No embedding_dim for method '{method}' in config")
                emb_dim = cfg.embedding_dim_map[method]

                monitor.log_event(f"Prepare {method}")
                embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)
                embedder.prepare(
                    adata=adata, adata_path=str(infile), batch_key=cfg.batch_key
                )

                monitor.log_event(f"Embed {method}")
                obsm_key = f"X_{method}"
                adata = embedder.embed(
                    adata=adata,
                    obsm_key=obsm_key,
                    batch_key=cfg.batch_key,
                    batch_size=cfg.batch_size,
                )
                monitor.log_event(f"Finished {method}")

            # Write combined AnnData with all embeddings
            logger.info("Writing combined output to %s", outfile)
            safe_write_zarr(adata=adata, target=outfile)
            logger.info("Saved combined embeddings to %s", outfile)

    except Exception as e:
        logger.exception("Embedding pipeline failed, with error: %s", e)
        raise e
    finally:
        monitor.stop()
        monitor.print_summary()
        monitor.save(hydra_run_dir)
        monitor.plot_metrics(hydra_run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in embed_adata.py")
        sys.exit(1)
