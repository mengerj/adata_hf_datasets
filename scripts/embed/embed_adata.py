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
from typing import Union

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
import zarr
import anndata as ad
import numpy as np

from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.file_utils import safe_read_h5ad
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.sys_monitor import SystemMonitor
from hydra.core.hydra_config import HydraConfig

logger = setup_logging()


def check_existing_embeddings(file_path: Path) -> set[str]:
    """
    Check which embeddings already exist in the file without loading the entire dataset.

    Parameters
    ----------
    file_path : Path
        Path to the AnnData file (.h5ad or .zarr)

    Returns
    -------
    set[str]
        Set of existing obsm keys
    """
    if file_path.suffix == ".zarr":
        # For zarr, we can check the obsm group directly
        store = zarr.DirectoryStore(file_path)
        root = zarr.group(store=store)
        if "obsm" in root:
            return set(root["obsm"].keys())
        return set()
    else:
        # For h5ad, we need to load the file to check obsm
        adata = safe_read_h5ad(file_path, copy_local=False)
        return set(adata.obsm.keys())


def append_embedding(
    adata_path: Union[str, Path],
    embedding: np.ndarray,
    outfile: Union[str, Path],
    obsm_key: str,
    chunk_rows: int = 16_384,
) -> Path:
    """
    Append or overwrite an embedding matrix in an AnnData file.

    This function efficiently adds or updates embeddings in AnnData files. For zarr files, it streams
    the data chunk-by-chunk; for h5ad files, it loads the object into memory.

    If the input and output paths differ, the input file is first copied to the
    output location before adding the embedding.

    Parameters
    ----------
    adata_path : Union[str, Path]
        Path to the source AnnData file (.zarr or .h5ad).
    embedding : np.ndarray
        Array of shape (n_cells, n_components) containing the computed embedding.
        Must be in the same cell order as the input file.
    outfile : Union[str, Path]
        Path where the updated AnnData file should be written.
        If different from adata_path, the file will be copied first.
    obsm_key : str
        Target key in adata.obsm, e.g. "X_pca" or "X_scvi_fm".
    chunk_rows : int, default=16384
        Number of cells written per chunk when streaming into Zarr.
        More rows means fewer, larger writes. Tune based on your I/O system.

    Returns
    -------
    Path
        Location of the updated AnnData file.

    Raises
    ------
    ValueError
        If the file format is not supported (.zarr or .h5ad).
    FileNotFoundError
        If the input file does not exist.
    """
    adata_path = Path(adata_path)
    outfile = Path(outfile)

    if not adata_path.exists():
        raise FileNotFoundError(f"Input file not found: {adata_path}")

    logger.info("Processing embedding %s (shape=%s)", obsm_key, embedding.shape)
    logger.info("Source file: %s", adata_path)
    logger.info("Target file: %s", outfile)

    # Handle file copying if needed
    if adata_path != outfile:
        logger.info("Copying %s to %s", adata_path, outfile)
        if adata_path.suffix == ".zarr":
            # For zarr, we need to copy the entire directory
            import shutil

            if outfile.exists():
                shutil.rmtree(outfile)
            shutil.copytree(adata_path, outfile)
        else:
            # For h5ad, we can use a simple file copy
            logger.info("H5AD are not copied, only zarr")
        logger.info("Finished copying file")

    if outfile.suffix == ".zarr":
        logger.info("Opening zarr file in read-write mode")
        root = zarr.open_group(str(outfile), mode="r+")

        # ensure the obsm group exists
        logger.info("Creating/accessing obsm group")
        obsm_grp = root.require_group("obsm")

        # create or overwrite the dataset
        logger.info("Creating/overwriting dataset %s", obsm_key)
        ds = obsm_grp.require_dataset(
            name=obsm_key,
            shape=embedding.shape,
            dtype=embedding.dtype,
            chunks=(chunk_rows, embedding.shape[1]),
            overwrite=True,
        )

        # stream row chunks into the dataset
        logger.info("Streaming embedding data in chunks of %d rows", chunk_rows)
        for start in range(0, embedding.shape[0], chunk_rows):
            stop = min(start + chunk_rows, embedding.shape[0])
            ds[start:stop] = embedding[start:stop]
            logger.debug("Wrote rows %d to %d", start, stop)

        root.store.close()
        logger.info("Successfully stored %s in zarr file", obsm_key)
        return outfile

    if outfile.suffix == ".h5ad":
        logger.info("Loading h5ad file into memory")
        adata = ad.read_h5ad(adata_path)
        logger.info("Adding embedding to AnnData object")
        adata.obsm[obsm_key] = embedding
        logger.info("Writing updated file to %s", outfile)
        adata.write_h5ad(outfile)
        logger.info("Successfully stored %s in h5ad file", obsm_key)
        return outfile

    raise ValueError(
        f"Unsupported file type: {outfile.suffix}. Only .zarr and .h5ad are supported."
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="embed_adata")
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
            if outfile.exists():
                logger.info("Loading existing combined file %s", outfile)
                infile = outfile

            # Determine which methods still need to run
            methods_to_run = []
            existing_obsm_keys = check_existing_embeddings(infile)

            for method in cfg.methods:
                obsm_key = f"X_{method}"
                if obsm_key in existing_obsm_keys and not cfg.overwrite:
                    logger.info("Skipping existing embedding '%s'", obsm_key)
                else:
                    methods_to_run.append(method)

            if not methods_to_run:
                logger.info("All embeddings present for %s; skipping write.", infile)
                continue

            # Load the data only if we need to compute new embeddings
            # logger.info("Loading AnnData from %s", infile)
            # adata = safe_read_h5ad(infile, copy_local=False)
            # logger.info(
            #   "Loaded AnnData with %d cells and %d genes", adata.n_obs, adata.n_vars
            # )

            # Compute missing embeddings
            for method in methods_to_run:
                if method not in cfg.embedding_dim_map:
                    raise KeyError(f"No embedding_dim for method '{method}' in config")
                emb_dim = cfg.embedding_dim_map[method]

                monitor.log_event(f"Prepare {method}")
                embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)
                embedder.prepare(adata_path=str(infile), batch_key=cfg.batch_key)

                monitor.log_event(f"Embed {method}")
                obsm_key = f"X_{method}"
                emb_matrix = embedder.embed(
                    adata_path=str(infile),
                    obsm_key=obsm_key,
                    batch_key=cfg.batch_key,
                    batch_size=cfg.batch_size,
                )
                monitor.log_event(f"Finished {method}")

                append_embedding(
                    adata_path=str(infile),
                    embedding=emb_matrix,
                    outfile=str(outfile),
                    obsm_key=obsm_key,
                )

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
        logger.info("Embedding python script completed successfully.")
    except Exception:
        logger.exception("Fatal error in embed_adata.py")
        sys.exit(1)
