#!/usr/bin/env python
"""
Apply initial embeddings to preprocessed AnnData files.

This script:
  1. Reads one or more preprocessed .h5ad or .zarr files.
  2. For each file, loads it into memory once (or loads an existing combined output).
  3. Loops over cfg.methods, computing any missing embeddings.
  4. Stores each embedding in adata.obsm["X_<method>"].
  5. Writes out a single file with all embeddings attached.
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


def check_existing_embeddings(file_path: Path, input_format: str = "auto") -> set[str]:
    """
    Check which embeddings already exist in the file without loading the entire dataset.

    Parameters
    ----------
    file_path : Path
        Path to the AnnData file (.h5ad or .zarr)
    input_format : str
        Format specification: "auto", "h5ad", or "zarr"

    Returns
    -------
    set[str]
        Set of existing obsm keys
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


def get_output_path(
    input_path: Path, output_format: str = "zarr", output_dir: Path = None
) -> Path:
    """
    Generate output path based on input path and desired output format.

    Parameters
    ----------
    input_path : Path
        Path to input file
    output_format : str
        Desired output format: "zarr" or "h5ad"
    output_dir : Path, optional
        Custom output directory. If None, creates processed_with_emb directory

    Returns
    -------
    Path
        Output file path
    """
    if output_dir is None:
        # Default behavior: replace "processed" with "processed_with_emb"
        out_dir = Path(
            str(input_path.parent).replace("processed", "processed_with_emb")
        )
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Set output extension based on format
    if output_format == "zarr":
        output_suffix = ".zarr"
    elif output_format == "h5ad":
        output_suffix = ".h5ad"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return out_dir / f"{input_path.stem}{output_suffix}"


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

    # ---------------------------------------------------------------------
    # Copy (or convert) the source only once, when the target store does
    # not yet exist.  Subsequent calls will append to the same Zarr.
    # ---------------------------------------------------------------------
    if adata_path != outfile and not outfile.exists():
        logger.info("Creating target store %s from %s", outfile, adata_path)

        if adata_path.suffix == ".zarr":
            # ---- Source is already Zarr → directory copy is enough
            import shutil

            shutil.copytree(adata_path, outfile)
            logger.debug("Directory-based Zarr copy completed")

        elif adata_path.suffix == ".h5ad":
            # ---- Source is H5AD → convert to Zarr once
            tmp_adata = ad.read_h5ad(adata_path, backed="r")
            tmp_adata.write_zarr(outfile, compressor=None)  # keep original chunks
            tmp_adata.file.close()
            logger.debug("One-time H5AD → Zarr conversion completed")

        else:
            raise ValueError(
                "Unsupported source type %s. Only .h5ad or .zarr are accepted."
                % adata_path.suffix
            )

        logger.info("Target store initialised; subsequent embeddings will be appended")

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
        Paths to preprocessed files (.h5ad or .zarr).
    cfg.input_format : str, optional
        Input file format: "auto", "h5ad", or "zarr". Default is "auto".
    cfg.output_format : str, optional
        Output file format: "zarr" or "h5ad". Default is "zarr".
    cfg.output_dir : str, optional
        Custom output directory. If None, creates processed_with_emb directory.
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

    # Get format specifications with defaults
    input_format = getattr(cfg, "input_format", "auto")
    output_format = getattr(cfg, "output_format", "zarr")
    output_dir = getattr(cfg, "output_dir", None)

    monitor = SystemMonitor(logger=logger)
    monitor.start()

    try:
        for input_file in cfg.input_files:
            infile = Path(input_file)
            logger.info("Processing file: %s", infile)
            if not infile.exists():
                raise FileNotFoundError(f"Input file not found: {infile}")

            # Generate output path based on configuration
            outfile = get_output_path(
                infile, output_format, Path(output_dir) if output_dir else None
            )
            logger.info("Output file: %s", outfile)

            # Load existing combined file if present (and not overwrite), else raw
            if outfile.exists():
                logger.info("Loading existing combined file %s", outfile)
                file_to_check = outfile
                format_to_check = output_format
            else:
                file_to_check = infile
                format_to_check = input_format

            # Determine which methods still need to run
            methods_to_run = []
            existing_obsm_keys = check_existing_embeddings(
                file_to_check, format_to_check
            )

            for method in cfg.methods:
                obsm_key = f"X_{method}"
                if obsm_key in existing_obsm_keys and not cfg.overwrite:
                    logger.info("Skipping existing embedding '%s'", obsm_key)
                else:
                    methods_to_run.append(method)

            if not methods_to_run:
                logger.info("All embeddings present for %s; skipping.", file_to_check)
                continue

            # Use the existing output file as input if it exists, otherwise use original input
            input_for_processing = outfile if outfile.exists() else infile

            # Compute missing embeddings
            for method in methods_to_run:
                if method not in cfg.embedding_dim_map:
                    raise KeyError(f"No embedding_dim for method '{method}' in config")
                emb_dim = cfg.embedding_dim_map[method]

                monitor.log_event(f"Prepare {method}")
                embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)
                embedder.prepare(
                    adata_path=str(input_for_processing), batch_key=cfg.batch_key
                )

                monitor.log_event(f"Embed {method}")
                obsm_key = f"X_{method}"
                emb_matrix = embedder.embed(
                    adata_path=str(input_for_processing),
                    obsm_key=obsm_key,
                    batch_key=cfg.batch_key,
                    batch_size=cfg.batch_size,
                )
                monitor.log_event(f"Finished {method}")

                append_embedding(
                    adata_path=str(input_for_processing),
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
