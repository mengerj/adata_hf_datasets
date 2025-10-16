#!/usr/bin/env python
"""
Apply initial embeddings to preprocessed AnnData files.

This script supports two modes of operation:

1. Full Pipeline Mode (prepare_only=False, default):
   - Reads one or more preprocessed .h5ad or .zarr files.
   - For each file, loads it into memory once (or loads an existing combined output).
   - Loops over cfg.methods, computing any missing embeddings.
   - Stores each embedding in adata.obsm["X_<method>"].
   - Writes out a single file with all embeddings attached.

2. Prepare-Only Mode (prepare_only=True):
   - Reads one or more preprocessed .h5ad or .zarr files.
   - For each file, loads it into memory once.
   - Loops over cfg.methods, calling only `InitialEmbedder.prepare` to do CPU-intensive setup.
   - Does not write out any AnnData; cached results live internally in your embedder.
   - Useful for GPU-dependent embedders where prepare step is more efficient on CPU.

The prepare-only mode is especially useful for embedders that rely on GPU, as their prepare step
is more efficient on the CPU and would otherwise block the precious GPU for a long time.
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
from adata_hf_datasets.embed import InitialEmbedder
from adata_hf_datasets.sys_monitor import SystemMonitor
from adata_hf_datasets.workflow import apply_all_transformations, validate_config
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
        return ad.read_zarr(file_path)
    elif format_to_use == "h5ad":
        return safe_read_h5ad(file_path)
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
        adata = safe_read_h5ad(file_path)
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


@hydra.main(
    version_base=None,
    config_path=None,  # Will be set dynamically by the launcher
    config_name=None,  # Will be set dynamically by the launcher
)
def main(cfg: DictConfig):
    """
    Apply multiple embedding methods to one or more AnnData files.

    This function supports two modes of operation:
    1. prepare_only=True: Only run the prepare() step for each method without saving embeddings
    2. prepare_only=False: Run the full pipeline (prepare + embed + save)

    The prepare_only parameter can be set via command line:
    - ++prepare_only=true  # Run only prepare step
    - ++prepare_only=false # Run full pipeline (default)

    This function now works with dataset-centric config structure where:
    - cfg.embedding contains all embedding parameters (selected by launcher)
    - Common keys (batch_key, etc.) are at the top level
    - Paths are auto-generated from dataset metadata

    The config is automatically transformed to include:
    - Generated paths (input_files, output_dir, etc.)
    - Propagated common keys to workflow sections
    """
    # Apply all transformations to the resolved config
    logger.info("Applying config transformations...")
    cfg = apply_all_transformations(cfg)

    # Validate the transformed config
    logger.info("Validating config...")
    validate_config(cfg)

    # Select the appropriate embedding configuration section
    # This replaces the unified config approach with direct section selection
    embedding_config_section = getattr(cfg, "embedding_config_section", None)
    if embedding_config_section:
        logger.info(
            f"🔧 Config section selection requested: {embedding_config_section}"
        )

        if hasattr(cfg, embedding_config_section):
            embedding_cfg = getattr(cfg, embedding_config_section)

            # Log the configuration details before copying
            logger.info(f"📋 Found {embedding_config_section} section with:")
            logger.info(f"   - Methods: {getattr(embedding_cfg, 'methods', 'NOT SET')}")
            logger.info(
                f"   - Input files: {getattr(embedding_cfg, 'input_files', 'NOT SET')}"
            )
            logger.info(
                f"   - Batch size: {getattr(embedding_cfg, 'batch_size', 'NOT SET')}"
            )
            logger.info(
                f"   - Output dir: {getattr(embedding_cfg, 'output_dir', 'NOT SET')}"
            )

            # Create a unified embedding config by copying the selected section
            # This maintains compatibility with the rest of the code
            cfg.embedding = embedding_cfg
            logger.info(
                f"✅ Successfully selected {embedding_config_section} configuration"
            )
        else:
            available_sections = [
                key for key in cfg.keys() if key.startswith("embedding")
            ]
            raise ValueError(
                f"Embedding config section '{embedding_config_section}' not found in config. "
                f"Available embedding sections: {available_sections}"
            )
    else:
        # Fallback to legacy unified embedding config
        if hasattr(cfg, "embedding") and cfg.embedding is not None:
            embedding_cfg = cfg.embedding
            logger.info("📋 Using unified embedding configuration (legacy mode)")
        else:
            available_sections = [
                key for key in cfg.keys() if key.startswith("embedding")
            ]
            raise ValueError(
                f"No embedding configuration found and no embedding_config_section specified. "
                f"Available embedding sections: {available_sections}"
            )

    # Now embedding_cfg points to the correct configuration
    # Get prepare_only from command line override (defaults to False)
    prepare_only = getattr(cfg, "prepare_only", False)

    # Log the configuration being used
    logger.info("Dataset: %s", cfg.dataset.name)
    logger.info(
        "Operation mode: %s", "prepare_only" if prepare_only else "full_pipeline"
    )
    logger.info("Embedding methods: %s", embedding_cfg.methods)
    logger.info("Input files: %s", embedding_cfg.input_files)
    logger.info("Output directory: %s", embedding_cfg.output_dir)
    logger.info("Batch key: %s", embedding_cfg.batch_key)

    # Validate that all required embedding parameters are present
    required_embedding_params = [
        "methods",
        "input_files",
        "output_dir",
        "batch_key",
        "batch_size",
        "embedding_dim_map",
    ]
    missing_params = []
    for param in required_embedding_params:
        if not hasattr(embedding_cfg, param) or getattr(embedding_cfg, param) is None:
            missing_params.append(param)

    if missing_params:
        raise ValueError(
            f"Missing required embedding parameters: {missing_params}. "
            f"These should be defined in the dataset config or inherited from defaults."
        )

    # Validate that all methods have corresponding embedding dimensions
    for method in embedding_cfg.methods:
        if method not in embedding_cfg.embedding_dim_map:
            raise KeyError(
                f"Method '{method}' specified in methods but not found in embedding_dim_map. "
                f"Available methods: {list(embedding_cfg.embedding_dim_map.keys())}"
            )

    load_dotenv(override=True)
    hydra_run_dir = HydraConfig.get().run.dir

    # Get format specifications with defaults
    input_format = getattr(embedding_cfg, "input_format", "auto")
    output_format = getattr(embedding_cfg, "output_format", "zarr")
    output_dir_base = getattr(embedding_cfg, "output_dir", None)

    monitor = SystemMonitor(logger=logger)
    monitor.start()

    try:
        for input_file in embedding_cfg.input_files:
            infile = Path(input_file)

            logger.info("Processing file: %s", infile)
            if not infile.exists():
                raise FileNotFoundError(f"Input file not found: {infile}")

            # Load AnnData with format detection
            # adata = load_adata_file(infile, input_format)
            # logger.info(
            #    "Loaded AnnData with %d cells, %d vars", adata.n_obs, adata.n_vars
            # )

            if prepare_only:
                # PREPARE_ONLY MODE: Only run prepare() step
                logger.info(
                    "Running in prepare_only mode - no embeddings will be saved"
                )
                for method in embedding_cfg.methods:
                    if method not in embedding_cfg.embedding_dim_map:
                        raise KeyError(
                            f"No embedding_dim for method '{method}' in config"
                        )
                    emb_dim = embedding_cfg.embedding_dim_map[method]

                    monitor.log_event(f"Prepare {method}")
                    embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)
                    embedder.prepare(
                        adata_path=str(infile),
                        batch_key=embedding_cfg.batch_key,
                    )
                    logger.info("Prepared embedding resources for '%s'", method)
                    monitor.log_event(f"Finished prepare {method}")

                logger.info(
                    "All preparations complete for %s; results cached internally.",
                    infile,
                )

            else:
                # FULL PIPELINE MODE: Run prepare + embed + save
                logger.info("Running full embedding pipeline")

                # Generate output path based on configuration
                # get the split name from the input file and add it to the output dir
                split_name = infile.parent.name
                output_dir = Path(output_dir_base) / split_name
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

                for method in embedding_cfg.methods:
                    obsm_key = f"X_{method}"
                    if obsm_key in existing_obsm_keys and not embedding_cfg.overwrite:
                        logger.info("Skipping existing embedding '%s'", obsm_key)
                    else:
                        methods_to_run.append(method)

                if not methods_to_run:
                    logger.info(
                        "All embeddings present for %s; skipping.", file_to_check
                    )
                    continue

                # Use the existing output file as input if it exists, otherwise use original input
                input_for_processing = outfile if outfile.exists() else infile

                # Compute missing embeddings
                for method in methods_to_run:
                    if method not in embedding_cfg.embedding_dim_map:
                        raise KeyError(
                            f"No embedding_dim for method '{method}' in config"
                        )
                    emb_dim = embedding_cfg.embedding_dim_map[method]

                    monitor.log_event(f"Prepare {method}")
                    embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)
                    embedder.prepare(
                        adata_path=str(input_for_processing),
                        batch_key=embedding_cfg.batch_key,
                    )

                    monitor.log_event(f"Embed {method}")
                    obsm_key = f"X_{method}"

                    # Add robust retry logic for GPU-dependent methods
                    max_retries = 3
                    retry_delay = 30  # seconds

                    for attempt in range(max_retries):
                        try:
                            logger.info(
                                f"Embedding attempt {attempt + 1}/{max_retries} for method '{method}'"
                            )
                            emb_matrix = embedder.embed(
                                adata_path=str(input_for_processing),
                                obsm_key=obsm_key,
                                batch_key=embedding_cfg.batch_key,
                                batch_size=embedding_cfg.batch_size,
                            )
                            logger.info(
                                f"✓ Embedding successful for method '{method}' on attempt {attempt + 1}"
                            )
                            break

                        except RuntimeError as e:
                            error_msg = str(e).lower()
                            is_cuda_error = any(
                                cuda_keyword in error_msg
                                for cuda_keyword in [
                                    "cuda",
                                    "gpu",
                                    "device",
                                    "driver initialization failed",
                                    "out of memory",
                                    "cudnn",
                                    "cublas",
                                ]
                            )

                            if is_cuda_error and attempt < max_retries - 1:
                                logger.warning(
                                    f"CUDA-related error on attempt {attempt + 1}/{max_retries} for method '{method}': {e}"
                                )
                                logger.info(f"Retrying in {retry_delay} seconds...")

                                # Clear CUDA cache if available
                                try:
                                    import torch

                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        logger.info("Cleared CUDA cache")
                                except ImportError:
                                    pass

                                import time

                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                # Non-CUDA error or final attempt - re-raise
                                logger.error(
                                    f"Final attempt failed for method '{method}': {e}"
                                )
                                raise

                        except Exception as e:
                            # Non-RuntimeError exceptions - don't retry
                            logger.error(
                                f"Non-retryable error for method '{method}': {e}"
                            )
                            raise

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
        logger.exception("Fatal error in embed_core.py")
        sys.exit(1)
