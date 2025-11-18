# src/my_pipeline/preprocessing.py
import logging
from pathlib import Path
from .qc import pp_quality_control
from .general import pp_adata_general
from .geneformer import pp_adata_geneformer
from .utils import (
    ensure_raw_counts_layer,
    prepend_instrument_to_description,
    delete_layers,
)
from .bimodal import split_if_bimodal
from .sra import maybe_add_sra_metadata
from .loader import BatchChunkLoader
from ..file_utils import sanitize_zarr_keys
import numpy as np
from anndata import concat
import pandas as pd
import zarr

logger = logging.getLogger(__name__)


def preprocess_adata(
    adata,
    *,
    min_cells: int = 10,
    min_genes: int = 200,
    batch_key: str = "batch",
    count_layer_key: str = "counts",
    n_top_genes: int = 1_000,
    consolidation_categories: list[str] | str | None = None,
    category_threshold: int = 1,
    remove_low_frequency: bool = True,
    geneformer_pp: bool = True,
    sra_chunk_size: int | None = None,
    sra_extra_cols: list[str] | None = None,
    skip_sra_fetch: bool = False,
    sra_max_retries: int = 3,
    sra_continue_on_fail: bool = False,
    instrument_key: str | None = None,
    description_key: str | None = None,
    bimodal_col: str | None = None,
    split_bimodal: bool = False,
    layers_to_delete: list[str] | None = None,
):
    """
    Preprocess an AnnData object in memory with quality control, filtering, and normalization.

    This function applies a full preprocessing pipeline to an AnnData object, including:
      1. Ensuring raw counts are properly stored
      2. Optional layer deletion
      3. Optional bimodal splitting and independent normalization
      4. SRA metadata fetching
      5. Quality control filtering
      6. General preprocessing (filtering, HVG selection, category consolidation)
      7. Optional Geneformer preprocessing (includes ensembl IDs)
      8. Optional instrument description prepending

    If `split_bimodal` is True, the data will be split based on a technical covariate,
    each split will be normalized independently, and then merged back together. This is
    useful when a technical factor (e.g., sequencing depth, batch effect) dominates the
    variance and normalization should be applied separately to each subgroup.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to preprocess.
    min_cells : int, optional
        Minimum cells per gene for filtering (default 10).
    min_genes : int, optional
        Minimum genes per cell for filtering (default 200).
    batch_key : str, optional
        Column in `.obs` for batch labels (default "batch").
    count_layer_key : str, optional
        Key for the raw counts layer in `.layers` (default "counts").
    n_top_genes : int, optional
        Number of HVGs to select (default 1000).
    consolidation_categories : list[str] | str | None, optional
        List of categories in adata.obs to consolidate low frequency categories.
        If None, no consolidation is performed.
    category_threshold : int, optional
        Frequency threshold for consolidating categories (default 1).
    remove_low_frequency : bool, optional
        If True, remove rows (cells) with low-frequency categories. Otherwise, merges them into "unknown".
    geneformer_pp : bool, optional
        Whether to run the Geneformer step (default True).
    sra_chunk_size : int, optional
        If provided, chunk size for SRA metadata queries.
    sra_extra_cols : list[str], optional
        Additional columns to fetch from SRA.
    skip_sra_fetch : bool, optional
        If True, skip fetching SRA metadata altogether.
    sra_max_retries : int, optional
        Maximum number of retries for SRA connections.
    sra_continue_on_fail : bool, optional
        If True, continue processing even if SRA fetching fails.
    instrument_key : str, optional
        Key in `.obs` holding instrument names. If provided, the instrument name will be prepended to the sample description.
    description_key : str, optional
        Key in `.obs` holding sample descriptions.
    bimodal_col : str, optional
        Column in `.obs` for bimodal splitting. Required if `split_bimodal` is True.
    split_bimodal : bool, optional
        If True, splits the data into two subgroups based on `bimodal_col`, processes
        each subgroup independently (with separate normalization), and then merges them
        back together. This is only used when a technical factor dominates the variance
        and independent normalization is desired (default False).
    layers_to_delete : list[str] | None, optional
        List of layer names to delete from adata.layers. If None, no layers are deleted.

    Returns
    -------
    AnnData
        The preprocessed AnnData object. If `split_bimodal` was True, this is the merged
        result of independently processed splits. Otherwise, it is the directly processed
        AnnData object.

    Notes
    -----
    When `split_bimodal` is True, the merging step combines the independently processed
    splits back into a single AnnData object. This merging preserves all genes from all
    splits (using an outer join) and manually preserves var attributes across splits.
    If `split_bimodal` is False, no merging occurs and the function returns the directly
    processed AnnData object.
    """
    # Make sure X contains raw counts, and "counts" layer is set
    ensure_raw_counts_layer(adata, raw_layer_key=count_layer_key)

    # Delete specified layers early in the process
    adata = delete_layers(adata, layers_to_delete)

    processed_splits = []
    if split_bimodal and bimodal_col in adata.obs:
        # log-transform the covariate (use pandas or numpy, not sc.pp.log1p on a Series)
        log_col = f"{bimodal_col}_log"
        adata.obs[log_col] = np.log1p(adata.obs[bimodal_col].values)
        adata_splits = split_if_bimodal(adata, column_name=log_col, backed_path=None)
    else:
        adata_splits = {"all": adata}

    # Process each split (not training/val but based on bimodality)
    for _split_label, ad_sub in adata_splits.items():
        # Process each chunk
        if sra_chunk_size and sra_extra_cols:
            ad_sub = maybe_add_sra_metadata(
                ad_sub,
                chunk_size=sra_chunk_size,
                new_cols=sra_extra_cols,
                skip_sra_fetch=skip_sra_fetch,
                max_retries=sra_max_retries,
                continue_on_fail=sra_continue_on_fail,
            )
        else:
            logger.info("Skipping SRA metadata fetching as requested.")
        ad_sub = pp_quality_control(ad_sub)
        ad_sub = pp_adata_general(
            ad_sub,
            min_cells=min_cells,
            min_genes=min_genes,
            batch_key=batch_key,
            n_top_genes=n_top_genes,
            categories=consolidation_categories,
            category_threshold=category_threshold,
            remove=remove_low_frequency,
        )
        if geneformer_pp:
            ad_sub = pp_adata_geneformer(ad_sub)
        if instrument_key and description_key:
            prepend_instrument_to_description(
                ad_sub,
                instrument_key=instrument_key,
                description_key=description_key,
            )
        processed_splits.append(ad_sub)

    # Re-concatenate splits back into a single AnnData
    # This merging step is only performed if bimodal splitting was used
    # It merges obs and var, stacking the cells back together after independent normalization
    if len(processed_splits) > 1:
        # First collect all var attributes from all splits
        var_attrs = {}
        for split in processed_splits:
            for attr_name in split.var.keys():
                if attr_name not in var_attrs:
                    var_attrs[attr_name] = {}
                for gene in split.var_names:
                    if gene in split.var[attr_name]:
                        var_attrs[attr_name][gene] = split.var[attr_name][gene]

        # Now concatenate as usual but with minimal merge
        adata_merged = concat(
            processed_splits,
            join="outer",  # outer join on vars to keep all genes
            label="bimodal_split",  # adds an .obs['bimodal_split'] column
            fill_value=0,  # fill missing values with 0
            merge=None,  # don't try to merge var attributes automatically
        )

        # Manually add back the var attributes
        for attr_name, attr_dict in var_attrs.items():
            # Create a Series with the right index
            adata_merged.var[attr_name] = pd.Series(
                {gene: attr_dict.get(gene, None) for gene in adata_merged.var_names}
            )
    else:
        adata_merged = processed_splits[0]

    return adata_merged


def preprocess_h5ad(
    infile: Path,
    outdir: Path,
    *,
    chunk_size: int = 1_000,
    min_cells: int = 10,
    min_genes: int = 200,
    batch_key: str = "batch",
    count_layer_key: str = "counts",
    n_top_genes: int = 1_000,
    consolidation_categories: list[str] | str | None = None,
    category_threshold: int = 1,
    remove_low_frequency: bool = True,
    geneformer_pp: bool = True,
    sra_chunk_size: int | None = None,
    sra_extra_cols: list[str] | None = None,
    skip_sra_fetch: bool = False,
    sra_max_retries: int = 3,
    sra_continue_on_fail: bool = False,
    instrument_key: str | None = None,
    description_key: str | None = None,
    bimodal_col: str | None = None,
    split_bimodal: bool = False,
    output_format: str = "zarr",
    layers_to_delete: list[str] | None = None,
) -> None:
    """
    Preprocess a large AnnData file in chunks and writes each chunk to disk.

    This function handles chunked processing of large AnnData files that may not fit
    in memory. It:
      1. Iterates over smaller AnnData chunks loaded from disk.
      2. Calls `preprocess_adata` to apply preprocessing steps to each chunk.
      3. Writes each processed chunk back to disk.

    The actual preprocessing logic (QC, filtering, normalization, etc.) is handled by
    `preprocess_adata`. This function is a wrapper that provides chunked I/O for large files.

    Parameters
    ----------
    infile : Path
        Path to the raw AnnData file (.h5ad or .zarr).
    outdir : Path
        Path to the directory where the chunked output files will be saved.
    chunk_size : int, optional
        Number of cells per chunk (default 1000).
    min_cells : int, optional
        Minimum cells per gene for filtering (default 10). Passed to `preprocess_adata`.
    min_genes : int, optional
        Minimum genes per cell for filtering (default 200). Passed to `preprocess_adata`.
    batch_key : str, optional
        Column in `.obs` for batch labels (default "batch"). Passed to `preprocess_adata`.
    count_layer_key : str, optional
        Key for the raw counts layer in `.layers` (default "counts"). Passed to `preprocess_adata`.
    n_top_genes : int, optional
        Number of HVGs to select (default 1000). Passed to `preprocess_adata`.
    consolidation_categories : list[str] | str | None, optional
        List of categories in adata.obs to consolidate low frequency categories.
        If None, no consolidation is performed. Passed to `preprocess_adata`.
    category_threshold : int, optional
        Frequency threshold for consolidating categories (default 1). Passed to `preprocess_adata`.
    remove_low_frequency : bool, optional
        If True, remove rows (cells) with low-frequency categories. Otherwise, merges them into "unknown".
        Passed to `preprocess_adata`.
    geneformer_pp : bool, optional
        Whether to run the Geneformer step (default True). Passed to `preprocess_adata`.
    sra_chunk_size : int, optional
        If provided, chunk size for SRA metadata queries. Passed to `preprocess_adata`.
    sra_extra_cols : list[str], optional
        Additional columns to fetch from SRA. Passed to `preprocess_adata`.
    skip_sra_fetch : bool, optional
        If True, skip fetching SRA metadata altogether. Passed to `preprocess_adata`.
    sra_max_retries : int, optional
        Maximum number of retries for SRA connections. Passed to `preprocess_adata`.
    sra_continue_on_fail : bool, optional
        If True, continue processing even if SRA fetching fails. Passed to `preprocess_adata`.
    instrument_key : str, optional
        Key in `.obs` holding instrument names. If provided, the instrument name will be prepended to the sample description. Passed to `preprocess_adata`.
    description_key : str, optional
        Key in `.obs` holding sample descriptions. Passed to `preprocess_adata`.
    bimodal_col : str, optional
        Column in `.obs` for bimodal splitting. Passed to `preprocess_adata`.
        See `preprocess_adata` documentation for details on bimodal splitting.
    split_bimodal : bool, optional
        If True, splits the data into two subgroups based on `bimodal_col`, processes
        each subgroup independently (with separate normalization), and then merges them
        back together. This merging is only performed when `split_bimodal` is True.
        Used when a technical factor dominates the variance and independent normalization
        is desired. Passed to `preprocess_adata` (default False).
    output_format : str, default="zarr"
        Format to write the output file. Must be either "zarr" or "h5ad".
    layers_to_delete : list[str] | None, optional
        List of layer names to delete from adata.layers. If None, no layers are deleted.
        Passed to `preprocess_adata`.

    Notes
    -----
    This function is a wrapper around `preprocess_adata` that provides chunked I/O
    capabilities for large files. For processing AnnData objects that already fit in
    memory, use `preprocess_adata` directly.

    The merging step mentioned in `preprocess_adata` documentation only occurs when
    `split_bimodal` is True. In that case, each chunk's splits are independently
    normalized and then merged back together before writing to disk.

    References
    ----------
    - Data source: single-cell RNA‑seq count matrix stored in H5AD or Zarr format.
    """
    if output_format not in ["zarr", "h5ad"]:
        raise ValueError("output_format must be either 'zarr' or 'h5ad'")

    chunk_dir = outdir
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Initialize loader based on input file format
    if infile.suffix == ".zarr":
        logger.info("Using zarr format for input file")
        loader = BatchChunkLoader(
            infile, chunk_size, batch_key=batch_key, file_format="zarr"
        )
    else:
        logger.info("Using h5ad format for input file")
        loader = BatchChunkLoader(
            infile, chunk_size, batch_key=batch_key, file_format="h5ad"
        )

    for i, adata in enumerate(loader):
        try:
            logger.info("Preprocessing chunk %d", i)
            # Process the chunk using the main preprocessing function
            adata_merged = preprocess_adata(
                adata,
                min_cells=min_cells,
                min_genes=min_genes,
                batch_key=batch_key,
                count_layer_key=count_layer_key,
                n_top_genes=n_top_genes,
                consolidation_categories=consolidation_categories,
                category_threshold=category_threshold,
                remove_low_frequency=remove_low_frequency,
                geneformer_pp=geneformer_pp,
                sra_chunk_size=sra_chunk_size,
                sra_extra_cols=sra_extra_cols,
                skip_sra_fetch=skip_sra_fetch,
                sra_max_retries=sra_max_retries,
                sra_continue_on_fail=sra_continue_on_fail,
                instrument_key=instrument_key,
                description_key=description_key,
                bimodal_col=bimodal_col,
                split_bimodal=split_bimodal,
                layers_to_delete=layers_to_delete,
            )

            # Write chunk with appropriate format
            chunk_path = chunk_dir / f"chunk_{i}.{output_format}"
            logger.info("Writing chunk %d to %s", i, chunk_path)
            if output_format == "zarr":
                # Sanitize column names to remove forward slashes (not allowed in Zarr keys)
                sanitize_zarr_keys(adata_merged)
                adata_merged.write_zarr(chunk_path)
                # Consolidate metadata to avoid KeyError when reading
                zarr.consolidate_metadata(chunk_path)
            else:
                adata_merged.write_h5ad(chunk_path)

        except Exception as e:
            logger.error(f"Error processing chunk {i}: {e}")
            continue

    logger.info("Finished processing all chunks")
