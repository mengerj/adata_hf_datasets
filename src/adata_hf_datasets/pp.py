import pybiomart
import anndata
import pandas as pd
import logging
import scanpy as sc
from adata_hf_datasets.utils import (
    consolidate_low_frequency_categories,
    stable_numeric_id,
)
import numpy as np
import scipy.sparse as sp
from scipy.stats import median_abs_deviation
from pysradb import SRAweb
from sklearn.mixture import GaussianMixture
from pathlib import Path
import os
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


def pp_adata(
    adata: anndata.AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    categories: list[str] = None,
    category_threshold: int = 1,
    remove: bool = True,
    call_geneformer: bool = True,
    tag: str | None = None,
) -> anndata.AnnData:
    """
    Create an initial preprocessed AnnData in memory, ready for embeddings.

    This function:
    1. Ensures that `adata.X` is raw counts (if not, raises an error).
       Also stores the raw counts in `adata.layers["counts"]`.
    2. Optionally inserts a `tag` into `adata.uns`.
    3. Runs basic quality control (e.g., removing outlier cells).
    4. Runs general preprocessing (e.g., filtering genes/cells,
       consolidating low-frequency categories).
    5. Optionally calls a geneformer-like step for final touches.
    6. Returns the modified `adata` in memory.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to preprocess in memory.
    min_cells : int, optional
        Minimum number of cells for gene filtering.
    min_genes : int, optional
        Minimum number of genes for cell filtering.
    categories : List[str] | None, optional
        Categories in `adata.obs` to consolidate low-frequency categories.
    category_threshold : int, optional
        Frequency threshold for category consolidation.
    remove : bool, optional
        If True, remove rows (cells) containing low-frequency categories entirely.
        Otherwise, relabel them as 'remaining <col>'. Defaults to True.
    call_geneformer : bool, optional
        If True, call a Geneformer-like in-memory step for additional metadata or steps.
    tag : str | None, optional
        Optional tag to include into the uns object of the AnnData object.

    Returns
    -------
    anndata.AnnData
        The preprocessed AnnData object.

    Notes
    -----
    - This function modifies `adata` in place but also returns it.
    - If you have a file-based workflow, you can read your file externally, call
      this function, then write out the result.
    - The sub-steps `pp_quality_control`, `pp_adata_general`, and `pp_adata_geneformer`
      are assumed to be your previously defined in-memory functions.

    Examples
    --------
    >>> adata = sc.read_h5ad("my_raw_data.h5ad")
    >>> adata = pp_adata_inmemory(adata, min_cells=10, min_genes=200, ...)
    >>> adata.write("my_preprocessed_data.h5ad")
    """
    logger.info(
        "Starting preprocessing. Raw data has %d cells and %d genes.",
        adata.n_obs,
        adata.n_vars,
    )
    if "counts" in adata.layers:
        logger.info("Using pre-existing `adata.layers['counts']` as raw counts.")
        adata.X = adata.layers["counts"]
    elif is_raw_counts(adata.X):
        logger.info("Storing adata.X as raw counts in `adata.layers['counts']`.")
        adata.layers["counts"] = adata.X.copy()
    else:
        logger.error("X does not contain raw counts. Cannot create 'counts' layer.")
        raise ValueError("X does not contain raw counts. Cannot create 'counts' layer.")

    # 1) Optionally add a 'tag' into adata.uns
    if tag:
        logger.info("Storing tag='%s' in adata.uns.", tag)
        adata.uns["tag"] = tag

    try:
        # 2) Run quality control
        logger.info("Running quality control on data.")
        adata = pp_quality_control(
            adata,
            nmads_main=3,
            nmads_mt=3,
            pct_counts_mt_threshold=8.0,
            percent_top=[20, 50, 100],
        )

        # 3) Basic in-memory preprocessing
        logger.info("Running general preprocessing (filtering, low-frequency cat).")
        adata = pp_adata_general(
            adata=adata,
            min_cells=min_cells,
            min_genes=min_genes,
            categories=categories,
            category_threshold=category_threshold,
            remove=remove,
        )

        # 4) Optionally call Geneformer preprocessing
        if call_geneformer:
            logger.info("Running Geneformer in-memory step.")
            adata = pp_adata_geneformer(adata)

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

    logger.info(
        "Preprocessing done. Now there are %d cells and %d genes left.",
        adata.n_obs,
        adata.n_vars,
    )
    return adata


def pp_adata_general(
    adata: anndata.AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    categories: list[str] | None = None,
    category_threshold: int = 1,
    remove: bool = True,
) -> anndata.AnnData:
    """
    Create an initial preprocessed AnnData object in memory ready for embeddings.

    This function performs the following steps:
    1. Makes gene and cell names unique.
    2. Filters out genes expressed in fewer than `min_cells` cells and cells
       expressing fewer than `min_genes` genes.
    3. Consolidates low-frequency categories in specified categories of `adata.obs`.
    4. Checks if `adata.X` contains raw counts. If so, stores a copy in
       `adata.layers["counts"]`; otherwise raises an error.
    5. Normalizes and log-transforms the data (in place).

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to preprocess.
    min_cells : int, optional
        Minimum number of cells in which a gene must be expressed to keep that gene.
    min_genes : int, optional
        Minimum number of genes a cell must express to keep that cell.
    categories : List[str] | None, optional
        categories in `adata.obs` to consolidate low-frequency categories.
        If None, no consolidation is performed.
    category_threshold : int, optional
        Frequency threshold for consolidating categories. Categories with fewer
        than this many occurrences are either removed or renamed to 'remaining <col>'.
    remove : bool, optional
        If True, remove rows (cells) containing low-frequency categories. Otherwise,
        rename them. Defaults to True.

    Returns
    -------
    anndata.AnnData
        The preprocessed AnnData object (modified in place, but also returned).

    References
    ----------
    Data is assumed to be single-cell RNA-seq counts in `adata.X` before processing.
    """
    logger.info("Starting in-memory preprocessing for initial embeddings.")

    # 0) Make ids unique
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # 1) Remove genes and cells below thresholds
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # 2) Consolidate low-frequency categories if categories is not None
    if categories is not None:
        logger.info(
            "Consolidating low-frequency categories in categories: %s with threshold=%d remove=%s",
            categories,
            category_threshold,
            remove,
        )
        adata = consolidate_low_frequency_categories(
            adata, categories, category_threshold, remove=remove
        )
    # verify that there are still cells left
    if adata.n_obs == 0:
        logger.error("No cells left after filtering. Exiting.")
        raise ValueError("No cells left after filtering. Exiting.")
    # 3) Store counts in a new layer if X is raw
    if "counts" not in adata.layers:
        if is_raw_counts(adata.X):
            logger.info("Storing raw counts in adata.layers['counts']")
            adata.layers["counts"] = adata.X.copy()
        else:
            logger.error("X does not contain raw counts. Cannot create 'counts' layer.")
            raise ValueError(
                "X does not contain raw counts. Cannot create 'counts' layer."
            )

    # 4) Normalize and log-transform (in place)
    ensure_log_norm(adata)

    logger.info("In-memory preprocessing complete.")
    return adata


def pp_adata_geneformer(
    adata: anndata.AnnData,
) -> anndata.AnnData:
    """
    Preprocess an AnnData object for Geneformer embeddings, in memory.

    The following steps are performed:
    1. Add ensembl IDs if not present in `adata.var['ensembl_id']`.
    2. Add `n_counts` to `adata.obs` if not already present, by running QC metrics.
    3. Add a stable sample index in `adata.obs['sample_index']` if it doesn't exist.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be preprocessed for Geneformer embeddings.

    Returns
    -------
    anndata.AnnData
        The modified AnnData object (in place, but also returned).

    References
    ----------
    Data is typically in log-transformed or raw form. This function itself
    doesn't enforce a particular transformation, but it expects an
    AnnData with standard .obs and .var annotations.
    """
    logger.info("Preprocessing in-memory AnnData for Geneformer.")

    # 1. Add ensembl IDs if not present
    if "ensembl_id" not in adata.var.columns:
        logger.info("Adding 'ensembl_id' to adata.var.")
        add_ensembl_ids(adata)  # user-provided function

    # 2. Add n_counts if not present
    if "n_counts" not in adata.obs.columns:
        logger.info("Calculating n_counts, which requires scanning the data once.")
        n_genes = adata.n_vars
        percent_top = []
        for p in [50, 100, 200, 500]:
            if p < n_genes:
                percent_top.append(p)

        sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=percent_top)
        adata.obs["n_counts"] = adata.obs["total_counts"]

    # Convert obs_names (which might be string-based) to stable numeric IDs:
    adata.obs["sample_index"] = [stable_numeric_id(str(idx)) for idx in adata.obs_names]

    logger.info("Geneformer in-memory preprocessing complete.")
    return adata


def pp_quality_control(
    adata: anndata.AnnData,
    nmads_main: int = 5,
    nmads_mt: int = 3,
    pct_counts_mt_threshold: float = 8.0,
    percent_top: list[int] | None = None,
    log1p_for_qc: bool = True,
) -> anndata.AnnData:
    """
    Perform quality control filtering on single-cell RNA-seq data based on various metrics.

    This function:
    1. Labels certain genes (mitochondrial, ribosomal, hemoglobin) in `adata.var`.
    2. Calculates QC metrics for these gene sets (e.g., % mitochondrial reads).
    3. Identifies outlier cells using a median absolute deviation (MAD) heuristic
       on metrics like total counts, number of genes, % counts in top genes, etc.
    4. Filters out low-quality cells (e.g., outliers, high % mt).

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing single-cell data.
    nmads_main : int, optional
        The number of MADs to define outliers for main QC metrics (total counts,
        number of genes, etc.). Default is 5.
    nmads_mt : int, optional
        The number of MADs to define outliers for mitochondrial percent. Default is 3.
    pct_counts_mt_threshold : float, optional
        An absolute threshold for % mitochondrial counts. Cells above this are flagged
        as outliers. Default is 8.0 (i.e., 8%).
    percent_top : list of int, optional
        A list of top gene counts for sc.pp.calculate_qc_metrics (e.g., [20, 50, 100]).
        If None, defaults to [20].
    log1p_for_qc : bool, optional
        Whether to compute log1p values for the QC metrics. Default is True.

    Returns
    -------
    anndata.AnnData
        A filtered AnnData object with low-quality cells removed.

    References
    ----------
    Typical single-cell QC steps might also include:
    * Checking for doublets (e.g., with Scrublet or scDblFinder).
    * Checking for additional gene families (RBC, cell-cycle markers, etc.).
    * Using platform/chemistry-specific thresholds (10x vs. Smart-seq).
    * Considering isotype controls or other library-specific signals.

    Examples
    --------
    >>> adata = sc.read_h5ad("my_raw_data.h5ad")
    >>> adata = pp_quality_control(adata, nmads_main=5, nmads_mt=3, pct_counts_mt_threshold=8.0)
    >>> print(adata)
    """

    logger.info("Starting quality control checks.")

    # 1. Label relevant gene categories in adata.var
    logger.info("Labeling mitochondrial, ribosomal, and hemoglobin genes in adata.var")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
    # For hemoglobin, we exclude parentheses and 'P' if you want to skip "HBP"
    # Adjust the regex if needed for your species naming convention
    adata.var["hb"] = adata.var_names.str.upper().str.contains(r"^HB[^P]")

    # 2. Calculate QC metrics
    if percent_top is None:
        percent_top = [20]
    logger.info(
        "Calculating QC metrics with percent_top=%s, log1p=%s, for gene sets [mt, ribo, hb].",
        percent_top,
        log1p_for_qc,
    )
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=percent_top,
        log1p=log1p_for_qc,
    )

    # 3. Define a helper function for outlier detection using median absolute deviation
    def is_outlier(metric_values: np.ndarray, nmads: int) -> pd.Series:
        """
        Returns a boolean mask where True indicates the cell is an outlier
        based on the given metric array and nmads.
        """
        M = metric_values
        med = np.median(M)
        mad = median_abs_deviation(M)
        lower_bound = med - nmads * mad
        upper_bound = med + nmads * mad
        outlier_mask = (M < lower_bound) | (M > upper_bound)
        return outlier_mask

    # 4. Main QC outliers: total counts, number of genes, % counts in top X genes
    # Because we used `log1p=True`, relevant columns are "log1p_total_counts" & "log1p_n_genes_by_counts".
    logger.info(
        "Flagging outliers for total counts, number of genes, and %% in top genes."
    )
    outlier_main = is_outlier(
        adata.obs["log1p_total_counts"].values, nmads_main
    ) | is_outlier(adata.obs["log1p_n_genes_by_counts"].values, nmads_main)
    # e.g. 'pct_counts_in_top_20_genes' if percent_top=[20].
    if f"pct_counts_in_top_{percent_top[0]}_genes" in adata.obs.columns:
        outlier_main |= is_outlier(
            adata.obs[f"pct_counts_in_top_{percent_top[0]}_genes"].values, nmads_main
        )

    # 5. Mitochondrial outliers
    logger.info("Flagging outliers for mitochondrial fraction.")
    # The metric from sc.pp.calculate_qc_metrics is "pct_counts_mt"
    mt_metric = adata.obs["pct_counts_mt"].values
    outlier_mt = is_outlier(mt_metric, nmads_mt) | (mt_metric > pct_counts_mt_threshold)

    # Combine all outliers
    adata.obs["outlier"] = outlier_main
    adata.obs["mt_outlier"] = outlier_mt

    # 6. Filter out the outliers
    n_before = adata.n_obs
    adata = adata[~adata.obs["outlier"] & ~adata.obs["mt_outlier"]].copy()
    n_after = adata.n_obs

    logger.info(
        "Filtered out %d cells as outliers (main or mt). Remaining cells: %d.",
        n_before - n_after,
        n_after,
    )

    logger.info("QC filtering complete.")
    return adata


def prepend_instrument_to_description(
    adata: anndata.AnnData, instrument_key: str, description_key: str
) -> None:
    """Prepend instrument information to the description field of an AnnData object.

    This function modifies `adata.obs[description_key]` in-place by prepending
    a string indicating the instrument used for the measurement. The instrument
    is sourced from `adata.obs[instrument_key]` for each observation.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Typically of shape (n_obs, n_vars).
    instrument_key : str
        Column name in `adata.obs` containing instrument identifiers.
    description_key : str
        Column name in `adata.obs` containing the existing description to be updated.

    Raises
    ------
    KeyError
        If `instrument_key` or `description_key` are not found in `adata.obs`.

    Notes
    -----
    The updated description will be of the form:
    "This measurement was conducted with <instrument>. <original_description>"
    """
    if instrument_key not in adata.obs.columns:
        raise KeyError(f"'{instrument_key}' not found in adata.obs columns.")

    if description_key not in adata.obs.columns:
        raise KeyError(f"'{description_key}' not found in adata.obs columns.")

    logger.info(
        "Prepending instrument information from '%s' to description in '%s'.",
        instrument_key,
        description_key,
    )

    adata.obs[description_key] = (
        "This measurement was conducted with "
        + adata.obs[instrument_key].astype(str)
        + ". "
        + adata.obs[description_key].astype(str)
    )
    # print a random example
    logger.info(
        "Example: %s",
        adata.obs[description_key].sample(1).values[0],
    )


def maybe_add_sra_metadata(
    adata,
    new_cols: str | list[str] = ["library_layout", "library_source", "instrument"],
    sample_id_key: str = "accession",
    sra_key: str = "sample_accession",
    exp_id_key: str = "experiment_accession",
    chunk_size: int = 10000,
):
    """
    Check if the data is from SRA and fetch metadata if so.

    This function checks if the first entry in adata.obs.index starts with "SRX".
    If so, it assumes the data is from SRA and calls fetch_sra_metadata.
    Otherwise, it does nothing.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    new_cols : str or list of str, optional
        Metadata columns to copy from the SRA results into `adata.obs`.
    sample_id_key : str, optional
        The column in `adata.obs` containing SRA-based IDs (e.g., SRR or other sample-level accessions).
    sra_key : str, optional
        The column in the returned SRA DataFrame to match your IDs against.
        Defaults to "sample_accession".
    exp_id_key : str, optional
        Has to be present in adata.obs and contain SRX IDs. Will be used to merge with the SRA metadata.
    chunk_size : int, optional
        Number of unique IDs to process per chunk. Defaults to 10000.
    """
    adata.obs[exp_id_key] = adata.obs.index
    # will be false if no srx ids are found
    if filter_invalid_sra_ids(adata, srx_column=exp_id_key, srs_column=sample_id_key):
        fetch_sra_metadata(
            adata,
            sample_id_key=sample_id_key,
            sra_key=sra_key,
            exp_id_key=exp_id_key,
            new_cols=new_cols,
            chunk_size=chunk_size,
        )
    else:
        logger.info("Data does not appear to be from SRA. Skipping metadata fetching.")


def filter_invalid_sra_ids(
    adata: anndata.AnnData,
    srx_column: str | None = None,
    srs_column: str | None = None,
    pct_tolerate: float = 0.2,
) -> anndata.AnnData:
    """
    Filter out cells from `adata` where the provided accession columns do not
    contain valid IDs. For SRX IDs, a valid ID starts with 'SRX' followed by digits.
    For SRS IDs, a valid ID starts with 'SRS' followed by digits.

    If both are provided, a cell is kept only if it has valid IDs in both columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    srx_column : str | None, optional
        Column in `adata.obs` to check for valid SRX IDs.
    srs_column : str | None, optional
        Column in `adata.obs` to check for valid SRS IDs.
    pct_tolerate : float, optional
        Fraction of cells that can be invalid before raising an error.
        Defaults to 0.2.

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object containing only cells with valid IDs in all
        specified accession columns.

    Raises
    ------
    KeyError
        If any provided column is not found in `adata.obs`.
    ValueError
        If more than `pct_tolerate * 100` percent of cells in any column are invalid.
    """
    # Create an initial mask that is True for all cells.
    final_mask = pd.Series([True] * adata.n_obs, index=adata.obs.index)

    # Function to check a column against a regex, update final_mask
    def _check_and_filter(col_name: str, pattern_str: str):
        if col_name not in adata.obs.columns:
            return False
        pattern = re.compile(pattern_str)
        # Cast to string and match
        mask = adata.obs[col_name].astype(str).str.match(pattern)
        n_invalid = (~mask).sum()
        n_total = adata.n_obs
        n_tolerated = int(n_total * pct_tolerate)
        if n_invalid == n_total:
            raise ValueError(f"All IDs in '{col_name}' are invalid.")
        if n_invalid > n_tolerated:
            logger.error(
                "More than %.1f%% of IDs in column '%s' are invalid. Example invalid ID: %s",
                pct_tolerate * 100,
                col_name,
                adata.obs[~mask].iloc[0],
            )
            raise ValueError(
                f"More than {pct_tolerate * 100:.1f}% of IDs in column '{col_name}' are invalid."
            )
        if n_invalid > 0:
            logger.warning(
                "Removing %d invalid IDs in column '%s' out of %d cells.",
                n_invalid,
                col_name,
                n_total,
            )
        # Return the boolean mask for valid IDs in this column.
        return mask

    # If srx_column is provided, check and update final_mask.
    if srx_column is not None:
        mask_srx = _check_and_filter(srx_column, r"^SRX\d+$")
        final_mask &= mask_srx

    # If srs_column is provided, check and update final_mask.
    if srs_column is not None:
        mask_srs = _check_and_filter(srs_column, r"^SRS\d+$")
        final_mask &= mask_srs

    adata = adata[final_mask]
    logger.info(
        "After filtering, %d cells remain out of %d.",
        adata.n_obs,
    )
    return True


def fetch_sra_metadata(
    adata: anndata.AnnData,
    sample_id_key: str = "accession",
    sra_key: str = "sample_accession",
    exp_id_key: str = "experiment_accession",
    new_cols: str | list[str] = [
        "library_layout",
        "library_source",
        "instrument",
    ],
    fallback: str = "unknown",
    chunk_size: int = 10000,
) -> None:
    """
    Fetch various metadata fields (e.g., 'library_layout', 'library_source', 'instrument_model')
    from SRA for all unique IDs in `adata.obs[sample_id_key]`, processing in chunks,
    and store them in `adata.obs[new_cols]`.

    This function:
    1) Extracts all unique IDs from `adata.obs[sample_id_key]` and experimental accessions from `adata.obs[exp_id_key]`.
    2) Splits the IDs into chunks (default size: 10,000) and calls `db.sra_metadata` for each chunk.
    3) Concatenates the returned metadata into one DataFrame.
    4) Checks that every requested experimental ID is found in the SRA results (and logs or sets fallback if missing).
    5) Removes extra rows from the SRA results that do not correspond to your unique experimental IDs.
    6) Merges and assigns the requested columns to `adata.obs`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with IDs in `adata.obs[sample_id_key]` and experimental IDs in `adata.obs[exp_id_key]`.
    sample_id_key : str, optional
        The column in `adata.obs` containing SRA-based IDs (e.g., SRR or other sample-level accessions).
    sra_key : str, optional
        The column in the returned SRA DataFrame to match your IDs against.
        Defaults to "sample_accession".
    exp_id_key : str, optional
        Has to be present in adata.obs and contain SRX IDs. Will be used to merge with the SRA metadata.
    new_cols : str or list of str, optional
        Metadata columns to copy from the SRA results into `adata.obs`.
    fallback : str, optional
        Value to use if a column is missing or if some IDs are not found. Defaults to "unknown".
    chunk_size : int, optional
        Number of unique IDs to process per chunk. Defaults to 10000.

    Returns
    -------
    None
        Modifies `adata.obs[new_cols]` in place.

    Examples
    --------
    >>> adata = sc.read_h5ad("my_data.h5ad")
    >>> fetch_sra_metadata(adata, sample_id_key="accession", sra_key="sample_accession",
                           exp_id_key="experiment_accession")
    >>> adata.obs["library_layout"].value_counts()
    """

    if isinstance(new_cols, str):
        new_cols = [new_cols]

    logger.info("Fetching SRA metadata for %d samples.", adata.n_obs)

    if sample_id_key not in adata.obs.columns:
        raise ValueError(f"Column '{sample_id_key}' not found in adata.obs.")

    if exp_id_key not in adata.obs.columns:
        raise ValueError(f"Column '{exp_id_key}' not found in adata.obs.")

    # Deduplicate samples if needed (assuming deduplicate_samples_by_id is defined)
    adata = deduplicate_samples_by_id(adata, sample_id_key)

    # 1) Extract all unique IDs and experimental accession IDs
    unique_ids = adata.obs[sample_id_key].dropna().unique().tolist()
    experiment_accessions = adata.obs[exp_id_key].dropna().unique().tolist()
    logger.info("Found %d unique IDs in adata.obs[%s].", len(unique_ids), sample_id_key)

    if not unique_ids:
        msg = f"No unique IDs found in adata.obs[{sample_id_key}]. Cannot proceed."
        logger.error(msg)
        raise ValueError(msg)

    db = SRAweb()

    # 2) Process the unique_ids in chunks.
    chunks = [
        unique_ids[i : i + chunk_size] for i in range(0, len(unique_ids), chunk_size)
    ]
    logger.info(
        "Processing %d chunks of approximately %d IDs each.", len(chunks), chunk_size
    )

    df_list = []
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        try:
            df_chunk = db.sra_metadata(chunk)
            if df_chunk is not None and not df_chunk.empty:
                df_list.append(df_chunk)
        except Exception as e:
            logger.error("Failed to fetch metadata for chunk: %s", e)
            # Optionally, you can raise here or continue to process the others.
            raise ValueError(f"Failed to fetch metadata for a chunk: {e}")

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
    else:
        msg = "No metadata returned in any chunk. Check if the IDs are valid or if SRA is accessible."
        logger.error(msg)
        raise ValueError(msg)

    # 3) Check if the expected SRA column(s) exist.
    if sra_key not in df_all.columns:
        logger.error(
            "sra_key='%s' not in returned columns. Found: %s", sra_key, df_all.columns
        )
        raise ValueError(f"sra_key='{sra_key}' not in returned metadata columns.")

    # 4) Identify the set of unique experimental IDs.
    desired_experiment_ids = set(experiment_accessions)
    returned_experiment_ids = set(df_all[exp_id_key].unique())
    logger.info(
        "Returned %d unique experimental IDs from SRA metadata.",
        len(returned_experiment_ids),
    )

    # Check for missing experimental IDs.
    missing_ids = desired_experiment_ids - returned_experiment_ids
    if missing_ids:
        logger.warning(
            "Some experimental IDs in adata.obs[%s] were not found in the SRA metadata: %s",
            exp_id_key,
            missing_ids,
        )
        logger.warning("These will be assigned fallback='%s'.", fallback)

    # 5) Remove extra rows from df_all that do not correspond to your desired experimental IDs.
    extra_ids = returned_experiment_ids - desired_experiment_ids
    if extra_ids:
        logger.info(
            "Removing %d extra experimental IDs not present in adata.obs[%s].",
            len(extra_ids),
            exp_id_key,
        )
        df_all = df_all[~df_all[exp_id_key].isin(extra_ids)]

    # 6) Ensure that the requested new columns exist, and drop duplicates so each exp_id appears only once.
    missing_cols = [col for col in new_cols if col not in df_all.columns]
    if missing_cols:
        logger.warning(
            "Some requested columns are missing in metadata: %s", missing_cols
        )

    keep_cols = [col for col in new_cols if col in df_all.columns]
    df_map = df_all[[exp_id_key] + keep_cols].copy()
    df_map = df_map.drop_duplicates(subset=exp_id_key, keep="first")

    # 7) Merge the SRA metadata into the adata.obs.
    obs_reset = adata.obs.reset_index(drop=False)
    # Drop any conflicting columns before merging so we don't get suffixes.
    cols_to_drop = [c for c in keep_cols if c in obs_reset.columns]
    obs_reset = obs_reset.drop(columns=cols_to_drop, errors="ignore")
    merged = obs_reset.merge(
        df_map,
        how="left",
        left_on=exp_id_key,
        right_on=exp_id_key,
    )

    # 8) Assign the new columns to adata.obs, filling missing values with fallback.
    for col in new_cols:
        if col in merged.columns:
            adata.obs[col] = merged[col].fillna(fallback).values
        else:
            adata.obs[col] = fallback
            logger.warning(
                "Column '%s' not found in SRA metadata. Using fallback='%s'.",
                col,
                fallback,
            )

    # logger.info(
    #    "Successfully added columns %s to adata.obs using fallback='%s'.",
    #    new_cols, fallback
    # )


def deduplicate_samples_by_id(
    adata: anndata.AnnData, sample_id_key: str
) -> anndata.AnnData:
    """Deduplicate samples in `adata` based on `sample_id_key`.

    Rows with missing IDs (NA) are dropped.
    If multiple observations share the same ID, one is selected randomly.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    sample_id_key : str
        Column in `adata.obs` that contains sample IDs.

    Returns
    -------
    AnnData
        Deduplicated AnnData object.
    """
    if sample_id_key not in adata.obs.columns:
        raise KeyError(f"'{sample_id_key}' not found in adata.obs.")

    obs = adata.obs

    n_obs_before = adata.n_obs
    obs_no_na = obs.dropna(subset=[sample_id_key])
    unique_ids = obs_no_na[sample_id_key].unique().tolist()

    logger.info("Found %d unique IDs in adata.obs[%s].", len(unique_ids), sample_id_key)

    if len(unique_ids) == len(obs_no_na):
        logger.info(
            "No duplicates found. Dropping %d samples with missing IDs.",
            n_obs_before - len(obs_no_na),
        )
        return adata[obs_no_na.index].copy()

    logger.warning(
        "Found %d samples but only %d unique IDs. Deduplicating and dropping %d samples with missing IDs.",
        n_obs_before,
        len(unique_ids),
        n_obs_before - len(obs_no_na),
    )

    # Randomly select one sample per unique ID
    chosen_indices = (
        obs_no_na.groupby(sample_id_key, group_keys=False)
        .apply(lambda x: x.sample(1, random_state=42))
        .index
    )

    logger.info("Keeping %d samples after deduplication.", len(chosen_indices))

    return adata[chosen_indices].copy()


def is_log_transformed(X: np.ndarray, tol: float = 1e-3) -> bool:
    """
    Check if the data is likely log-transformed using a naive heuristic:
    * We expect few or no negative values,
    * The maximum value typically less than ~50 for log1p scRNA-seq data.

    Parameters
    ----------
    X : numpy.ndarray
        The expression matrix to check.
    tol : float, optional
        Tolerance for negative values. If any value is below -tol, we say it's not log-transformed.

    Returns
    -------
    bool
        True if the data appears log-transformed, False otherwise.
    """
    if np.min(X) < -tol:
        return False
    max_val = np.max(X)
    return max_val < 100


def is_normalized(X: np.ndarray, axis: int = 1, var_threshold: float = 0.2) -> bool:
    """
    Check if the data is likely normalized by testing row-sum consistency.
    Specifically, we calculate the ratio of standard deviation to mean of
    all row sums and compare it against a threshold.

    If row sums are relatively consistent (i.e., ratio < var_threshold),
    we assume the data is "normalized."

    Parameters
    ----------
    X : numpy.ndarray
        The expression matrix to check.
        By default, rows are cells and columns are genes.
    axis : int, optional
        The axis along which to sum. Default is 1 (summing each row/cell).
    var_threshold : float, optional
        The cutoff for the ratio (std/mean) of row sums.
        If std/mean < var_threshold, we declare the data normalized.

    Returns
    -------
    bool
        True if the data appears normalized, False otherwise.

    Notes
    -----
    * This is a naive heuristic. For real single-cell data, row sums
      can still vary significantly, depending on the pipeline.
    * Adjust `var_threshold` based on domain knowledge or empirical observations.
    """
    # Ensure X is dense for quick stats
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X

    sums = X_dense.sum(axis=axis)
    sums = np.array(sums).ravel()
    mean_sum = sums.mean()
    std_sum = sums.std()

    if mean_sum == 0:
        logger.warning(
            "Row sums are zero. Data might be empty or already transformed in a different way."
        )
        return False

    ratio = std_sum / mean_sum
    logger.debug("Row sum ratio: std/mean=%.3f, threshold=%.3f", ratio, var_threshold)
    return ratio < var_threshold


def ensure_log_norm(
    adata: anndata.AnnData, in_place: bool = True, var_threshold: float = 0.2
) -> None:
    """
    Checks if `adata.X` is log-transformed and normalized.
    If not, applies sc.pp.log1p() and sc.pp.normalize_total() in place.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object whose .X matrix will be checked/preprocessed.
    in_place : bool, optional
        If True, modifies `adata.X` in place.
    var_threshold : float, optional
        Threshold used by `is_normalized` to decide if data appears normalized.

    Raises
    ------
    ValueError
        If data cannot be processed or if no modifications can be done.

    Notes
    -----
    * The checks used here are naive heuristics. Adjust them for your data as needed.
    """
    logger.info("Checking if data in adata.X appears log-transformed and normalized.")
    # check if adata is backed and load to memory
    if adata.isbacked:
        adata = adata.to_memory()
    if sp.issparse(adata.X):
        X_arr = adata.X.copy().toarray()
    else:
        X_arr = adata.X.copy()

    already_log = is_log_transformed(X_arr)
    already_norm = is_normalized(X_arr, var_threshold=var_threshold)

    if not already_norm:
        logger.info(
            "Data does not appear to be normalized. Applying sc.pp.normalize_total() in place."
        )
        sc.pp.normalize_total(adata)  # modifies adata.X in place
    else:
        logger.info("Data already appears to be normalized.")

    if not already_log:
        logger.info(
            "Data does not appear to be log-transformed. Applying sc.pp.log1p() in place."
        )
        sc.pp.log1p(adata)  # modifies adata.X in place
    else:
        logger.info("Data already appears to be log-transformed.")


def is_data_scaled(
    X,
    sample_genes: int = 1000,
    sample_cells: int = 1000,
    mean_tol: float = 0.2,
    std_tol: float = 0.2,
    fraction_thresh: float = 0.8,
) -> bool:
    """
    Checks if data is likely z-score scaled (per-gene mean ~ 0, std ~ 1).
    Operates by sampling a subset of cells and genes (optional) to
    compute means & stds per gene.

    Parameters
    ----------
    X : np.ndarray or scipy.sparse.spmatrix
        Expression matrix with shape (n_cells, n_genes).
    sample_genes : int, optional
        Number of genes to randomly sample (if total genes > sample_genes).
    sample_cells : int, optional
        Number of cells to randomly sample (if total cells > sample_cells).
    mean_tol : float, optional
        Tolerance around 0 for the mean. For example, 0.2 means a gene's
        mean must be in [-0.2, 0.2] to be considered "scaled."
    std_tol : float, optional
        Tolerance around 1 for the std. For example, 0.2 means a gene's
        std must be in [0.8, 1.2] to be considered "scaled."
    fraction_thresh : float, optional
        Fraction of genes that must meet the above mean/std criteria
        for the dataset to be considered scaled. For example, 0.8 means
        at least 80% of sampled genes must have mean in [-mean_tol, mean_tol]
        and std in [1-std_tol, 1+std_tol].

    Returns
    -------
    bool
        True if data is likely scaled, False otherwise.

    Notes
    -----
    * If the data is huge, we convert to dense after subselecting some rows
      and columns. For extremely large data, consider chunked approaches
      or adjust sampling.
    * For single-cell data, typical z-score scaling is done per gene
      (columns) after normalization and log transform.
    """
    n_cells, n_genes = X.shape

    # 1) Randomly sample cells and genes if needed
    if n_genes > sample_genes:
        gene_idx = np.random.choice(n_genes, sample_genes, replace=False)
    else:
        gene_idx = np.arange(n_genes)

    if n_cells > sample_cells:
        cell_idx = np.random.choice(n_cells, sample_cells, replace=False)
    else:
        cell_idx = np.arange(n_cells)

    # Subset the matrix
    if sp.issparse(X):
        X_sub = X[cell_idx, :][:, gene_idx].toarray()
    else:
        X_sub = X[cell_idx][:, gene_idx]

    # 2) Compute mean and std for each gene (columns)
    gene_means = X_sub.mean(axis=0)
    gene_stds = X_sub.std(axis=0, ddof=1)  # unbiased estimator

    # 3) Check how many genes fall in the acceptable range
    mean_mask = np.abs(gene_means) <= mean_tol
    std_mask = (gene_stds >= (1 - std_tol)) & (gene_stds <= (1 + std_tol))
    combined_mask = mean_mask & std_mask

    fraction_scaled = np.mean(combined_mask)
    logger.debug("Fraction of genes that appear scaled: %.3f", fraction_scaled)

    return fraction_scaled >= fraction_thresh


def is_raw_counts(
    X,
    min_val: float = -1e-6,
    max_noninteger_fraction: float = 0.01,
    check_sparsity: bool = True,
) -> bool:
    """
    Check if a matrix `X` likely contains raw (integer) single-cell counts.

    Parameters
    ----------
    X : np.ndarray or sparse matrix
        The expression matrix to check. Rows are cells, columns are genes.
    min_val : float, optional
        Minimum allowed value (just under 0 to account for floating point noise).
        If we detect values significantly below 0, we assume it's not raw counts.
    max_noninteger_fraction : float, optional
        Maximum fraction of entries that can be non-integer before we say
        it's not raw counts.
        For example, 0.01 means only up to 1% of entries can be non-integer.
    check_sparsity : bool, optional
        Whether to expect a decent fraction of zeros for typical scRNA data.
        If True, we also confirm that at least e.g. 30% of entries are zeros.
        This is a naive heuristic that can be toggled off for non-sparse data.

    Returns
    -------
    bool
        True if `X` likely contains raw (integer) counts, False otherwise.

    Notes
    -----
    This function uses several heuristics:
      1. Negligible negative values (min(X) >= min_val).
      2. The fraction of non-integer entries is below `max_noninteger_fraction`.
      3. (Optional) The data is somewhat sparse (for scRNA-seq, typically many zeros).

    Adjust thresholds according to your dataset’s characteristics.
    """
    # Convert to a dense array for checks if it's sparse.
    # For large datasets, you may want to sample cells/genes instead of converting fully!
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = X

    # 1. Check for negative values
    if np.min(X_dense) < min_val:
        return False

    # 2. Check fraction of non-integer values
    #    We'll say a value is "integer" if |val - round(val)| < some tiny epsilon
    diffs = np.abs(X_dense - np.round(X_dense))
    # we can treat near-zero as integer
    epsilon = 1e-6
    noninteger_mask = diffs > epsilon
    fraction_noninteger = np.mean(noninteger_mask)

    if fraction_noninteger > max_noninteger_fraction:
        return False

    # 3. Optional check for sparsity (only if typical scRNA-seq data)
    if check_sparsity:
        # e.g. expect at least 30% zeros in typical scRNA raw count data
        zero_fraction = np.mean(X_dense == 0)
        if zero_fraction < 0.3:
            return False

    return True


def check_ensembl_ids(ensembl_ids):
    """
    Checks if all Ensembl gene IDs start with "ENS" and logs a warning for missing IDs.

    Parameters
    ----------
    ensembl_ids : list of str
        A list of Ensembl gene IDs.

    Returns
    -------
    bool
        True if all non-empty IDs start with "ENS", False otherwise.
    """
    missing_count = sum(1 for eid in ensembl_ids if eid in ("", None))
    invalid_count = sum(1 for eid in ensembl_ids if eid and not eid.startswith("ENS"))

    if missing_count > 0:
        logger.warning(
            f"{missing_count} genes do not have an Ensembl ID (empty or None)."
        )

    if invalid_count > 0:
        logger.warning(f"{invalid_count} genes have IDs that do not start with 'ENS'.")

    return invalid_count == 0  # Returns True if all valid IDs start with "ENS"


def add_ensembl_ids(
    adata: anndata.AnnData,
    ensembl_col: str = "ensembl_id",
    species: str = "hsapiens",
    use_cache: bool = False,
):
    """Add Ensembl IDs to an AnnData object based on gene symbols.

    Maps gene symbols in ``adata.var[var_key]`` to Ensembl IDs via biomart and stores them
    in a new column, ``adata.var[ensembl_col]``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object whose var DataFrame contains a column with gene symbols.
    ensembl_col : str, optional
        Column name under which Ensembl IDs will be stored, by default "ensembl_id".
    species : str, optional
        Species name passed to ``sc.queries.biomart_annotations()``. Typically "hsapiens"
        (human) or "mmusculus" (mouse), by default "hsapiens".
    use_cache : bool, optional
        Whether to allow caching for ``sc.queries.biomart_annotations()``. Set to False
        if you run into concurrency issues, by default False.

    Returns
    -------
    None
        The function modifies ``adata`` in-place by adding a new column ``adata.var[ensembl_col]``.

    Notes
    -----
    - This uses the built-in Scanpy function ``sc.queries.biomart_annotations`` to fetch
      gene annotations directly from Ensembl's biomart service.
    - If your gene names are ambiguous or do not match exactly, some mappings may be missing.
    - The gene annotation data is sourced from [Ensembl biomart](https://www.ensembl.org/info/data/biomart/index.html).

    Examples
    --------
    >>> add_ensembl_ids(adata, var_key="gene_name", ensembl_col="ensembl_id")
    >>> adata.var.head()  # Now contains a new column 'ensembl_id'
    """
    # Check that entries in .var_names are valid gene symbols
    logger.info(
        "Gene symbols are exptected to be found in .var_names. An example gene symbol: %s",
        adata.var_names[0],
    )
    # if entries in .var_names are ensmebl ids, directly store them in the ensembl_col
    if adata.var_names[0].startswith("ENS"):
        adata.var[ensembl_col] = adata.var_names
        logger.info(
            f"Directly storing Ensembl IDs from row index in adata.var['{ensembl_col}']."
        )
        return

    logger.info("Fetching biomart annotations from Ensembl. This may take a moment...")

    dataset = pybiomart.Dataset(
        name=f"{species}_gene_ensembl", host="http://www.ensembl.org"
    )
    biomart_df = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])

    # Drop duplicates so that each gene symbol maps to exactly one Ensembl ID
    biomart_df = biomart_df.drop_duplicates(subset="Gene name").set_index("Gene name")

    # Prepare list for the mapped Ensembl IDs
    gene_symbols = adata.var_names
    ensembl_ids = []
    for symbol in gene_symbols:
        if symbol in biomart_df.index:
            ensembl_ids.append(biomart_df.loc[symbol, "Gene stable ID"])
        else:
            ensembl_ids.append("")
    # check that ensembl_ids contain "ENSG" IDs and are of same length adata
    if not check_ensembl_ids(ensembl_ids):
        raise ValueError(
            "Ensembl IDs are not valid. Please check the index column or provide a correct {ensembl_col} column."
        )
    adata.var[ensembl_col] = ensembl_ids

    # Optionally drop rows (genes) with missing Ensembl IDs:
    # missing_mask = adata.var[ensembl_col].isna()
    # if missing_mask.any():
    #     n_missing = missing_mask.sum()
    #     logger.warning(f"{n_missing} genes have no valid Ensembl ID. Dropping them.")
    #     adata._inplace_subset_var(~missing_mask)

    logger.info(
        f"Added column '{ensembl_col}' to adata.var with Ensembl IDs. "
        f"Total genes with mapped IDs: {(~pd.isna(adata.var[ensembl_col])).sum()}/"
        f"{len(adata.var)}."
    )


def remove_zero_variance_genes(adata):
    """Remove genes with zero variance from an AnnData object."""
    logger = logging.getLogger(__name__)
    if sp.issparse(adata.X):
        # For sparse matrices
        gene_variances = np.array(
            adata.X.power(2).mean(axis=0) - np.square(adata.X.mean(axis=0))
        ).flatten()
    else:
        # For dense matrices
        gene_variances = np.var(adata.X, axis=0)
    zero_variance_genes = gene_variances == 0
    num_zero_variance_genes = np.sum(zero_variance_genes)

    if np.any(zero_variance_genes):
        adata = adata[:, ~zero_variance_genes]
        logger.info(f"Removed {num_zero_variance_genes} genes with zero variance.")
        return adata
    else:
        logger.info("No genes with zero variance found.")
        return adata


def remove_zero_variance_cells(adata):
    """Check for cells with zero variance in an AnnData object."""
    logger = logging.getLogger(__name__)
    if sp.issparse(adata.X):
        cell_variances = np.array(
            adata.X.power(2).mean(axis=1) - np.square(adata.X.mean(axis=1))
        ).flatten()
    else:
        cell_variances = np.var(adata.X, axis=1)
    zero_variance_cells = cell_variances == 0
    num_zero_variance_cells = np.sum(zero_variance_cells)
    if np.any(zero_variance_cells):
        adata = adata[~zero_variance_cells, :]
        logger.info(f"Removed {num_zero_variance_cells} cells with zero variance.")
        return adata
    else:
        logger.info("No cells with zero variance found.")
        return adata


def fit_GMM(
    adata: anndata.AnnData,
    column_name: str = "n_genes_by_counts",
    n_components: int = 2,
) -> None:
    """
    Fit a Gaussian Mixture Model (GMM) to the specified column in adata.obs and label
    each sample as 'low' or 'high' based on their cluster mean.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing single-cell or bulk RNA-seq data.
    column_name : str
        The key in `adata.obs` for the column to fit the GMM to.
    n_components : int
        Number of components for the GMM.

    Returns
    -------
    None
        Modifies `adata.obs` in place by adding a column with cluster labels.

    Notes
    -----
    - The cluster with the lower mean is labeled 'low' and the cluster with the
      higher mean is labeled 'high'.
    - Samples with missing values in `column_name` get NaN in the label column.
    """
    label_prefix = column_name

    if column_name not in adata.obs.columns:
        raise ValueError(f"{column_name} not found in adata.obs.")

    # Extract the numeric array
    arr = adata.obs[column_name].dropna().values.reshape(-1, 1)

    # Fit a GMM with n_components
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(arr)

    # Predict cluster labels (0, 1, etc.)
    labels_numeric = gmm.predict(arr)

    # Compute the means of each cluster
    means = []
    for cluster_id in range(n_components):
        cluster_points = arr[labels_numeric == cluster_id]
        means.append((cluster_id, cluster_points.mean()))
    # Sort by mean to identify which cluster is "low" and which is "high"
    means_sorted = sorted(means, key=lambda x: x[1])  # sort by the mean value

    # The cluster with the smaller mean
    low_cluster_id = means_sorted[0][0]
    # The cluster with the larger mean
    high_cluster_id = means_sorted[-1][0]

    # Map numeric labels to string labels
    str_labels = []
    for lbl in labels_numeric:
        if lbl == low_cluster_id:
            str_labels.append(f"low_{column_name}")
        elif lbl == high_cluster_id:
            str_labels.append(f"high_{column_name}")
        else:
            # If n_components > 2, you might need to label these differently
            str_labels.append(f"cluster_{lbl}")
    # log how many samples are in each cluster
    counts = pd.Series(str_labels).value_counts()
    logger.info(f"Cluster counts: {counts.to_dict()}")
    # Attach these labels back to adata.obs
    valid_index = adata.obs[column_name].dropna().index
    adata.obs.loc[valid_index, f"{label_prefix}_label"] = str_labels

    # For samples with missing column_name, set them to NaN
    mask_missing = adata.obs[column_name].isna()
    adata.obs.loc[mask_missing, f"{label_prefix}_label"] = np.nan


def is_bimodal_by_gmm(arr, random_state=42):
    gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(arr)
    gmm2 = GaussianMixture(n_components=2, random_state=random_state).fit(arr)
    # Compare BIC or AIC
    if gmm2.bic(arr) < gmm1.bic(arr):
        return True
    return False


def maybe_fit_bimodality(
    adata: anndata.AnnData, column_name: str = "readsaligned_log"
) -> bool:
    """
    If `column_name` exists in adata.obs, create a log-transformed version,
    then fit a 2-component GMM using 'fit_GMM' to label cells as 'low'/'high'.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to be modified in place.
    column_name : str, optional
        Column in `adata.obs` with the raw alignment counts (e.g., 'readsaligned').

    Returns
    -------
    None
        Modifies `adata.obs` in place by adding a new column with labels ('low'/'high')
        if `column_name` is found.

    Notes
    -----
    - If `column_name` is missing, does nothing (we assume it's not a bulk dataset).
    - We always fit a 2-component GMM. Cells are assigned 'low' or 'high'
      in `adata.obs[label_col]`.
    - If you want a more rigorous check for "bimodality," you could compare
      GMM(2) vs. GMM(1) BIC/AIC inside this function.
    """
    label_col = f"{column_name}_label"
    if column_name not in adata.obs:
        logger.info(
            "No '%s' found in adata.obs. Assuming not bulk; skipping GMM.", column_name
        )
        return False

    arr = np.log1p(adata.obs[column_name].dropna().values.reshape(-1, 1))
    if not is_bimodal_by_gmm(arr):
        logger.info("readsaligned_log not strongly bimodal. Skipping GMM labeling.")
        return False

    # Fit a 2-component GMM using the function you provided
    logger.info(
        "Fitting GMM (n_components=2) on '%s' to label 'low'/'high'.", column_name
    )
    fit_GMM(adata, column_name=column_name, n_components=2)

    # The above will create e.g. "readsaligned_log_label" with 'low', 'high' in adata.obs.
    # If you had a unimodal distribution, you'll still get 2 clusters,
    # but one might be tiny or they'd be close in means.
    logger.info("GMM labeling done. New column in adata.obs['%s'] created.", label_col)
    return True


def split_adata_by_label(
    adata: anndata.AnnData,
    label_col: str = "readsaligned_log_label",
    backed_path: str | Path | None = None,
) -> dict[str, anndata.AnnData]:
    """
    Split the input AnnData into multiple subsets based on a categorical column (e.g., 'low'/'high').

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to split.
    label_col : str
        Column in `adata.obs` used to group cells (e.g., 'readsaligned_log_label').
    labels : list of str, optional
        The label values to split by. If None, automatically use unique values in `label_col`.
    backed_path : str or Path, optional
        If provided, the path to save the split AnnData objects. If None, uses in-memory copies.

    Returns
    -------
    dict[str, anndata.AnnData]
        Mapping from label value -> subset AnnData.

    Notes
    -----
    - If `label_col` doesn't exist or is all missing, returns { 'all': adata }.
    - If you only want certain label values, specify `labels`.
    - This function can be used for e.g. 'low' vs. 'high' splitting after GMM labeling.
    """
    if label_col not in adata.obs.columns:
        logger.warning(
            "Label column '%s' not in adata.obs. Returning original adata only.",
            label_col,
        )
        return {"all": adata}

    unique_vals = adata.obs[label_col].dropna().unique().tolist()
    if not unique_vals:
        logger.warning(
            "Label column '%s' is present but all values are NaN. Returning original adata only.",
            label_col,
        )
        return {"all": adata}

    labels = sorted(unique_vals)

    subsets = {}
    if backed_path is not None:
        logger.info(
            "Intermediate saving of AnnData to %s to avoid double in memory adata",
            backed_path,
        )
        adata.write(backed_path)
        del adata
        for val in labels:
            # If some label doesn't actually appear in the data, skip or create an empty subset
            if val not in unique_vals:
                logger.warning(
                    "Label '%s' not found in '%s'. Skipping.", val, label_col
                )
                continue
            adata_backed = sc.read_h5ad(backed_path, backed="r")
            mask = adata_backed.obs[label_col] == val
            sub_view = adata_backed[mask]
            print(f"Subset has shape: {sub_view.shape} for label='{val}'")
            subsets[val] = sub_view.to_memory()
        adata_backed.file.close()
        os.remove(backed_path)
        del adata_backed
    else:
        for val in labels:
            # If some label doesn't actually appear in the data, skip or create an empty subset
            if val not in unique_vals:
                logger.warning(
                    "Label '%s' not found in '%s'. Skipping.", val, label_col
                )
                continue
            mask = adata.obs[label_col] == val
            subsets[val] = adata[mask].copy()
            print(f"Subset has shape: {subsets[val].shape} for label='{val}'")

    return subsets


def split_if_bimodal(
    adata: anndata.AnnData,
    column_name: str = "readsaligned_log",
    backed_path: str | Path | None = None,
) -> dict[str, anndata.AnnData]:
    """
    Uses `maybe_fit_bimodality` on `column_name` to check if data is bimodal.
    If so, calls `split_adata_by_label` to split into 'low'/'high' clusters (and any extra if n_components>2).
    Otherwise returns {"all": adata}.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to potentially split.
    column_name : str
        The column in `adata.obs` that is checked for bimodality. e.g. 'readsaligned_log'.
    backed_path : str or Path, optional
        If provided, the path to save the split AnnData objects. This is needed for very large datasets to avoid having it in memory twice.
        If None, uses in-memory copies.

    Returns
    -------
    dict[str, anndata.AnnData]
        If `maybe_fit_bimodality` returns True, we get a dict of subsets (e.g., {"low": adata_low, "high": adata_high}).
        If not bimodal, returns {"all": adata}.

    Notes
    -----
    - This function expects that `maybe_fit_bimodality`, `split_adata_by_label`, etc.
      are already defined and imported in the same scope.
    - 'maybe_fit_bimodality' itself calls 'is_bimodal_by_gmm' (BIC check) and if True, calls 'fit_GMM'.
    """
    is_bi = maybe_fit_bimodality(adata, column_name=column_name)
    if is_bi:
        # The label column is typically f"{column_name}_label" in maybe_fit_bimodality.
        label_col = f"{column_name}_label"
        return split_adata_by_label(adata, label_col=label_col, backed_path=backed_path)
    else:
        return {"all": adata}
