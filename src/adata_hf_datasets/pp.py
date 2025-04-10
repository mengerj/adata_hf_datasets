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
        logger.info("Running quality control on data.")
        adata = pp_quality_control(
            adata,
            nmads_main=5,
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


def fetch_sra_metadata(
    adata: anndata.AnnData,
    sample_id_key: str = "accession",
    sra_key: str = "sample_accession",
    exp_id_key: str = "experiment_accession",
    new_cols: str | list[str] = [
        "library_layout",
        "library_source",
        "instrument_model",
    ],
    fallback: str = "unknown",
) -> None:
    """
    Fetch various metadata fields (e.g., 'library_layout', 'library_source')
    from SRA for all unique IDs in `adata.obs[sample_id_key]`, and store them in
    `adata.obs[new_cols]`.

    This function:
    1) Extracts all unique IDs from `adata.obs[sample_id_key]`.
    2) Calls `db.sra_metadata` once to get metadata in a single batch.
    3) Ensures that every requested ID is found in the SRA results (and logs or sets fallback if missing).
    4) Also removes extra rows from the SRA results that do not correspond to your unique IDs.
    5) Merges and assigns the requested columns to `adata.obs`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with IDs in `adata.obs[sample_id_key]`.
    sample_id_key : str, optional
        The column in `adata.obs` containing SRA-based IDs (e.g., SRR).
    sra_key : str, optional
        The column in the returned SRA DataFrame to match your IDs against.
        Defaults to "run_accession".
    exp_id_key : str, optional
        Has to be present in adata.obs and contain SRX IDs. Will be used to match with the db.
    new_cols : str or List[str], optional
        Metadata columns to copy from SRA results into `adata.obs`.
    fallback : str, optional
        Value to use if a column is missing or if some IDs are not found.
        Defaults to "unknown".

    Returns
    -------
    None
        Modifies `adata.obs[new_cols]` in place.

    Examples
    --------
    >>> adata = sc.read_h5ad("my_data.h5ad")
    >>> fetch_sra_metadata(adata, sample_id_key="accession", sra_key="run_accession")
    >>> adata.obs["library_layout"].value_counts()
    """

    if isinstance(new_cols, str):
        new_cols = [new_cols]

    logger.info("Fetching SRA metadata for %d samples.", adata.n_obs)

    if sample_id_key not in adata.obs.columns:
        raise ValueError(f"Column '{sample_id_key}' not found in adata.obs.")

    # 1) Extract all unique IDs
    unique_ids = adata.obs[sample_id_key].dropna().unique().tolist()
    experiment_accession = adata.obs[exp_id_key].dropna().unique().tolist()
    logger.info("Found %d unique IDs in adata.obs[%s].", len(unique_ids), sample_id_key)

    if not unique_ids:
        msg = f"No unique IDs found in adata.obs[{sample_id_key}]. Cannot proceed."
        logger.error(msg)
        raise ValueError(msg)

    db = SRAweb()

    # 2) Single batch query
    try:
        df_all = db.sra_metadata(unique_ids)
    except Exception as e:
        logger.error("Failed to fetch metadata for IDs: %s", e)
        raise ValueError(f"Failed to fetch metadata for IDs: {e}")

    if df_all is None or df_all.empty:
        msg = (
            "No metadata returned. Check if the IDs are valid or if SRA is accessible."
        )
        logger.error(msg)
        raise ValueError(msg)

    # 3) Check if sra_key is in df_all
    if sra_key not in df_all.columns:
        logger.error(
            "sra_key='%s' not in returned columns. Found: %s", sra_key, df_all.columns
        )
        raise ValueError(f"sra_key='{sra_key}' not in returned metadata columns.")

    # Identify the set of unique experimental ids
    desired_ids = set(experiment_accession)
    # ...and the set of IDs returned
    returned_ids = set(df_all[exp_id_key].unique())
    logger.info("Returned %d unique IDs from SRA metadata.", len(returned_ids))

    # 4) Check for missing IDs
    missing_ids = desired_ids - returned_ids
    if missing_ids:
        logger.warning(
            "Some IDs in adata.obs[%s] were not found in the SRA metadata: %s",
            sample_id_key,
            missing_ids,
        )
        logger.warning("These will be assigned fallback='%s'.", fallback)

    # 5) Filter out extra rows in df_all that we do not actually need
    extra_ids = returned_ids - desired_ids
    if extra_ids:
        logger.info(
            "Removing %d extra IDs not present in adata.obs[%s].",
            len(extra_ids),
            sample_id_key,
        )
        df_all = df_all[~df_all[exp_id_key].isin(extra_ids)]

    # Now df_all only has rows that correspond to the IDs we asked for (though some might be missing).
    # Next, check if new_cols exist. We'll apply fallback for missing columns.
    missing_cols = [col for col in new_cols if col not in df_all.columns]
    if missing_cols:
        logger.warning(
            "Some requested columns are missing in metadata: %s", missing_cols
        )
        # We'll still create them in adata.obs with fallback

    # Subset the SRA DataFrame to only the columns that exist
    keep_cols = [c for c in new_cols if c in df_all.columns]
    df_map = df_all[[exp_id_key] + keep_cols].copy()
    # some ids might map to several entries in the database, eg multiple runs for the same sample.
    # Since we cant really assess which would be the correct entry, we will just keep the first and expect our metadata to be the same in all cases
    df_map = df_map.drop_duplicates(subset=exp_id_key, keep="first")

    # Merge
    # if the columns are already in adata.obs, remove them first

    obs_reset = adata.obs.reset_index(drop=False)
    cols_to_drop = [c for c in keep_cols if c in obs_reset.columns]
    obs_reset = obs_reset.drop(columns=cols_to_drop, errors="ignore")
    merged = obs_reset.merge(
        df_map,
        how="left",
        left_on=exp_id_key,
        right_on=exp_id_key,
    )

    # 6) Assign columns to adata.obs, fill fallback for missing rows or columns
    for col in new_cols:
        if col in merged.columns:
            adata.obs[col] = merged[col].fillna(fallback).values
        else:
            # This means the column was never in df_all
            adata.obs[col] = fallback
            logger.warning(
                "Column '%s' not found in SRA metadata. Using fallback='%s'.",
                col,
                fallback,
            )

    logger.info(
        "Successfully added columns %s to adata.obs using fallback='%s'.",
        new_cols,
        fallback,
    )


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
    label_prefix: str = None,
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
    label_prefix : str, optional
        Prefix for the label column in `adata.obs`. If None, defaults to
        f"{column_name}_label".

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
    if label_prefix is None:
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
            str_labels.append("low")
        elif lbl == high_cluster_id:
            str_labels.append("high")
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
