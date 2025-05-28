import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def consolidate_low_frequency_categories(
    adata: AnnData, columns: list | str, threshold: int, remove=False
):
    """Consolidates low frequency categories in specified columns of an AnnData object.

    Modifies the AnnData object's .obs by setting entries in specified columns
    to 'remaining {column_name}' or removing them if their frequency is below a specified threshold.
    Converts columns to non-categorical if necessary to adjust the categories dynamically.

    Parameters
    ----------
    adata
        The AnnData object to be processed.
    columns
        List of column names in adata.obs to check for low frequency.
    threshold
        Frequency threshold below which categories are considered low.
    remove
        If True, categories below the threshold are removed entirely.

    Returns
    -------
    anndata.AnnData: The modified AnnData object.
    """
    # Ensure the object is loaded into memory if it's in backed mode
    if adata.isbacked:
        adata = adata.to_memory()
    adata_cut = adata.copy()
    if not isinstance(columns, list):
        columns = [columns]
    for col in columns:
        if col in adata_cut.obs.columns:
            # Convert column to string if it's categorical
            if isinstance(adata_cut.obs[col].dtype, pd.CategoricalDtype):
                as_string = adata_cut.obs[col].astype(str)
                adata_cut.obs[col] = as_string

            # Calculate the frequency of each category
            freq = adata_cut.obs[col].value_counts()

            # Identify low frequency categories
            low_freq_categories = freq[freq < threshold].index

            if remove:
                # Remove entries with low frequency categories entirely
                mask = ~adata_cut.obs[col].isin(low_freq_categories)
                adata_cut._inplace_subset_obs(mask)
                # Convert column back to categorical with new categories
                adata_cut.obs[col] = pd.Categorical(adata_cut.obs[col])
            else:
                # Update entries with low frequency categories to 'remaining {col}'
                adata_cut.obs.loc[adata_cut.obs[col].isin(low_freq_categories), col] = (
                    f"remaining {col}"
                )

                # Convert column back to categorical with new categories
                adata_cut.obs[col] = pd.Categorical(adata_cut.obs[col])

        else:
            print(f"Column {col} not found in adata_cut.obs")

    return adata_cut


def prepend_instrument_to_description(
    adata: AnnData, instrument_key: str, description_key: str
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


def check_enough_genes_per_batch(
    adata: AnnData,
    batch_key: str,
    min_genes: int,
    var_threshold: float = 1e-6,
) -> None:
    """
    Check whether each batch has at least `min_genes` genes with variance > var_threshold.
    If any batch doesn't meet this, raise a ValueError suggesting ways to fix the issue.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object. Must have `adata.obs[batch_key]`.
    batch_key : str
        Column in `adata.obs` used to identify batches.
    min_genes : int
        The requested number of HVGs (n_top_genes).
        Each batch must have at least this many candidate genes.
    var_threshold : float, optional
        Minimal variance to consider a gene "non-negligible". Defaults to 1e-8.

    Raises
    ------
    ValueError
        If any batch has fewer than `min_genes` genes above the variance threshold.
    """
    if batch_key not in adata.obs.columns:
        logger.warning(
            f"batch_key='{batch_key}' not found in adata.obs. Skipping batch check."
        )
        return adata

    cell_counts = adata.obs[batch_key].value_counts()
    logger.info(
        "Checking that each batch has at least %d genes with variance > %g ...",
        min_genes,
        var_threshold,
    )

    # For each batch, compute how many genes pass the variance threshold
    for bval in cell_counts.index:
        sub = adata[adata.obs[batch_key] == bval]
        if sub.n_obs == 0:
            continue  # no cells here; skip

        X = sub.X.toarray() if sp.issparse(sub.X) else sub.X
        variances = X.var(axis=0)
        var_nonzero = (variances > var_threshold).sum()

        if var_nonzero < min_genes:
            msg = (
                f"Batch '{bval}' has only {var_nonzero} genes with variance > {var_threshold}, "
                f"which is fewer than the requested {min_genes}. "
                "Scanpy will fail to select HVGs in this batch.\n"
                "Removing this batch from the dataset."
            )
            logger.warning(msg)
            # remove this batch from the dataset
            adata = adata[adata.obs[batch_key] != bval]
            adata.uns[f"batch_removal_warning_{bval}"] = msg
        else:
            logger.info(
                "All batches have at least %d genes above variance threshold %g.",
                min_genes,
                var_threshold,
            )
        return adata


def deduplicate_samples_by_id(adata: AnnData, sample_id_key: str) -> AnnData:
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
        return adata[obs_no_na.index]

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
    # if all are true, there is no need to subset
    if len(chosen_indices) == adata.n_obs:
        logger.info("No samples were dropped during deduplication.")
        return adata
    return adata[chosen_indices]


def ensure_raw_counts_layer(
    adata: AnnData,
    raw_layer_key: str | None = None,
    raise_on_missing: bool = False,
) -> None:
    """
    Guarantee that `adata.X` and `adata.layers['counts']` contain the raw count matrix.

    This will (in order):
      1. If `raw_layer_key` is provided and exists in `adata.layers`,
         copy that layer into `'counts'` and set `adata.X` to it.
      2. Else if `adata.X` itself appears to be raw counts (via `is_raw_counts`),
         store a copy as `adata.layers['counts']`.
      3. Else if a `'counts'` layer already exists and is valid,
         set `adata.X` to that layer.
      4. Otherwise log an error (and optionally raise).

    After one of steps 1–3, it re‑validates that `adata.layers['counts']`
    really look like integer counts and logs an error if not.

    Parameters
    ----------
    adata
        AnnData to check/modify in place.
    raw_layer_key
        Optional key in `adata.layers` from which to source raw counts.
    raise_on_missing
        If True, raises ValueError when no valid raw counts are found;
        otherwise just logs an error and leaves `adata` unchanged.

    Raises
    ------
    ValueError
        If `raise_on_missing` is True and no raw counts could be located.
    """
    # 1) Prefer the user‐specified layer
    if raw_layer_key and raw_layer_key in adata.layers:
        logger.info("Using layer '%s' for raw counts", raw_layer_key)
        adata.X = adata.layers[raw_layer_key]
        adata.layers["counts"] = adata.layers[raw_layer_key]
    # 2) Detect if X is raw counts
    elif adata.raw is not None and is_raw_counts(adata.raw.X):
        logger.info("Detected raw counts in adata.raw.X; saving to layer 'counts'")
        adata.layers["counts"] = adata.raw.X.copy()
    elif is_raw_counts(adata.X):
        logger.info("Detected raw counts in adata.X; saving to layer 'counts'")
        adata.layers["counts"] = adata.X.copy()
    # 3) Fall back to an existing 'counts' layer
    elif "counts" in adata.layers and is_raw_counts(adata.layers["counts"]):
        logger.info("Using existing 'counts' layer for raw counts")
        adata.X = adata.layers["counts"]
    else:
        msg = (
            "Could not find raw counts: "
            f"no layer '{raw_layer_key}', adata.X and adata.raw.X not raw, and no valid 'counts' layer."
        )
        logger.error(msg)
        if raise_on_missing:
            raise ValueError(msg)
        return

    # 4) Re‑validate that 'counts' really contains integer counts
    if not is_raw_counts(adata.layers["counts"]):
        logger.error(
            "Layer 'counts' does not appear to contain integer raw counts; "
            "downstream steps may fail."
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
    adata: AnnData, in_place: bool = True, var_threshold: float = 0.2
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

    if not np.issubdtype(adata.X.dtype, np.floating):
        logger.info("Casting adata.X to float64 to ensure HVG works.")
        adata.X = adata.X.astype(np.float64)


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

    return True


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
