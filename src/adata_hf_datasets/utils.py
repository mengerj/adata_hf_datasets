import logging

import anndata
import numpy as np

from datetime import datetime
import os

from pathlib import Path
import importlib
from huggingface_hub import HfApi
import tempfile
from string import Template
import scipy.sparse as sp
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def split_anndata(adata: anndata.AnnData, train_size: float = 0.8):
    """
    Splits an AnnData object into training and validation sets.

    Parameters
    ----------
    adata
        The complete AnnData object to be split.
    train_size
        The proportion of the dataset to include in the train split. Should be between 0 and 1.

    Returns
    -------
    anndata.AnnData: The training AnnData object.
    anndata.AnnData: The validation AnnData object.
    """
    # Ensure train_size is a valid proportion
    if not 0 < train_size < 1:
        raise ValueError("train_size must be a float between 0 and 1.")

    # Generate random indices
    indices = np.arange(adata.n_obs)
    np.random.shuffle(indices)

    # Calculate the number of observations for the train set
    train_indices_count = int(train_size * adata.n_obs)

    # Split indices for train and validation sets
    train_indices = indices[:train_indices_count]
    val_indices = indices[train_indices_count:]

    # Subset the AnnData object
    train_adata = adata[train_indices]
    val_adata = adata[val_indices]

    return train_adata, val_adata


def consolidate_low_frequency_categories(
    adata: anndata.AnnData, columns: list | str, threshold: int, remove=False
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


def fix_non_numeric_nans(adata: anndata.AnnData) -> None:
    """
    For each column in ``adata.obs`` that is not strictly numeric,
    replace NaN with 'unknown'. This prevents mixed float/string
    issues that SCVI can run into when sorting categorical columns.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to fix in-place. Must have .obs attribute.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype

    for col in adata.obs.columns:
        if is_numeric_dtype(adata.obs[col]):
            # strictly numeric -> do nothing
            continue

        if is_categorical_dtype(adata.obs[col]):
            # For a categorical column, we must add a new category
            # before filling with it
            if "unknown" not in adata.obs[col].cat.categories:
                adata.obs[col] = adata.obs[col].cat.add_categories(["unknown"])
            adata.obs[col] = adata.obs[col].fillna("unknown")
        else:
            # For object/string columns, cast to str, then fillna
            adata.obs[col] = adata.obs[col].astype(str).fillna("unknown")


def setup_logging():
    """Set up logging configuration for the module.

    This function configures the root logger to display messages in the console and to write them to a file
    named by the day. The log level is set to INFO.
    """

    # Create the logs directory
    os.makedirs("logs", exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (important if function is called multiple times)
    if not logger.hasHandlers():
        # Create a file handler
        log_file = logging.FileHandler(
            f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        log_file.setLevel(logging.INFO)

        # Create a console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file.setFormatter(formatter)
        console.setFormatter(formatter)

        # Add the handlers to the root logger
        logger.addHandler(log_file)
        logger.addHandler(console)

    return logger


def annotate_and_push_dataset(
    dataset,
    repo_id: str | None = None,
    private: bool = False,
    readme_template_name: str | None = None,
    embedding_generation: str | None = None,
    caption_generation: str | None = None,
    dataset_type_explanation: str | None = None,
) -> None:
    """Annotates and pushes the dataset to Hugging Face.

    A README.md is dynamically created based on a template and a given unique description.
    The README is stored in a temporary directory and uploaded before being deleted.

    Parameters
    ----------
    repo_id (str, optional):
        Repository ID for Hugging Face. If provided, the dataset will be pushed to Hugging Face.
    private (bool, optional):
        If True, the dataset will be private on Hugging Face. Default is False.
    readme_template_name (str, optional):
        The name of the README template to use. Has to be stored in the package resources.
    embedding_generation (str, optional):
        A description of how the embeddings stored in .obsm of the adata files were generated.
    caption_generation (str, optional):
        A description of how the captions stored in .obs of the adata files were generated.
    dataset_type_explanation (str, optional):
        A description of the dataset type. E.g. "pairs" or "multiplets".
    """

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        readme_path = Path(temp_dir) / "README.md"

        # Write the dynamically generated README file
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                _generate_readme(
                    readme_template_name=readme_template_name,
                    repo_id=repo_id,
                    embedding_generation=embedding_generation,
                    caption_generation=caption_generation,
                    dataset_type_explanation=dataset_type_explanation,
                )
            )

        # Push dataset with README
        dataset.push_to_hub(repo_id, private=private)

        # Upload README file
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )


def _generate_readme(
    readme_template_name,
    repo_id,
    embedding_generation,
    caption_generation=None,
    dataset_type_explanation=None,
) -> str:
    """
    Fills the README template with dataset-specific details.

    Returns
    -------
    str
        The formatted README content.
    readme_template_name
        The name of the template file to use. E.g cellwhisperer
    embedding_generation
        A description of how the embeddings stored in .obsm of the adata files were generated.
    caption_generation
        A description of how the captions stored in .obs of the adata files were generated. Not needed for inference datasets.
    """
    if caption_generation is None:
        caption_info = ""
    else:
        caption_info = f"""The caption entry of the dataset contains a textual description of the dataset, it was generated like this:{caption_generation}"""
    readme_template = _load_readme_template(readme_template_name=readme_template_name)
    readme_filled = Template(readme_template).safe_substitute(
        repo_id=repo_id,
        embedding_generation=embedding_generation,
        caption_generation=caption_info,
        dataset_type_explanation=dataset_type_explanation,
    )
    return readme_filled


def _load_readme_template(readme_template_name) -> str:
    """Generate a README.md file for the dataset."""
    # Load the template stored locally at {project_dir}/dataset_readmes/{readme_template_name}.md
    package_name = "adata_hf_datasets.templates"  # Adjust this to your package name

    try:
        with (
            importlib.resources.files(package_name)
            .joinpath(f"{readme_template_name}.md")
            .open("r", encoding="utf-8") as f
        ):
            return f.read()
    except FileNotFoundError:
        raise ValueError(
            f"Template {readme_template_name} not found in package resources"
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
