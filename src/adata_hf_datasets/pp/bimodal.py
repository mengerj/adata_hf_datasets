import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def fit_GMM(
    adata: AnnData,
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


def maybe_fit_bimodality(adata: AnnData, column_name: str = "readsaligned_log") -> bool:
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
    adata: AnnData,
    label_col: str = "readsaligned_log_label",
    backed_path: str | Path | None = None,
) -> dict[str, AnnData]:
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
    adata: AnnData,
    column_name: str = "readsaligned_log",
    backed_path: str | Path | None = None,
) -> dict[str, AnnData]:
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
