import logging
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


def pp_quality_control(
    adata: AnnData,
    nmads_main: int = 5,
    nmads_mt: int = 3,
    pct_counts_mt_threshold: float = 8.0,
    percent_top: list[int] | None = None,
    log1p_for_qc: bool = True,
) -> AnnData:
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
    adata : AnnData
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
    AnnData
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
    # max of percent top can't be greater than n_vars
    for p_top in percent_top:
        if p_top > adata.n_vars:
            logger.error(
                f"max(percent_top)={max(percent_top)} cannot be greater than n_vars={adata.n_vars}. Setting to n_vars - 1."
            )
        percent_top = [adata.n_vars - 1]

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
