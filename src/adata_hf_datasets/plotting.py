import logging
import os
from typing import List, Optional

import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def qc_evaluation_plots(
    adata: anndata.AnnData,
    subset_cells: int = 5000,
    batch_key: Optional[str] = None,
    qc_vars: Optional[List[str]] = None,
    percent_top: Optional[List[int]] = None,
    log1p_for_qc: bool = True,
    metrics_of_interest: Optional[List[str]] = None,
    save_plots: bool = False,
    save_dir: Optional[str] = None,
) -> None:
    """
    Subset an AnnData to up to 5k cells, compute QC metrics, run PCA,
    and generate multiple plots to assess data quality.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to evaluate. Should contain raw or preprocessed data.
    subset_cells : int, optional
        Maximum number of cells to keep in the subset for plotting. Default is 5000.
    batch_key : str, optional
        Column in `adata.obs` specifying batch labels. If provided, we will
        color PCA plots by batch and produce a plot of n_genes_by_counts vs. batch.
    qc_vars : list of str, optional
        List of boolean columns in `adata.var` for which to calculate QC metrics
        (e.g. ["mt", "ribo", "hb"]). If None, defaults to ["mt", "ribo", "hb"].
    percent_top : list of int, optional
        Values for `percent_top` in `sc.pp.calculate_qc_metrics`. Defaults to [20].
    log1p_for_qc : bool, optional
        If True, `sc.pp.calculate_qc_metrics` will compute log1p metrics
        (e.g., 'log1p_total_counts'). Default is True.
    metrics_of_interest : list of str, optional
        Keys in `adata.obs` to color PCA plots and produce distribution plots.
        If None, defaults to typical metrics:
        ["total_counts", "n_genes_by_counts", "pct_counts_mt"].
    save_plots : bool, optional
        If True, saves each generated plot to `save_dir`. If False, displays them interactively.
    save_dir : str, optional
        Directory where plots will be saved (only if `save_plots=True`).
        If None, defaults to the current directory.

    Returns
    -------
    None
        Generates and optionally saves multiple QC figures.

    References
    ----------
    * Mitochondrial (MT-), ribosomal (RPS/RPL), and hemoglobin (HB) genes
      are common QC features in scRNA-seq data.
    * For massive datasets, consider chunked or backed mode to avoid large memory usage.
    * This function is for quick interactive QC and does not permanently alter
      the original `adata` object (beyond storing computed QC metrics in `.obs`).

    Examples
    --------
    >>> adata = sc.read("my_raw_data.h5ad")
    >>> qc_evaluation_plots(adata, subset_cells=5000, save_plots=True, save_dir="qc_plots")
    """

    logger.info("Starting QC evaluation plots.")

    if save_plots and save_dir is not None:
        # 1) Configure Scanpy
        sc.settings.figdir = save_dir  # Where to save
        sc.settings.file_format_figs = "png"  # Save in PNG
        sc.settings.autosave = True  # Will save automatically
        sc.settings.autoshow = False  # Don't display inline

    # 1) Subset to up to 5k cells
    n_obs = adata.n_obs
    if n_obs > subset_cells:
        logger.info("Subsetting data from %d cells to %d cells.", n_obs, subset_cells)
        idx = np.random.choice(n_obs, subset_cells, replace=False)
        adata_sub = adata[idx, :].copy()
    else:
        logger.info("Data has <= %d cells, no subsetting needed.", subset_cells)
        adata_sub = adata.copy()

    # 2) Label QC genes if needed
    if qc_vars is None:
        qc_vars = ["mt", "ribo", "hb"]  # typical sets
    logger.info("Labeling known QC genes (mt, ribo, hb) if not already labeled.")
    if "mt" not in adata_sub.var.columns:
        adata_sub.var["mt"] = adata_sub.var_names.str.upper().str.startswith("MT-")
    if "ribo" not in adata_sub.var.columns:
        adata_sub.var["ribo"] = adata_sub.var_names.str.upper().str.startswith(
            ("RPS", "RPL")
        )
    if "hb" not in adata_sub.var.columns:
        adata_sub.var["hb"] = adata_sub.var_names.str.upper().str.contains(r"^HB[^P]")

    # 3) Calculate QC metrics
    if percent_top is None:
        percent_top = [20]

    logger.info(
        "Calculating QC metrics with qc_vars=%s, percent_top=%s, log1p=%s",
        qc_vars,
        percent_top,
        log1p_for_qc,
    )
    sc.pp.calculate_qc_metrics(
        adata_sub,
        qc_vars=qc_vars,
        inplace=True,
        percent_top=percent_top,
        log1p=log1p_for_qc,
    )

    # 4) Define metrics_of_interest if none given
    if metrics_of_interest is None:
        # total_counts and n_genes_by_counts always come from sc.pp.calculate_qc_metrics
        # "pct_counts_mt" is typical if "mt" in qc_vars
        metrics_of_interest = ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
    logger.info("Metrics of interest for plotting: %s", metrics_of_interest)

    # 5) PCA: We'll assume data is log-transformed, but let's do minimal steps if needed.
    logger.info("Running PCA for the subset data. This is a quick approximate check.")
    # If the data is not log-transformed, consider calling sc.pp.log1p(adata_sub) here.
    # If you want scaling, consider sc.pp.scale(adata_sub).
    sc.pp.pca(adata_sub)

    # 6) Generate and optionally save multiple plots

    # 6A) PCA scatter, colored by QC metrics
    logger.info("Generating PCA plots for metrics of interest...")
    for metric in metrics_of_interest:
        if metric not in adata_sub.obs.columns:
            logger.warning(
                "Metric %s not found in adata.obs, skipping PCA color plot.", metric
            )
            continue

        sc.pl.pca(adata_sub, color=metric, show=not save_plots, save=f"_{metric}")

    # 6B) Violin plots of metrics
    logger.info("Generating violin plots for metrics of interest...")
    sc.pl.violin(
        adata_sub,
        keys=metrics_of_interest,
        groupby=None,  # no grouping, just overall distribution
        rotation=45,
        show=not save_plots,
        save="_violin",
    )

    # 6C) Histograms of metrics
    logger.info("Generating histogram distributions for metrics of interest...")
    for metric in metrics_of_interest:
        if metric not in adata_sub.obs.columns:
            continue
        plt.figure()
        plt.hist(adata_sub.obs[metric], bins=50, edgecolor="k")
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        if save_plots and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"hist_{metric}.png"), dpi=150)
        else:
            plt.show()
        plt.close()

    # 6D) Example scatter: total_counts vs. pct_counts_mt
    if (
        "total_counts" in adata_sub.obs.columns
        and "pct_counts_mt" in adata_sub.obs.columns
    ):
        logger.info("Generating scatter plot of total_counts vs. pct_counts_mt.")
        plt.figure()
        plt.scatter(
            adata_sub.obs["total_counts"],
            adata_sub.obs["pct_counts_mt"],
            s=10,
            alpha=0.5,
        )
        plt.xlabel("Total Counts")
        plt.ylabel("% Counts MT")
        plt.title("Total Counts vs. % MT")
        if save_plots and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "scatter_counts_mt.png"), dpi=150)
        else:
            plt.show()
        plt.close()

    # --------------------------------------------------------
    # NEW CHUNK: Batch-specific QC inspection
    # --------------------------------------------------------
    if batch_key is not None and batch_key in adata_sub.obs.columns:
        logger.info("Investigating the effect of batch label '%s'.", batch_key)

        # 1) PCA colored by batch
        sc.pl.pca(
            adata_sub, color=batch_key, show=not save_plots, save=f"_pca_{batch_key}"
        )

        # 2) Distribution of n_genes_by_counts by batch
        #    (This helps see if certain batches systematically have more genes.)
        if "n_genes_by_counts" in adata_sub.obs.columns:
            sc.pl.violin(
                adata_sub,
                keys="n_genes_by_counts",
                groupby=batch_key,
                rotation=45,
                stripplot=True,  # add dots on top
                jitter=0.4,
                show=not save_plots,
                save=f"_violin_{batch_key}_n_genes",
            )
        else:
            logger.warning(
                "`n_genes_by_counts` not found in adata.obs. Skipping violin by batch."
            )

    logger.info("Finished generating QC evaluation plots.")


def _move_scanpy_figure(filename, save_dir, new_name=None):
    """
    Internal helper to move the default scanpy output figure to `save_dir`.
    By default, scanpy saves figure outputs to the local `figures` folder
    with `save=True` in sc.pl.* calls.

    Parameters
    ----------
    filename : str
        The name of the figure file that was saved by scanpy (e.g., 'pca_metric.png').
    save_dir : str
        The directory to which to move the figure.
    new_name : str, optional
        Rename the figure file when moving. If None, keep the original name.
    """
    import shutil

    source_path = os.path.join("figures", filename)
    if not os.path.exists(source_path):
        # Might be in the current directory if the user changed scanpy settings
        if os.path.exists(filename):
            source_path = filename
        else:
            return  # Can't locate the figure

    if new_name is None:
        new_name = filename
    target_path = os.path.join(save_dir, new_name)

    os.makedirs(save_dir, exist_ok=True)
    try:
        shutil.move(source_path, target_path)
        logger.info("Moved figure from %s to %s", source_path, target_path)
    except Exception as e:
        logger.warning("Could not move figure: %s", e)
