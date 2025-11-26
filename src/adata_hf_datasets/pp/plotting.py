import logging
import os
from typing import List, Optional

import numpy as np
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def qc_evaluation_plots(
    adata: anndata.AnnData,
    subset_cells: int = 5000,
    qc_vars: Optional[List[str]] = None,
    percent_top: Optional[List[int]] = None,
    log1p_for_qc: bool = True,
    metrics_of_interest: Optional[List[str]] = None,
    categories_of_interest: Optional[List[str]] = None,
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
    categories_of_interest : list of str, optional
        Keys in `adata.obs` to color PCA plots and group violin plots of the metrics of interest
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
        sc.set_figure_params()
        sc.settings.figdir = save_dir  # Where to save
        sc.settings.file_format_figs = "png"  # Save in PNG
        sc.settings.autosave = True  # Will save automatically
        sc.settings.autoshow = False  # Don't display inline

        # 1) Decide whether to subset
        n_obs = adata.n_obs
        if n_obs > subset_cells:
            logger.info(
                "Subsetting data from %d cells to %d cells.", n_obs, subset_cells
            )
            idx = np.random.choice(n_obs, subset_cells, replace=False)

            # 2) If backed, slice first (lazy) then load selection into memory
            if getattr(adata, "isbacked", False):
                # this view still references the on‐disk file
                view = adata[idx, :]
                # now pull just these cells into memory
                adata_sub = view.to_memory()
                view.file.close()
            else:
                adata_sub = adata[idx, :].copy()
        else:
            logger.info("Data has <= %d cells, no subsetting needed.", subset_cells)
            if getattr(adata, "isbacked", False):
                adata_sub = adata.to_memory()
                adata.file.close()
            else:
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
        metrics_of_interest = ["total_counts", "n_genes_by_counts"]
    logger.info("Metrics of interest for plotting: %s", metrics_of_interest)

    # 5) PCA: We'll assume data is log-transformed, but let's do minimal steps if needed.
    logger.info("Running PCA for the subset data. This is a quick approximate check.")
    # If the data is not log-transformed, consider calling sc.pp.log1p(adata_sub) here.
    # If you want scaling, consider sc.pp.scale(adata_sub).
    sc.pp.pca(adata_sub)

    # 6) Generate and optionally save multiple plots

    # 6A) PCA scatter, colored by QC metrics
    logger.info("Generating PCA plots for metrics of interest...")
    for metric in metrics_of_interest + categories_of_interest:
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

        # Get the data and clean it thoroughly
        data = adata_sub.obs[metric]

        # Remove NaN and infinite values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) == 0:
            logger.warning(f"No valid data for metric {metric}, skipping histogram.")
            continue

        # Check for variation
        data_range = data.max() - data.min()
        if data_range == 0 or not np.isfinite(data_range):
            logger.warning(
                f"Metric {metric} has no variation or invalid range, skipping histogram."
            )
            continue

        # For very small datasets or very limited unique values, skip histogram
        n_unique = len(data.unique())
        if n_unique <= 1:
            logger.warning(
                f"Metric {metric} has only {n_unique} unique value(s), skipping histogram."
            )
            continue

        plt.figure()
        try:
            # Try multiple binning strategies with full error handling
            if n_unique <= 10:
                # Very discrete data - use exact unique values
                bins = n_unique
                logger.debug(
                    f"Using {bins} bins for very discrete metric {metric} (n_unique={n_unique})"
                )
            elif data_range < 1:
                # Very small range - use minimal bins
                bins = min(10, n_unique)
                logger.debug(
                    f"Using {bins} bins for metric {metric} with small range ({data_range})"
                )
            else:
                # Normal data - use automatic binning
                bins = "auto"
                logger.debug(f"Using 'auto' bins for metric {metric}")

            plt.hist(data, bins=bins, edgecolor="k", alpha=0.7)

            plt.title(f"Distribution of {metric}")
            plt.xlabel(metric)
            plt.ylabel("Frequency")

            if save_plots and save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"hist_{metric}.png"), dpi=150)
            else:
                plt.show()

        except (ValueError, RuntimeError) as e:
            # If histogram creation fails for any reason, just skip it and log
            logger.warning(
                f"Could not create histogram for metric {metric} (range={data_range:.4f}, "
                f"n_unique={n_unique}, n_samples={len(data)}). Error: {e}. Skipping."
            )
        finally:
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
    for category in categories_of_interest:
        if category is not None and category in adata_sub.obs.columns:
            # Check if the entries in adata.obs are categorial
            if not adata_sub.obs[category].dtype.name.startswith("category"):
                logger.info("Converting %s to categorical type.", category)
                adata_sub.obs[category] = adata_sub.obs[category].astype("category")

            logger.info("Investigating the effect of categorial label '%s'.", category)

            for metric in metrics_of_interest:
                # Distribution of n_genes_by_counts by batch (Seaborn violin)
                if metric in adata_sub.obs.columns:
                    logger.info(
                        "Plotting '%s' by '%s' with violinplot.", metric, category
                    )

                    # Prepare a small DataFrame for Seaborn
                    df = adata_sub.obs[[category, metric]].copy()
                    df.dropna(inplace=True)

                    plt.figure(figsize=(6, 5))

                    # Main violin (horizontal)
                    sns.violinplot(
                        data=df,
                        y=category,  # group categories on y-axis
                        x=metric,  # numeric metric on x-axis
                        orient="h",  # horizontal orientation
                        color="white",  # base color for violins
                        edgecolor="black",
                    )

                    # Optional stripplot on top for individual points
                    sns.stripplot(
                        data=df,
                        y=category,
                        x=metric,
                        orient="h",
                        color="black",
                        alpha=0.4,
                        size=2,
                        jitter=0.4,
                    )

                    plt.title(f"{metric} by {category}")
                    plt.tight_layout()

                    if save_plots and save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        outpath = os.path.join(
                            save_dir, f"violin_{category}_{metric}.png"
                        )
                        plt.savefig(outpath, dpi=150)
                        plt.close()
                    else:
                        plt.show()

                else:
                    logger.warning(
                        f"{metric} not found in adata.obs. Skipping violin by batch."
                    )

        else:
            logger.info("No batch key provided or not found in adata.obs.")
    logger.info("Finished generating QC evaluation plots.")
