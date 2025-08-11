import logging
import scanpy as sc
from anndata import AnnData
from adata_hf_datasets.pp.pybiomart_utils import add_ensembl_ids
import numpy as np

logger = logging.getLogger(__name__)


def pp_adata_geneformer(
    adata: AnnData,
) -> AnnData:
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
        add_ensembl_ids(
            adata, ensembl_col="ensembl_id", species="hsapiens"
        )  # user-provided function

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

    # sample_index should already be present
    # add a numeric sample index to obs, which is needed for geneformer
    adata.obs["sample_index"] = np.arange(adata.shape[0])
    if "sample_index" not in adata.obs.columns:
        raise ValueError(
            "sample_index not found in adata.obs. Please add it before calling this function. Add it before splitting the data."
        )

    logger.info("Geneformer in-memory preprocessing complete.")
    return adata
