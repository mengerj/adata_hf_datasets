import logging
import scanpy as sc
from anndata import AnnData
from adata_hf_datasets.pp.utils import (
    ensure_log_norm,
    consolidate_low_frequency_categories,
    check_enough_genes_per_batch,
)

logger = logging.getLogger(__name__)


def pp_adata_general(
    adata: AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    batch_key: str = "batch",
    n_top_genes: int = 1000,
    categories: list[str] | None = None,
    category_threshold: int = 1,
    remove: bool = True,
) -> AnnData:
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
    batch_key : str, optional
        Key in `adata.obs` for batch information. Used to ensure each batch has enough variable genes for highly variable gene selection.
    n_top_genes : int, optional
        Number of top variable genes to select. Will only be marked and not filtered.
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
    # adata.var_names_make_unique() #had issues when var names are ensembl ids
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

    # 4) Normalize and log-transform (in place)
    ensure_log_norm(adata)
    if batch_key in adata.obs.columns:
        # check if each batch has at least 1000 variable genes
        adata = check_enough_genes_per_batch(
            adata, batch_key=batch_key, min_genes=n_top_genes
        )
        # perform highly variable gene selection
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)
    else:
        logging.warning(
            "Batch key not found in adata.obs. Selecting highly variable genes without batch correction."
        )
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    logger.info("In-memory preprocessing complete.")
    return adata
