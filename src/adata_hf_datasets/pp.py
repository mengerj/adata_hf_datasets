import pybiomart
import anndata
import pandas as pd
import logging
import scanpy as sc
import os
from adata_hf_datasets.utils import (
    consolidate_low_frequency_categories,
    is_raw_counts,
    ensure_log_norm,
)
import numpy as np
import scipy.sparse as sp
import gc

logger = logging.getLogger(__name__)


def pp_adata(
    infile: str,
    outfile: str,
    min_cells: int = 10,
    min_genes: int = 200,
    columns: list[str] | None = None,
    category_threshold: int = 1,
    remove: bool = True,
    call_geneformer: bool = True,
) -> None:
    """
    Create an initial preprocessed AnnData file ready for embeddings.

    This function:
    1. Reads the input AnnData from `infile`.
    2. Runs `pp_adata_inmemory` on the in-memory object.
    3. Optionally calls `pp_geneformer_inmemory` (if `call_geneformer=True`).
    4. Writes the result to `outfile`.

    Parameters
    ----------
    infile : str
        Path to the input AnnData file (H5AD).
    outfile : str
        Path to the output AnnData file after preprocessing.
    min_cells : int, optional
        Minimum number of cells for gene filtering.
    min_genes : int, optional
        Minimum number of genes for cell filtering.
    columns : List[str] | None, optional
        Columns in `adata.obs` to consolidate low-frequency categories.
    category_threshold : int, optional
        Frequency threshold for category consolidation.
    remove : bool, optional
        If True, remove low-frequency categories entirely.
    call_geneformer : bool, optional
        If True, call `pp_adata_geneformer` on the AnnData after general preprocessing.

    Returns
    -------
    None
        Writes the final AnnData object to `outfile`.

    References
    ----------
    Data is read from disk, processed fully in memory, then written to disk.
    """
    logger.info("Reading AnnData from %s", infile)
    adata = sc.read(infile)
    try:
        # 1) Basic preprocessing in memory
        adata = pp_adata_general(
            adata=adata,
            min_cells=min_cells,
            min_genes=min_genes,
            columns=columns,
            category_threshold=category_threshold,
            remove=remove,
        )

        # 2) Optionally call Geneformer preprocessing
        if call_geneformer:
            adata = pp_adata_geneformer(adata)

        # 3) Write final output
        logger.info("Writing final preprocessed AnnData to %s", outfile)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        adata.write(outfile)

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        del adata
        gc.collect()

    logger.info("Done. Final preprocessed file: %s", outfile)


def pp_adata_general(
    adata: anndata.AnnData,
    min_cells: int = 10,
    min_genes: int = 200,
    columns: list[str] | None = None,
    category_threshold: int = 1,
    remove: bool = True,
) -> anndata.AnnData:
    """
    Create an initial preprocessed AnnData object in memory ready for embeddings.

    This function performs the following steps:
    1. Makes gene and cell names unique.
    2. Filters out genes expressed in fewer than `min_cells` cells and cells
       expressing fewer than `min_genes` genes.
    3. Consolidates low-frequency categories in specified columns of `adata.obs`.
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
    columns : List[str] | None, optional
        Columns in `adata.obs` to consolidate low-frequency categories.
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

    # 2) Consolidate low-frequency categories if columns is not None
    if columns is not None:
        logger.info(
            "Consolidating low-frequency categories in columns: %s with threshold=%d remove=%s",
            columns,
            category_threshold,
            remove,
        )
        adata = consolidate_low_frequency_categories(
            adata, columns, category_threshold, remove=remove
        )

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

    # 3. Attach a stable sample index
    if "sample_index" not in adata.obs.columns:
        logger.info("Adding a stable sample_index to obs.")
        n_obs = adata.shape[0]
        adata.obs["sample_index"] = range(n_obs)

    logger.info("Geneformer in-memory preprocessing complete.")
    return adata


'''
def pp_adata(
    infile: str,
    outfile: str,
    min_cells: int = 10,
    min_genes: int = 200,
    columns: list[str] | None = None,
    category_threshold: int = 1,
    remove: bool = True,
) -> None:
    """
    Create an initial preprocessed AnnData file ready for embeddings.

    This function performs the following steps:
    1. Calls `pp_geneformer` on the input file to ensure stable metadata.
    2. Reads the output of `pp_geneformer`, removes zero-variance cells and genes,
       and consolidates low-frequency categories in specified columns.
    3. Normalizes, log-transforms, and scales the entire dataset (no PCA).

    The final preprocessed data is written to `outfile`.

    Parameters
    ----------
    infile : str
        Path to the input AnnData file (H5AD).
    outfile : str
        Path to the output AnnData file after preprocessing.
    columns : List[str] | None, optional
        Columns in `adata.obs` to consolidate low-frequency categories.
        If None, no columns are processed for consolidation.
    category_threshold : int, optional
        Frequency threshold for low-frequency category consolidation.
        Defaults to 1 (i.e., categories with <1 occurrence are consolidated/removed).
    remove : bool, optional
        If True, remove rows containing low-frequency categories entirely.
        Otherwise, relabel them as 'remaining <col>'.
        Defaults to True.

    Returns
    -------
    None
        Writes the final AnnData object to `outfile`.

    Notes
    -----
    - Zero-variance genes/cells are removed by explicitly checking variance across
      each gene and each cell.
    - The entire dataset is scaled (no PCA is performed here).
    - You can adapt to chunked or backed-mode reading if memory is limited.
    """
    logger.info("Starting combined preprocessing for initial embeddings.")

    try:
        adata = sc.read(infile)
        # 0) Make ids unique
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        # 1) Remove genes and cells with low amount of cells and genes
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        # 2) Consolidate low-frequency categories if columns is not None
        if columns is not None:
            logger.info(
                "Consolidating low-frequency categories in columns: %s with threshold=%d remove=%s",
                columns,
                category_threshold,
                remove,
            )
            adata = consolidate_low_frequency_categories(
                adata, columns, category_threshold, remove=remove
            )

        # 3) Store counts in a new layer
        if "counts" not in adata.layers:
            # Check if X contains raw counts
            if is_raw_counts(adata.X):
                adata.layers["counts"] = adata.X.copy()
            else:
                logger.error("X does not contain raw counts. Cannot create 'counts' layer.")
                raise ValueError("X does not contain raw counts. Cannot create 'counts' layer.")
        # 4) Normalize and log-transform
        ensure_log_norm(adata)

        # write to temp file with is used for geneformer
        temp_file = outfile.replace(".h5ad", "_temp.h5ad")
        logger.info("Writing preprocessed AnnData to %s", temp_file)
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        adata.write(temp_file)
        # Explicitly close the AnnData object
        adata.file.close() if hasattr(
            adata, "file"
        ) and adata.file is not None else None
        del adata

        # 5) Call pp_geneformer and write result to file
        pp_geneformer(infile=temp_file, outfile=outfile, overwrite=True)

        # Clean up temporary file
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        # Clean up temporary file in case of error
        temp_file = outfile.replace(".h5ad", "_temp.h5ad")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file after error: {temp_file}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        raise

    logger.info("Done. Final preprocessed file: %s", outfile)


def pp_geneformer(
    infile: str,
    outfile: str,
    overwrite: bool = False,
):
    """
    Preprocess an AnnData file for Geneformer embeddings.

    Parameters
    ----------
    infile : str
        Path to the input AnnData file (H5AD).
    outfile : str
        Path to the output AnnData file after preprocessing.
    overwrite : bool, optional
        If True, overwrite the output file if it exists.

    Notes
    -----
    - If your data is absolutely huge, you may need an advanced chunked approach
      (e.g. reading partial slices). For moderate data, 'backed' mode
      might suffice if you have enough memory for the partial steps.
    - This function modifies obs and var. It must be able to write changes to disk,
      so be mindful of anndata's limitations in 'backed' mode.
    - One simpler approach is to read the entire data if you have enough memory to at
      least hold obs and var (but not necessarily X). Then write out the new file.
    """
    if Path(outfile).exists() and not overwrite:
        raise FileExistsError(
            f"Output file {outfile} already exists. Set overwrite=True to replace it."
        )

    logger.info("Loading AnnData from %s ...", infile)
    try:
        # A direct approach is to load in memory if you can handle obs,var in memory:
        adata = sc.read(infile)  # or None if you want to keep X on disk

        # 1. Add ensembl IDs if not present
        if "ensembl_id" not in adata.var.columns:
            logger.info("Adding 'ensembl_id' to adata.var.")
            add_ensembl_ids(adata)  # user-provided function

        # 2. Add n_counts if not present
        if "n_counts" not in adata.obs.columns:
            logger.info("Calculating n_counts, this requires scanning the data once.")
            # Adjust percent_top based on the number of genes to prevent IndexError
            n_genes = adata.n_vars
            percent_top = []
            # Only include percentages that make sense for the dataset size
            for p in [50, 100, 200, 500]:
                if p < n_genes:
                    percent_top.append(p)

            if len(percent_top) > 0:
                sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=percent_top)
            else:
                # If dataset is too small, skip percent_top calculations
                sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[])

            adata.obs["n_counts"] = adata.obs.total_counts

        # 3. Attach a stable sample index
        if "sample_index" not in adata.obs.columns:
            logger.info("Adding a stable sample_index to obs.")
            n_obs = adata.shape[0]
            adata.obs["sample_index"] = range(n_obs)

        # Force full write to a new file, removing the old .backed references
        logger.info("Writing preprocessed AnnData to %s", outfile)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        adata.write(outfile)

        # Explicitly close the AnnData object
        adata.file.close() if hasattr(
            adata, "file"
        ) and adata.file is not None else None
        del adata

    except Exception as e:
        logger.error(f"Error during Geneformer preprocessing: {e}")
        raise
    finally:
        # Ensure we clean up any remaining file handles
        import gc

        gc.collect()

    logger.info("Preprocessing done. Preprocessed file: %s", outfile)
'''


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
