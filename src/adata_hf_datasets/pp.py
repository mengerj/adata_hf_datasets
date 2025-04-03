import pybiomart
import anndata
import pandas as pd
import logging
import scanpy as sc
from pathlib import Path
import os
from adata_hf_datasets.utils import consolidate_low_frequency_categories
import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def pp_adata(
    infile: str,
    outfile: str,
    columns: list[str] | None = None,
    threshold: int = 1,
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
    threshold : int, optional
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
        # 1) Remove zero-variance genes and cells
        logger.info("Removing zero-variance genes and cells.")
        #    Genes with zero standard deviation across cells:
        adata = remove_zero_variance_genes(adata)
        #    Cells with zero standard deviation across genes
        adata = remove_zero_variance_cells(adata)

        # 2) Consolidate low-frequency categories if columns is not None
        if columns is not None:
            logger.info(
                "Consolidating low-frequency categories in columns: %s with threshold=%d remove=%s",
                columns,
                threshold,
                remove,
            )
            adata = consolidate_low_frequency_categories(
                adata, columns, threshold, remove=remove
            )

        # 3) Store counts in a new layer
        # if adata is a backed object
        adata.layers["counts"] = adata.X.copy()
        # 4) Normalize, log-transform, and scale the entire dataset
        logger.info("log-transforming the dataset.")
        sc.pp.log1p(adata)
        logger.info("normalizing a new layer of the dataset")
        adata.layers["log-norm"] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e4, layer="log-norm")
        adata.layers["log-norm-scaled"] = adata.layers["log-norm"].copy()
        sc.pp.scale(adata, layer="log-norm-scaled")

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
