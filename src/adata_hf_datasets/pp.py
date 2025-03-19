import pybiomart
import anndata
import pandas as pd
import logging
import scanpy as sc
from pathlib import Path
import os

logger = logging.getLogger(__name__)


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
    add_ensembl_func : callable
        A function that adds 'ensembl_id' to `adata.var`, e.g. 'add_ensembl_ids(adata)'
        from your code. Must work on the chunk's var as needed.
    chunk_size : int, optional
        Number of cells (rows) to process at a time if you are chunking.
        If your dataset is still too large, you can lower this.
    overwrite : bool, optional
        If True, overwrite the output file if it exists.

    Notes
    -----
    - If your data is absolutely huge, you may need an advanced chunked approach
      (e.g. reading partial slices). For moderate data, 'backed' mode
      might suffice if you have enough memory for the partial steps.
    - This function modifies obs and var. It must be able to write changes to disk,
      so be mindful of anndata’s limitations in 'backed' mode.
    - One simpler approach is to read the entire data if you have enough memory to at
      least hold obs and var (but not necessarily X). Then write out the new file.
    """
    if Path(outfile).exists() and not overwrite:
        raise FileExistsError(
            f"Output file {outfile} already exists. Set overwrite=True to replace it."
        )

    logger.info("Loading AnnData from %s ...", infile)
    # A direct approach is to load in memory if you can handle obs,var in memory:
    adata = sc.read(infile)  # or None if you want to keep X on disk

    # 1. Add ensembl IDs if not present
    if "ensembl_id" not in adata.var.columns:
        logger.info("Adding 'ensembl_id' to adata.var.")
        add_ensembl_ids(adata)  # user-provided function

    # 2. Add n_counts if not present
    if "n_counts" not in adata.obs.columns:
        logger.info("Calculating n_counts, this requires scanning the data once.")
        sc.pp.calculate_qc_metrics(adata, inplace=True)
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
