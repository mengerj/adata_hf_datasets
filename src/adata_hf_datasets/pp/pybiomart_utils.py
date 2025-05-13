import pybiomart
import logging
import anndata
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


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


def ensure_ensembl_index(
    adata: anndata.AnnData,
    ensembl_col: str = "ensembl_id",
    add_fn: callable = add_ensembl_ids,
) -> None:
    """
    Ensure that an AnnData.var index consists of Ensembl IDs,
    dropping rows with empty IDs and deduplicating.

    Parameters
    ----------
    adata
        AnnData whose .var.index should be Ensembl gene IDs.
    ensembl_col
        Column in adata.var to use if the index is not already Ensembl IDs.
    add_fn
        Optional function to call as add_fn(adata, ensembl_col=ensembl_col)
        if neither the index nor ensembl_col provide valid IDs.

    Raises
    ------
    ValueError
        If, after attempting to use `ensembl_col` or calling `add_fn`, the var.index
        still does not consist of valid Ensembl IDs.
    """
    # 1) If index is already ENSG…
    first = adata.var_names[0]
    if isinstance(first, str) and first.startswith("ENS"):
        logger.info("AnnData.var.index already Ensembl IDs; skipping reindex.")
    else:
        # 2) Try existing var[ensembl_col]
        if ensembl_col in adata.var.columns:
            logger.info(f"Reindexing from var['{ensembl_col}'].")
            adata.var.index = adata.var[ensembl_col].astype(str)
        # 3) Fallback: call add_fn
        elif add_fn is not None:
            logger.info(f"Calling {add_fn.__name__} to populate '{ensembl_col}'.")
            add_fn(adata, ensembl_col=ensembl_col)
            adata.var.index = adata.var[ensembl_col].astype(str)
        else:
            raise ValueError(
                "Cannot ensure Ensembl index: neither index nor "
                f"var['{ensembl_col}'] present, and no add_fn supplied."
            )

        # Verify
        if not adata.var_names[0].startswith("ENS"):
            raise ValueError(
                "After reindexing, AnnData.var.index does not start with 'ENS'."
            )

    # --- CLEANUP DUPLICATES & EMPTIES ------------------------------------------------

    # a) Drop any empty-string indices
    names = adata.var_names.astype(str)
    empty_mask = names == ""
    if empty_mask.any():
        n_empty = int(empty_mask.sum())
        logger.warning(f"Dropping {n_empty} features with empty Ensembl IDs.")
        adata._inplace_subset_var(~empty_mask)
        names = adata.var_names.astype(str)

    # b) Deduplicate any remaining valid ENSG IDs
    #    For each duplicated ID, randomly drop all but one.
    is_dup = pd.Series(names).duplicated(keep=False).to_numpy()
    if is_dup.any():
        dup_ids = np.unique(names[is_dup])
        to_drop = []
        for gene in dup_ids:
            # positions of this gene
            positions = np.where(names == gene)[0]
            # keep one at random, drop the rest
            drop_positions = np.random.choice(
                positions, size=len(positions) - 1, replace=False
            )
            to_drop.extend(drop_positions.tolist())

        logger.warning(
            f"Found {len(dup_ids)} genes with duplicate Ensembl IDs; "
            f"dropping {len(to_drop)} random duplicates."
        )
        keep_mask = np.ones(len(names), dtype=bool)
        keep_mask[to_drop] = False
        adata._inplace_subset_var(keep_mask)

    logger.info(f"Final var.index has {adata.n_vars} unique Ensembl IDs.")


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
