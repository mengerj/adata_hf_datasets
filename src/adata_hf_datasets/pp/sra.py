import logging
import pandas as pd
import re
from pysradb import SRAweb
from anndata import AnnData
from tqdm import tqdm
from adata_hf_datasets.pp.utils import deduplicate_samples_by_id

logger = logging.getLogger(__name__)


def maybe_add_sra_metadata(
    adata,
    new_cols: str | list[str] = ["library_layout", "library_source", "instrument"],
    sample_id_key: str = "accession",
    sra_key: str = "sample_accession",
    exp_id_key: str = "experiment_accession",
    chunk_size: int = 10000,
):
    """
    Check if the data is from SRA and fetch metadata if so.

    This function checks if the first entry in adata.obs.index starts with "SRX".
    If so, it assumes the data is from SRA and calls fetch_sra_metadata.
    Otherwise, it does nothing.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    new_cols : str or list of str, optional
        Metadata columns to copy from the SRA results into `adata.obs`.
    sample_id_key : str, optional
        The column in `adata.obs` containing SRA-based IDs (e.g., SRR or other sample-level accessions).
    sra_key : str, optional
        The column in the returned SRA DataFrame to match your IDs against.
        Defaults to "sample_accession".
    exp_id_key : str, optional
        Has to be present in adata.obs and contain SRX IDs. Will be used to merge with the SRA metadata.
    chunk_size : int, optional
        Number of unique IDs to process per chunk. Defaults to 10000.
    """
    adata.obs[exp_id_key] = adata.obs.index
    # will be false if no srx ids are found
    if filter_invalid_sra_ids(adata, srx_column=exp_id_key, srs_column=sample_id_key):
        adata = fetch_sra_metadata(
            adata,
            sample_id_key=sample_id_key,
            sra_key=sra_key,
            exp_id_key=exp_id_key,
            new_cols=new_cols,
            chunk_size=chunk_size,
        )
    else:
        logger.info("Data does not appear to be from SRA. Skipping metadata fetching.")
    return adata


def filter_invalid_sra_ids(
    adata: AnnData,
    srx_column: str | None = None,
    srs_column: str | None = None,
    pct_tolerate: float = 0.2,
) -> AnnData:
    """
    Filter out cells from `adata` where the provided accession columns do not
    contain valid IDs. For SRX IDs, a valid ID starts with 'SRX' followed by digits.
    For SRS IDs, a valid ID starts with 'SRS' followed by digits.

    If both are provided, a cell is kept only if it has valid IDs in both columns.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    srx_column : str | None, optional
        Column in `adata.obs` to check for valid SRX IDs.
    srs_column : str | None, optional
        Column in `adata.obs` to check for valid SRS IDs.
    pct_tolerate : float, optional
        Fraction of cells that can be invalid before raising an error.
        Defaults to 0.2.

    Returns
    -------
    anndata.AnnData
        Filtered AnnData object containing only cells with valid IDs in all
        specified accession columns.

    Raises
    ------
    KeyError
        If any provided column is not found in `adata.obs`.
    ValueError
        If more than `pct_tolerate * 100` percent of cells in any column are invalid.
    """
    # Create an initial mask that is True for all cells.
    final_mask = pd.Series([True] * adata.n_obs, index=adata.obs.index)

    # Function to check a column against a regex, update final_mask
    def _check_and_filter(col_name: str, pattern_str: str):
        if col_name not in adata.obs.columns:
            raise KeyError(f"Column '{col_name}' not found in adata.obs.")
        pattern = re.compile(pattern_str)
        # Cast to string and match
        mask = adata.obs[col_name].astype(str).str.match(pattern)
        n_invalid = (~mask).sum()
        n_total = adata.n_obs
        n_tolerated = int(n_total * pct_tolerate)
        if n_invalid == n_total:
            raise ValueError(
                f"All IDs in column '{col_name}' are invalid. Check the data."
            )
        if n_invalid > n_tolerated:
            logger.error(
                "More than %.1f%% of IDs in column '%s' are invalid. Example invalid ID: %s",
                pct_tolerate * 100,
                col_name,
                adata.obs[~mask].iloc[0],
            )
            raise ValueError(
                f"More than {pct_tolerate * 100:.1f}% of IDs in column '{col_name}' are invalid."
            )
        if n_invalid > 0:
            logger.warning(
                "Removing %d invalid IDs in column '%s' out of %d cells.",
                n_invalid,
                col_name,
                n_total,
            )
        # Return the boolean mask for valid IDs in this column.
        return mask

    # If srx_column is provided, check and update final_mask.
    if srx_column is not None:
        try:
            mask_srx = _check_and_filter(srx_column, r"^SRX\d+$")
            final_mask &= mask_srx
        except ValueError or KeyError as e:
            logger.error("Error checking SRX IDs: %s", e)
            return False

    # If srs_column is provided, check and update final_mask.
    if srs_column is not None:
        try:
            mask_srs = _check_and_filter(srs_column, r"^SRS\d+$")
            final_mask &= mask_srs
        except ValueError or KeyError as e:
            logger.error("Error checking SRS IDs: %s", e)
            return False
    # if all are true, dont subset.
    if final_mask.all():
        logger.info("All IDs are valid.")
        return True
    adata = adata[final_mask]
    logger.info("After filtering, %d cells remain", adata.n_obs)
    return True


def fetch_sra_metadata(
    adata: AnnData,
    sample_id_key: str = "accession",
    sra_key: str = "sample_accession",
    exp_id_key: str = "experiment_accession",
    new_cols: str | list[str] = [
        "library_layout",
        "library_source",
        "instrument",
    ],
    fallback: str = "unknown",
    chunk_size: int = 10000,
) -> None:
    """
    Fetch various metadata fields (e.g., 'library_layout', 'library_source', 'instrument_model')
    from SRA for all unique IDs in `adata.obs[sample_id_key]`, processing in chunks,
    and store them in `adata.obs[new_cols]`.

    This function:
    1) Extracts all unique IDs from `adata.obs[sample_id_key]` and experimental accessions from `adata.obs[exp_id_key]`.
    2) Splits the IDs into chunks (default size: 10,000) and calls `db.sra_metadata` for each chunk.
    3) Concatenates the returned metadata into one DataFrame.
    4) Checks that every requested experimental ID is found in the SRA results (and logs or sets fallback if missing).
    5) Removes extra rows from the SRA results that do not correspond to your unique experimental IDs.
    6) Merges and assigns the requested columns to `adata.obs`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with IDs in `adata.obs[sample_id_key]` and experimental IDs in `adata.obs[exp_id_key]`.
    sample_id_key : str, optional
        The column in `adata.obs` containing SRA-based IDs (e.g., SRR or other sample-level accessions).
    sra_key : str, optional
        The column in the returned SRA DataFrame to match your IDs against.
        Defaults to "sample_accession".
    exp_id_key : str, optional
        Has to be present in adata.obs and contain SRX IDs. Will be used to merge with the SRA metadata.
    new_cols : str or list of str, optional
        Metadata columns to copy from the SRA results into `adata.obs`.
    fallback : str, optional
        Value to use if a column is missing or if some IDs are not found. Defaults to "unknown".
    chunk_size : int, optional
        Number of unique IDs to process per chunk. Defaults to 10000.

    Returns
    -------
    None
        Modifies `adata.obs[new_cols]` in place.

    Examples
    --------
    >>> adata = sc.read_h5ad("my_data.h5ad")
    >>> fetch_sra_metadata(adata, sample_id_key="accession", sra_key="sample_accession",
                           exp_id_key="experiment_accession")
    >>> adata.obs["library_layout"].value_counts()
    """

    if isinstance(new_cols, str):
        new_cols = [new_cols]

    logger.info("Fetching SRA metadata for %d samples.", adata.n_obs)

    if sample_id_key not in adata.obs.columns:
        raise ValueError(f"Column '{sample_id_key}' not found in adata.obs.")

    if exp_id_key not in adata.obs.columns:
        raise ValueError(f"Column '{exp_id_key}' not found in adata.obs.")

    # Deduplicate samples if needed (assuming deduplicate_samples_by_id is defined)
    adata = deduplicate_samples_by_id(adata, sample_id_key)

    # 1) Extract all unique IDs and experimental accession IDs
    unique_ids = adata.obs[sample_id_key].dropna().unique().tolist()
    experiment_accessions = adata.obs[exp_id_key].dropna().unique().tolist()
    logger.info("Found %d unique IDs in adata.obs[%s].", len(unique_ids), sample_id_key)

    if not unique_ids:
        msg = f"No unique IDs found in adata.obs[{sample_id_key}]. Cannot proceed."
        logger.error(msg)
        raise ValueError(msg)

    db = SRAweb()

    # 2) Process the unique_ids in chunks.
    chunks = [
        unique_ids[i : i + chunk_size] for i in range(0, len(unique_ids), chunk_size)
    ]
    logger.info(
        "Processing %d chunks of approximately %d IDs each.", len(chunks), chunk_size
    )

    df_list = []
    for chunk in tqdm(chunks, desc="Processing Chunks"):
        try:
            df_chunk = db.sra_metadata(chunk)
            if df_chunk is not None and not df_chunk.empty:
                df_list.append(df_chunk)
        except Exception as e:
            logger.error("Failed to fetch metadata for chunk: %s", e)
            # Optionally, you can raise here or continue to process the others.
            raise ValueError(f"Failed to fetch metadata for a chunk: {e}")

    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
    else:
        msg = "No metadata returned in any chunk. Check if the IDs are valid or if SRA is accessible."
        logger.error(msg)
        raise ValueError(msg)

    # 3) Check if the expected SRA column(s) exist.
    if sra_key not in df_all.columns:
        logger.error(
            "sra_key='%s' not in returned columns. Found: %s", sra_key, df_all.columns
        )
        raise ValueError(f"sra_key='{sra_key}' not in returned metadata columns.")

    # 4) Identify the set of unique experimental IDs.
    desired_experiment_ids = set(experiment_accessions)
    returned_experiment_ids = set(df_all[exp_id_key].unique())
    logger.info(
        "Returned %d unique experimental IDs from SRA metadata.",
        len(returned_experiment_ids),
    )

    # Check for missing experimental IDs.
    missing_ids = desired_experiment_ids - returned_experiment_ids
    if missing_ids:
        logger.warning(
            "Some experimental IDs in adata.obs[%s] were not found in the SRA metadata: %s",
            exp_id_key,
            missing_ids,
        )
        logger.warning("These will be assigned fallback='%s'.", fallback)

    # 5) Remove extra rows from df_all that do not correspond to your desired experimental IDs.
    extra_ids = returned_experiment_ids - desired_experiment_ids
    if extra_ids:
        logger.info(
            "Removing %d extra experimental IDs not present in adata.obs[%s].",
            len(extra_ids),
            exp_id_key,
        )
        df_all = df_all[~df_all[exp_id_key].isin(extra_ids)]

    # 6) Ensure that the requested new columns exist, and drop duplicates so each exp_id appears only once.
    missing_cols = [col for col in new_cols if col not in df_all.columns]
    if missing_cols:
        logger.warning(
            "Some requested columns are missing in metadata: %s", missing_cols
        )

    keep_cols = [col for col in new_cols if col in df_all.columns]
    df_map = df_all[[exp_id_key] + keep_cols].copy()
    df_map = df_map.drop_duplicates(subset=exp_id_key, keep="first")

    # 7) Merge the SRA metadata into the adata.obs.
    obs_reset = adata.obs.reset_index(drop=False)
    # Drop any conflicting columns before merging so we don't get suffixes.
    cols_to_drop = [c for c in keep_cols if c in obs_reset.columns]
    obs_reset = obs_reset.drop(columns=cols_to_drop, errors="ignore")
    merged = obs_reset.merge(
        df_map,
        how="left",
        left_on=exp_id_key,
        right_on=exp_id_key,
    )

    # 8) Assign the new columns to adata.obs, filling missing values with fallback.
    for col in new_cols:
        if col in merged.columns:
            adata.obs[col] = merged[col].fillna(fallback).values
        else:
            adata.obs[col] = fallback
            logger.warning(
                "Column '%s' not found in SRA metadata. Using fallback='%s'.",
                col,
                fallback,
            )
    return adata

    # logger.info(
    #    "Successfully added columns %s to adata.obs using fallback='%s'.",
    #    new_cols, fallback
    # )
