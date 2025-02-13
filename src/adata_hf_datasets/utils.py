import logging

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from datetime import datetime
import os

from pathlib import Path
import importlib
from huggingface_hub import HfApi
import tempfile
from string import Template
import scipy.sparse as sp

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
    biomart_df = sc.queries.biomart_annotations(
        species, ["ensembl_gene_id", "external_gene_name"], use_cache=use_cache
    )

    # Drop duplicates so that each gene symbol maps to exactly one Ensembl ID
    biomart_df = biomart_df.drop_duplicates(subset="external_gene_name").set_index(
        "external_gene_name"
    )

    # Prepare list for the mapped Ensembl IDs
    gene_symbols = adata.var_names
    ensembl_ids = []
    for symbol in gene_symbols:
        if symbol in biomart_df.index:
            ensembl_ids.append(biomart_df.loc[symbol, "ensembl_gene_id"])
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


def split_anndata(adata: anndata.AnnData, train_size: float = 0.8):
    """
    Splits an AnnData object into training and validation sets.

    Parameters
    ----------
    adata
        The complete AnnData object to be split.
    train_size
        The proportion of the dataset to include in the train split. Should be between 0 and 1.

    Returns
    -------
    anndata.AnnData: The training AnnData object.
    anndata.AnnData: The validation AnnData object.
    """
    # Ensure train_size is a valid proportion
    if not 0 < train_size < 1:
        raise ValueError("train_size must be a float between 0 and 1.")

    # Generate random indices
    indices = np.arange(adata.n_obs)
    np.random.shuffle(indices)

    # Calculate the number of observations for the train set
    train_indices_count = int(train_size * adata.n_obs)

    # Split indices for train and validation sets
    train_indices = indices[:train_indices_count]
    val_indices = indices[train_indices_count:]

    # Subset the AnnData object
    train_adata = adata[train_indices]
    val_adata = adata[val_indices]

    return train_adata, val_adata


def setup_logging():
    """Set up logging configuration for the module.

    This function configures the root logger to display messages in the console and to write them to a file
    named by the day. The log level is set to INFO.
    """

    # Create the logs directory
    os.makedirs("logs", exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (important if function is called multiple times)
    if not logger.hasHandlers():
        # Create a file handler
        log_file = logging.FileHandler(
            f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
        )
        log_file.setLevel(logging.INFO)

        # Create a console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # Create a formatter and set it for the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_file.setFormatter(formatter)
        console.setFormatter(formatter)

        # Add the handlers to the root logger
        logger.addHandler(log_file)
        logger.addHandler(console)

    return logger


def annotate_and_push_dataset(
    dataset,
    repo_id: str | None = None,
    private: bool = False,
    readme_template_name: str | None = None,
    embedding_generation: str | None = None,
    caption_generation: str | None = None,
    dataset_type_explanation: str | None = None,
) -> None:
    """Annotates and pushes the dataset to Hugging Face.

    A README.md is dynamically created based on a template and a given unique description.
    The README is stored in a temporary directory and uploaded before being deleted.

    Parameters
    ----------
    repo_id (str, optional):
        Repository ID for Hugging Face. If provided, the dataset will be pushed to Hugging Face.
    private (bool, optional):
        If True, the dataset will be private on Hugging Face. Default is False.
    readme_template_name (str, optional):
        The name of the README template to use. Has to be stored in the package resources.
    embedding_generation (str, optional):
        A description of how the embeddings stored in .obsm of the adata files were generated.
    caption_generation (str, optional):
        A description of how the captions stored in .obs of the adata files were generated.
    dataset_type_explanation (str, optional):
        A description of the dataset type. E.g. "pairs" or "multiplets".
    """

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        readme_path = Path(temp_dir) / "README.md"

        # Write the dynamically generated README file
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                _generate_readme(
                    readme_template_name=readme_template_name,
                    repo_id=repo_id,
                    embedding_generation=embedding_generation,
                    caption_generation=caption_generation,
                    dataset_type_explanation=dataset_type_explanation,
                )
            )

        # Push dataset with README
        dataset.push_to_hub(repo_id, private=private)

        # Upload README file
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )


def _generate_readme(
    readme_template_name,
    repo_id,
    embedding_generation,
    caption_generation=None,
    dataset_type_explanation=None,
) -> str:
    """
    Fills the README template with dataset-specific details.

    Returns
    -------
    str
        The formatted README content.
    readme_template_name
        The name of the template file to use. E.g cellwhisperer
    embedding_generation
        A description of how the embeddings stored in .obsm of the adata files were generated.
    caption_generation
        A description of how the captions stored in .obs of the adata files were generated. Not needed for inference datasets.
    """
    if caption_generation is None:
        caption_info = ""
    else:
        caption_info = f"""The caption entry of the dataset contains a textual description of the dataset, it was generated like this:{caption_generation}"""
    readme_template = _load_readme_template(readme_template_name=readme_template_name)
    readme_filled = Template(readme_template).safe_substitute(
        repo_id=repo_id,
        embedding_generation=embedding_generation,
        caption_generation=caption_info,
        dataset_type_explanation=dataset_type_explanation,
    )
    return readme_filled


def _load_readme_template(readme_template_name) -> str:
    """Generate a README.md file for the dataset."""
    # Load the template stored locally at {project_dir}/dataset_readmes/{readme_template_name}.md
    package_name = "adata_hf_datasets.templates"  # Adjust this to your package name

    try:
        with (
            importlib.resources.files(package_name)
            .joinpath(f"{readme_template_name}.md")
            .open("r", encoding="utf-8") as f
        ):
            return f.read()
    except FileNotFoundError:
        raise ValueError(
            f"Template {readme_template_name} not found in package resources"
        )


'''
import gc
import sys
import logging
from typing import List, Tuple

def get_top_n_memory_objects(n: int = 5) -> List[Tuple[str, int]]:
    """
    Retrieve the top-n objects by memory usage (in bytes),
    as seen by Python's garbage collector and sys.getsizeof.

    Parameters
    ----------
    n : int, optional
        Number of objects to return (sorted from largest to smallest),
        by default 5.

    Returns
    -------
    list of tuple
        A list of (type_name, size_in_bytes), sorted by size in descending order.

    Notes
    -----
    - sys.getsizeof does not account for nested object references, so
      these numbers can be incomplete for complex objects.
    - Some objects might raise TypeError when calling sys.getsizeof.
      Such objects are simply skipped.
    """
    all_objs = gc.get_objects()
    objects_and_sizes = []

    for obj in all_objs:
        try:
            size = sys.getsizeof(obj)
            type_name = type(obj).__name__
            objects_and_sizes.append((type_name, size))
        except TypeError:
            # Some objects (especially extension types) might not work with getsizeof
            pass

    # Sort descending by size
    objects_and_sizes.sort(key=lambda x: x[1], reverse=True)

    # Return only top-n
    return objects_and_sizes[:n]


def log_top_n_memory_objects(n: int = 5) -> None:
    """
    Log (using Python's logging) the top-n objects by memory usage.

    Parameters
    ----------
    n : int, optional
        Number of objects to log, by default 5

    Returns
    -------
    None
    """
    top_objs = get_top_n_memory_objects(n=n)
    logger.info("Top %d objects by approximate memory usage (sys.getsizeof):", n)
    for rank, (type_name, size_bytes) in enumerate(top_objs, start=1):
        logger.info("%d) %s: %.2f MB", rank, type_name, size_bytes / (1024 * 1024))


def log_top_n_objects_pympler(n: int = 5):
    """
    Log the top-n objects by deep memory usage using Pympler's asizeof.
    """
    all_objs = gc.get_objects()
    objects_and_sizes = []
    for obj in all_objs:
        try:
            size = asizeof.asizeof(obj)
            objects_and_sizes.append((type(obj).__name__, size, obj))
        except Exception:
            # Some extension objects might still fail
            pass

    objects_and_sizes.sort(key=lambda x: x[1], reverse=True)
    top_n = objects_and_sizes[:n]
    logger.info("Top %d objects by deep memory usage (Pympler):", n)
    for i, (type_name, size_bytes, obj_ref) in enumerate(top_n, start=1):
        logger.info("%d) %s: %.2f MB [object reference: %r]",
                    i, type_name, size_bytes / (1024*1024), obj_ref)
'''


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
