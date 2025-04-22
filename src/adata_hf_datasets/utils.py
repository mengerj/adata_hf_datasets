import logging

import anndata
import numpy as np

from datetime import datetime
import os
from pathlib import Path
import importlib
from huggingface_hub import HfApi
import tempfile
from string import Template
import scipy.sparse as sp
import pandas as pd
import hashlib
from typing import Iterator, List
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)


class BatchChunkLoader:
    """
    Lazy loader for .h5ad files that yields AnnData chunks
    composed of whole batches up to a target size.

    Each chunk will contain one or more entire batches,
    accumulating batches until at least `chunk_size` cells
    are reached (but never splitting a batch).

    Parameters
    ----------
    path : Path
        Path to the H5AD file.
    chunk_size : int
        Minimum number of cells per chunk.
    batch_key : str
        Column in `.obs` to group by (batches will stay intact).
    """

    def __init__(self, path: Path, chunk_size: int, batch_key: str):
        self.path = path
        self.chunk_size = chunk_size
        self.batch_key = batch_key

    def __iter__(self) -> Iterator[AnnData]:
        """
        Yields
        ------
        AnnData
            Next chunk loaded into memory, containing whole batches.
        """
        logger.info("Opening H5AD in backed mode: %s", self.path)
        adata_backed = sc.read(self.path, backed="r")

        # preserve the order of batches as they appear
        obs = adata_backed.obs
        if self.batch_key not in obs.columns:
            raise KeyError(f"batch_key '{self.batch_key}' not in adata.obs")

        batch_labels: List = obs[self.batch_key].tolist()
        # unique in original order
        seen = set()
        unique_batches = [b for b in batch_labels if not (b in seen or seen.add(b))]

        current_batches: List = []
        current_count = 0

        for batch in unique_batches:
            # count cells in this batch
            n_cells = (obs[self.batch_key] == batch).sum()
            # if adding this batch would overshoot—and we already have some batches—flush
            if current_batches and (current_count + n_cells > self.chunk_size):
                yield self._load_chunk(adata_backed, current_batches)
                current_batches = []
                current_count = 0

            current_batches.append(batch)
            current_count += n_cells

        # any remaining batches
        if current_batches:
            yield self._load_chunk(adata_backed, current_batches)

        # close backed file
        adata_backed.file.close()

    def _load_chunk(self, adata_backed: AnnData, batches: List) -> AnnData:
        """
        Load selected batches from the backed AnnData into memory.

        Parameters
        ----------
        adata_backed : AnnData
            AnnData in backed mode.
        batches : list
            Batch labels to include.

        Returns
        -------
        AnnData
            In‑memory AnnData containing only those batches.
        """
        logger.info(
            "Loading chunk with %d batches (%s)",
            len(batches),
            ", ".join(map(str, batches)),
        )
        # boolean mask selection preserves obs/var slicing on backed AnnData
        mask = adata_backed.obs[self.batch_key].isin(batches)
        return adata_backed[mask].to_adata()


def stable_numeric_id(s: str) -> int:
    """
    Convert a string to a stable 64-bit numeric ID using part of an MD5 hash.

    Parameters
    ----------
    s : str
        Input string to be hashed.

    Returns
    -------
    int
        A 64-bit integer derived from the hash of `s`.
    """
    # Compute MD5 hash of the string
    md5_bytes = hashlib.md5(s.encode("utf-8")).digest()
    # Take the first 4 bytes of the MD5 digest and interpret them as an unsigned big-endian integer
    numeric_id = int.from_bytes(md5_bytes[:4], byteorder="big", signed=False)
    return numeric_id


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


def consolidate_low_frequency_categories(
    adata: anndata.AnnData, columns: list | str, threshold: int, remove=False
):
    """Consolidates low frequency categories in specified columns of an AnnData object.

    Modifies the AnnData object's .obs by setting entries in specified columns
    to 'remaining {column_name}' or removing them if their frequency is below a specified threshold.
    Converts columns to non-categorical if necessary to adjust the categories dynamically.

    Parameters
    ----------
    adata
        The AnnData object to be processed.
    columns
        List of column names in adata.obs to check for low frequency.
    threshold
        Frequency threshold below which categories are considered low.
    remove
        If True, categories below the threshold are removed entirely.

    Returns
    -------
    anndata.AnnData: The modified AnnData object.
    """
    # Ensure the object is loaded into memory if it's in backed mode
    if adata.isbacked:
        adata = adata.to_memory()
    adata_cut = adata.copy()
    if not isinstance(columns, list):
        columns = [columns]
    for col in columns:
        if col in adata_cut.obs.columns:
            # Convert column to string if it's categorical
            if isinstance(adata_cut.obs[col].dtype, pd.CategoricalDtype):
                as_string = adata_cut.obs[col].astype(str)
                adata_cut.obs[col] = as_string

            # Calculate the frequency of each category
            freq = adata_cut.obs[col].value_counts()

            # Identify low frequency categories
            low_freq_categories = freq[freq < threshold].index

            if remove:
                # Remove entries with low frequency categories entirely
                mask = ~adata_cut.obs[col].isin(low_freq_categories)
                adata_cut._inplace_subset_obs(mask)
                # Convert column back to categorical with new categories
                adata_cut.obs[col] = pd.Categorical(adata_cut.obs[col])
            else:
                # Update entries with low frequency categories to 'remaining {col}'
                adata_cut.obs.loc[adata_cut.obs[col].isin(low_freq_categories), col] = (
                    f"remaining {col}"
                )

                # Convert column back to categorical with new categories
                adata_cut.obs[col] = pd.Categorical(adata_cut.obs[col])

        else:
            print(f"Column {col} not found in adata_cut.obs")

    return adata_cut


def fix_non_numeric_nans(adata: anndata.AnnData) -> None:
    """
    For each column in ``adata.obs`` that is not strictly numeric,
    replace NaN with 'unknown'. This prevents mixed float/string
    issues that SCVI can run into when sorting categorical columns.

    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to fix in-place. Must have .obs attribute.
    """
    from pandas.api.types import is_numeric_dtype, is_categorical_dtype

    for col in adata.obs.columns:
        if is_numeric_dtype(adata.obs[col]):
            # strictly numeric -> do nothing
            continue

        if is_categorical_dtype(adata.obs[col]):
            # For a categorical column, we must add a new category
            # before filling with it
            if "unknown" not in adata.obs[col].cat.categories:
                adata.obs[col] = adata.obs[col].cat.add_categories(["unknown"])
            adata.obs[col] = adata.obs[col].fillna("unknown")
        else:
            # For object/string columns, cast to str, then fillna
            adata.obs[col] = adata.obs[col].astype(str).fillna("unknown")


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
    metadata: dict | None = None,
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
                    share_info=metadata,
                )
            )

        # Push dataset with README
        # Step 3: Define metadata with custom share_link
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
    share_info=None,
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
        share_info=share_info,
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
