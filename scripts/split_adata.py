import os
import anndata
import logging

# from adata_hf_datasets.utils import setup_logging
logger = logging.getLogger(__name__)


def split_anndata_file(file_path: str, output_dir: str, n_splits: int = 10) -> None:
    """
    Split an AnnData dataset (H5AD file) into n_splits parts and save each part
    to the specified output directory.

    Parameters
    ----------
    file_path : str
        Path to the input .h5ad file. This data is sourced from your local file system.
    output_dir : str
        Path to the directory where the split .h5ad files will be saved.
    n_splits : int, optional
        Number of splits to create (default is 10).

    Returns
    -------
    None
        Saves the split files to the output directory.

    Notes
    -----
    This function loads the AnnData object from the provided H5AD file.
    The observations (cells) are split approximately equally among the parts.
    If the total number of observations is not divisible by n_splits, some splits
    will have one additional observation.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the entire AnnData object
    logger.info(f"Loading AnnData from {file_path}")
    adata = anndata.read_h5ad(file_path)

    n_obs = adata.n_obs
    logger.info(f"AnnData contains {n_obs} observations.")

    # Compute base chunk size and remainder to distribute among the first splits
    base_chunk_size = n_obs // n_splits
    remainder = n_obs % n_splits

    start_idx = 0
    for i in range(n_splits):
        # If there is a remainder, distribute one extra observation to current split
        current_split_size = base_chunk_size + (1 if i < remainder else 0)

        # Define slice boundaries
        end_idx = start_idx + current_split_size

        # Create the subset of adata
        split_adata = adata[start_idx:end_idx].copy()
        logger.info(
            f"Split {i + 1}/{n_splits}: Index range [{start_idx}:{end_idx}], "
            f"Size {current_split_size}."
        )

        # Save to a new H5AD file
        output_filename = os.path.join(output_dir, f"part_{i + 1}.h5ad")
        logger.info(f"Writing split {i + 1} to {output_filename}")
        split_adata.write_h5ad(output_filename)

        # Update start index for next split
        start_idx = end_idx


if __name__ == "__main__":
    # Example usage:
    input_file = "data/RNA/raw/train/cellxgene_pseudo_bulk_full.h5ad"
    output_dir = "data/RNA/raw/train/cellxgene_pseudo_bulk_splits/"
    os.makedirs(output_dir, exist_ok=True)
    split_anndata_file(input_file, output_dir, n_splits=10)
