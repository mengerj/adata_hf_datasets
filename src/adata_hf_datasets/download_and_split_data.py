import json
import logging
import os

import anndata as ad
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_figshare_file(url: str, download_dir: str, file_name: str) -> str:
    """
    Download data from a Figshare link with a progress bar and save to the specified directory.

    Parameters
    ----------
    url : str
        The Figshare URL of the dataset.
    download_dir : str
        The directory to save the downloaded file.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    logger.info("Starting download from Figshare...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        logger.error(f"Failed to fetch the data. Status code: {response.status_code}")
        raise ValueError("Failed to download data from Figshare.")

    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_name)
    # check if file already exists and skip download
    if os.path.exists(file_path):
        logger.info(f"File already exists at {file_path}. Skipping download.")
        return file_path
    # Total file size in bytes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(file_path, "wb") as f, tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading") as t:
        for chunk in response.iter_content(chunk_size=block_size):
            t.update(len(chunk))
            f.write(chunk)

    logger.info(f"Data downloaded successfully and saved to {file_path}.")
    return file_path


def split_train_test(data: ad.AnnData, split_by: str = "random", test_ratio: float = 0.1) -> tuple:
    """
    Split the AnnData object into train and test sets.

    Parameters
    ----------
    data : anndata.AnnData
        The input dataset in AnnData format.
    split_by : str, optional
        The splitting strategy: "random" or "dataset_id".
    test_ratio : float, optional
        The proportion of the test set.

    Returns
    -------
    tuple
        Train and test AnnData objects.
    """
    if split_by == "random":
        train_indices, test_indices = train_test_split(range(data.shape[0]), test_size=test_ratio, random_state=42)
    elif split_by == "dataset_id":
        if "dataset_id" not in data.obs:
            logger.error("dataset_id not found in .obs.")
            raise KeyError("dataset_id must be present in .obs to use 'dataset_id' splitting.")

        dataset_groups = data.obs.groupby("dataset_id").size()
        test_datasets = []
        test_size = 0
        for dataset_id, count in dataset_groups.iteritems():
            if test_size >= test_ratio * len(data):
                break
            test_datasets.append(dataset_id)
            test_size += count

        is_test = data.obs["dataset_id"].isin(test_datasets)
        test_indices = np.where(is_test)[0]
        train_indices = np.where(~is_test)[0]
    else:
        logger.error(f"Invalid split_by option: {split_by}")
        raise ValueError("split_by must be 'random' or 'dataset_id'.")

    train_data = data[train_indices].copy()
    test_data = data[test_indices].copy()

    logger.info(f"Split completed: {len(train_indices)} train samples, {len(test_indices)} test samples.")
    return train_data, test_data


def save_splits(train_data: ad.AnnData, test_data: ad.AnnData, output_dir: str, file_name, format: str = "h5ad"):
    """
    Save train and test splits in the specified directory.

    Parameters
    ----------
    train_data : anndata.AnnData
        The train dataset.
    test_data : anndata.AnnData
        The test dataset.
    output_dir : str
        The output directory to save the splits.
    file_name : str
        The base name of the files to save.
    format : str, optional
        The format to save the data, either "h5ad" or "zarr".
    """
    # check if file_name has a suffix and remove it
    if "." in file_name:
        file_name = file_name.split(".")[0]
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_path = os.path.join(train_dir, f"{file_name}.{format}")
    test_path = os.path.join(test_dir, f"{file_name}.{format}")

    if format == "h5ad":
        train_data.write_h5ad(train_path)
        test_data.write_h5ad(test_path)
    elif format == "zarr":
        train_data.write_zarr(train_path)
        test_data.write_zarr(test_path)
    else:
        logger.error(f"Unsupported format: {format}")
        raise ValueError("Format must be 'h5ad' or 'zarr'.")

    logger.info(f"Train data saved to {train_path}")
    logger.info(f"Test data saved to {test_path}")


def download_figshare_data(
    download_dir: str = ".",
    figshare_id: str = "12420968",
    base_url="https://api.figshare.com/v2",
    out_format: str = "h5ad",
):
    """Download the data from Figshare and split it into train and test sets.

    Parameters
    ----------
    download_dir : str, optional
        The folder to where the data will be downloaded.
    figshare_id : str, optional
        The Figshare ID of the dataset.
    base_url : str, optional
        The base URL of the Figshare API.
    out_format : str, optional
        The format to save the data, either "h5ad" or "zarr".
    """
    # Configure the logger
    logging.basicConfig(level=logging.INFO)
    os.makedirs(download_dir, exist_ok=True)

    # Step 1: Download the data

    r = requests.get(base_url + "/articles/" + figshare_id)
    # Load the metadata as JSON
    if r.status_code != 200:
        raise ValueError("Request to figshare failed:", r.content)
    else:
        metadata = json.loads(r.text)
    # View metadata:
    files_meta = metadata["files"]
    data_paths = {}
    for file_meta in files_meta:
        download_url = file_meta["download_url"]
        file_size = file_meta["size"]
        file_name = file_meta["name"]
        # Format size in GB for readability
        file_size_gb = file_size / 1024**3
        logger.info(f"Downloading File: {file_name}, Size: {file_size_gb:.2f} GB")
        try:
            data_paths[file_name] = download_figshare_file(download_url, download_dir, file_name=file_name)
        except ValueError as e:
            logger.error(e)
            return
    for file_name in data_paths.keys():
        # Step 2: Load the data (assuming extracted files)
        data_path = data_paths[file_name]
        if not os.path.exists(data_path):
            logger.error(f"File {data_path} not found.")
            return
        if data_path.endswith(".h5ad"):
            data = ad.read_h5ad(data_path)
        elif data_path.endswith(".zarr"):
            data = ad.read_zarr(data_path)
        else:
            logger.error(f"Unsupported file format: {data_path}")
            return

        logger.info(f"Loaded AnnData object with {data.shape[0]} samples and {data.shape[1]} features.")
        return
