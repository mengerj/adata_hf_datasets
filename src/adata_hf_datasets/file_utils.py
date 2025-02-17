import h5py
import os
import logging
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import anndata as ad
import pandas as pd
import numpy as np
from typing import Optional


def download_from_link(url, save_path):
    """
    Download a file with a progress bar.

    Parameters
    ----------
    url : str
        The direct URL to the file.
    save_path : str
        The local file path to save the downloaded file.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for failed requests (e.g., 404, 403)

        # Get total file size from headers
        total_size = int(response.headers.get("content-length", 0))

        # Download with progress bar
        with (
            open(save_path, "wb") as file,
            tqdm(
                desc=save_path,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                bar.update(len(chunk))  # Update progress bar

        print(f"\nDownload complete: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nDownload failed: {e}")
        return False


def save_and_upload_adata(
    adata, local_path, nextcloud_config=None, create_share_link=True
):
    """
    Saves an AnnData object to a file and optionally uploads it to a Nextcloud server based on provided configuration.

    Parameters:
        adata (AnnData): The AnnData object to save.
        local_path (str): Local path to save the .h5ad file.
        nextcloud_config (dict, optional): Configuration dictionary for Nextcloud which contains:
                                           'url' (str): URL to the Nextcloud server.
                                           'username' (str): Username for Nextcloud.
                                           'password' (str): Password for Nextcloud.
                                           'remote_path' (str): Remote path in Nextcloud where the file will be uploaded.

    Example:
        adata = AnnData(np.random.rand(10, 10))  # Example AnnData object
        save_and_upload_adata(adata, 'local_file.h5ad', nextcloud_config={
                              'url': 'https://nxc-fredato.imbi.uni-freiburg.de',
                              'username': 'your_username',
                              'password': 'your_password',
                              'remote_path': '/path/on/nextcloud/file.h5ad.gz'})
    """
    # Save the AnnData object to a local .h5ad file
    adata.write(local_path, compression="gzip")
    logging.info(f"File saved locally at {local_path}")

    # Upload the file to Nextcloud if configuration is provided
    if nextcloud_config:
        try:
            create_nested_directories(
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )

            response = upload_file_to_nextcloud(
                local_path,
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            logging.info(
                f"File uploaded to Nextcloud at {nextcloud_config['remote_path']} with status code {response.status_code}"
            )
        except Exception as e:
            logging.error(f"Failed to upload file to Nextcloud: {e}")
        if create_share_link:
            share_url = get_share_link(
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            return share_url


def save_embedding_data(
    data: pd.DataFrame,
    local_path: str,
    nextcloud_config: Optional[dict] = None,
    create_share_link: bool = True,
) -> Optional[str]:
    """
    Save embedding data to a local file and optionally upload it to Nextcloud.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with sample IDs as index and embedding vectors as rows.
    local_path : str
        Local file path where the embedding data is saved (e.g. ending with .npz).
    nextcloud_config : dict, optional
        Nextcloud configuration dictionary.
    create_share_link : bool, optional
        Whether to create and return a share link for the file.

    Returns
    -------
    str or None
        The share link URL if the file is uploaded and a link is created, otherwise None.

    Notes
    -----
    For simplicity, we use numpy.savez_compressed to store the data.
    The saved file will contain two arrays: "data" (the embedding matrix) and "sample_ids" (the row labels).
    """
    # Save using NumPy compressed format.
    np.savez_compressed(local_path, data=data.values, sample_ids=data.index.values)
    logger.info("Embedding data saved locally at %s", local_path)

    # Upload to Nextcloud if configured.
    if nextcloud_config:
        try:
            # Import functions from your file utils module.
            from .file_utils import (
                create_nested_directories,
                upload_file_to_nextcloud,
                get_share_link,
            )

            create_nested_directories(
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            response = upload_file_to_nextcloud(
                local_path,
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            logger.info(
                "Embedding file uploaded to Nextcloud at %s with status code %s",
                nextcloud_config["remote_path"],
                response.status_code,
            )
        except Exception as e:
            logger.error("Failed to upload embedding data to Nextcloud: %s", e)
            return None
        if create_share_link:
            share_url = get_share_link(
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            return share_url
    return None


def upload_file_to_nextcloud(file_path, nextcloud_url, username, password, remote_path):
    """
    Uploads a file to a Nextcloud server via WebDAV.

    Parameters:
        file_path (str): Path to the local file to upload.
        nextcloud_url (str): URL to your personal nextcloud instance. Can be found in the settings of your nextcloud account. (File Seetings -> WebDAV)
        username (str): Username for Nextcloud authentication.
        password (str): Password for Nextcloud authentication.
        remote_path (str): Path in Nextcloud where the file will be stored.

    Returns:
        requests.Response: The response object from the requests library.

    Example:
        response = upload_file_to_nextcloud('path/to/local/file.txt',
                                            'https://nxc-fredato.imbi.uni-freiburg.de',
                                            'your_username',
                                            'your_password',
                                            '/path/to/save/file.txt')
        print(response.status_code)
    """
    # Complete URL to access the WebDAV interface
    full_url = f"{nextcloud_url}/remote.php/dav/files/{username}/{remote_path}"

    # Open the file in binary mode
    with open(file_path, "rb") as file_content:
        # Make the PUT request to upload the file
        response = requests.put(
            full_url, data=file_content, auth=HTTPBasicAuth(username, password)
        )
    return response


def create_nested_directories(nextcloud_url, username, password, remote_path):
    """
    Creates nested directories in Nextcloud one by one.

    Parameters:
        nextcloud_url (str): URL to the Nextcloud server.
        username (str): Username for Nextcloud.
        password (str): Password for Nextcloud.
        remote_path (str): Remote directory path where directories need to be created.
    """
    base_url = f"{nextcloud_url}/remote.php/dav/files/{username}"
    segments = remote_path.strip("/").split("/")
    path = ""
    for segment in segments[:-1]:  # Exclude the last segment assuming it's a file
        path += f"/{segment}"
        folder_url = f"{base_url}{path}"
        response = requests.request(
            "MKCOL", folder_url, auth=HTTPBasicAuth(username, password)
        )
        if response.status_code == 201:
            print(f"Created directory: {segment}")
        elif response.status_code in [
            405,
            301,
        ]:  # Directory already exists or moved permanently
            print(f"Directory already exists: {segment}")
        else:
            print(
                f"Error {response.status_code} creating directory {segment}: {response.reason}"
            )
            return False
    return True


def get_share_link(nextcloud_url, username, password, remote_path):
    """
    Creates a public share link for a file on Nextcloud.

    Parameters:
        nextcloud_url (str): URL to the Nextcloud server.
        username (str): Username for Nextcloud authentication.
        password (str): Password for Nextcloud authentication.
        remote_path (str): Path in Nextcloud where the file is stored, starting from the root of the user's files directory.

    Returns:
        str: URL of the shared file if successful, None otherwise.
    """
    api_url = f"{nextcloud_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
    data = {
        "shareType": 3,  # This specifies a public link
        "path": remote_path,
        "permissions": 1,  # Read-only permissions
    }
    headers = {
        "OCS-APIRequest": "true",
        "Accept": "application/xml",  # Set header to accept XML
    }

    response = requests.post(
        api_url, data=data, headers=headers, auth=HTTPBasicAuth(username, password)
    )

    if response.status_code == 200:
        try:
            # Parse the XML response
            root = ET.fromstring(response.text)
            token = root.find(".//token").text
            share_url = (
                f"{nextcloud_url}/s/{token}/download"  # Ensure direct file download
            )
            return share_url
        except ET.ParseError:
            print("Failed to parse the XML data.")
            return None
        except AttributeError:
            print("Failed to find the required elements in the XML.")
            return None
    else:
        print(f"Failed to create share link: {response.status_code}")
        print("Response details:", response.text)
        return None


def download_file_from_share_link(share_link, save_path, chunk_size=8192):
    """
    Downloads a file from a Nextcloud share link and checks if it's a valid .h5ad file.

    Parameters:
        share_link (str): The full share link URL to the file.
        save_path (str): The local path where the file should be saved.

    Returns:
        bool: True if the download was successful and the file is a valid .h5ad, False otherwise.

    Example:
        success = download_file_from_share_link('https://nxc-fredato.imbi.uni-freiburg.de/s/Zs6pAa8P5ynDTiP',
                                                'path/to/save/file.h5ad')
        print("Download successful:", success)
    """
    # Send a GET request to the share link
    try:
        # Step 1: Stream download the file
        with requests.get(share_link, stream=True) as response:
            response.raise_for_status()

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)

        # Step 2: Validate the file structure
        try:
            with h5py.File(save_path, "r") as h5_file:
                # Check if required keys exist in the HDF5 structure
                required_keys = ["X", "obs", "var"]  # Common in .h5ad files
                if all(key in h5_file for key in required_keys):
                    print("✅ File is a valid .h5ad file.")
                    return True
                else:
                    print("⚠️ File is an HDF5 file but missing required .h5ad keys.")
                    return False
        except Exception as e:
            print(f"❌ Error while checking the file: {e}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download the file: {e}")
        return False

    finally:
        # Optional: Delete file if invalid
        if not os.path.exists(save_path) or not h5py.is_hdf5(save_path):
            print("❌ Deleting invalid file.")
            os.remove(save_path)
            return False


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

    with (
        open(file_path, "wb") as f,
        tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading") as t,
    ):
        for chunk in response.iter_content(chunk_size=block_size):
            t.update(len(chunk))
            f.write(chunk)

    logger.info(f"Data downloaded successfully and saved to {file_path}.")
    return file_path


def download_figshare_data(
    download_dir: str = ".",
    figshare_id: str = "12420968",
    base_url="https://api.figshare.com/v2",
    out_format: str = "h5ad",
    wanted_file_name: str | None = None,
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
    wanted_file_name: str, optional
        If you only want to download a certain file. Otherwise all files in the remote directory will be downloaded.
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
        if wanted_file_name is not None and file_name != wanted_file_name:
            continue
        # Format size in GB for readability
        file_size_gb = file_size / 1024**3
        logger.info(f"Downloading File: {file_name}, Size: {file_size_gb:.2f} GB")
        try:
            data_paths[file_name] = download_figshare_file(
                download_url, download_dir, file_name=file_name
            )
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

        logger.info(
            f"Loaded AnnData object with {data.shape[0]} samples and {data.shape[1]} features."
        )
        return
