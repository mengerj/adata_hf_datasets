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
from typing import Optional, Union, Any
import tempfile
from pathlib import Path
import random
import anndata
import shutil
import gc
from anndata.abc import CSRDataset

logger = logging.getLogger(__name__)


def add_obs_column_to_h5ad(
    infile: Union[str, Path],
    temp_out: Union[str, Path],
    column_name: str = "sample_index",
    column_data: Optional[np.ndarray] = None,
    dtype: np.dtype = np.int64,
    is_categorical: bool = False,
) -> Path:
    """
    Copy an .h5ad on disk and inject a column into the obs dataframe
    with minimal memory usage by using backed mode.

    Parameters
    ----------
    infile : Union[str, Path]
        Path to the original .h5ad file.
    temp_out : Union[str, Path]
        Path where the modified copy will be written.
    column_name : str, optional
        Name of the column to add to obs, by default "sample_index".
    column_data : Optional[np.ndarray], optional
        Data for the column. If None and column_name is "sample_index",
        will use np.arange(n_obs). Must be provided for other column names.
    dtype : np.dtype, optional
        Data type for the column, by default np.int64.
    is_categorical : bool, optional
        Whether to mark the column as categorical, by default False.
        Useful for string data that represents categories.

    Returns
    -------
    Path
        The path to `temp_out`, now containing the new column in obs.

    Notes
    -----
    - Uses AnnData's backed mode to minimize memory usage.
    - Only loads the metadata into memory, not the full X matrix.
    - If you need to handle files too large for even this approach, consider
      a more specialized solution with pure h5py.
    """
    # avoid POSIX locking
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    infile = Path(infile)
    temp_out = Path(temp_out)

    # Check if we're modifying in-place
    in_place = infile == temp_out

    if not in_place:
        logger.info(f"Copying {infile} → {temp_out}")
        shutil.copyfile(str(infile), str(temp_out))
    else:
        logger.info(f"Modifying {infile} in-place")

    # Open in backed mode - only loads metadata, not the full matrix
    logger.info("Opening in backed mode to add column")

    adata = None
    try:
        adata = anndata.read_h5ad(temp_out, backed="r+")

        # Get number of observations
        n_obs = adata.n_obs
        logger.info(f"File has {n_obs} observations")

        if column_name in adata.obs.columns:
            logger.info(f"Column '{column_name}' already exists, skipping")
            return temp_out

        # Generate data if needed
        if column_data is None:
            if column_name == "sample_index":
                logger.info("Generating sample_index data")
                column_data = np.arange(n_obs, dtype=dtype)
            else:
                raise ValueError(
                    f"column_data must be provided for column '{column_name}'"
                )

        # Ensure data has the right length
        if len(column_data) != n_obs:
            raise ValueError(
                f"column_data length ({len(column_data)}) doesn't match n_obs ({n_obs})"
            )

        # Add the column
        logger.info(f"Adding column '{column_name}' to obs")
        if is_categorical:
            # Convert to categorical if requested
            from pandas import Categorical

            adata.obs[column_name] = Categorical(column_data)
        else:
            adata.obs[column_name] = column_data

        # ---- FIX: check for missing raw.X ----
        # This is a workaround for a bug that writing to h5ad from backed mode gives if raw is empty/corrupted
        is_backed_object = getattr(adata, "isbacked", False)
        raw_is_csr_hdf5_group = (
            adata.raw is not None
            and isinstance(adata.raw.X, CSRDataset)  # ← catches wrapper
            # defensive fallback in case class name changes
            or (
                hasattr(adata, "file")
                and isinstance(adata.file["raw"]["X"], h5py.Group)
            )
        )

        if is_backed_object and raw_is_csr_hdf5_group:
            logger.info(
                "Backed object with sparse .raw – materialising .raw in memory."
            )
            adata = (
                adata.to_memory()
            )  # move the whole object into memory and hope it isnt to big
        # -------------------------------------------------------------------
        # Write changes to disk
        logger.info("Writing changes to disk")
        adata.write_h5ad(temp_out, compression="gzip")

        logger.info(f"Done adding column '{column_name}'")
    except Exception as e:
        logger.error(f"Error while adding column: {e}")
        raise
    finally:
        # Explicitly close the AnnData object to release file handles
        if adata is not None:
            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            del adata
            # Force garbage collection to ensure file handles are released
            gc.collect()

    # Verify the file can be opened after modification
    try:
        with h5py.File(temp_out, "r") as f:
            if "obs" in f and column_name in f["obs"]:
                logger.info(
                    f"Verified column '{column_name}' exists in the output file"
                )
            else:
                logger.warning(
                    f"Column '{column_name}' was not found in the output file"
                )
    except Exception as e:
        logger.error(f"Error verifying output file: {e}")

    return temp_out


# Legacy function for backward compatibility
def add_sample_index_to_h5ad(
    infile: Union[str, Path],
    temp_out: Union[str, Path],
) -> Path:
    """
    Legacy function that calls add_obs_column_to_h5ad with sample_index as column_name.
    Kept for backward compatibility.
    """
    return add_obs_column_to_h5ad(infile, temp_out, column_name="sample_index")


def load_adata_from_hf_dataset(
    test_dataset,
    sample_size=10,
):
    """
    Load an AnnData object from a Hugging Face test dataset that contains a share link to an external `.h5ad` file.

    This function downloads the file to a temporary directory
    and reads it into memory.

    Parameters
    ----------
    test_dataset : dict
        A dictionary-like object representing the HF dataset split (e.g., `test_dataset["train"]`).
        It must contain an 'anndata_ref' field, where each element is a JSON string with a "file_path" key.
    sample_size : int, optional
        Number of random rows to check for file path consistency before download, by default 10.

    Returns
    -------
    anndata.AnnData
        The AnnData object read from the downloaded `.h5ad` file.

    Notes
    -----
    - Data is assumed to come from a Hugging Face dataset with a single unique `file_path` for all rows.
    - The function downloads the file to a temporary directory, which is removed when this function returns.
    - If multiple rows have different `file_path` values, the function raises an error.
    """
    # If the dataset split is large, reduce the sample size to the dataset size
    size_of_dataset = len(test_dataset)
    sample_size = min(sample_size, size_of_dataset)

    # Randomly sample rows to ensure all file paths match
    indices_to_check = random.sample(range(size_of_dataset), sample_size)
    paths = []
    for idx in indices_to_check:
        adata_ref = test_dataset[idx]["anndata_ref"]
        paths.append(adata_ref["file_record"]["dataset_path"])

    # Ensure that all random rows have the same file path
    first_path = paths[0]
    for p in paths[1:]:
        if p != first_path:
            raise ValueError(
                "Not all sampled rows contain the same file path. Please verify the dataset consistency."
            )

    # Download the file from the share link into a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test.h5ad"
        download_file_from_share_link(first_path, str(save_path))
        adata = ad.read_h5ad(save_path)

    return adata


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


# -----------------------------------------------------------------------------#
# Helper: Very fast sanity-check of a Nextcloud share link                      #
# -----------------------------------------------------------------------------#
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"  # First 8 bytes of every HDF5 file (.h5, .h5ad)
NUMPY_MAGIC = b"\x93NUMPY"  # First 6 bytes of a .npy file
ZIP_MAGIC = b"PK\x03\x04"  # First 4 bytes of a zip/npz
_CHUNK_RANGE_HEADER = {"Range": "bytes=0-255"}  # 256-byte range


def verify_share_link(
    share_link: str,
    expected_suffix: str | None = None,
    timeout: int = 10,
) -> bool:
    """
    Cheap on-the-wire verification that a Nextcloud share link is alive and
    points to a *downloadable* file of the expected type.

    The function sends _two_ minimal requests:

    1. **HEAD** – confirms HTTP 2xx and collects basic metadata.
    2. **Range GET** (first ≤256 bytes) – checks the file signature (“magic”).

    Parameters
    ----------
    share_link
        Public Nextcloud share URL.
    expected_suffix
        File‐extension (e.g. ``".h5ad"``).  When *None*, the suffix is inferred
        from the URL.
    timeout
        Network timeout in seconds for each request.

    Returns
    -------
    bool
        ``True`` if the link is reachable *and* the leading bytes match the
        signature of ``expected_suffix`` (if supplied); ``False`` otherwise.

    Notes
    -----
    *No full download is performed.*  The largest transfer is 256 bytes.
    """
    suffix = (expected_suffix or Path(share_link).suffix).lower()

    # --- 1) HEAD -------------------------------------------------------------#
    try:
        head = requests.head(share_link, allow_redirects=True, timeout=timeout)
        head.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("HEAD request failed for %s → %s", share_link, exc)
        return False

    # --- 2) tiny Range-GET ---------------------------------------------------#
    try:
        rng = requests.get(
            share_link,
            headers=_CHUNK_RANGE_HEADER,
            allow_redirects=True,
            timeout=timeout,
        )
        rng.raise_for_status()
        magic = rng.content[:8]
    except requests.RequestException as exc:
        logger.warning("Range request failed for %s → %s", share_link, exc)
        return False

    # --- 3) Signature check --------------------------------------------------#
    if suffix == ".h5ad":
        return magic.startswith(HDF5_MAGIC)
    if suffix == ".npy":
        return magic.startswith(NUMPY_MAGIC)
    if suffix == ".npz":
        return magic.startswith(ZIP_MAGIC)
    # Unknown type → we can at least confirm it is downloadable
    return True


def upload_folder_to_nextcloud(
    data_folder: Union[str, Path],
    nextcloud_config: dict[str, Any],
    create_share_link: bool = True,
) -> Path:
    """
    Upload every file under *data_folder* to Nextcloud **unless** a valid link
    already exists in ``share_map.json``.

    For each (new or re-uploaded) file a fresh share link is obtained via
    :pyfunc:`save_and_upload_adata`.  All final links – verified existing ones
    *and* newly created ones – are (over)written to one
    ``data_folder/share_map.json``.

    Parameters
    ----------
    data_folder
        Folder that contains the files to be uploaded.
    nextcloud_config
        Dict forwarded to :pyfunc:`save_and_upload_adata`.
        At minimum it must contain *remote_path*; this field is overwritten
        internally for each file.
    create_share_link
        Forwarded to :pyfunc:`save_and_upload_adata`.

    Returns
    -------
    Path
        Absolute path of the written ``share_map.json``.

    Raises
    ------
    ValueError
        If *data_folder* is not an existing directory.

    References
    ----------
    All data originate from files under *data_folder* and are stored to the
    Nextcloud instance configured in *nextcloud_config*.
    """

    data_folder = Path(data_folder)
    if not data_folder.is_dir():
        raise ValueError(f"{data_folder!r} is not a directory")

    # ---------------------------------------------------------------------#
    # 1) Load existing mapping (if any)                                     #
    # ---------------------------------------------------------------------#
    mapping_path = data_folder / "share_map.json"
    if mapping_path.exists():
        logger.info("Found existing share_map.json → loading")
        share_map: dict[str, str] = json.loads(mapping_path.read_text("utf-8"))
    else:
        logger.info("No share_map.json present → starting fresh")
        share_map = {}

    # ---------------------------------------------------------------------#
    # 2) Walk over every file                                               #
    # ---------------------------------------------------------------------#
    for root, _, files in os.walk(data_folder):
        root_path = Path(root)
        for fname in files:
            local_path = root_path / fname
            if local_path.name == "share_map.json":
                continue  # skip the mapping file itself

            rel_path = local_path.relative_to(data_folder).as_posix()
            remote_path = f"data/{rel_path}"

            link_is_valid = False
            if rel_path in share_map:
                candidate_link = share_map[rel_path]
                logger.info("Verifying stored share link for %s …", rel_path)
                link_is_valid = verify_share_link(candidate_link, local_path.suffix)
                if link_is_valid:
                    logger.info("✓ link OK – skipping upload")
                else:
                    logger.info("✗ link broken – will re-upload")

            if link_is_valid:
                continue

            # ----------------------------------------------------------------#
            # 2a) Upload / re-upload                                          #
            # ----------------------------------------------------------------#
            nc_cfg = {**nextcloud_config, "remote_path": remote_path}
            logger.info("Uploading %s → %s …", local_path, remote_path)
            share_link = save_and_upload_adata(
                str(local_path),
                nc_cfg,
                create_share_link=create_share_link,
            )
            logger.info(" → got new share link %s", share_link)
            share_map[rel_path] = share_link

    # ---------------------------------------------------------------------#
    # 3) Persist (possibly updated) share_map                               #
    # ---------------------------------------------------------------------#
    logger.info("Writing share_map.json to %s", mapping_path)
    mapping_path.write_text(json.dumps(share_map, indent=2), encoding="utf-8")
    logger.info("Done.")
    return mapping_path


def save_and_upload_adata(local_path, nextcloud_config=None, create_share_link=True):
    """
    Saves an AnnData object to a file and optionally uploads it to a Nextcloud server based on provided configuration.

    Parameters:
        local_path (str): Local path where AnnData object is saved.
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
    if not nextcloud_config:
        return None  # dont return a share link if nextcloud config is not provided
    # extract the filename and the last dir name from the local path
    filename = os.path.basename(local_path)
    last_dir = os.path.basename(os.path.dirname(local_path))
    # replace the filename from the remotepath with the dir and the new filename
    old_file_name = os.path.basename(nextcloud_config["remote_path"])
    remote_path = nextcloud_config["remote_path"].replace(
        old_file_name, last_dir + "/" + filename
    )

    # Upload to Nextcloud if configured.
    try:
        # Import functions from your file utils module.
        from .file_utils import (
            create_nested_directories,
            upload_file_to_nextcloud,
            get_share_link,
        )

        # create a remote path by replacing
        create_nested_directories(
            nextcloud_config["url"],
            os.getenv(nextcloud_config["username"]),
            os.getenv(nextcloud_config["password"]),
            remote_path,
        )
        response = upload_file_to_nextcloud(
            local_path,
            nextcloud_config["url"],
            os.getenv(nextcloud_config["username"]),
            os.getenv(nextcloud_config["password"]),
            remote_path,
        )
        logger.info(
            "Embedding file uploaded to Nextcloud at %s with status code %s",
            remote_path,
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
            remote_path,
        )
        return share_url
    return None


class ReadWithProgress:
    """
    Wrap a file object to iterate over fixed‐size chunks, updating a tqdm bar,
    and providing a __len__ so that requests can send a Content-Length header.
    """

    def __init__(self, f, total_size, chunk_size=1024 * 1024, desc=None):
        self.f = f
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.tqdm = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            leave=True,
        )

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.f.read(self.chunk_size)
        if not chunk:
            self.tqdm.close()
            raise StopIteration
        self.tqdm.update(len(chunk))
        return chunk

    def __len__(self):
        # This lets requests detect the total length.
        return self.total_size


def upload_file_to_nextcloud(
    file_path: str,
    nextcloud_url: str,
    username: str,
    password: str,
    remote_path: str,
    chunk_size: int = 1024 * 1024,
    timeout: tuple = (5, 60),
) -> requests.Response:
    """
    Upload a file to Nextcloud via WebDAV, showing a tqdm progress bar (with speed),
    while ensuring a proper Content-Length header (no chunked transfer).

    Parameters
    ----------
    file_path : str
    nextcloud_url : str
    username : str
    password : str
    remote_path : str
    chunk_size : int, optional
    timeout : tuple, optional

    Returns
    -------
    requests.Response
    """
    full_url = f"{nextcloud_url.rstrip('/')}/remote.php/dav/files/{username}/{remote_path.lstrip('/')}"
    total_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        reader = ReadWithProgress(
            f, total_size, chunk_size, desc=f"Uploading {filename}"
        )
        # Manually set Content-Length so requests does NOT chunk
        headers = {"Content-Length": str(total_size)}
        response = requests.put(
            full_url,
            data=reader,
            auth=HTTPBasicAuth(username, password),
            headers=headers,
            timeout=timeout,
        )

    if response.ok:
        tqdm.write(f"✔️  Uploaded {filename}")
    else:
        tqdm.write(f"❌  Upload failed ({response.status_code}) for {filename}")

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
    Downloads a file from a Nextcloud share link and validates it based on its suffix.

    Parameters
    ----------
    share_link : str
        The full share link URL to the file.
    save_path : str
        The local path where the file should be saved.
    chunk_size : int, optional
        Size of each chunk in bytes during streaming; defaults to 8192.

    Returns
    -------
    bool
        True if the download was successful and the file is valid based on its suffix;
        False otherwise.

    References
    ----------
    Data is expected to come from a Nextcloud share link and is validated in memory.
    """
    # Step 1: Stream download the file
    try:
        with requests.get(share_link, stream=True) as response:
            response.raise_for_status()

            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download the file from '{share_link}': {e}")
        return False

    # Step 2: Validate based on suffix
    file_suffix = os.path.splitext(save_path)[1].lower()

    try:
        if file_suffix == ".h5ad":
            # Validate as an anndata-compatible HDF5 file
            with h5py.File(save_path, "r") as h5_file:
                required_keys = ["X", "obs", "var"]  # Common in .h5ad
                if all(key in h5_file for key in required_keys):
                    logger.info("File is a valid .h5ad file.")
                    return True
                else:
                    logger.warning(
                        "File is an HDF5 file but missing required .h5ad keys."
                    )
                    return False

        elif file_suffix == ".npz":
            # Validate as a .npz file (we can at least confirm we can load it)
            try:
                np.load(save_path, allow_pickle=True)
                logger.info("File is a valid .npz file.")
                return True
            except Exception as e:
                logger.error(f"Error while validating the downloaded file: {e}")
                return False

        elif file_suffix == ".npy":
            # Validate as a .npy file
            try:
                np.load(save_path, allow_pickle=True)
                logger.info("File is a valid .npy file.")
                return True
            except Exception as e:
                logger.error(f"Error while validating the downloaded file: {e}")
                return False

        else:
            # If your use-case requires more file types, add them here
            logger.warning(
                f"No specific validation logic for files of type '{file_suffix}'. "
                "Skipping validation."
            )
            return True

    except Exception as e:
        logger.error(f"Error while validating the downloaded file: {e}")
        return False


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
