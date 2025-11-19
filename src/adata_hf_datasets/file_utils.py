import h5py
import os
import logging
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET
from pathlib import PurePosixPath
from tqdm import tqdm
import json
import anndata as ad
import pandas as pd
import numpy as np
from typing import Optional, Union, Any
import tempfile
from pathlib import Path
import random
import shutil
import gc
from anndata.abc import CSRDataset
import errno
import time
import uuid
import zarr

logger = logging.getLogger(__name__)


def sanitize_zarr_keys(adata: ad.AnnData) -> None:
    """
    Sanitize column names in .obs and .var to be compatible with Zarr.

    Zarr does not allow forward slashes in keys. This function removes
    columns and layers with forward slashes from the AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to sanitize (modified in-place).
    """
    # Remove .obs columns with forward slashes
    obs_columns_to_remove = [col for col in adata.obs.columns if "/" in col]
    if obs_columns_to_remove:
        logger.warning(
            f"Removing .obs columns with forward slashes for Zarr compatibility: {obs_columns_to_remove}"
        )
        adata.obs = adata.obs.drop(columns=obs_columns_to_remove)

    # Remove .var columns with forward slashes
    var_columns_to_remove = [col for col in adata.var.columns if "/" in col]
    if var_columns_to_remove:
        logger.warning(
            f"Removing .var columns with forward slashes for Zarr compatibility: {var_columns_to_remove}"
        )
        adata.var = adata.var.drop(columns=var_columns_to_remove)

    # Remove layers with forward slashes
    layers_to_remove = [layer for layer in adata.layers.keys() if "/" in layer]
    if layers_to_remove:
        logger.warning(
            f"Removing layers with forward slashes for Zarr compatibility: {layers_to_remove}"
        )
        for layer_name in layers_to_remove:
            del adata.layers[layer_name]


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
        adata = ad.read_h5ad(temp_out, backed="r+")

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
        if adata.raw is not None:
            raw_is_csr_hdf5_group = (
                adata.raw is not None
                and isinstance(adata.raw.X, CSRDataset)  # ← catches wrapper
                # defensive fallback in case class name changes
                or (
                    hasattr(adata, "file")
                    and isinstance(adata.file["raw"]["X"], h5py.Group)
                )
            )
        else:
            raw_is_csr_hdf5_group = False

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
        # Use urllib directly to avoid any requests_cache issues
        import urllib.request
        import urllib.error

        # Download with progress bar using urllib
        with tqdm(
            desc=save_path,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:

            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    bar.total = total_size
                bar.update(block_size)

            urllib.request.urlretrieve(url, save_path, progress_hook)

        print(f"\nDownload complete: {save_path}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"\nDownload failed: {e}")
        return False
    except Exception as e:
        print(f"\nDownload failed with unexpected error: {e}")
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
    if suffix == {".npz", ".zip"}:
        return magic.startswith(ZIP_MAGIC)
    # Unknown type → we can at least confirm it is downloadable
    return True


def upload_folder_to_nextcloud(
    data_folder: str | Path,
    nextcloud_config: dict[str, Any],
    max_workers: int = 4,
    *,
    force_reupload: bool = False,
) -> dict[str, str]:
    """
    Upload *.zarr directories and *.h5ad files to Nextcloud and create share links.

    This function will:
    • Convert *.zarr directories and *.h5ad files to ZIP format
    • Create missing parent directories on Nextcloud
    • Upload each ZIP file with robust error handling and retry logic
    • Create public share links for each uploaded file
    • Write/extend share_map.json with the mapping of local files to share links

    Parameters
    ----------
    data_folder : str | Path
        Local directory containing *.zarr directories and/or *.h5ad files to upload.
    nextcloud_config : dict[str, Any]
        Configuration dictionary containing environment variable names (not actual values).
        Required keys:
        - "url": Environment variable name containing the Nextcloud base URL
        - "username": Environment variable name containing the Nextcloud username
        - "password": Environment variable name containing the Nextcloud password
        Optional keys:
        - "chunk_size": Upload chunk size for single-PUT streaming in bytes (default: 1MB)
        - "chunked_threshold_bytes": If file size ≥ this, use WebDAV chunked upload (default: 1_000_000_000)
        - "force_chunked": Force WebDAV chunked upload for all files (default: False)
        - "chunk_bytes": Size of each WebDAV chunk (default: 50 MiB)
    max_workers : int, optional
        Maximum number of worker threads for parallel uploads (default: 4).

    Returns
    -------
    dict[str, str]
        Mapping of local file paths (relative to data_folder) to public share URLs.

    Notes
    -----
    - Environment variables are read using os.getenv() at runtime
    - Files are converted to ZIP format before upload for better compression
    - Share links are created as public read-only links
    - Progress is saved incrementally to share_map.json
    - Failed uploads will raise RuntimeError with details

    Examples
    --------
    >>> # Set environment variables first
    >>> os.environ["NC_URL"] = "https://cloud.example.com"
    >>> os.environ["NC_USER"] = "myusername"
    >>> os.environ["NC_PASS"] = "mypassword"
    >>>
    >>> # Then call with env var names in config
    >>> config = {
    ...     "url": "NC_URL",
    ...     "username": "NC_USER",
    ...     "password": "NC_PASS"
    ... }
    >>> share_map = upload_folder_to_nextcloud("/path/to/data", config)
    """
    import json
    import os
    import shutil
    import requests
    from pathlib import Path
    import time

    data_folder = Path(data_folder).resolve()
    mapping_path = data_folder / "share_map.json"
    share_map = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}

    # ---------- 1) turn stores into zip files --------------------------
    def ensure_zip(p: Path) -> Path:
        if p.suffix == ".zip":
            return p
        if p.is_dir() and p.suffix == ".zarr":
            z = p.with_suffix(".zarr.zip")
            shutil.make_archive(z.with_suffix(""), "zip", p)
            return z
        if p.is_file() and p.suffix == ".h5ad":
            z = p.with_suffix(".h5ad.zip")
            shutil.make_archive(
                z.with_suffix(""), "zip", root_dir=p.parent, base_dir=p.name
            )
            return z
        return p  # ignore everything else

    zip_paths = [
        ensure_zip(p) for p in data_folder.iterdir() if p.name != "share_map.json"
    ]

    # ---------- 2) build remote paths & create dirs --------------------
    nc_url = os.getenv(nextcloud_config["url"]).rstrip("/")
    auth = requests.auth.HTTPBasicAuth(
        os.getenv(nextcloud_config["username"]),
        os.getenv(nextcloud_config["password"]),
    )

    remote_dirs = set()
    uploads = []
    for z in zip_paths:
        rel_key = z.relative_to(data_folder).as_posix()

        # keep original hierarchy after first "data/"
        remote_tail = rel_key if "data/" not in str(z) else str(z).split("data/", 1)[1]
        remote_rel = f"data/{remote_tail}"
        parent = Path(remote_tail).parent
        while parent.as_posix() not in ("", "."):
            remote_dirs.add(f"data/{parent.as_posix()}")
            parent = parent.parent

        if (
            force_reupload
            or rel_key not in share_map
            or not verify_share_link(share_map.get(rel_key, ""), ".zip")
        ):
            uploads.append((z, remote_rel))

    # create parent dirs sequentially with better error handling
    base = f"{nc_url}/remote.php/dav/files/{auth.username}"
    for d in sorted(remote_dirs, key=lambda p: p.count("/")):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with requests.Session() as session:
                    # Configure session for better reliability
                    session.mount(
                        "https://",
                        requests.adapters.HTTPAdapter(
                            max_retries=requests.packages.urllib3.util.retry.Retry(
                                total=3,
                                backoff_factor=1,
                                status_forcelist=[500, 502, 503, 504],
                            )
                        ),
                    )
                    r = session.request("MKCOL", f"{base}/{d}", auth=auth, timeout=30)
                    if r.status_code not in (201, 405):
                        logger.warning("MKCOL %s → %s", d, r.status_code)
                    break
            except (requests.exceptions.RequestException, TimeoutError) as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "MKCOL retry %d/%d for %s: %s", attempt + 1, max_retries, d, e
                    )
                    time.sleep(2**attempt)  # exponential backoff
                else:
                    logger.error(
                        "MKCOL failed after %d attempts for %s: %s", max_retries, d, e
                    )

    # ---------- 3) upload each ZIP sequentially with robust error handling ------------------------
    failed_uploads = []

    for local_p, remote_rel in tqdm(uploads, desc="WebDAV upload", unit="file"):
        full_url = f"{base}/{remote_rel}"
        size = os.path.getsize(local_p)

        max_retries = 3
        upload_success = False

        for attempt in range(max_retries):
            try:
                # Create a fresh session for each upload to avoid connection issues
                with requests.Session() as session:
                    # Configure session for large file uploads
                    adapter = requests.adapters.HTTPAdapter(
                        max_retries=requests.packages.urllib3.util.retry.Retry(
                            total=0,  # We handle retries manually
                            backoff_factor=1,
                            status_forcelist=[500, 502, 503, 504],
                        )
                    )
                    session.mount("https://", adapter)
                    session.mount("http://", adapter)

                    # Decide whether to use chunked WebDAV upload
                    use_chunked = nextcloud_config.get(
                        "force_chunked", False
                    ) or size >= nextcloud_config.get(
                        "chunked_threshold_bytes", 1_000_000_000
                    )
                    logger.info(f"use_chunked: {use_chunked}")

                    if use_chunked:
                        try:
                            upload_file_to_nextcloud_chunked(
                                file_path=str(local_p),
                                nextcloud_url=nc_url,
                                username=auth.username,
                                password=auth.password,
                                remote_path=remote_rel,
                                chunk_bytes=nextcloud_config.get(
                                    "chunk_bytes", 50 * 1024 * 1024
                                ),
                                timeout=(30, 1800),
                            )
                            logger.info(
                                "✅ Chunked upload successful: %s", local_p.name
                            )
                            upload_success = True

                            # Save successful upload immediately to share_map
                            rel_key = local_p.relative_to(data_folder).as_posix()
                            remote_tail = (
                                rel_key
                                if "data/" not in str(local_p)
                                else str(local_p).split("data/", 1)[1]
                            )
                            share_link = get_share_link(
                                nc_url,
                                auth.username,
                                auth.password,
                                f"data/{remote_tail}",
                            )
                            if share_link:
                                share_map[rel_key] = share_link
                                mapping_path.write_text(json.dumps(share_map, indent=2))
                                logger.info(
                                    "💾 Saved progress: %s → %s", rel_key, share_link
                                )

                            break
                        except Exception as e:
                            logger.warning(
                                "Chunked upload failed for %s on attempt %d: %s",
                                local_p.name,
                                attempt + 1,
                                e,
                            )
                    else:
                        # Use single-PUT streaming upload with progress bar
                        with open(local_p, "rb") as fh:
                            reader = ReadWithProgress(
                                fh,
                                size,
                                chunk_size=nextcloud_config.get(
                                    "chunk_size", 1024 * 1024
                                ),
                                desc=f"Uploading {local_p.name} (attempt {attempt + 1})",
                            )

                            timeout = (30, 1800)  # (connect, read)

                            r = session.put(
                                full_url,
                                data=reader,
                                headers={"Content-Length": str(size)},
                                auth=auth,
                                timeout=timeout,
                            )

                            if r.ok:
                                logger.info("✅ Upload successful: %s", local_p.name)
                                upload_success = True

                                # Save successful upload immediately to share_map
                                rel_key = local_p.relative_to(data_folder).as_posix()
                                remote_tail = (
                                    rel_key
                                    if "data/" not in str(local_p)
                                    else str(local_p).split("data/", 1)[1]
                                )
                                share_link = get_share_link(
                                    nc_url,
                                    auth.username,
                                    auth.password,
                                    f"data/{remote_tail}",
                                )
                                if share_link:
                                    share_map[rel_key] = share_link
                                    # Save progress immediately
                                    mapping_path.write_text(
                                        json.dumps(share_map, indent=2)
                                    )
                                    logger.info(
                                        "💾 Saved progress: %s → %s",
                                        rel_key,
                                        share_link,
                                    )

                                break
                            else:
                                logger.warning(
                                    "Upload failed (HTTP %s) for %s on attempt %d",
                                    r.status_code,
                                    local_p.name,
                                    attempt + 1,
                                )

            except (requests.exceptions.RequestException, TimeoutError) as e:
                logger.warning(
                    "Upload attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    max_retries,
                    local_p.name,
                    e,
                )
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    sleep_time = (2**attempt) + random.uniform(0, 1)
                    logger.info("Retrying in %.1f seconds...", sleep_time)
                    time.sleep(sleep_time)

        if not upload_success:
            logger.error(
                "❌ Upload failed after %d attempts: %s", max_retries, local_p.name
            )
            failed_uploads.append(local_p.name)

    # ---------- 4) create / refresh share links for any remaining files ------------------------
    for z in zip_paths:
        rel_key = z.relative_to(data_folder).as_posix()
        if rel_key not in share_map or not verify_share_link(
            share_map[rel_key], ".zip"
        ):
            # Only try to create share links for files that weren't just uploaded
            # (since we already created them in step 3)
            if rel_key not in [
                f.relative_to(data_folder).as_posix() for f, _ in uploads
            ]:
                remote_tail = (
                    rel_key if "data/" not in str(z) else str(z).split("data/", 1)[1]
                )
                share_link = get_share_link(
                    nc_url, auth.username, auth.password, f"data/{remote_tail}"
                )
                if share_link:
                    share_map[rel_key] = share_link

    # Final save of share_map
    mapping_path.write_text(json.dumps(share_map, indent=2))

    # Report results
    if failed_uploads:
        logger.error(
            "❌ %d files failed to upload: %s", len(failed_uploads), failed_uploads
        )
        logger.info(
            "✅ %d files uploaded successfully and saved to share_map.json",
            len([k for k in share_map.keys() if k.endswith(".zip")]),
        )
        logger.info("🔄 Re-run the script to retry failed uploads")
        raise RuntimeError(
            f"Upload failed for {len(failed_uploads)} files: {failed_uploads}"
        )

    logger.info("✅ All uploads completed successfully!")
    return share_map


def upload_folder_to_zenodo(
    data_folder: str | Path,
    zenodo_token: str,
    zenodo_config: dict[str, Any],
    max_workers: int = 4,
    *,
    force_reupload: bool = False,
    dataset_name: str | None = None,
    dataset_description: str | None = None,
    raw_data_link: str | None = None,
    obsm_keys: list[str] | None = None,
    hf_repo_id: str | None = None,
    shared_mapping_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Upload *.zarr directories and *.h5ad files to Zenodo and return download links.

    This function will:
    • Convert *.zarr directories and *.h5ad files to ZIP format
    • Create a new Zenodo deposit (or reuse existing from share_map.json)
    • Upload each ZIP file to the deposit
    • Return a mapping of local files to Zenodo download URLs

    Parameters
    ----------
    data_folder : str | Path
        Local directory containing *.zarr directories and/or *.h5ad files to upload.
    zenodo_token : str
        Zenodo access token (from ZENODO_TOKEN environment variable).
    zenodo_config : dict[str, Any]
        Configuration dictionary. Required keys:
        - "sandbox": bool, whether to use Zenodo sandbox (default: False)
    max_workers : int, optional
        Not used (kept for API compatibility). Uploads are performed sequentially.
    force_reupload : bool, optional
        If True, re-upload files even if they exist in share_map.json (default: False).
    dataset_name : str, optional
        Name of the dataset (used as deposit title).
    dataset_description : str, optional
        Description of the dataset.
    raw_data_link : str, optional
        URL to the raw/original data source.
    obsm_keys : list[str], optional
        List of .obsm layer keys present in the AnnData objects.
    hf_repo_id : str, optional
        HuggingFace repository ID (e.g., "jo-mengr/dataset-name") for linking.
    shared_mapping_path : str | Path, optional
        Path to a shared share_map.json file. If provided, this will be used instead
        of creating one in data_folder. Useful for sharing a single deposit across
        multiple folders (e.g., train/val splits).

    Returns
    -------
    dict[str, str]
        Mapping of local file paths (relative to data_folder) to Zenodo download URLs.

    Notes
    -----
    - Files are converted to ZIP format before upload for better compression
    - Deposit metadata is minimal; you can edit it later on Zenodo
    - Progress is saved incrementally to share_map.json
    - Failed uploads will raise RuntimeError with details

    Examples
    --------
    >>> import os
    >>> token = os.getenv("ZENODO_TOKEN")
    >>> config = {"sandbox": False}
    >>> share_map = upload_folder_to_zenodo("/path/to/data", token, config)
    """
    data_folder = Path(data_folder).resolve()

    # Use shared mapping path if provided, otherwise use local one
    if shared_mapping_path:
        mapping_path = Path(shared_mapping_path).resolve()
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        mapping_path = data_folder / "share_map.json"

    share_map = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}

    # Determine Zenodo API base URL
    sandbox = zenodo_config.get("sandbox", False)
    api_base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"

    # ---------- 1) turn stores into zip files --------------------------
    def ensure_zip(p: Path) -> Path:
        if p.suffix == ".zip":
            return p
        if p.is_dir() and p.suffix == ".zarr":
            z = p.with_suffix(".zarr.zip")
            if not z.exists() or p.stat().st_mtime > z.stat().st_mtime:
                logger.info(f"Creating zip for {p.name}...")
                shutil.make_archive(z.with_suffix(""), "zip", p)
            return z
        if p.is_file() and p.suffix == ".h5ad":
            z = p.with_suffix(".h5ad.zip")
            if not z.exists() or p.stat().st_mtime > z.stat().st_mtime:
                logger.info(f"Creating zip for {p.name}...")
                shutil.make_archive(
                    z.with_suffix(""), "zip", root_dir=p.parent, base_dir=p.name
                )
            return z
        return p  # ignore everything else

    zip_paths = [
        ensure_zip(p) for p in data_folder.iterdir() if p.name != "share_map.json"
    ]

    # ---------- 2) Get or create deposit --------------------
    deposit_id = share_map.get("_zenodo_deposit_id")
    deposit_links = share_map.get("_zenodo_deposit_links", {})
    is_new_deposit = False

    if deposit_id and not force_reupload:
        logger.info(f"Using existing Zenodo deposit: {deposit_id}")
        # Verify deposit still exists
        try:
            r = requests.get(
                f"{api_base}/deposit/depositions/{deposit_id}",
                params={"access_token": zenodo_token},
                timeout=30,
            )
            if r.status_code == 200:
                deposit_links = r.json().get("links", {})
            else:
                logger.warning(
                    f"Deposit {deposit_id} not found or inaccessible, creating new one"
                )
                deposit_id = None
                is_new_deposit = True
        except Exception as e:
            logger.warning(
                f"Error checking deposit {deposit_id}: {e}, creating new one"
            )
            deposit_id = None
            is_new_deposit = True
    else:
        is_new_deposit = True

    if is_new_deposit:
        logger.info("Creating new Zenodo deposit...")

        # Build comprehensive description
        title = dataset_name if dataset_name else f"Dataset: {data_folder.name}"

        description_parts = []

        if dataset_description:
            description_parts.append(dataset_description)

        description_parts.append(
            "\n\n## Processed AnnData Objects\n\n"
            "This deposit contains processed AnnData objects (single-cell genomics data) "
            "that were processed using the `adata_hf_datasets` Python package. "
            "The files are in HDF5 (.h5ad) or Zarr format and contain preprocessed "
            "gene expression data along with computed embeddings."
        )

        if obsm_keys:
            obsm_list = ", ".join([f"`{key}`" for key in obsm_keys])
            description_parts.append(
                f"\n\n### Available Embeddings (.obsm layers)\n\n"
                f"The AnnData objects contain the following embedding layers: {obsm_list}."
            )

        if hf_repo_id:
            hf_url = f"https://huggingface.co/datasets/{hf_repo_id}"
            description_parts.append(
                f"\n\n## Related Resources\n\n"
                f"- **HuggingFace Dataset**: [{hf_repo_id}]({hf_url})"
            )

        if raw_data_link:
            description_parts.append(
                f"- **Raw Data Source**: [Download]({raw_data_link})"
            )

        description_parts.append(
            "\n\n## Package Information\n\n"
            "These files were processed using the `adata_hf_datasets` package. "
            "For more information about the processing pipeline, please refer to "
            "the associated HuggingFace dataset repository."
        )

        full_description = "".join(description_parts)

        # Create a new deposit with enhanced metadata
        deposit_data = {
            "metadata": {
                "title": title,
                "upload_type": "dataset",
                "description": full_description,
                "creators": [{"name": "jo-mengr", "affiliation": "HuggingFace"}],
                "keywords": [
                    "single-cell",
                    "genomics",
                    "anndata",
                    "processed-data",
                    "embeddings",
                ],
            }
        }

        # Add related identifiers if we have HF repo
        if hf_repo_id:
            deposit_data["metadata"]["related_identifiers"] = [
                {
                    "identifier": f"https://huggingface.co/datasets/{hf_repo_id}",
                    "relation": "isSupplementTo",
                    "scheme": "url",
                }
            ]
        r = requests.post(
            f"{api_base}/deposit/depositions",
            params={"access_token": zenodo_token},
            json=deposit_data,
            timeout=30,
        )
        if r.status_code not in (200, 201):
            error_msg = f"Failed to create Zenodo deposit: {r.status_code} {r.text}"
            if r.status_code == 401:
                error_msg += (
                    "\n\nAuthentication failed. Please check your ZENODO_TOKEN in .env file. "
                    "You can create a token at https://zenodo.org/account/settings/applications/"
                )
            raise RuntimeError(error_msg)
        deposit = r.json()
        deposit_id = deposit["id"]
        deposit_links = deposit.get("links", {})
        share_map["_zenodo_deposit_id"] = deposit_id
        share_map["_zenodo_deposit_links"] = deposit_links
        logger.info(f"Created Zenodo deposit: {deposit_id}")

    # Get upload URL from deposit links
    upload_url = deposit_links.get("bucket")
    if not upload_url:
        raise RuntimeError(
            f"Deposit {deposit_id} does not have an upload bucket URL. "
            "The deposit may have been published already."
        )

    # ---------- 3) Upload files --------------------
    uploads = []
    for z in zip_paths:
        # When using shared mapping, prefix filename with split name to avoid conflicts
        # e.g., "train_chunk_0.zarr.zip" instead of just "chunk_0.zarr.zip"
        if shared_mapping_path:
            # Create prefixed filename for Zenodo upload
            zenodo_filename = f"{data_folder.name}_{z.name}"
            # Use prefixed filename as key in mapping
            rel_key = zenodo_filename
        else:
            zenodo_filename = z.name  # Use original filename
            rel_key = z.relative_to(data_folder).as_posix()

        if (
            force_reupload
            or rel_key not in share_map
            or not share_map[rel_key].startswith("http")
        ):
            uploads.append((z, rel_key, zenodo_filename))

    if not uploads:
        logger.info("All files already uploaded to Zenodo")
        # Return filtered results if using shared mapping
        if shared_mapping_path:
            folder_name = data_folder.name
            filtered_map = {
                k: v
                for k, v in share_map.items()
                if not k.startswith("_") and k.startswith(f"{folder_name}_")
            }
            result = {}
            for k, v in filtered_map.items():
                if k.startswith(f"{folder_name}_"):
                    result[k[len(folder_name) + 1 :]] = v
                else:
                    result[k] = v
            return result
        else:
            return {k: v for k, v in share_map.items() if not k.startswith("_")}

    failed_uploads = []

    # Upload files sequentially to avoid API rate limits and connection issues
    for zip_path, rel_key, zenodo_filename in uploads:
        file_size = zip_path.stat().st_size

        try:
            logger.info(
                f"Uploading {zip_path.name} as {zenodo_filename} ({file_size / 1024 / 1024:.1f} MB)..."
            )

            # Upload file to Zenodo bucket with the prefixed filename
            with open(zip_path, "rb") as f:
                upload_response = requests.put(
                    f"{upload_url}/{zenodo_filename}",
                    data=f,
                    params={"access_token": zenodo_token},
                    timeout=(30, 1800),  # 30s connect, 30min read
                )

            if upload_response.status_code not in (200, 201):
                error_msg = f"Upload failed for {zenodo_filename}: {upload_response.status_code} {upload_response.text}"
                if upload_response.status_code == 401:
                    error_msg += "\nAuthentication failed. Please check your ZENODO_TOKEN in .env file."
                logger.error(error_msg)
                failed_uploads.append(rel_key)
                continue

            logger.info(
                f"File {zenodo_filename} uploaded successfully (status {upload_response.status_code})"
            )

            # Construct download URL directly from deposit_id and filename
            # Format: https://zenodo.org/api/records/{deposit_id}/draft/files/{filename}/content
            # For sandbox: https://sandbox.zenodo.org/api/records/{deposit_id}/draft/files/{filename}/content
            download_url = (
                f"{api_base}/records/{deposit_id}/draft/files/{zenodo_filename}/content"
            )

            # Success - save the download URL
            share_map[rel_key] = download_url
            # Save progress incrementally
            mapping_path.write_text(json.dumps(share_map, indent=2))
            logger.info(f"✅ Uploaded {zenodo_filename} - Download URL: {download_url}")

        except Exception as e:
            logger.error(f"Error uploading {zenodo_filename}: {e}", exc_info=True)
            failed_uploads.append(rel_key)

    # Final save of share_map
    mapping_path.write_text(json.dumps(share_map, indent=2))

    # Report results
    if failed_uploads:
        logger.error(
            "❌ %d files failed to upload: %s", len(failed_uploads), failed_uploads
        )
        logger.info(
            "✅ %d files uploaded successfully and saved to share_map.json",
            len(
                [
                    k
                    for k in share_map.keys()
                    if k.endswith(".zip") and not k.startswith("_")
                ]
            ),
        )
        logger.info("🔄 Re-run the script to retry failed uploads")
        raise RuntimeError(
            f"Upload failed for {len(failed_uploads)} files: {failed_uploads}"
        )

    logger.info("✅ All uploads completed successfully!")
    # Return only file mappings, not metadata
    # If using shared mapping, filter to only return entries for current data_folder
    if shared_mapping_path:
        # Filter to only return entries for the current split/folder
        # Keys are now in format "split_name_filename.zip" (e.g., "train_chunk_0.zarr.zip")
        folder_name = data_folder.name
        filtered_map = {
            k: v
            for k, v in share_map.items()
            if not k.startswith("_") and k.startswith(f"{folder_name}_")
        }
        # Remove split prefix from keys for easier lookup in build_split_dataset
        # e.g., "train_chunk_0.zarr.zip" -> "chunk_0.zarr.zip"
        result = {}
        for k, v in filtered_map.items():
            if k.startswith(f"{folder_name}_"):
                # Remove "split_name_" prefix from key for return value
                result[k[len(folder_name) + 1 :]] = v
            else:
                result[k] = v
        return result
    else:
        return {k: v for k, v in share_map.items() if not k.startswith("_")}


# ------------------------------------------------------------------ #
# helper: zip target if necessary and return Path to zip
# ------------------------------------------------------------------ #
def _ensure_zip(local_path: Path) -> Path:
    """
    If *local_path* is
        - a directory called *.zarr*   → create <name>.zarr.zip
        - a file     called *.h5ad*    → create <name>.h5ad.zip
        - already *.zip*               → return as-is
    The zip is created next to the input and overwritten if outdated.
    """
    if local_path.suffix == ".zip":
        return local_path

    if local_path.is_dir() and local_path.suffix == ".zarr":
        zip_path = local_path.with_suffix(".zarr.zip")
    elif local_path.is_file() and local_path.suffix == ".h5ad":
        zip_path = local_path.with_suffix(".h5ad.zip")
    else:
        # leave all other files unchanged (e.g. share_map.json)
        return local_path

    if not zip_path.exists() or zip_path.stat().st_mtime < local_path.stat().st_mtime:
        shutil.make_archive(
            base_name=zip_path.with_suffix(""), format="zip", root_dir=local_path
        )
    return zip_path


def _mk_remote_dirs(
    nc_url: str,
    auth: HTTPBasicAuth,
    remote_dirs: set[str],
    timeout: int = 15,
) -> None:
    """
    Create each directory in *remote_dirs* **sequentially**.

    Parameters
    ----------
    nc_url
        Base Nextcloud URL *without* trailing slash.
    auth
        `HTTPBasicAuth(username, password)`.
    remote_dirs
        Iterable of paths like ``"data/RNA/processed_with_emb/train"``.
    timeout
        Seconds to wait for each MKCOL.

    Notes
    -----
    • Ignores 405 (already exists).
    • Logs any other response code but keeps going.
    """
    base = f"{nc_url.rstrip('/')}/remote.php/dav/files/{auth.username}"
    for rel in sorted(
        remote_dirs, key=lambda p: PurePosixPath(p).as_posix().count("/")
    ):
        url = f"{base}/{rel.lstrip('/')}"
        try:
            r = requests.request("MKCOL", url, auth=auth, timeout=timeout)
            if r.status_code not in (201, 405):
                logger.warning("MKCOL %s → %s", url, r.status_code)
        except requests.RequestException as exc:
            logger.warning("MKCOL %s failed: %s", url, exc)


def _upload_one_file(
    local_remote: tuple[Path, str],
    nc_url: str,
    session: requests.Session,
    auth: HTTPBasicAuth,
    chunk: int = 1024 * 1024,
    timeout: tuple[int, int] = (30, 1800),  # Increased timeouts
    max_retries: int = 3,
):
    """PUT one file with streaming upload and retry logic."""
    local_path, remote_rel = local_remote
    full_url = f"{nc_url.rstrip('/')}/remote.php/dav/files/{auth.username}/{remote_rel}"
    size = os.path.getsize(local_path)

    for attempt in range(max_retries):
        try:
            with open(local_path, "rb") as fh:
                # Create progress reader for streaming upload
                reader = ReadWithProgress(
                    fh,
                    size,
                    chunk_size=chunk,
                    desc=f"Uploading {local_path.name} (attempt {attempt + 1})",
                )
                headers = {"Content-Length": str(size)}
                res = session.put(
                    full_url, data=reader, headers=headers, timeout=timeout, auth=auth
                )
                if res.ok:
                    logger.info("✅ Upload successful: %s", local_path.name)
                    return remote_rel
                else:
                    logger.warning(
                        "Upload failed (HTTP %s) for %s on attempt %d",
                        res.status_code,
                        local_path.name,
                        attempt + 1,
                    )
                    if attempt == max_retries - 1:
                        res.raise_for_status()

        except (requests.exceptions.RequestException, TimeoutError) as e:
            logger.warning(
                "Upload attempt %d/%d failed for %s: %s",
                attempt + 1,
                max_retries,
                local_path.name,
                e,
            )
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                sleep_time = (2**attempt) + random.uniform(0, 1)
                logger.info("Retrying in %.1f seconds...", sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error(
                    "Upload failed after %d attempts: %s", max_retries, local_path
                )
                raise

    return remote_rel


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
                os.getenv(nextcloud_config["url"]),
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )

            response = upload_file_to_nextcloud(
                local_path,
                os.getenv(nextcloud_config["url"]),
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
                os.getenv(nextcloud_config["url"]),
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
            os.getenv(nextcloud_config["url"]),
            os.getenv(nextcloud_config["username"]),
            os.getenv(nextcloud_config["password"]),
            remote_path,
        )
        response = upload_file_to_nextcloud(
            local_path,
            os.getenv(nextcloud_config["url"]),
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
            os.getenv(nextcloud_config["url"]),
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
    timeout: tuple = (30, 1800),  # Increased timeouts: 30s connect, 30min read
    max_retries: int = 3,
) -> requests.Response:
    """
    Upload a file to Nextcloud via WebDAV, showing a tqdm progress bar (with speed),
    while ensuring a proper Content-Length header (no chunked transfer).
    Includes retry logic for robust uploads.

    Parameters
    ----------
    file_path : str
    nextcloud_url : str
    username : str
    password : str
    remote_path : str
    chunk_size : int, optional
    timeout : tuple, optional
        (connect_timeout, read_timeout) in seconds
    max_retries : int, optional
        Number of retry attempts for failed uploads

    Returns
    -------
    requests.Response
    """
    full_url = f"{nextcloud_url.rstrip('/')}/remote.php/dav/files/{username}/{remote_path.lstrip('/')}"
    total_size = os.path.getsize(file_path)
    filename = os.path.basename(file_path)

    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                reader = ReadWithProgress(
                    f,
                    total_size,
                    chunk_size,
                    desc=f"Uploading {filename} (attempt {attempt + 1})",
                )
                # Manually set Content-Length so requests does NOT chunk
                headers = {"Content-Length": str(total_size)}

                # Create session with retry configuration
                with requests.Session() as session:
                    adapter = requests.adapters.HTTPAdapter(
                        max_retries=requests.packages.urllib3.util.retry.Retry(
                            total=0,  # We handle retries manually
                            backoff_factor=1,
                            status_forcelist=[500, 502, 503, 504],
                        )
                    )
                    session.mount("https://", adapter)
                    session.mount("http://", adapter)

                    response = session.put(
                        full_url,
                        data=reader,
                        auth=HTTPBasicAuth(username, password),
                        headers=headers,
                        timeout=timeout,
                    )

            if response.ok:
                tqdm.write(f"✔️  Uploaded {filename}")
                return response
            else:
                tqdm.write(
                    f"❌  Upload failed ({response.status_code}) for {filename} on attempt {attempt + 1}"
                )
                if attempt == max_retries - 1:
                    response.raise_for_status()

        except (requests.exceptions.RequestException, TimeoutError) as e:
            tqdm.write(
                f"⚠️  Upload attempt {attempt + 1}/{max_retries} failed for {filename}: {e}"
            )
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                sleep_time = (2**attempt) + random.uniform(0, 1)
                tqdm.write(f"🔄  Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                tqdm.write(
                    f"❌  Upload failed after {max_retries} attempts for {filename}"
                )
                raise

    return response


def upload_file_to_nextcloud_chunked(
    file_path: str | Path,
    nextcloud_url: str,
    username: str,
    password: str,
    remote_path: str,
    chunk_bytes: int = 50 * 1024 * 1024,
    timeout: tuple[int, int] = (30, 1800),
    move_max_retries: int = 5,
) -> None:
    """
    Upload a file using Nextcloud's WebDAV chunked upload API.

    Flow:
      1) MKCOL /remote.php/dav/uploads/{user}/{upload_id}
      2) PUT chunks to .../{upload_id}/{offset:016d}
      3) MOVE assembled upload to /remote.php/dav/files/{user}/{remote_path}
    """
    import requests

    base = nextcloud_url.rstrip("/")
    auth = requests.auth.HTTPBasicAuth(username, password)

    uploads_base = f"{base}/remote.php/dav/uploads/{username}"
    files_base = f"{base}/remote.php/dav/files/{username}"

    upload_id = uuid.uuid4().hex
    session_dir = f"{uploads_base}/{upload_id}"

    # 1) Create upload session directory
    r = requests.request("MKCOL", session_dir, auth=auth, timeout=30)
    if r.status_code not in (201, 405):
        raise RuntimeError(f"MKCOL failed ({r.status_code}): {r.text}")

    size = os.path.getsize(file_path)
    offset = 0
    logging.info("Chunked upload")
    logger.info(f"file_path: {file_path}")
    logger.info(f"size: {size}")
    with (
        open(file_path, "rb") as fh,
        tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Chunked upload {Path(file_path).name}",
        ) as bar,
    ):
        while True:
            buf = fh.read(chunk_bytes)
            if not buf:
                break
            chunk_name = f"{offset:016d}"
            put_url = f"{session_dir}/{chunk_name}"
            res = requests.put(
                put_url,
                data=buf,
                headers={"Content-Length": str(len(buf))},
                auth=auth,
                timeout=timeout,
            )
            if not res.ok:
                raise RuntimeError(
                    f"PUT chunk failed ({res.status_code}) at offset {offset}"
                )
            offset += len(buf)
            bar.update(len(buf))

    # 3) Finalize: MOVE to files area
    # Build a proper URL; do not use pathlib on URLs (it collapses // after scheme)
    dest_path = f"{files_base}/{str(remote_path).lstrip('/')}"

    # Try moving the virtual '.file' first, then the session dir as fallback
    move_sources = (f"{session_dir}/.file", session_dir)
    last_status = None
    last_text = None
    for src in move_sources:
        for attempt in range(1, move_max_retries + 1):
            try:
                res = requests.request(
                    "MOVE",
                    src,
                    headers={"Destination": dest_path, "Overwrite": "T"},
                    auth=auth,
                    timeout=timeout,
                )
                last_status = res.status_code
                last_text = res.text
                if res.status_code in (201, 204):
                    return
            except requests.exceptions.ReadTimeout:
                # Server may still be finalizing; probe destination
                pass
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Finalize MOVE error on attempt %d/%d for %s: %s",
                    attempt,
                    move_max_retries,
                    Path(file_path).name,
                    e,
                )

            # Probe destination: if present, treat as success
            try:
                head = requests.head(dest_path, auth=auth, timeout=30)
                if head.status_code in (200, 204):
                    return
            except requests.RequestException:
                pass

            time.sleep(min(30, 2 ** (attempt - 1)))

    raise RuntimeError(f"Finalize MOVE failed: {last_status} {last_text or ''}".strip())


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


def _extract_token_from_xml(xml_text: str) -> str:
    """Return the <token> value or raise ValueError."""
    root = ET.fromstring(xml_text)
    token_el = root.find(".//token")
    if token_el is None or not token_el.text:
        raise ValueError("No <token> element in XML")
    return token_el.text


def _extract_token_from_json(json_obj: dict) -> str:
    """Return the token from a JSON OCS answer or raise ValueError."""
    try:
        return json_obj["ocs"]["data"]["token"]
    except (KeyError, TypeError):
        raise ValueError("No token field in JSON response")


def get_share_link(
    nextcloud_url: str,
    username: str,
    password: str,
    remote_path: str,
    permissions: int = 1,
) -> Optional[str]:
    """
    Get (or create) a **public share link** for *remote_path* on Nextcloud.

    Works for both files **and** folders.  If the item is already shared,
    the existing link is reused instead of creating duplicates.

    Parameters
    ----------
    nextcloud_url
        Base URL, e.g. ``https://cloud.example.org``.
    username / password
        Basic-auth credentials.
    remote_path
        Path *inside your Files area* (no leading slash), e.g.
        ``"data/train/sample_1.zarr"`` **or** ``"data/train/"``.
    permissions
        1 = read-only, 15 = all (see Nextcloud OCS docs).

    Returns
    -------
    str or None
        Public URL (no ``/download`` suffix) or *None* on failure.
    """
    base = nextcloud_url.rstrip("/")
    api = f"{base}/ocs/v2.php/apps/files_sharing/api/v1/shares"
    auth = HTTPBasicAuth(username, password)
    headers = {"OCS-APIRequest": "true", "Accept": "application/json"}

    # Ensure POSIX style path & strip redundant leading slash
    remote_path = str(PurePosixPath(remote_path).as_posix()).lstrip("/")

    # ------------------------------------------------------------------ #
    # 1) Try to CREATE a share                                           #
    # ------------------------------------------------------------------ #
    payload = {"shareType": 3, "path": remote_path, "permissions": permissions}
    resp = requests.post(api, data=payload, headers=headers, auth=auth)
    if resp.status_code in (200, 201):
        token = _extract_token_from_json(resp.json())
        return f"{base}/s/{token}"
    elif resp.status_code == 400 and "already shared" in resp.text.lower():
        logger.info("Path already shared – retrieving existing link.")
    else:
        logger.warning(
            "Share creation failed (%d): %s", resp.status_code, resp.text.strip()
        )

    # ------------------------------------------------------------------ #
    # 2) Fallback: LIST shares for that path and reuse first token       #
    # ------------------------------------------------------------------ #
    list_params = {"path": remote_path, "reshares": "true", "subfiles": "true"}
    resp = requests.get(api, params=list_params, headers=headers, auth=auth)
    if resp.status_code != 200:
        logger.error("Could not list shares (%d): %s", resp.status_code, resp.text)
        return None

    try:
        token = _extract_token_from_json(resp.json()["ocs"]["data"][0])
        return f"{base}/s/{token}"
    except (IndexError, ValueError, KeyError):
        logger.error("No existing share link found for %s", remote_path)
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


def safe_read_h5ad(
    path: str | Path,
    *,
    max_retry: int = 3,
    copy_local: bool = True,
    verify: bool = True,
    sleep: int = 5,
) -> ad.AnnData:
    """
    Robustly read an .h5ad file, retrying on transient I/O errors.

    Parameters
    ----------
    path
        Path to the .h5ad on the shared filesystem.
    max_retry
        Max attempts before giving up (default 3).
    copy_local
        On retry, copy the file to $TMPDIR and read from there.
    verify
        After reading, open with h5py and touch ``/X/data`` to make sure
        the file isn't truncated.
    sleep
        Seconds to wait between retries.
    """
    path = Path(path)
    attempt = 0
    local_copy: Path | None = None

    while attempt < max_retry:
        attempt += 1
        try:
            src = local_copy or path
            adata = ad.read_h5ad(src)

            if verify:
                with h5py.File(src, "r") as f:
                    # cheap probe, raises if dataset missing/corrupt
                    _ = f["X/data"][0:1]

            return adata

        except (OSError, IOError) as e:
            # only retry on low-level I/O errors
            if getattr(e, "errno", None) not in (errno.EIO, errno.ESTALE):
                raise

            if attempt >= max_retry:
                logger.error("safe_read: giving up after %d attempts – %s", attempt, e)
                raise

            logger.warning(
                "safe_read: %s on %s – retry %d/%d", e, path, attempt, max_retry
            )
            time.sleep(sleep)

            # first retry ⇒ make a local tmp copy
            if copy_local and local_copy is None:
                tmpdir = Path(os.getenv("TMPDIR", tempfile.gettempdir()))
                local_copy = tmpdir / f"{path.stem}_{uuid.uuid4().hex}.h5ad"
                try:
                    shutil.copy2(path, local_copy)
                    logger.info("safe_read: copied to %s for local access", local_copy)
                except Exception as cp_err:
                    logger.warning(
                        "safe_read: local copy failed (%s); continue without copy",
                        cp_err,
                    )
                    local_copy = None

    # Should never reach here
    raise RuntimeError("safe_read_h5ad: unexpected exit")


def safe_write_h5ad(adata, target, max_retry=3, **kw):
    tmp = Path(tempfile.mktemp(dir=target.parent, suffix=".h5ad.tmp"))
    for attempt in range(1, max_retry + 1):
        try:
            adata.write_h5ad(tmp, **kw)
            os.replace(tmp, target)  # atomic on POSIX
            return
        except OSError as e:
            if e.errno == errno.EIO and attempt < max_retry:
                logger.warning(
                    "EIO on write (%s). Retry %d/%d …", target, attempt, max_retry
                )
                time.sleep(5)
                continue
            raise
    # should not reach here


def _atomic_overwrite(src_dir: Path, dst_dir: Path) -> None:
    """
    Atomically replace *dst_dir* with *src_dir*.

    If *dst_dir* already exists and is non-empty, move it to a temporary
    backup directory first, then rename *src_dir* into place.  Finally
    remove the backup.
    """
    backup: Optional[Path] = None
    try:
        if dst_dir.exists():
            backup = Path(tempfile.mkdtemp(dir=dst_dir.parent, suffix=".zarr.bak"))
            # dst_dir -> backup  (atomic)
            os.replace(dst_dir, backup)

        # tmp -> dst_dir  (atomic)
        os.replace(src_dir, dst_dir)

        # success ➜ purge backup
        if backup:
            shutil.rmtree(backup, ignore_errors=True)

    except Exception:
        # on failure try to restore original store
        if backup and not dst_dir.exists():
            os.replace(backup, dst_dir)
        elif backup:
            shutil.rmtree(backup, ignore_errors=True)
        raise


def safe_write_zarr(
    adata: ad.AnnData,
    target: Path,
    *,
    max_retry: int = 3,
) -> None:
    """
    Atomically write *adata* to *target* (a directory-Zarr store),
    **overwriting** any existing store at the same path.
    """
    # Sanitize column names to remove forward slashes (not allowed in Zarr keys)
    # Create a copy to avoid modifying the original
    adata = adata.copy()
    sanitize_zarr_keys(adata)

    for attempt in range(1, max_retry + 1):
        tmp_dir = Path(tempfile.mkdtemp(dir=target.parent, suffix=".zarr.tmp"))
        try:
            adata.write_zarr(str(tmp_dir))
            zarr.consolidate_metadata(tmp_dir)

            _atomic_overwrite(tmp_dir, target)  # ← handles overwrite

            return  # success
        except OSError as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if e.errno == errno.EIO and attempt < max_retry:
                time.sleep(5)
                continue
            raise
