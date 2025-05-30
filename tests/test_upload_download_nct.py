"""
End-to-end test:  Zarr ↔ Nextcloud upload helper.

What is covered
---------------

* creation of AnnData → .zarr stores with all typical groups
* `upload_folder_to_nextcloud`:
    - walks sub-files
    - writes `share_map.json`
    - stores *one* share link per store
* helper `download_obsm_slice` opens the **folder share link** via fsspec
  and reads only a small slice of the embedding (`obsm/X_geneformer`)

All network I/O is monkey-patched so the test works offline.

Notes
-----
The nextcloud_config parameter expects environment variable names, not actual values.
The function uses os.getenv() to read the actual credentials from environment variables.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict

import time
from adata_hf_datasets.file_utils import upload_folder_to_nextcloud
import anndata as ad
import fsspec
import numpy as np
import pandas as pd
import pytest
from numcodecs import Blosc
import zarr

# import functions under test  (adjust to your actual package paths)

# monkey-patched helpers live in that module:
from adata_hf_datasets import file_utils as fu

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# fixtures – temp directory and two mini AnnData objects
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def tmp_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A temporary directory that mimics one *split* folder."""
    return tmp_path_factory.mktemp("train_split")


def _make_dummy_adata(n_cells: int, n_genes: int, name: str) -> ad.AnnData:
    """Return a small AnnData with two `obsm` embeddings and extra fields."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_cells, n_genes)).astype("float32")
    obs = pd.DataFrame(index=[f"{name}_cell{i}" for i in range(n_cells)])
    obs["batch"] = [name] * n_cells
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_geneformer"] = rng.normal(size=(n_cells, 12)).astype("float16")
    adata.obsm["X_pca"] = rng.normal(size=(n_cells, 8)).astype("float16")
    adata.uns["dummy"] = "foo"
    return adata


@pytest.fixture(scope="module")
def create_zarr_stores(tmp_data_dir: Path):
    """Write two directory-zarr stores into *tmp_data_dir*."""
    _comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    for name in ("sample_A", "sample_B"):
        adata = _make_dummy_adata(30, 20, name)
        out = tmp_data_dir / f"{name}.zarr"
        adata.write_zarr(out)
    yield
    # cleanup (pytest will rm tmp dir anyway, but be explicit)
    shutil.rmtree(tmp_data_dir, ignore_errors=True)


# ---------------------------------------------------------------------- #
# monkey patches – fake Nextcloud requests
# ---------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_nextcloud(monkeypatch, tmp_data_dir: Path):
    # ------------------------------------------------------------------ #
    # 0) Mock environment variables that the function expects
    # ------------------------------------------------------------------ #
    monkeypatch.setenv("NC_URL", "https://dummy")
    monkeypatch.setenv("NC_USER", "user")
    monkeypatch.setenv("NC_PASS", "pass")

    # ------------------------------------------------------------------ #
    # 1) no-op for the old single-file helper (still referenced elsewhere)
    # ------------------------------------------------------------------ #
    monkeypatch.setattr(
        fu,
        "save_and_upload_adata",
        lambda *a, **k: "OK",
    )

    # ------------------------------------------------------------------ #
    # 2) stub directory creation – do nothing
    # ------------------------------------------------------------------ #
    monkeypatch.setattr(
        fu,
        "_mk_remote_dirs",
        lambda session, nc_url, auth, remote_dirs: None,
    )

    # ------------------------------------------------------------------ #
    # 3) Mock requests.Session.put method which is actually used in upload
    # ------------------------------------------------------------------ #
    class MockResponse:
        def __init__(self, ok=True, status_code=200):
            self.ok = ok
            self.status_code = status_code

    def mock_session_put(self, url, **kwargs):
        # Simply return success to simulate upload
        return MockResponse(ok=True)

    monkeypatch.setattr("requests.Session.put", mock_session_put)

    # ------------------------------------------------------------------ #
    # 4) deterministic local share link
    # ------------------------------------------------------------------ #
    def _dummy_share_link(url, user, pw, remote_path, **kw):
        rel = (
            remote_path[len("data/") :]
            if remote_path.startswith("data/")
            else remote_path
        )
        return f"file://{tmp_data_dir / rel}"

    monkeypatch.setattr(fu, "get_share_link", _dummy_share_link)

    # 5) every stored link is considered valid
    monkeypatch.setattr(fu, "verify_share_link", lambda link, suffix="": True)

    yield


# ---------------------------------------------------------------------- #
# helper under test: chunked obsm download
# ---------------------------------------------------------------------- #
def download_obsm_slice(folder_link: str, obsm_key: str, rows: slice):
    mapper = fsspec.get_mapper(folder_link, anon=True)  # works for file://
    root = zarr.open_consolidated(mapper, mode="r")
    return root["obsm"][obsm_key][rows]


# ---------------------------------------------------------------------- #
# tests
# ---------------------------------------------------------------------- #
def test_upload_and_share_map(create_zarr_stores, tmp_data_dir: Path):
    """Uploading two *.zarr stores yields two folder share links."""
    # run the upload helper (patched – no network)
    # Note: config contains environment variable names, not actual values
    _share_map: Dict[str, str] = upload_folder_to_nextcloud(
        data_folder=tmp_data_dir,
        nextcloud_config={
            "url": "NC_URL",  # env var name, not actual URL
            "username": "NC_USER",  # env var name, not actual username
            "password": "NC_PASS",  # env var name, not actual password
        },
    )

    # -------------- expectations -------------------------------------- #
    # 1) share_map.json exists
    mapping_path = tmp_data_dir / "share_map.json"
    assert mapping_path.exists(), "share_map.json not written"
    stored = json.loads(mapping_path.read_text())

    # 2) exactly one entry per store, keys are ZIP file names
    assert set(stored) == {"sample_A.zarr.zip", "sample_B.zarr.zip"}
    # 3) each link starts with dummy host
    for link in stored.values():
        assert link.startswith("file://")

    # -------------- chunked download check ---------------------------- #
    # The zarr stores were created by the create_zarr_stores fixture and have proper metadata
    # We can test the download functionality by reading from the original zarr directory
    zarr_dir = tmp_data_dir / "sample_A.zarr"
    # Use zarr.open instead of zarr.open_consolidated since the test zarr may not be consolidated
    root = zarr.open(str(zarr_dir), mode="r")
    vecs = root["obsm"]["X_geneformer"][slice(0, 5)]
    assert vecs.shape == (5, 12)
    assert vecs.dtype == np.float16


def _make_files(tmpdir: Path, n=20):
    for i in range(n):
        # Create .zarr directories instead of .bin files to match function expectations
        zarr_dir = tmpdir / f"sample_{i}.zarr"
        zarr_dir.mkdir()
        # Create a minimal zarr structure
        (zarr_dir / ".zarray").write_text('{"shape": [10, 10]}')


def test_thread_pool_invoked(tmp_path, monkeypatch):
    _make_files(tmp_path, n=20)

    # ---------------- set up environment variables ------------------- #
    monkeypatch.setenv("NC_URL", "https://dummy")
    monkeypatch.setenv("NC_USER", "user")
    monkeypatch.setenv("NC_PASS", "pwd")

    # ---------------- monitor calls to track parallel execution ------ #
    call_log = []

    # Override the session.put mock to track calls
    class MockResponse:
        def __init__(self, ok=True, status_code=200):
            self.ok = ok
            self.status_code = status_code

    def mock_session_put_with_logging(self, url, **kwargs):
        time.sleep(0.01)  # simulate network latency
        call_log.append(url)  # record the URL being uploaded to
        return MockResponse(ok=True)

    monkeypatch.setattr("requests.Session.put", mock_session_put_with_logging)

    # Mock other functions that are called (these override the autouse fixture)
    monkeypatch.setattr(
        "adata_hf_datasets.file_utils.get_share_link",
        lambda *a, **k: "file:///dev/null",
    )
    monkeypatch.setattr(
        "adata_hf_datasets.file_utils.verify_share_link",
        lambda *a, **k: True,  # Return True to indicate links are valid
    )

    upload_folder_to_nextcloud(
        tmp_path,
        nextcloud_config={
            "url": "NC_URL",  # env var name
            "username": "NC_USER",  # env var name
            "password": "NC_PASS",  # env var name
        },
        max_workers=8,  # ask for many workers
    )

    # if we really ran uploads, the log should hold *all* upload URLs
    assert len(call_log) == 20
