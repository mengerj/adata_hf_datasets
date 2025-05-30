# tests/test_safe_write_zarr.py
from pathlib import Path

import anndata as ad
import numpy as np

from adata_hf_datasets.file_utils import safe_write_zarr


# helper ────────────────────────────────────────────────────────────────
def _make_adata(n_cells=20, n_genes=10) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_cells, n_genes)).astype("float32")
    adata = ad.AnnData(X)
    adata.obsm["X_geneformer"] = rng.normal(size=(n_cells, 12)).astype("float16")
    return adata


# test ──────────────────────────────────────────────────────────────────
def test_overwrite_existing_zarr(tmp_path: Path):
    """A second call to safe_write_zarr should atomically replace the store."""
    zpath = tmp_path / "overwrite_test.zarr"

    # 1️⃣  initial write with one embedding
    adata1 = _make_adata()
    safe_write_zarr(adata1, target=zpath)
    saved1 = ad.read_zarr(zpath)
    assert "X_geneformer" in saved1.obsm

    # 2️⃣  modify: add a new embedding and overwrite
    saved1.obsm["X_pca"] = np.random.randn(saved1.n_obs, 5).astype("float16")
    safe_write_zarr(saved1, target=zpath)  # ← overwrite same path

    # 3️⃣  read back and check both embeddings are present
    saved2 = ad.read_zarr(zpath)
    assert "X_geneformer" in saved2.obsm
    assert "X_pca" in saved2.obsm
    assert saved2.obsm["X_pca"].shape == (saved2.n_obs, 5)
