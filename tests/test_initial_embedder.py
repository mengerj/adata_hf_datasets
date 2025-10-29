import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad, AnnData, read_zarr

from adata_hf_datasets.embed import InitialEmbedder
from adata_hf_datasets.embed import InitialEmbedder as ie_mod


@pytest.fixture
def raw_adata_path(tmp_path):
    """Write a tiny AnnData with a 'batch' column to disk."""
    X = np.arange(20).reshape(10, 2)
    adata = AnnData(X=X)
    adata.obs["batch"] = pd.Categorical(["A"] * 5 + ["B"] * 5)
    p = tmp_path / "raw.h5ad"
    adata.write_h5ad(p)
    return p


@pytest.fixture
def raw_adata_zarr_path(tmp_path):
    """Write a tiny AnnData with a 'batch' column to disk in zarr format."""
    X = np.arange(20).reshape(10, 2)
    adata = AnnData(X=X)
    adata.obs["batch"] = pd.Categorical(["A"] * 5 + ["B"] * 5)
    p = tmp_path / "raw.zarr"
    adata.write_zarr(p)
    return p


class DummyEmbedder:
    """
    Stand‑in embedder.  Its embed() method takes an AnnData or path,
    writes an obsm key, and returns the embedding matrix.
    """

    requires_mem_adata = True

    def __init__(self, embedding_dim: int, **kwargs):
        self.embedding_dim = embedding_dim

    def prepare(self, adata=None, adata_path: str = None, **kwargs):
        # no-op
        pass

    def embed(self, adata=None, adata_path=None, obsm_key=None, **kwargs):
        # Load if needed
        if adata is None and adata_path is not None:
            if str(adata_path).endswith(".zarr"):
                adata = read_zarr(adata_path)
            else:
                adata = read_h5ad(adata_path)

        # Create a dummy embedding matrix
        n = adata.n_obs
        mat = np.full(
            (n, self.embedding_dim), fill_value=self.embedding_dim, dtype=float
        )
        # Attach it if adata is provided
        if adata is not None:
            adata.obsm[obsm_key] = mat
        return mat


@pytest.fixture(autouse=True)
def patch_embedders(monkeypatch):
    """
    Replace the real embedder classes so InitialEmbedder always uses DummyEmbedder.
    """
    for name in (
        "SCVIEmbedderFM",
        "GeneformerEmbedder",
        "PCAEmbedder",
        "HighlyVariableGenesEmbedder",
    ):
        monkeypatch.setattr(ie_mod, name, DummyEmbedder)
    yield


def test_inmemory_looping_over_methods_attaches_obsm(raw_adata_path):
    methods = ["hvg", "pca", "scvi_fm", "geneformer"]
    embedding_dim_map = {"hvg": 4, "pca": 2, "scvi_fm": 3, "geneformer": 5}
    batch_key = "batch"

    # We'll call embed() on each method, chaining into the same AnnData
    adata = read_h5ad(raw_adata_path)
    for method in methods:
        dim = embedding_dim_map[method]
        emb = InitialEmbedder(method=method, embedding_dim=dim)

        _emb_matrix = emb.embed(
            adata=adata,
            obsm_key=f"X_{method}",
            batch_key=batch_key,
        )

    # Now assert that every embedding is in adata.obsm with correct shape & value
    for method, dim in embedding_dim_map.items():
        key = f"X_{method}"
        assert key in adata.obsm, f"{key} missing in .obsm"
        mat = adata.obsm[key]
        assert mat.shape == (adata.n_obs, dim)
        # each cell should see the dummy fill value
        assert np.all(mat == float(dim)), f"{key} values incorrect"


def test_file_based_embedding_h5ad(raw_adata_path):
    """Test embedding using h5ad file path."""
    methods = ["hvg", "pca", "scvi_fm", "geneformer"]
    embedding_dim_map = {"hvg": 4, "pca": 2, "scvi_fm": 3, "geneformer": 5}
    batch_key = "batch"

    for method in methods:
        dim = embedding_dim_map[method]
        emb = InitialEmbedder(method=method, embedding_dim=dim)

        # Test with file path
        emb_matrix = emb.embed(
            adata_path=raw_adata_path,
            obsm_key=f"X_{method}",
            batch_key=batch_key,
        )

        # Verify the returned matrix
        assert isinstance(emb_matrix, np.ndarray)
        assert emb_matrix.shape == (10, dim)  # 10 cells in our test data
        assert np.all(emb_matrix == float(dim))


def test_file_based_embedding_zarr(raw_adata_zarr_path):
    """Test embedding using zarr file path."""
    methods = ["hvg", "pca", "scvi_fm", "geneformer"]
    embedding_dim_map = {"hvg": 4, "pca": 2, "scvi_fm": 3, "geneformer": 5}
    batch_key = "batch"

    for method in methods:
        dim = embedding_dim_map[method]
        emb = InitialEmbedder(method=method, embedding_dim=dim)

        # Test with file path
        emb_matrix = emb.embed(
            adata_path=raw_adata_zarr_path,
            obsm_key=f"X_{method}",
            batch_key=batch_key,
        )

        # Verify the returned matrix
        assert isinstance(emb_matrix, np.ndarray)
        assert emb_matrix.shape == (10, dim)  # 10 cells in our test data
        assert np.all(emb_matrix == float(dim))


def test_mixed_embedding_approaches(raw_adata_path):
    """Test mixing in-memory and file-based approaches."""
    method = "pca"
    dim = 2
    batch_key = "batch"
    emb = InitialEmbedder(method=method, embedding_dim=dim)

    # First embed with file path
    emb_matrix1 = emb.embed(
        adata_path=raw_adata_path,
        obsm_key=f"X_{method}",
        batch_key=batch_key,
    )

    # Then embed with in-memory AnnData
    adata = read_h5ad(raw_adata_path)
    emb_matrix2 = emb.embed(
        adata=adata,
        obsm_key=f"X_{method}",
        batch_key=batch_key,
    )

    # Verify both approaches give same results
    assert np.array_equal(emb_matrix1, emb_matrix2)
    assert isinstance(emb_matrix1, np.ndarray)
    assert isinstance(emb_matrix2, np.ndarray)


def test_embedding_without_obsm_storage(raw_adata_path):
    """Test that embedding works without storing in AnnData.obsm."""
    method = "pca"
    dim = 2
    batch_key = "batch"
    emb = InitialEmbedder(method=method, embedding_dim=dim)

    # Embed without providing AnnData
    emb_matrix = emb.embed(
        adata_path=raw_adata_path,
        batch_key=batch_key,
    )

    # Verify the returned matrix
    assert isinstance(emb_matrix, np.ndarray)
    assert emb_matrix.shape == (10, dim)
    assert np.all(emb_matrix == float(dim))


def test_invalid_file_format(tmp_path):
    """Test that invalid file formats raise appropriate errors."""
    method = "pca"
    dim = 2
    batch_key = "batch"
    emb = InitialEmbedder(method=method, embedding_dim=dim)

    # Create an invalid file
    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_text("not an anndata file")

    # Test invalid file format
    with pytest.raises(
        ValueError,
        match="Unsupported file format.*Only .h5ad and .zarr formats are supported",
    ):
        emb.embed(
            adata_path=invalid_path,
            batch_key=batch_key,
        )

    # Test non-existent file
    nonexistent_path = tmp_path / "nonexistent.h5ad"
    with pytest.raises(FileNotFoundError, match="File not found"):
        emb.embed(
            adata_path=nonexistent_path,
            batch_key=batch_key,
        )
