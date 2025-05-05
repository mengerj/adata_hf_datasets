import numpy as np
import pandas as pd
import pytest
from anndata import read_h5ad, AnnData

import adata_hf_datasets.initial_embedder as ie_mod
from adata_hf_datasets.initial_embedder import InitialEmbedder


@pytest.fixture
def raw_adata_path(tmp_path):
    """Write a tiny AnnData with a 'batch' column to disk."""
    X = np.arange(20).reshape(10, 2)
    adata = AnnData(X=X)
    adata.obs["batch"] = pd.Categorical(["A"] * 5 + ["B"] * 5)
    p = tmp_path / "raw.h5ad"
    adata.write_h5ad(p)
    return p


class DummyEmbedder:
    """
    Stand‑in embedder.  Its embed() method takes an AnnData or path,
    writes an obsm key, and returns the AnnData.
    """

    requires_mem_adata = True

    def __init__(self, embedding_dim: int, **kwargs):
        self.embedding_dim = embedding_dim

    def prepare(self, adata, adata_path: str, **kwargs):
        # no-op
        pass

    def embed(self, adata=None, obsm_key=None, **kwargs):
        # Load if needed
        # Create a dummy embedding matrix
        n = adata.n_obs
        mat = np.full(
            (n, self.embedding_dim), fill_value=self.embedding_dim, dtype=float
        )
        # Attach it
        adata.obsm[obsm_key] = mat
        return adata


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

    # We’ll call embed() on each method, chaining into the same AnnData
    adata = read_h5ad(raw_adata_path)
    for method in methods:
        dim = embedding_dim_map[method]
        emb = InitialEmbedder(method=method, embedding_dim=dim)

        adata = emb.embed(
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
