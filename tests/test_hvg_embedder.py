import numpy as np
import scipy.sparse as sp
import scanpy as sc
import pytest
from anndata import AnnData

from adata_hf_datasets.initial_embedder import HighlyVariableGenesEmbedder  # ← adjust


# ---------------------------------------------------------------------
def _make_dense_adata(n_cells=30, n_genes=100):
    rng = np.random.default_rng(0)
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype(float)
    ad = AnnData(X)
    ad.obs["batch"] = ["A"] * (n_cells // 2) + ["B"] * (n_cells - n_cells // 2)
    return ad


@pytest.fixture
def toy_adata_dense():
    return _make_dense_adata()


@pytest.fixture
def toy_adata_sparse(toy_adata_dense):
    ad = toy_adata_dense.copy()
    ad.X = sp.csr_matrix(ad.X)
    return ad


# ---------------------------------------------------------------------
def test_hvg_no_precomputed(toy_adata_dense):
    emb = HighlyVariableGenesEmbedder(embedding_dim=20)
    emb.prepare(toy_adata_dense)  # no-op for now
    emb_array = emb.embed(adata=toy_adata_dense, batch_key="batch")
    assert emb_array.shape == (toy_adata_dense.n_obs, 20)


def test_hvg_precomputed_exact(toy_adata_sparse):
    ad = toy_adata_sparse
    # Precompute exactly 15 HVGs
    sc.pp.highly_variable_genes(ad, n_top_genes=15, batch_key=None)
    emb = HighlyVariableGenesEmbedder(embedding_dim=15)
    emb_array = emb.embed(adata=ad, obsm_key="X_hvg_pre")
    assert emb_array.shape == (ad.n_obs, 15)


def test_hvg_precomputed_too_many():
    ad = _make_dense_adata()
    sc.pp.highly_variable_genes(ad, n_top_genes=50, batch_key=None)
    emb = HighlyVariableGenesEmbedder(embedding_dim=20)

    # When pre-computed HVGs > requested, embedder trims to exact n
    emb_array = emb.embed(adata=ad, obsm_key="X_hvg_trim")
    assert emb_array.shape == (ad.n_obs, 20)
    assert ad.var["highly_variable"].sum() == 20


def test_hvg_precomputed_too_few():
    ad = _make_dense_adata()
    sc.pp.highly_variable_genes(ad, n_top_genes=10, batch_key=None)
    emb = HighlyVariableGenesEmbedder(embedding_dim=20)
    emb_array = emb.embed(adata=ad)
    assert emb_array.shape == (ad.n_obs, 20)
    assert ad.var["highly_variable"].sum() == 20  # should redo the HVG selection


def test_hvg_infinite_values_handled():
    ad = _make_dense_adata()
    ad.X[0, 0] = np.inf  # inject an infinity
    emb = HighlyVariableGenesEmbedder(embedding_dim=10)
    emb_array = emb.embed(adata=ad)  # should succeed (drops gene)
    assert emb_array.shape[1] == 10
