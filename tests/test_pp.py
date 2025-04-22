import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from anndata import AnnData

from adata_hf_datasets.pp.loader import BatchChunkLoader
from adata_hf_datasets.pp.utils import ensure_raw_counts_layer
from adata_hf_datasets.pp.qc import pp_quality_control
from adata_hf_datasets.pp.general import pp_adata_general


@pytest.fixture
def toy_adata_dense(tmp_path) -> AnnData:
    """
    Create a small dense AnnData with raw integer counts, two genes and three cells.
    Cells 'A','B','C' belong to batches 1,1,2.
    """
    X = np.array([[1, 0], [5, 0], [0, 10]], dtype=int)
    obs = {"batch": ["batch1", "batch1", "batch2"]}
    adata = AnnData(X=X, obs=obs)
    return adata


@pytest.fixture
def toy_adata_sparse() -> AnnData:
    """
    Same as toy_adata_dense but with a sparse matrix.
    """
    X = sp.csr_matrix([[1, 0], [0, 2]])
    obs = {"batch": ["a", "b"]}
    adata = AnnData(X=X, obs=obs)
    return adata


def test_ensure_raw_counts_from_layer(toy_adata_dense):
    """If 'counts' layer exists, it's used and copied to X."""
    ad = toy_adata_dense.copy()
    # set up a custom layer
    ad.layers["my_counts"] = ad.X.copy()
    # erase X to something non-raw
    ad.X = ad.X.astype(float) / 10.0
    ensure_raw_counts_layer(ad, raw_layer_key="my_counts", raise_on_missing=True)
    # after call, 'counts' must be present, X equals that layer, and dtype integer
    assert "counts" in ad.layers
    np.testing.assert_array_equal(
        ad.X.A if sp.issparse(ad.X) else ad.X,
        ad.layers["my_counts"].A
        if sp.issparse(ad.layers["my_counts"])
        else ad.layers["my_counts"],
    )
    assert np.issubdtype(ad.X.dtype, np.integer)


def test_ensure_raw_counts_detects_X(tmp_path):
    """If no layer but X is raw, we copy X into 'counts'."""
    ad = AnnData(X=np.array([[2, 3], [0, 1]], dtype=int), obs={"batch": ["x", "y"]})
    ensure_raw_counts_layer(ad, raw_layer_key=None, raise_on_missing=True)
    assert "counts" in ad.layers
    np.testing.assert_array_equal(
        ad.layers["counts"].toarray()
        if sp.issparse(ad.layers["counts"])
        else ad.layers["counts"],
        ad.X.toarray() if sp.issparse(ad.X) else ad.X,
    )


def test_ensure_raw_counts_error(tmp_path):
    """If neither layer nor X are raw, raise when requested."""
    ad = AnnData(X=np.array([[0.1, 0.2]]), obs={})
    with pytest.raises(ValueError):
        ensure_raw_counts_layer(ad, raw_layer_key="nope", raise_on_missing=True)


@pytest.fixture
def realistic_adata_general() -> AnnData:
    """
    Simulate a small but realistic scRNA‑seq AnnData for pp_adata_general tests:
      - 50 cells × 200 genes
      - Counts ~ Poisson(1), with about half the genes expressed in ≥10 cells
      - Three batches in obs['batch']
    """
    n_cells, n_genes = 50, 200
    # simulate counts
    X = np.random.poisson(lam=1.0, size=(n_cells, n_genes)).astype(int)
    # randomly zero out half the genes in most cells to create low-frequency genes
    zero_mask = np.random.rand(n_cells, n_genes) < 0.5
    X[zero_mask] = 0
    # define batches
    batches = np.random.choice(
        ["batch1", "batch2", "batch3"], size=n_cells, p=[0.4, 0.4, 0.2]
    )
    obs = pd.DataFrame({"batch": batches})
    ad = AnnData(X=X, obs=obs)
    ad.var_names = [f"gene{i}" for i in range(n_genes)]
    return ad


def test_pp_adata_general_filters_and_hvg_realistic(realistic_adata_general):
    ad = realistic_adata_general.copy()
    # 1) Ensure raw counts layer is created
    ensure_raw_counts_layer(ad, raw_layer_key=None, raise_on_missing=True)
    # 1) Remove genes expressed in < 10 cells
    filtered = pp_adata_general(
        ad.copy(),
        min_cells=10,
        min_genes=0,
        batch_key="batch",
        n_top_genes=20,
    )
    # check that no gene in filtered is expressed in fewer than 10 cells
    counts_per_gene = np.array((filtered.layers["counts"] > 0).sum(axis=0)).ravel()
    assert (counts_per_gene >= 10).all()

    # 2) Now make min_genes so high that no cell survives
    with pytest.raises(ValueError):
        pp_adata_general(
            ad.copy(),
            min_cells=0,
            min_genes=1000,
            batch_key="batch",
            n_top_genes=20,
        )


@pytest.fixture
def realistic_loader(tmp_path) -> Path:
    """
    Create an on‑disk .h5ad with 30 cells from 3 batches:
      - 'A': 10 cells
      - 'B': 12 cells
      - 'C': 8 cells
    """
    n_cells = 30
    n_genes = 50
    X = sp.random(n_cells, n_genes, density=0.2, format="csr", random_state=0)
    batches = ["A"] * 10 + ["B"] * 12 + ["C"] * 8
    ad = AnnData(X=X, obs={"batch": batches})
    ad.var_names = [f"g{i}" for i in range(n_genes)]
    out = tmp_path / "realistic.h5ad"
    ad.write_h5ad(out)
    return out


def test_batch_chunk_loader_groups_realistic(realistic_loader):
    loader = BatchChunkLoader(realistic_loader, chunk_size=15, batch_key="batch")
    chunks = list(loader)
    # with chunk_size=15 we expect:
    #  - first chunk: batches A (10) + B (need 5 of B) → but B must stay whole, so chunk1 = A only (10)
    #  - chunk2: B (12) alone (12)
    #  - chunk3: C (8) alone (8)
    assert len(chunks) == 3
    sizes = [ch.n_obs for ch in chunks]
    assert sizes == [10, 12, 8]
    # ensure no batch is split
    for chunk in chunks:
        assert len(set(chunk.obs["batch"])) == 1


def test_returned_adata_is_copy_realistic(realistic_adata_general):
    ad = realistic_adata_general.copy()
    # simulate the preprocessing steps
    ensure_raw_counts_layer(ad, raw_layer_key=None, raise_on_missing=True)
    original_X_id = id(ad.X)
    qc = pp_quality_control(
        ad.copy(),
        percent_top=[10],
        nmads_main=2,
        nmads_mt=2,
        pct_counts_mt_threshold=20,
    )
    general = pp_adata_general(
        ad.copy(), min_cells=5, min_genes=50, batch_key="batch", n_top_genes=20
    )
    # ensure .X buffers differ
    assert id(qc.X) != original_X_id
    assert id(general.X) != original_X_id
