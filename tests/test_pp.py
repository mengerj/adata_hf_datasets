import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path
from anndata import AnnData
import anndata as ad

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
    ad = ensure_raw_counts_layer(ad, raw_layer_key="my_counts", raise_on_missing=True)
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
    ad = ensure_raw_counts_layer(ad, raw_layer_key=None, raise_on_missing=True)
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
    ad = ensure_raw_counts_layer(ad, raw_layer_key=None, raise_on_missing=True)
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


@pytest.fixture
def adata_with_metadata(tmp_path) -> Path:
    """
    Create an on-disk .h5ad with cells that have unique identifiers and metadata.
    Each cell has:
    - A unique cell_id in obs.index
    - cell_type metadata
    - batch metadata
    - Other metadata columns

    This allows us to verify that random chunking preserves the correspondence
    between obs.index and obs columns.
    """
    n_cells = 50
    n_genes = 30

    # Create count matrix
    X = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=42)

    # Create unique cell IDs
    cell_ids = [f"cell_{i:03d}" for i in range(n_cells)]

    # Create metadata with known patterns
    cell_types = np.array(["T_cell", "B_cell", "NK_cell", "Monocyte", "DC"])[
        np.arange(n_cells) % 5
    ]
    batches = np.array(["batch_A", "batch_B", "batch_C"])[np.arange(n_cells) % 3]

    # Create obs DataFrame with unique index and metadata
    obs = pd.DataFrame(
        {
            "cell_type": cell_types,
            "batch": batches,
            "patient_id": [f"patient_{i % 10:02d}" for i in range(n_cells)],
            "quality_score": np.random.uniform(0.5, 1.0, size=n_cells),
            "original_index": np.arange(n_cells),  # Track original position
        },
        index=cell_ids,  # Set unique cell IDs as index
    )

    # Create var DataFrame
    var = pd.DataFrame(
        index=[f"gene_{i}" for i in range(n_genes)],
        data={"gene_symbol": [f"GENE_{i}" for i in range(n_genes)]},
    )

    adata = AnnData(X=X, obs=obs, var=var)

    # Write to disk
    out = tmp_path / "adata_with_metadata.h5ad"
    adata.write_h5ad(out)
    return out


def test_random_chunk_loader_preserves_obs_alignment(adata_with_metadata):
    """
    Test that random chunking preserves the correspondence between obs.index
    and obs columns (e.g., cell_type, batch, etc.).

    This is critical because if indices are shuffled incorrectly, the metadata
    would become misaligned with the expression data.
    """
    # Load original data to get reference
    original = ad.read_h5ad(adata_with_metadata)

    # Create a mapping from cell_id to metadata for verification
    original_mapping = {}
    for cell_id in original.obs.index:
        original_mapping[cell_id] = {
            "cell_type": original.obs.loc[cell_id, "cell_type"],
            "batch": original.obs.loc[cell_id, "batch"],
            "patient_id": original.obs.loc[cell_id, "patient_id"],
            "original_index": original.obs.loc[cell_id, "original_index"],
        }

    # Test random chunking with a fixed seed for reproducibility
    loader = BatchChunkLoader(
        adata_with_metadata,
        chunk_size=15,
        batch_key=None,  # No batch key, should use random chunking
        random_chunking=True,
        random_seed=42,
    )

    # Collect all chunks
    chunks = list(loader)

    # Verify we got chunks
    assert len(chunks) > 0, "Loader should produce at least one chunk"

    # Collect all cells from all chunks
    all_cell_ids = []
    for chunk in chunks:
        all_cell_ids.extend(chunk.obs.index.tolist())

    # Verify we have all cells (no duplicates, no missing)
    assert len(all_cell_ids) == len(original.obs.index), (
        f"Expected {len(original.obs.index)} cells, got {len(all_cell_ids)}"
    )
    assert len(set(all_cell_ids)) == len(all_cell_ids), "Duplicate cell IDs found!"
    assert set(all_cell_ids) == set(original.obs.index), "Cell IDs don't match!"

    # Verify metadata alignment for each chunk
    for chunk_idx, chunk in enumerate(chunks):
        for cell_id in chunk.obs.index:
            # Verify cell_id exists in original
            assert cell_id in original_mapping, f"Cell {cell_id} not in original data"

            # Get expected metadata
            expected = original_mapping[cell_id]

            # Verify each metadata column matches
            assert chunk.obs.loc[cell_id, "cell_type"] == expected["cell_type"], (
                f"Chunk {chunk_idx}: cell_type mismatch for {cell_id}. "
                f"Expected {expected['cell_type']}, got {chunk.obs.loc[cell_id, 'cell_type']}"
            )
            assert chunk.obs.loc[cell_id, "batch"] == expected["batch"], (
                f"Chunk {chunk_idx}: batch mismatch for {cell_id}. "
                f"Expected {expected['batch']}, got {chunk.obs.loc[cell_id, 'batch']}"
            )
            assert chunk.obs.loc[cell_id, "patient_id"] == expected["patient_id"], (
                f"Chunk {chunk_idx}: patient_id mismatch for {cell_id}. "
                f"Expected {expected['patient_id']}, got {chunk.obs.loc[cell_id, 'patient_id']}"
            )
            assert (
                chunk.obs.loc[cell_id, "original_index"] == expected["original_index"]
            ), (
                f"Chunk {chunk_idx}: original_index mismatch for {cell_id}. "
                f"Expected {expected['original_index']}, got {chunk.obs.loc[cell_id, 'original_index']}"
            )

            # Verify expression data matches (check a few genes)
            original_idx = original.obs.index.get_loc(cell_id)
            chunk_idx_pos = chunk.obs.index.get_loc(cell_id)

            # Compare expression for first gene
            original_expr = original.X[original_idx, 0]
            chunk_expr = chunk.X[chunk_idx_pos, 0]

            if sp.issparse(original_expr):
                original_expr = original_expr.toarray()[0, 0]
            if sp.issparse(chunk_expr):
                chunk_expr = chunk_expr.toarray()[0, 0]

            assert original_expr == chunk_expr, (
                f"Chunk {chunk_idx}: Expression mismatch for {cell_id}, gene 0. "
                f"Expected {original_expr}, got {chunk_expr}"
            )


def test_random_chunk_loader_with_batch_key_force_random(adata_with_metadata):
    """
    Test that random chunking works even when batch_key is provided
    (force random mode).
    """
    original = ad.read_h5ad(adata_with_metadata)

    # Create loader with batch_key but force random chunking
    loader = BatchChunkLoader(
        adata_with_metadata,
        chunk_size=15,
        batch_key="batch",  # Provide batch key
        random_chunking=True,  # But force random chunking
        random_seed=123,
    )

    chunks = list(loader)
    assert len(chunks) > 0

    # Verify all cells are present
    all_cell_ids = []
    for chunk in chunks:
        all_cell_ids.extend(chunk.obs.index.tolist())

    assert len(all_cell_ids) == len(original.obs.index)
    assert set(all_cell_ids) == set(original.obs.index)

    # Verify metadata alignment (same as above)
    for chunk in chunks:
        for cell_id in chunk.obs.index:
            original_cell_type = original.obs.loc[cell_id, "cell_type"]
            chunk_cell_type = chunk.obs.loc[cell_id, "cell_type"]
            assert original_cell_type == chunk_cell_type, (
                f"Metadata mismatch for {cell_id}: "
                f"expected {original_cell_type}, got {chunk_cell_type}"
            )


def test_add_ensembl_ids_with_versioned_ensembl_index():
    """
    Test that add_ensembl_ids correctly handles var index with versioned Ensembl IDs.

    When the var index contains Ensembl IDs with version numbers (e.g., 'ENSG00000268903.1'),
    the function should create an 'ensembl_id' column with the version numbers stripped off.
    """
    from adata_hf_datasets.pp.pybiomart_utils import add_ensembl_ids

    # Create test data with versioned Ensembl IDs in the var index
    versioned_ensembl_ids = [
        "ENSG00000268903.1",
        "ENSG00000241860.6",
        "ENSG00000228463.10",
        "ENSG00000237094.12",
        "ENSG00000225972.1",
        "ENSG00000225630.1",
        "ENSG00000237973.1",
        "ENSG00000229344.1",
        "ENSG00000248527.1",
        "ENSG00000198744.5",
    ]

    # Expected Ensembl IDs without version numbers
    expected_ensembl_ids = [
        "ENSG00000268903",
        "ENSG00000241860",
        "ENSG00000228463",
        "ENSG00000237094",
        "ENSG00000225972",
        "ENSG00000225630",
        "ENSG00000237973",
        "ENSG00000229344",
        "ENSG00000248527",
        "ENSG00000198744",
    ]

    # Create a simple AnnData object with versioned Ensembl IDs as var index
    n_cells = 10
    n_genes = len(versioned_ensembl_ids)
    X = np.random.poisson(lam=2.0, size=(n_cells, n_genes)).astype(int)

    obs = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_cells)]})

    var = pd.DataFrame(
        index=versioned_ensembl_ids, data={"gene_type": ["protein_coding"] * n_genes}
    )

    adata = AnnData(X=X, obs=obs, var=var)

    # Verify initial state - no ensembl_id column should exist
    assert "ensembl_id" not in adata.var.columns
    assert adata.var_names.tolist() == versioned_ensembl_ids

    # Call add_ensembl_ids - this should detect the versioned Ensembl IDs and strip versions
    add_ensembl_ids(adata, ensembl_col="ensembl_id")

    # Verify that ensembl_id column was created
    assert "ensembl_id" in adata.var.columns

    # Verify that the ensembl_id column contains the expected IDs without version numbers
    actual_ensembl_ids = adata.var["ensembl_id"].tolist()
    assert actual_ensembl_ids == expected_ensembl_ids

    # Verify that the var index is still the original versioned IDs
    assert adata.var_names.tolist() == versioned_ensembl_ids

    # Verify that all ensembl_ids start with 'ENS' and don't contain dots
    for ensembl_id in actual_ensembl_ids:
        assert ensembl_id.startswith("ENS")
        assert "." not in ensembl_id
