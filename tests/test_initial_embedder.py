"""
Unit tests for InitialEmbedder and underlying embedders (HVG, PCA, SCVI, Geneformer).

We do *not* include full pipeline tests for Geneformer or SCVI, as they can be resource-intensive.
Instead, we ensure:
1) The data is preprocessed via `pp_adata(...)` so it contains required fields.
2) `prepare(...)` does not fail for any method (even if extra kwargs are present).
3) HVG/PCA can be embedded in-memory or in a chunk-based approach.
4) Geneformer is *not* chunk-based (since it always operates on a file path).
5) SCVI is tested only lightly (no actual model or reference data integration here).
"""

import pytest
import anndata
import numpy as np
import logging
import os
from unittest.mock import patch
import scanpy as sc

# If your real code is in a different file, adjust the import:
# from my_package.initial_embedder import InitialEmbedder

logger = logging.getLogger(__name__)


def pp_adata(infile: str, outfile: str):
    """
    Mock preprocessing function that ensures needed fields (sample_index, n_counts, ensembl_id, etc.)
    are present in the AnnData object, and writes out the processed data.

    This stands in for the actual 'pp_adata()' which might involve
    more complex operations like log-norm, ensembl ID assignment, etc.

    Parameters
    ----------
    infile : str
        Path to the input AnnData file (user-provided).
    outfile : str
        Output file path for writing the processed AnnData.

    Notes
    -----
    Data is user-provided in memory. This function simulates ensuring
    that geneformer/SCVI/HVG/PCA have the fields needed.
    """
    adata = anndata.read_h5ad(infile)
    # Make sure these columns are present for geneformer:
    if "sample_index" not in adata.obs.columns:
        adata.obs["sample_index"] = np.arange(adata.n_obs).astype(np.int32)

    if "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = np.sum(adata.X, axis=1).astype(np.int32)

    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = [f"ENSG_{i}" for i in range(adata.n_vars)]

    # Add layers which might be expected by the embedders
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Properly close and write the data
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    adata.write_h5ad(outfile)
    # Ensure file is properly closed
    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()
    del adata


@pytest.fixture
def raw_adata_path(tmp_path):
    """
    Create a small in-memory AnnData (200 cells, 100 genes) and write it to .h5ad.
    This simulates user-provided data before 'pp_adata' is applied.

    Returns
    -------
    Path
        Path to the raw (unprocessed) .h5ad file.
    """
    np.random.seed(42)
    # Create integer counts instead of floating point values for compatibility with Geneformer
    X = np.random.randint(
        0, 100000, size=(200, 100)
    )  # Integer values 0-9 instead of floats
    adata = anndata.AnnData(X=X)
    # (No guarantee it has sample_index, n_counts, etc.)
    raw_path = tmp_path / "raw_toy_data.h5ad"
    adata.write_h5ad(raw_path)
    return raw_path


@pytest.fixture
def processed_adata_path(raw_adata_path, tmp_path):
    """
    Run `pp_adata(infile, outfile)` to ensure required fields are present.

    Returns
    -------
    Path
        Path to the processed .h5ad file containing all required fields.
    """
    processed_path = tmp_path / "processed_toy_data.h5ad"
    pp_adata(infile=str(raw_adata_path), outfile=str(processed_path))
    # delete the raw file
    os.remove(raw_adata_path)
    return processed_path


@pytest.mark.parametrize("method", ["hvg", "pca", "geneformer"])
def test_prepare_runs_without_error(processed_adata_path, method):
    """
    Test that 'prepare(...)' completes without error for each method,
    even if additional arguments (like batch_key) are provided but not used.

    Parameters
    ----------
    processed_adata_path : Path
        Path to the processed AnnData file (ensures required fields).
    method : str
        Embedding method. Parametrized over "hvg", "pca", "scvi_fm", "geneformer".
    """
    from adata_hf_datasets.initial_embedder import InitialEmbedder  # adjust if needed

    embedder = InitialEmbedder(method=method, embedding_dim=16, some_extra_arg="foo")
    # 'batch_key' is relevant mostly for SCVI, but we'll pass it anyway:
    embedder.prepare(
        adata_path=str(processed_adata_path), batch_key="batch_maybe_unused"
    )


def test_hvg_in_memory_embed(processed_adata_path, tmp_path):
    """
    Test HVG embedding in a straightforward (in-memory) approach.

    Parameters
    ----------
    processed_adata_path : Path
        Path to the processed AnnData with required fields.
    tmp_path : Path
        Pytest-provided temporary directory for test outputs.

    Notes
    -----
    HVG just picks a subset of genes. We set embedding_dim=10 to confirm shape.
    """
    from adata_hf_datasets.initial_embedder import InitialEmbedder

    embedder = InitialEmbedder(method="hvg", embedding_dim=10)
    embedder.prepare(str(processed_adata_path))

    out_file = tmp_path / "hvg_embedded.h5ad"
    adata_emb = embedder.embed(
        adata_path=str(processed_adata_path),
        output_path=str(out_file),
        obsm_key="X_hvg_test",
    )
    assert out_file.exists(), "HVF embedding output was not written."

    # Check embedding shape: (n_obs, 10) because we selected 10 HVGs
    assert "X_hvg_test" in adata_emb.obsm
    emb_shape = adata_emb.obsm["X_hvg_test"].shape
    assert emb_shape[0] == 200, f"Expected 200 cells, got {emb_shape[0]}"
    assert emb_shape[1] == 10, f"Expected 10 HVGs, got {emb_shape[1]}"


def test_pca_chunk_based_embedding(processed_adata_path, tmp_path):
    """
    Test PCA embedding with chunk-based approach.

    Parameters
    ----------
    processed_adata_path : Path
        Path to the preprocessed .h5ad file with required fields.
    tmp_path : Path
        Temporary directory for test outputs.

    Notes
    -----
    We set batch_size < n_obs to force chunking.
    PCA embedder runs in memory, so chunk-based approach is used by the manager.
    """
    from adata_hf_datasets.initial_embedder import InitialEmbedder

    embedder = InitialEmbedder(method="pca", embedding_dim=5)
    embedder.prepare(str(processed_adata_path))

    out_file = tmp_path / "pca_chunked.h5ad"
    # Force chunk-based approach with small batch_size
    adata_emb = embedder.embed(
        adata_path=str(processed_adata_path),
        output_path=str(out_file),
        obsm_key="X_pca_test",
        batch_size=50,  # 200 cells, so 4 chunks
    )
    assert out_file.exists(), "PCA chunk-based embedding output was not written."

    # Check the embedding shape
    assert "X_pca_test" in adata_emb.obsm
    emb_shape = adata_emb.obsm["X_pca_test"].shape
    assert emb_shape == (200, 5), f"Expected 200 cells x 5 PCs, got {emb_shape}"


def test_geneformer_does_not_use_chunks(processed_adata_path, tmp_path):
    """
    Test that geneformer ignores chunk-based embedding and always uses file path.

    Parameters
    ----------
    processed_adata_path : Path
        Path to processed AnnData with fields needed by geneformer (sample_index, etc.).
    tmp_path : Path
        Temporary dir for test outputs.

    Notes
    -----
    We skip an actual 'geneformer' pipeline. We just ensure that specifying `batch_size`
    doesn't lead to chunked embedding for geneformer.
    """
    from adata_hf_datasets.initial_embedder import InitialEmbedder

    embedder = InitialEmbedder(method="geneformer", embedding_dim=512)

    out_file = tmp_path / "geneformer_test.h5ad"

    # Mock the actual embedding to avoid running the real geneformer model
    with (
        patch(
            "adata_hf_datasets.initial_embedder.GeneformerEmbedder.embed"
        ) as mock_embed,
        patch("anndata.experimental.AnnLoader") as mock_loader,
    ):
        # Set up the mock embed function to return a basic AnnData with fake embeddings
        mock_adata = anndata.read_h5ad(processed_adata_path, backed="r")
        mock_adata.obsm["X_geneformer_test"] = np.random.rand(mock_adata.n_obs, 512)
        mock_embed.return_value = mock_adata

        # If chunk-based approach was used, we'd see AnnLoader called.
        # We expect it NOT to be called for geneformer.
        adata_emb = embedder.embed(
            adata_path=str(processed_adata_path),
            output_path=str(out_file),
            obsm_key="X_geneformer_test",
            batch_size=30,  # forcibly ignored for geneformer
        )
        # Check the embedding shape
        assert "X_geneformer_test" in adata_emb.obsm
        emb_shape = adata_emb.obsm["X_geneformer_test"].shape
        assert emb_shape == (200, 512), (
            f"Expected 200 cells x 512 genes, got {emb_shape}"
        )
        # Check that adata_emb is backed
        assert adata_emb.isbacked, "Expected geneformer output to be backed."

        # Verify AnnLoader (chunking) was not called
        mock_loader.assert_not_called()

        # Verify the geneformer embedder was called directly
        mock_embed.assert_called_once()

    # We won't check real embedding shape because we skip the heavy pipeline.
    # We only confirm that an output file was written.
    assert out_file.exists(), "Geneformer embed was expected to write output."


def test_scvi_fm_prepare(processed_adata_path, tmp_path):
    """
    Test specific preparation of SCVI-FM embedder with proper mocking of S3 and model loading.

    Parameters
    ----------
    processed_adata_path : Path
        Path to the processed test AnnData file.
    tmp_path : Path
        Pytest-provided temporary directory.
    """
    from adata_hf_datasets.initial_embedder import InitialEmbedder

    # Create a temporary cache directory for the test
    cache_dir = tmp_path / "scvi_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # Mock both the S3 model loading and SCVI model loading
    with (
        patch("scvi.hub.HubModel.pull_from_s3") as mock_pull,
        patch("scvi.model.SCVI.load") as mock_load,
    ):
        # Set up mock model
        mock_model = mock_load.return_value
        # Create a small reference adata for the mock
        mock_model.adata = sc.read(processed_adata_path)
        mock_pull.return_value = mock_model

        # Initialize embedder with test cache directory
        embedder = InitialEmbedder(
            method="scvi_fm",
            embedding_dim=16,
        )

        # Prepare the embedder
        embedder.prepare(
            adata_path=str(processed_adata_path),
            batch_key="batch_maybe_unused",
            cache_dir=str(cache_dir),
        )

        # Verify the mocks were called correctly
        mock_pull.assert_called_once()
        # Verify the S3 parameters were correct
        _args, kwargs = mock_pull.call_args
        assert kwargs["s3_bucket"] == "cellxgene-contrib-public"
        assert kwargs["s3_path"] == "models/scvi/2024-02-12/homo_sapiens/modelhub"
        assert str(kwargs["cache_dir"]) == str(cache_dir)
        assert not kwargs[
            "pull_anndata"
        ]  # Should be False to avoid downloading reference data
