import pytest
import anndata
import numpy as np
from unittest.mock import patch

# Ensure geneformer is installed before running tests
try:
    import geneformer  # noqa: F401

    GENEFORMER_AVAILABLE = True
except ImportError:
    GENEFORMER_AVAILABLE = False

from adata_hf_datasets.initial_embedder import (
    GeneformerEmbedder,
    InitialEmbedder,
)  # Adjust the import path as needed

pytestmark = pytest.mark.skipif(
    not GENEFORMER_AVAILABLE,
    reason="Skipping tests because Geneformer is not installed.",
)


def create_small_adata():
    """Create a small synthetic AnnData object for testing."""
    X = np.random.rand(10, 2048)  # Small dataset with 10 cells and 2048 genes
    var_names = [f"Gene{i}" for i in range(2048)]
    obs_names = [f"Cell{i}" for i in range(10)]
    adata = anndata.AnnData(X)
    adata.var_names = var_names
    adata.obs_names = obs_names
    # Save time by already adding ensembl ids
    adata.var["ensembl_id"] = [f"ENSG{i}" for i in range(2048)]
    return adata


@pytest.mark.parametrize("model_input_size, num_layers", [(2048, 12), (4096, 20)])
def test_geneformer_embedder_init(model_input_size, num_layers):
    """Test initializing GeneformerEmbedder with valid model configurations."""
    embedder = GeneformerEmbedder(
        model_input_size=model_input_size, num_layers=num_layers
    )
    assert embedder.model_input_size == model_input_size


@pytest.mark.parametrize("model_input_size, num_layers", [(2048, 8), (4096, 15)])
def test_geneformer_embedder_invalid_layers(model_input_size, num_layers):
    """Test that an error is raised for invalid num_layers."""
    with pytest.raises(
        ValueError, match="Only .* layers are supported for a model dimension of"
    ):
        GeneformerEmbedder(model_input_size=model_input_size, num_layers=num_layers)


@pytest.mark.parametrize("model_input_size", [1024, 3000])
def test_geneformer_embedder_invalid_model_size(model_input_size):
    """Test that an error is raised for unsupported model input sizes."""
    with pytest.raises(
        ValueError, match="Only embedding dimensions of 2048 and 4096 are supported"
    ):
        GeneformerEmbedder(model_input_size=model_input_size)


def test_geneformer_embedder_fit():
    """Test that calling fit() does nothing but maintains compatibility with the interface."""
    embedder = GeneformerEmbedder()
    assert embedder.fit() is None  # Should do nothing


def test_geneformer_embedder_embedding():
    """Test that embed() runs without errors and modifies the input AnnData."""
    import pandas as pd

    embedder = GeneformerEmbedder()
    adata = create_small_adata()
    # The embedding output is expected to be a dataframe with a column sample_index with values ordered from 0 to n_samples

    n_samples = adata.n_obs  # Number of samples
    embedding_dim = 512  # Embedding size

    # Create a mock DataFrame for embeddings
    mock_embeddings = pd.DataFrame(
        np.random.rand(n_samples, embedding_dim),  # Random embeddings
        columns=[f"dim_{i}" for i in range(embedding_dim)],
    )
    mock_embeddings["sample_index"] = np.arange(n_samples)
    # randomly rearrange the order of sample_idx to test if the embeddings are correctly assigned to the samples
    mock_embeddings["sample_index"] = np.random.permutation(
        mock_embeddings["sample_index"]
    )

    with (
        patch("geneformer.TranscriptomeTokenizer.tokenize_data", return_value=None),
        patch("geneformer.EmbExtractor.extract_embs", return_value=mock_embeddings),
    ):
        adata = embedder.embed(adata)

    assert "X_geneformer" in adata.obsm
    assert adata.obsm["X_geneformer"].shape == (10, 512)  # Expected embedding shape
    # ------------------- Order Verification -------------------

    # 1. Extract the original sample_index from mock_embeddings before it's dropped
    original_order = mock_embeddings["sample_index"].values

    # 2. Reorder the rows of the embedding matrix according to original_order
    reordered_embeddings = adata.obsm["X_geneformer"][original_order, :]

    # 3. Assert that the reordered embeddings match the original mock_embeddings (without the sample_index column)
    np.testing.assert_array_equal(reordered_embeddings, mock_embeddings.iloc[:, :-1])


def test_initial_embedder_geneformer():
    """Test that InitialEmbedder correctly initializes GeneformerEmbedder."""
    embedder = InitialEmbedder(method="geneformer")
    assert isinstance(embedder.embedder, GeneformerEmbedder)


def test_initial_embedder_unknown_method():
    """Test that InitialEmbedder raises an error for an unknown method."""
    with pytest.raises(ValueError, match="Unknown embedding method: unknown_method"):
        InitialEmbedder(method="unknown_method")
