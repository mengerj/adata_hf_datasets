#!/usr/bin/env python3
"""
Test the new PCA embedder with pre-trained model

This test verifies that the new PCAEmbedder correctly:
- Loads the pre-trained model and gene list
- Subsets data to the required genes in correct order
- Applies the PCA transformation
- Handles missing genes by filling with zeros
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
import os

# Add the src directory to the path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adata_hf_datasets.embed.initial_embedder import PCAEmbedder


class TestNewPCAEmbedder:
    """Test the new PCA embedder functionality."""

    def setup_method(self):
        """Set up test data."""
        # Create test gene list (subset of the actual 8k genes)
        self.test_genes = [
            "ENSG00000233576",  # From the actual gene list
            "ENSG00000268895",  # From the actual gene list
            "ENSG00000175899",  # From the actual gene list
            "ENSG00000166535",  # From the actual gene list
            "ENSG00000114771",  # From the actual gene list
        ]

        # Create test datasets with different gene sets
        self.dataset1_genes = [
            "ENSG00000233576",
            "ENSG00000268895",
            "ENSG00000175899",
        ]  # First 3
        self.dataset2_genes = [
            "ENSG00000166535",
            "ENSG00000114771",
            "ENSG00000233576",
        ]  # Last 3 + first

        # Create test data
        self.dataset1 = self._create_test_adata(self.dataset1_genes, 100, "dataset1")
        self.dataset2 = self._create_test_adata(self.dataset2_genes, 150, "dataset2")

    def _create_test_adata(self, genes, n_cells, dataset_name):
        """Create a test AnnData object."""
        # Create random expression data
        X = np.random.poisson(5, size=(n_cells, len(genes))).astype(np.float32)

        # Create var DataFrame
        var_df = pd.DataFrame(index=genes)
        var_df["gene_name"] = genes
        var_df["ensembl_id"] = genes

        # Create obs DataFrame
        obs_df = pd.DataFrame(
            index=[f"{dataset_name}_cell_{i}" for i in range(n_cells)]
        )
        obs_df["dataset"] = dataset_name

        return ad.AnnData(X=X, obs=obs_df, var=var_df)

    def test_pca_embedder_initialization(self):
        """Test that the PCA embedder initializes correctly."""
        # Test with default paths
        embedder = PCAEmbedder(embedding_dim=50)
        assert embedder.embedding_dim == 50
        assert "cellxgene_geo_pca_10000_to_50.pkl" in embedder.model_path
        assert "gene_selection_10k.txt" in embedder.gene_list_path

        # Test with custom resources_dir
        custom_resources_dir = "custom_resources"
        embedder = PCAEmbedder(
            embedding_dim=50,
            resources_dir=custom_resources_dir,
        )
        assert custom_resources_dir in embedder.model_path
        assert custom_resources_dir in embedder.gene_list_path

        # Test with custom file names
        custom_model_file = "custom_model.pkl"
        custom_gene_list_file = "custom_genes.txt"
        embedder = PCAEmbedder(
            embedding_dim=50,
            model_file=custom_model_file,
            gene_list_file=custom_gene_list_file,
        )
        assert custom_model_file in embedder.model_path
        assert custom_gene_list_file in embedder.gene_list_path

        # Test with both resources_dir and file names
        embedder = PCAEmbedder(
            embedding_dim=50,
            resources_dir=custom_resources_dir,
            model_file=custom_model_file,
            gene_list_file=custom_gene_list_file,
        )
        assert embedder.model_path == f"{custom_resources_dir}/{custom_model_file}"
        assert (
            embedder.gene_list_path == f"{custom_resources_dir}/{custom_gene_list_file}"
        )

    def test_model_loading(self):
        """Test that the model loads correctly."""
        embedder = PCAEmbedder(embedding_dim=50)

        # Test prepare method
        embedder.prepare()

        # Check that model components are loaded
        assert embedder.pca_model is not None
        assert embedder.gene_order is not None
        assert len(embedder.gene_order) > 0
        assert embedder.pca_model.n_components_ == 50

        print("✓ Model loaded successfully")
        print(f"  - PCA components: {embedder.pca_model.n_components_}")
        print(f"  - Gene order length: {len(embedder.gene_order)}")
        print(f"  - First 5 genes: {embedder.gene_order[:5]}")

    def test_gene_subsetting(self):
        """Test that genes are subsetted correctly."""
        embedder = PCAEmbedder(embedding_dim=50)
        embedder.prepare()

        # Test with dataset that has some matching genes
        subsetted = embedder._subset_and_order_genes(self.dataset1)

        # Check that the subsetted dataset has the correct gene order
        assert list(subsetted.var_names) == embedder.gene_order
        assert subsetted.n_vars == len(embedder.gene_order)

        # Check that available genes have data, missing genes are zeros
        for j, gene in enumerate(embedder.gene_order):
            if gene in self.dataset1.var_names:
                # Gene should have original data
                assert not np.allclose(subsetted.X[:, j], 0), (
                    f"Gene {gene} is all zeros but should have data"
                )
            else:
                # Gene should be all zeros
                assert np.allclose(subsetted.X[:, j], 0), (
                    f"Gene {gene} is not all zeros but should be"
                )

        print("✓ Gene subsetting works correctly")

    def test_pca_application(self):
        """Test that PCA is applied correctly."""
        embedder = PCAEmbedder(embedding_dim=50)
        embedder.prepare()

        # Apply PCA to test dataset
        embedding = embedder.embed(self.dataset1, obsm_key="X_pca_test")

        # Check embedding dimensions
        assert embedding.shape[0] == self.dataset1.n_obs  # Same number of cells
        assert embedding.shape[1] == 50  # 50 PCA components

        # Check that results are stored in adata
        assert "X_pca_test" in self.dataset1.obsm
        assert self.dataset1.obsm["X_pca_test"].shape == embedding.shape

        print("✓ PCA application works correctly")
        print(f"  - Input shape: {self.dataset1.shape}")
        print(f"  - Output shape: {embedding.shape}")

    def test_missing_genes_handling(self):
        """Test that missing genes are handled correctly."""
        embedder = PCAEmbedder(embedding_dim=50)
        embedder.prepare()

        # Create dataset with no matching genes (should fail)
        no_matching_genes = ["GENE_NOT_IN_LIST_1", "GENE_NOT_IN_LIST_2"]
        dataset_no_match = self._create_test_adata(no_matching_genes, 50, "no_match")

        with pytest.raises(
            ValueError, match="Dataset has no genes from the required gene set"
        ):
            embedder._subset_and_order_genes(dataset_no_match)

        print("✓ Missing genes handling works correctly")

    def test_embedding_dimension_validation(self):
        """Test that embedding dimension validation works."""
        # Test with wrong embedding dimension
        embedder = PCAEmbedder(embedding_dim=100)  # Wrong dimension
        embedder.prepare()

        # Should automatically adjust to model's dimension
        assert embedder.embedding_dim == 50  # Should be adjusted to model's dimension

        print("✓ Embedding dimension validation works correctly")


if __name__ == "__main__":
    # Run tests
    test_instance = TestNewPCAEmbedder()
    test_instance.setup_method()

    print("Running new PCA embedder tests...")
    test_instance.test_pca_embedder_initialization()
    test_instance.test_model_loading()
    test_instance.test_gene_subsetting()
    test_instance.test_pca_application()
    test_instance.test_missing_genes_handling()
    test_instance.test_embedding_dimension_validation()
    print("All tests passed!")
