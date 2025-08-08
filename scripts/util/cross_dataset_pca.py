#!/usr/bin/env python3
"""
Cross-Dataset PCA Fitting for Consistent Dimensionality Reduction

This script fits a single PCA model across multiple datasets to ensure consistent
dimensionality reduction. The key requirement for reusable PCA is that all datasets
must have the EXACT same genes in the EXACT same order.

Key features:
- Loads multiple zarr chunk directories
- Aggregates and concatenates data with proper gene alignment
- Optional gene subsetting to a fixed gene list (in specific order)
- Fits PCA on combined data from all datasets
- Saves PCA model for reuse on new datasets
- Provides utilities for applying saved PCA consistently

Critical PCA Requirements:
- Same genes across all datasets (intersection or predefined list)
- Same gene order (maintained consistently)
- Same number of features (genes)
- Missing genes are filled with zeros to ensure consistent dimensionality
- Proper scaling before PCA fitting

Usage:
    python cross_dataset_pca.py --config pca_config.yaml

Author: Generated for adata_hf_datasets project
"""

import argparse
import os
import sys
import yaml
import logging
import pickle
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.pp.utils import is_data_scaled
from adata_hf_datasets.pp.pybiomart_utils import add_ensembl_ids, ensure_ensembl_index

# Configure scanpy
sc.settings.verbosity = 1


class CrossDatasetPCAFitter:
    """
    A class for fitting PCA across multiple datasets with consistent gene ordering.
    """

    def __init__(self, config_path: str):
        """
        Initialize the CrossDatasetPCAFitter with configuration from YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.combined_adata = None
        self.pca_model = None
        self.scaler = None  # Store scaler if we need to fit one
        self.final_gene_order = None  # Final gene order used
        self.pca_metadata = {}

    def _load_config(self) -> dict:
        """Load and validate configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["dataset_directories", "n_components", "output_dir"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration")

        # Validate dataset configurations
        if not isinstance(config["dataset_directories"], list):
            raise ValueError("'dataset_directories' must be a list")

        if len(config["dataset_directories"]) == 0:
            raise ValueError("'dataset_directories' cannot be empty")

        return config

    def _load_zarr_chunks(self, dataset_dir: Path) -> List[ad.AnnData]:
        """
        Load all zarr chunks from a dataset directory.

        Parameters
        ----------
        dataset_dir : Path
            Directory containing chunk_*.zarr files

        Returns
        -------
        List[ad.AnnData]
            List of loaded chunks
        """
        chunks = []
        zarr_files = sorted(dataset_dir.glob("chunk_*.zarr"))

        if not zarr_files:
            raise FileNotFoundError(f"No chunk_*.zarr files found in {dataset_dir}")

        self.logger.info(f"Loading {len(zarr_files)} zarr chunks from {dataset_dir}")

        for zarr_file in zarr_files:
            try:
                chunk = ad.read_zarr(zarr_file)
                chunks.append(chunk)
                self.logger.debug(f"Loaded {zarr_file}: {chunk.shape}")
            except Exception as e:
                self.logger.error(f"Failed to load {zarr_file}: {e}")
                raise

        return chunks

    def _combine_chunks_within_dataset(
        self, chunks: List[ad.AnnData], dataset_name: str
    ) -> ad.AnnData:
        """
        Combine chunks within a single dataset.

        Parameters
        ----------
        chunks : List[ad.AnnData]
            List of chunks from the same dataset
        dataset_name : str
            Name of the dataset for logging

        Returns
        -------
        ad.AnnData
            Combined dataset
        """
        if len(chunks) == 1:
            combined = chunks[0].copy()
        else:
            self.logger.info(
                f"Concatenating {len(chunks)} chunks for dataset {dataset_name}"
            )
            # Use outer join to keep all genes, fill missing with 0
            combined = ad.concat(chunks, join="outer", fill_value=0)

        self.logger.info(f"Dataset {dataset_name}: {combined.shape}")
        return combined

    def _find_common_genes(self, datasets: List[ad.AnnData]) -> List[str]:
        """
        Find genes common to all datasets.

        Parameters
        ----------
        datasets : List[ad.AnnData]
            List of datasets

        Returns
        -------
        List[str]
            List of common gene names
        """
        if not datasets:
            return []

        # Find intersection of all gene sets
        common_genes = set(datasets[0].var_names)
        for dataset in datasets[1:]:
            common_genes = common_genes.intersection(set(dataset.var_names))

        common_genes = sorted(list(common_genes))
        self.logger.info(f"Found {len(common_genes)} genes common to all datasets")

        return common_genes

    def _subset_to_genes(
        self, datasets: List[ad.AnnData], gene_list: List[str]
    ) -> List[ad.AnnData]:
        """
        Subset all datasets to the same gene list in the same order.
        Missing genes are filled with zeros to ensure consistent dimensionality.

        Parameters
        ----------
        datasets : List[ad.AnnData]
            List of datasets to subset
        gene_list : List[str]
            List of genes in desired order (ALL genes must be included)

        Returns
        -------
        List[ad.AnnData]
            List of subsetted datasets with identical gene sets
        """
        subsetted_datasets = []
        for i, adata in enumerate(datasets):
            # Ensure that all datasets have ensembl ids
            add_ensembl_ids(adata, ensembl_col="ensembl_id")
            ensure_ensembl_index(adata, ensembl_col="ensembl_id")

            # Create a new AnnData object with the exact gene list
            n_cells = adata.n_obs

            # Initialize data matrix with zeros for all genes
            if sp.issparse(adata.X):
                # For sparse data, create dense matrix with zeros
                X_new = np.zeros((n_cells, len(gene_list)), dtype=adata.X.dtype)
            else:
                X_new = np.zeros((n_cells, len(gene_list)), dtype=adata.X.dtype)

            # Create new var DataFrame with the exact gene list
            var_new = pd.DataFrame(index=gene_list)

            # Copy available genes from original dataset
            available_genes = [g for g in gene_list if g in adata.var_names]
            missing_genes = [g for g in gene_list if g not in adata.var_names]

            if len(available_genes) == 0:
                raise ValueError(
                    f"Dataset {i} has no genes from the provided gene list"
                )

            # Copy data for available genes
            for j, gene in enumerate(gene_list):
                if gene in adata.var_names:
                    # Find the column index in the original dataset
                    orig_idx = list(adata.var_names).index(gene)
                    X_new[:, j] = (
                        adata.X[:, orig_idx].toarray().flatten()
                        if sp.issparse(adata.X)
                        else adata.X[:, orig_idx]
                    )

                    # Copy var metadata for this gene
                    if gene in adata.var.columns:
                        for col in adata.var.columns:
                            if col not in var_new.columns:
                                var_new[col] = None
                            var_new.loc[gene, col] = adata.var.loc[gene, col]

            # Create new AnnData object
            adata_subset = ad.AnnData(
                X=X_new,
                obs=adata.obs.copy(),
                var=var_new,
                uns=adata.uns.copy() if adata.uns else {},
            )

            # Log information about missing genes
            if missing_genes:
                self.logger.warning(
                    f"Dataset {i}: {len(missing_genes)} genes missing, filled with zeros"
                )
                self.logger.debug(f"First few missing genes: {missing_genes[:5]}")

            subsetted_datasets.append(adata_subset)
            self.logger.info(
                f"Dataset {i} subsetted to {len(gene_list)} genes (exact order from file)"
            )

        return subsetted_datasets

    def _load_gene_list(self, gene_list_path: str) -> List[str]:
        """
        Load gene list from file.

        Parameters
        ----------
        gene_list_path : str
            Path to file containing gene list (one per line)

        Returns
        -------
        List[str]
            List of gene names
        """
        gene_list_path = Path(gene_list_path)

        if not gene_list_path.exists():
            raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

        with open(gene_list_path, "r") as f:
            genes = [line.strip() for line in f if line.strip()]

        self.logger.info(f"Loaded {len(genes)} genes from {gene_list_path}")
        return genes

    def _combine_datasets(self, datasets: List[ad.AnnData]) -> ad.AnnData:
        """
        Combine multiple datasets into a single AnnData object.

        Parameters
        ----------
        datasets : List[ad.AnnData]
            List of datasets with identical gene sets

        Returns
        -------
        ad.AnnData
            Combined dataset
        """
        if len(datasets) == 1:
            return datasets[0].copy()

        self.logger.info(f"Combining {len(datasets)} datasets")

        # Add dataset labels to track origin
        for i, adata in enumerate(datasets):
            adata.obs["dataset_index"] = f"dataset_{i}"

        # Concatenate along cell axis (axis=0)
        # Use inner join since we want identical gene sets
        combined = ad.concat(datasets, join="inner", fill_value=0)

        self.logger.info(f"Combined dataset shape: {combined.shape}")
        return combined

    def _prepare_data_for_pca(
        self, adata: ad.AnnData
    ) -> Tuple[np.ndarray, Optional[StandardScaler]]:
        """
        Prepare data for PCA fitting.

        Parameters
        ----------
        adata : ad.AnnData
            Input dataset

        Returns
        -------
        Tuple[np.ndarray, Optional[StandardScaler]]
            Prepared data matrix and scaler (if fitted)
        """
        self.logger.info("Preparing data for PCA fitting")

        # Convert to dense if sparse
        if sp.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X.copy()

        # Check if data is already scaled
        if is_data_scaled(X):
            self.logger.info("Data is already scaled, using as-is for PCA")
            return X, None
        else:
            self.logger.info("Data is not scaled, fitting StandardScaler")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return X_scaled, scaler

    def _fit_pca(self, X: np.ndarray) -> PCA:
        """
        Fit PCA model on prepared data.

        Parameters
        ----------
        X : np.ndarray
            Scaled data matrix (n_cells, n_genes)

        Returns
        -------
        PCA
            Fitted PCA model
        """
        n_components = self.config["n_components"]

        # Ensure we don't request more components than possible
        max_components = min(X.shape[0], X.shape[1]) - 1
        if n_components > max_components:
            self.logger.warning(
                f"Reducing n_components from {n_components} to {max_components}"
            )
            n_components = max_components

        self.logger.info(
            f"Fitting PCA with {n_components} components on data shape {X.shape}"
        )

        # Fit PCA
        pca = PCA(
            n_components=n_components, random_state=self.config.get("random_state", 42)
        )
        pca.fit(X)

        # Log explained variance
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)

        self.logger.info(
            f"Explained variance by first 5 components: {explained_var_ratio[:5]}"
        )
        self.logger.info(
            f"Cumulative explained variance (first 10): {cumulative_var[:10]}"
        )
        self.logger.info(f"Total explained variance: {cumulative_var[-1]:.3f}")

        return pca

    def run_fitting(self) -> None:
        """
        Run the complete cross-dataset PCA fitting process.
        """
        self.logger.info("Starting cross-dataset PCA fitting")

        # 1. Load all datasets
        datasets = []
        for i, dataset_dir in enumerate(self.config["dataset_directories"]):
            dataset_path = Path(dataset_dir)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

            # Load zarr chunks for this dataset
            chunks = self._load_zarr_chunks(dataset_path)
            combined = self._combine_chunks_within_dataset(chunks, f"dataset_{i}")
            datasets.append(combined)

        # 2. Determine final gene set
        if "gene_list_file" in self.config and self.config["gene_list_file"]:
            # Use provided gene list - ALL genes from file will be included
            # Missing genes in datasets will be filled with zeros
            gene_list = self._load_gene_list(self.config["gene_list_file"])
            datasets = self._subset_to_genes(datasets, gene_list)
            self.final_gene_order = gene_list  # Use complete gene list from file
        else:
            # Use intersection of all datasets
            common_genes = self._find_common_genes(datasets)
            if len(common_genes) == 0:
                raise ValueError("No genes common to all datasets")
            datasets = self._subset_to_genes(datasets, common_genes)
            self.final_gene_order = common_genes

        # 3. Combine all datasets
        self.combined_adata = self._combine_datasets(datasets)

        # 4. Prepare data and fit PCA
        X_prepared, scaler = self._prepare_data_for_pca(self.combined_adata)
        self.scaler = scaler
        self.pca_model = self._fit_pca(X_prepared)

        # 5. Store metadata
        self.pca_metadata = {
            "n_datasets": len(datasets),
            "total_cells": self.combined_adata.n_obs,
            "n_genes": len(self.final_gene_order),
            "n_components": self.pca_model.n_components_,
            "explained_variance_ratio": self.pca_model.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(
                np.sum(self.pca_model.explained_variance_ratio_)
            ),
            "gene_order": self.final_gene_order,
            "scaling_used": scaler is not None,
        }

        self.logger.info("Cross-dataset PCA fitting completed successfully")

    def save_results(self) -> None:
        """
        Save the PCA model and metadata to files.
        """
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.config.get("output_prefix", "cross_dataset_pca")

        # Save PCA model and scaler
        model_file = output_dir / f"{prefix}_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(
                {
                    "pca_model": self.pca_model,
                    "scaler": self.scaler,
                    "gene_order": self.final_gene_order,
                    "metadata": self.pca_metadata,
                    "config": self.config,
                },
                f,
            )

        # Save gene order as text file
        gene_file = output_dir / f"{prefix}_genes.txt"
        with open(gene_file, "w") as f:
            for gene in self.final_gene_order:
                f.write(f"{gene}\n")

        # Save metadata as JSON
        metadata_file = output_dir / f"{prefix}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.pca_metadata, f, indent=2)

        # Save example usage script
        usage_file = output_dir / f"{prefix}_usage_example.py"
        self._create_usage_example(usage_file, prefix)

        self.logger.info(f"Results saved to {output_dir}")
        self.logger.info(f"PCA model: {model_file}")
        self.logger.info(f"Gene list: {gene_file}")
        self.logger.info(f"Usage example: {usage_file}")

    def _create_usage_example(self, usage_file: Path, prefix: str) -> None:
        """Create an example script showing how to use the saved PCA model."""
        example_code = f'''#!/usr/bin/env python3
"""
Example usage of cross-dataset PCA model

This script demonstrates how to apply the fitted PCA model to new datasets.
"""

import pickle
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

def load_pca_model(model_path: str):
    """Load the saved PCA model and associated data."""
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

def apply_pca_to_dataset(adata: ad.AnnData,
                        pca_model,
                        gene_order: list,
                        scaler=None,
                        obsm_key: str = "X_pca"):
    """
    Apply fitted PCA to a new dataset.

    Parameters
    ----------
    adata : ad.AnnData
        New dataset to transform
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model
    gene_order : list
        Required gene order (ALL genes must be included, missing genes filled with zeros)
    scaler : sklearn.preprocessing.StandardScaler or None
        Fitted scaler (if scaling was used during fitting)
    obsm_key : str
        Key for storing PCA results in adata.obsm

    Returns
    -------
    ad.AnnData
        Dataset with PCA embedding added to obsm
    """

    # 1. Create data matrix with exact gene order (missing genes filled with zeros)
    n_cells = adata.n_obs

    # Initialize data matrix with zeros for all genes
    if sp.issparse(adata.X):
        X = np.zeros((n_cells, len(gene_order)), dtype=adata.X.dtype)
    else:
        X = np.zeros((n_cells, len(gene_order)), dtype=adata.X.dtype)

    # Copy data for available genes in the correct order
    missing_genes = []
    for j, gene in enumerate(gene_order):
        if gene in adata.var_names:
            # Find the column index in the original dataset
            orig_idx = list(adata.var_names).index(gene)
            X[:, j] = adata.X[:, orig_idx].toarray().flatten() if sp.issparse(adata.X) else adata.X[:, orig_idx]
        else:
            missing_genes.append(gene)

    if missing_genes:
        print(f"Warning: {{len(missing_genes)}} genes missing from dataset, filled with zeros")
        print(f"First few missing: {{missing_genes[:10]}}")

    # 2. Apply scaling if needed
    if scaler is not None:
        X = scaler.transform(X)

    # 3. Apply PCA transformation
    X_pca = pca_model.transform(X)

    # 4. Store results
    adata.obsm[obsm_key] = X_pca

    print(f"Applied PCA: {{adata.shape}} -> {{X_pca.shape}}")
    return adata

def main():
    # Load the fitted PCA model
    saved_data = load_pca_model("{prefix}_model.pkl")

    pca_model = saved_data['pca_model']
    scaler = saved_data['scaler']
    gene_order = saved_data['gene_order']
    metadata = saved_data['metadata']

    print("=== PCA Model Info ===")
    print(f"Components: {{metadata['n_components']}}")
    print(f"Genes required: {{metadata['n_genes']}}")
    print(f"Total explained variance: {{metadata['total_explained_variance']:.3f}}")
    print(f"Scaling used: {{metadata['scaling_used']}}")

    # Example: Apply to a new dataset
    # new_adata = ad.read_h5ad("new_dataset.h5ad")
    # new_adata = apply_pca_to_dataset(
    #     new_adata,
    #     pca_model,
    #     gene_order,
    #     scaler
    # )
    #
    # # Save result
    # new_adata.write("new_dataset_with_pca.h5ad")

if __name__ == "__main__":
    main()
'''

        with open(usage_file, "w") as f:
            f.write(example_code)


def main():
    """Main function to run cross-dataset PCA fitting from command line."""
    parser = argparse.ArgumentParser(
        description="Fit PCA across multiple datasets for consistent dimensionality reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fit PCA using configuration file
    python cross_dataset_pca.py --config pca_config.yaml

    # With custom log directory
    python cross_dataset_pca.py --config pca_config.yaml --log-dir /path/to/logs

Configuration file should specify:
    dataset_directories:
      - "path/to/dataset1_chunks/"
      - "path/to/dataset2_chunks/"

    n_components: 50
    gene_list_file: "stable_genes.txt"  # optional
    output_dir: "pca_results"
    output_prefix: "my_pca"
    random_state: 42
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "pca_config_example.yaml"),
        help="Path to YAML configuration file (default: scripts/util/pca_config_example.yaml)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs/)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting cross-dataset PCA fitting")

    try:
        # Initialize and run PCA fitting
        fitter = CrossDatasetPCAFitter(args.config)
        fitter.run_fitting()
        fitter.save_results()

        # Print summary
        logger.info("=== PCA Fitting Summary ===")
        logger.info(f"Datasets processed: {fitter.pca_metadata['n_datasets']}")
        logger.info(f"Total cells: {fitter.pca_metadata['total_cells']}")
        logger.info(f"Genes used: {fitter.pca_metadata['n_genes']}")
        logger.info(f"PCA components: {fitter.pca_metadata['n_components']}")
        logger.info(
            f"Total explained variance: {fitter.pca_metadata['total_explained_variance']:.3f}"
        )
        logger.info(f"Scaling used: {fitter.pca_metadata['scaling_used']}")

        logger.info("Cross-dataset PCA fitting completed successfully!")

    except Exception as e:
        logger.error(f"Cross-dataset PCA fitting failed: {e}")
        raise


if __name__ == "__main__":
    main()
