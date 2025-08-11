#!/usr/bin/env python3
"""
Stable Highly Variable Gene (HVG) Selection Across Multiple Datasets

This script performs batch-aware highly variable gene selection across multiple datasets
to create a stable, reproducible gene set that can be reused across different studies.

The approach:
1. Load multiple preprocessed datasets (h5ad or zarr format)
2. Run batch-aware HVG selection on each dataset individually
3. Find the intersection of HVGs across all datasets
4. If intersection is smaller than desired, fill with top-ranked genes from individual datasets
5. Save a stable gene set that can be reused for consistent analysis

Key benefits:
- Reproducible gene selection across studies
- Reduces batch effects between datasets
- Maintains biological relevance through cross-dataset validation
- Creates consistent feature space for downstream analysis

Usage:
    python stable_hvg_selection.py --config stable_hvg_config.yaml

Configuration file should specify:
- Dataset paths and their batch keys
- Gene name columns (e.g., Ensembl IDs)
- Number of genes to select per dataset and final count
- Output directory and file naming

Author: Generated for adata_hf_datasets project
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import logging
from typing import List, Tuple, Optional
import pickle
import json

import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc

# Add the src directory to the path to import utility functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.pp.utils import ensure_log_norm, check_enough_genes_per_batch

# Configure scanpy
sc.settings.verbosity = 1  # Reduce verbosity for cleaner output


class StableHVGSelector:
    """
    A class for performing stable highly variable gene selection across multiple datasets.
    """

    def __init__(self, config_path: str):
        """
        Initialize the StableHVGSelector with configuration from YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)

        # Results storage
        self.dataset_hvgs = {}  # Store HVG results for each dataset
        self.gene_rankings = {}  # Store gene rankings for each dataset
        self.final_gene_set = None  # Final stable gene set
        self.gene_metadata = {}  # Metadata about gene selection

    def _load_config(self) -> dict:
        """Load and validate configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = [
            "datasets",
            "n_genes_per_dataset",
            "final_gene_count",
            "output_dir",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in configuration")

        # Validate dataset configurations
        for i, dataset_config in enumerate(config["datasets"]):
            required_dataset_fields = ["path", "batch_key", "gene_name_column"]
            for field in required_dataset_fields:
                if field not in dataset_config:
                    raise ValueError(f"Missing required field '{field}' in dataset {i}")

        return config

    def _load_dataset(self, dataset_config: dict) -> ad.AnnData:
        """
        Load a single dataset from h5ad or zarr format.

        Parameters
        ----------
        dataset_config : dict
            Configuration for a single dataset

        Returns
        -------
        ad.AnnData
            Loaded AnnData object
        """
        path = Path(dataset_config["path"])

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.logger.info(f"Loading dataset: {path}")

        if path.suffix == ".h5ad":
            adata = ad.read_h5ad(path)
        elif path.suffix == ".zarr" or path.is_dir():
            adata = ad.read_zarr(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.logger.info(
            f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes"
        )
        return adata

    def _prepare_dataset_for_hvg(
        self, adata: ad.AnnData, dataset_config: dict
    ) -> ad.AnnData:
        """
        Prepare dataset for HVG selection by ensuring proper normalization and gene naming.

        Parameters
        ----------
        adata : ad.AnnData
            Input dataset
        dataset_config : dict
            Dataset configuration

        Returns
        -------
        ad.AnnData
            Prepared dataset
        """
        # Make a copy to avoid modifying the original
        adata = adata.copy()

        # Ensure we have the gene name column
        gene_name_col = dataset_config["gene_name_column"]
        if gene_name_col not in adata.var.columns:
            raise ValueError(
                f"Gene name column '{gene_name_col}' not found in dataset {dataset_config['path']}"
            )

        # Set gene names as index if requested
        if gene_name_col != adata.var.index.name and gene_name_col in adata.var.columns:
            # Store original var names
            adata.var["original_var_names"] = adata.var.index
            # Set new index
            adata.var.index = adata.var[gene_name_col].astype(str)
            adata.var.index.name = gene_name_col

        # Remove genes with empty or NaN names
        valid_genes = ~(
            adata.var.index.isna()
            | (adata.var.index == "")
            | (adata.var.index == "nan")
        )
        adata = adata[:, valid_genes]

        # Make gene names unique
        adata.var_names_make_unique()

        # Ensure data is properly normalized and log-transformed for HVG selection
        self.logger.info("Ensuring proper normalization for HVG selection...")
        ensure_log_norm(adata)

        return adata

    def _perform_hvg_selection(
        self, adata: ad.AnnData, dataset_config: dict
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Perform batch-aware highly variable gene selection on a single dataset.

        Parameters
        ----------
        adata : ad.AnnData
            Prepared dataset
        dataset_config : dict
            Dataset configuration

        Returns
        -------
        Tuple[List[str], pd.DataFrame]
            List of HVG names and DataFrame with gene statistics
        """
        batch_key = dataset_config["batch_key"]
        n_genes = self.config["n_genes_per_dataset"]

        self.logger.info(
            f"Performing HVG selection with batch_key='{batch_key}', n_genes={n_genes}"
        )

        # Check if batch key exists
        if batch_key not in adata.obs.columns:
            self.logger.warning(
                f"Batch key '{batch_key}' not found. Performing HVG selection without batch correction."
            )
            batch_key = None

        # Perform batch-aware quality checks if batch_key is available
        if batch_key is not None:
            try:
                adata = check_enough_genes_per_batch(adata, batch_key, n_genes)
            except Exception as e:
                self.logger.warning(
                    f"Batch quality check failed: {e}. Proceeding with available genes."
                )

        # Perform HVG selection
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_genes, batch_key=batch_key, inplace=True
            )
        except Exception as e:
            self.logger.error(f"HVG selection failed: {e}")
            raise

        # Extract HVG information
        if "highly_variable" not in adata.var.columns:
            raise ValueError("HVG selection failed - no 'highly_variable' column found")

        hvg_genes = adata.var[adata.var["highly_variable"]].index.tolist()

        # Create gene statistics DataFrame
        gene_stats = adata.var[["highly_variable"]].copy()
        if "dispersions_norm" in adata.var.columns:
            gene_stats["dispersions_norm"] = adata.var["dispersions_norm"]
        if "variances_norm" in adata.var.columns:
            gene_stats["variances_norm"] = adata.var["variances_norm"]

        # Add ranking information
        if "dispersions_norm" in gene_stats.columns:
            gene_stats["rank_dispersions"] = gene_stats["dispersions_norm"].rank(
                ascending=False, method="min"
            )
        if "variances_norm" in gene_stats.columns:
            gene_stats["rank_variances"] = gene_stats["variances_norm"].rank(
                ascending=False, method="min"
            )

        self.logger.info(f"Selected {len(hvg_genes)} highly variable genes")

        return hvg_genes, gene_stats

    def _find_gene_intersection_and_fill(self) -> List[str]:
        """
        Find intersection of HVGs across all datasets and fill to desired count.

        Returns
        -------
        List[str]
            Final stable gene set
        """
        final_count = self.config["final_gene_count"]

        # Find intersection of all HVG sets
        all_hvg_sets = [set(hvgs) for hvgs in self.dataset_hvgs.values()]
        intersection = set.intersection(*all_hvg_sets) if all_hvg_sets else set()

        self.logger.info(
            f"Found {len(intersection)} genes in intersection of all datasets"
        )

        if len(intersection) >= final_count:
            # If we have enough genes in intersection, prioritize by average ranking
            intersection_list = list(intersection)

            # Calculate average ranking across datasets for intersection genes
            gene_avg_ranks = {}
            for gene in intersection_list:
                ranks = []
                for dataset_name, gene_stats in self.gene_rankings.items():
                    if gene in gene_stats.index:
                        # Use dispersions_norm rank if available, else variances_norm rank
                        if "rank_dispersions" in gene_stats.columns:
                            ranks.append(gene_stats.loc[gene, "rank_dispersions"])
                        elif "rank_variances" in gene_stats.columns:
                            ranks.append(gene_stats.loc[gene, "rank_variances"])

                if ranks:
                    gene_avg_ranks[gene] = np.mean(ranks)

            # Sort by average rank and take top genes
            sorted_intersection = sorted(
                gene_avg_ranks.keys(), key=lambda x: gene_avg_ranks[x]
            )
            final_genes = sorted_intersection[:final_count]

            self.logger.info(
                f"Selected top {len(final_genes)} genes from intersection based on average ranking"
            )

        else:
            # Need to fill up with additional genes
            final_genes = list(intersection)
            needed = final_count - len(intersection)

            self.logger.info(
                f"Intersection has only {len(intersection)} genes. Need {needed} more genes."
            )

            # Collect all non-intersection HVGs with dataset counts and average ranks
            candidate_genes = {}

            for dataset_name, hvg_list in self.dataset_hvgs.items():
                gene_stats = self.gene_rankings[dataset_name]

                for gene in hvg_list:
                    if gene not in intersection:
                        if gene not in candidate_genes:
                            candidate_genes[gene] = {
                                "dataset_count": 0,
                                "ranks": [],
                                "datasets": [],
                            }

                        candidate_genes[gene]["dataset_count"] += 1
                        candidate_genes[gene]["datasets"].append(dataset_name)

                        # Add rank information
                        if gene in gene_stats.index:
                            if "rank_dispersions" in gene_stats.columns:
                                candidate_genes[gene]["ranks"].append(
                                    gene_stats.loc[gene, "rank_dispersions"]
                                )
                            elif "rank_variances" in gene_stats.columns:
                                candidate_genes[gene]["ranks"].append(
                                    gene_stats.loc[gene, "rank_variances"]
                                )

            # Calculate average ranks for candidates
            for gene, info in candidate_genes.items():
                if info["ranks"]:
                    info["avg_rank"] = np.mean(info["ranks"])
                else:
                    info["avg_rank"] = float("inf")

            # Sort candidates by: 1) dataset count (descending), 2) average rank (ascending)
            sorted_candidates = sorted(
                candidate_genes.items(),
                key=lambda x: (-x[1]["dataset_count"], x[1]["avg_rank"]),
            )

            # Add top candidates to final gene set
            additional_genes = [gene for gene, _ in sorted_candidates[:needed]]
            final_genes.extend(additional_genes)

            self.logger.info(
                f"Added {len(additional_genes)} additional genes based on cross-dataset ranking"
            )

        # Store metadata about gene selection
        self.gene_metadata = {
            "total_datasets": len(self.dataset_hvgs),
            "intersection_size": len(intersection),
            "final_gene_count": len(final_genes),
            "genes_from_intersection": len(
                [g for g in final_genes if g in intersection]
            ),
            "genes_from_candidates": len(final_genes)
            - len([g for g in final_genes if g in intersection]),
        }

        return final_genes

    def run_selection(self) -> None:
        """
        Run the complete stable HVG selection process.
        """
        self.logger.info("Starting stable HVG selection process")

        # Process each dataset
        for i, dataset_config in enumerate(self.config["datasets"]):
            dataset_name = dataset_config.get("name", f"dataset_{i}")

            try:
                # Load and prepare dataset
                adata = self._load_dataset(dataset_config)
                adata = self._prepare_dataset_for_hvg(adata, dataset_config)

                # Perform HVG selection
                hvg_genes, gene_stats = self._perform_hvg_selection(
                    adata, dataset_config
                )

                # Store results
                self.dataset_hvgs[dataset_name] = hvg_genes
                self.gene_rankings[dataset_name] = gene_stats

                self.logger.info(f"Completed HVG selection for {dataset_name}")

            except Exception as e:
                self.logger.error(f"Failed to process dataset {dataset_name}: {e}")
                raise

        # Find intersection and create final gene set
        self.final_gene_set = self._find_gene_intersection_and_fill()

        self.logger.info("Stable HVG selection completed successfully")

    def save_results(self) -> None:
        """
        Save the stable gene set and metadata to files.
        """
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.config.get("output_prefix", "stable_hvg")

        # Save final gene set as text file (one gene per line)
        gene_list_file = output_dir / f"{prefix}_gene_list.txt"
        with open(gene_list_file, "w") as f:
            for gene in self.final_gene_set:
                f.write(f"{gene}\n")

        # Save as Python pickle for easy loading
        pickle_file = output_dir / f"{prefix}_gene_set.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(
                {
                    "genes": self.final_gene_set,
                    "metadata": self.gene_metadata,
                    "config": self.config,
                    "dataset_hvgs": self.dataset_hvgs,
                    "gene_rankings": self.gene_rankings,
                },
                f,
            )

        # Save metadata as JSON
        metadata_file = output_dir / f"{prefix}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(self.gene_metadata, f, indent=2)

        # Save detailed results for each dataset
        for dataset_name, hvg_genes in self.dataset_hvgs.items():
            dataset_file = output_dir / f"{prefix}_{dataset_name}_hvgs.txt"
            with open(dataset_file, "w") as f:
                for gene in hvg_genes:
                    f.write(f"{gene}\n")

        # Save gene rankings
        rankings_file = output_dir / f"{prefix}_gene_rankings.xlsx"
        with pd.ExcelWriter(rankings_file) as writer:
            for dataset_name, rankings in self.gene_rankings.items():
                rankings.to_excel(
                    writer, sheet_name=dataset_name[:31]
                )  # Excel sheet name limit

        self.logger.info(f"Results saved to {output_dir}")
        self.logger.info(f"Final gene set: {gene_list_file}")
        self.logger.info(f"Reusable gene set object: {pickle_file}")

    def create_subsetting_function(self) -> callable:
        """
        Create a function that can be used to subset new datasets with the stable gene set.

        Returns
        -------
        callable
            Function that takes an AnnData object and returns it subsetted to stable genes
        """
        stable_genes = self.final_gene_set.copy()

        def subset_to_stable_genes(
            adata: ad.AnnData, gene_name_column: Optional[str] = None
        ) -> ad.AnnData:
            """
            Subset an AnnData object to the stable gene set.

            Parameters
            ----------
            adata : ad.AnnData
                Input dataset to subset
            gene_name_column : str, optional
                Column in adata.var containing gene names matching the stable set

            Returns
            -------
            ad.AnnData
                Subsetted dataset with genes in the same order as stable gene set
            """
            if gene_name_column and gene_name_column in adata.var.columns:
                # Use specified gene name column
                gene_mask = adata.var[gene_name_column].isin(stable_genes)
                subset_adata = adata[:, gene_mask].copy()

                # Reorder to match stable gene set order
                gene_order = {gene: i for i, gene in enumerate(stable_genes)}
                subset_adata.var["_stable_order"] = subset_adata.var[
                    gene_name_column
                ].map(gene_order)
                subset_adata = subset_adata[
                    :, subset_adata.var["_stable_order"].argsort()
                ]
                subset_adata.var.drop("_stable_order", axis=1, inplace=True)

            else:
                # Use var.index
                available_genes = [g for g in stable_genes if g in adata.var.index]
                subset_adata = adata[:, available_genes].copy()

            return subset_adata

        return subset_to_stable_genes


def main():
    """Main function to run stable HVG selection from command line."""
    parser = argparse.ArgumentParser(
        description="Perform stable highly variable gene selection across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with configuration file
    python stable_hvg_selection.py --config stable_hvg_config.yaml

    # Run with custom log directory
    python stable_hvg_selection.py --config my_config.yaml --log-dir /path/to/logs

The configuration file should be in YAML format with the following structure:
    datasets:
      - path: "path/to/dataset1.h5ad"
        name: "dataset1"  # optional
        batch_key: "batch"
        gene_name_column: "ensembl_id"
      - path: "path/to/dataset2.zarr"
        name: "dataset2"  # optional
        batch_key: "study_id"
        gene_name_column: "gene_ids"

    n_genes_per_dataset: 2000
    final_gene_count: 1500
    output_dir: "stable_hvg_results"
    output_prefix: "stable_hvg"
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="/Users/mengerj/repos/adata_hf_datasets/scripts/util/hvg_selection.yaml",
        help="Path to YAML configuration file (default: ../../conf/hvg_selection.yaml)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting stable HVG selection")

    try:
        # Initialize and run selector
        selector = StableHVGSelector(args.config)
        selector.run_selection()
        selector.save_results()

        # Print summary
        logger.info("=== Selection Summary ===")
        logger.info(
            f"Total datasets processed: {selector.gene_metadata['total_datasets']}"
        )
        logger.info(
            f"Genes in intersection: {selector.gene_metadata['intersection_size']}"
        )
        logger.info(
            f"Final stable gene count: {selector.gene_metadata['final_gene_count']}"
        )
        logger.info(
            f"Genes from intersection: {selector.gene_metadata['genes_from_intersection']}"
        )
        logger.info(
            f"Additional genes added: {selector.gene_metadata['genes_from_candidates']}"
        )

        logger.info("Stable HVG selection completed successfully!")

    except Exception as e:
        logger.error(f"Stable HVG selection failed: {e}")
        raise


if __name__ == "__main__":
    main()
