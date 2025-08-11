#!/usr/bin/env python3
"""
Script to fetch human gene symbols from local dictionary files.

This script reads gene names from two dictionary files in the resources folder:
1. A pickle file: gene_name_id_dict_gc95M.pkl
2. A JSON file: vocab.json

It returns the intersection of gene names (keys) from both dictionaries.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_genes_from_dictionaries(resources_dir: str = "resources") -> List[str]:
    """
    Load gene names from two dictionary files and return their intersection.

    Args:
        resources_dir: Directory containing the dictionary files

    Returns:
        List of gene names present in both dictionaries
    """
    resources_path = Path(resources_dir)

    # File paths
    pkl_file = resources_path / "gene_name_id_dict_gc95M.pkl"
    json_file = resources_path / "vocab.json"

    # Check if files exist
    if not pkl_file.exists():
        logger.error(f"Pickle file not found: {pkl_file}")
        return []

    if not json_file.exists():
        logger.error(f"JSON file not found: {json_file}")
        return []

    # Load pickle dictionary
    logger.info(f"Loading pickle dictionary from {pkl_file}")
    try:
        with open(pkl_file, "rb") as handle:
            pkl_dict = pickle.load(handle)
        pkl_genes = set(pkl_dict.keys())
        logger.info(f"Loaded {len(pkl_genes)} genes from pickle file")
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        return []

    # Load JSON dictionary
    logger.info(f"Loading JSON dictionary from {json_file}")
    try:
        with open(json_file, "r", encoding="utf-8") as handle:
            json_dict = json.load(handle)
        json_genes = set(json_dict.keys())
        logger.info(f"Loaded {len(json_genes)} genes from JSON file")
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return []

    # Get intersection of gene names
    common_genes = pkl_genes.intersection(json_genes)
    logger.info(f"Found {len(common_genes)} genes in common between both dictionaries")

    # Convert to sorted list
    return sorted(list(common_genes))


def fetch_genes_from_local_dicts() -> List[str]:
    """
    Fetch gene symbols from local dictionary files.

    Returns:
        List of gene symbols present in both dictionary files
    """
    logger.info("Fetching gene names from local dictionary files...")
    return load_genes_from_dictionaries()


def save_genes_to_file(
    genes: List[str], filename: str = "genes.txt", resources_dir: str = "resources"
):
    """
    Save gene symbols to a txt file in the resources directory.

    Args:
        genes: List of gene symbols
        filename: Name of the output file
        resources_dir: Directory to save the file in
    """
    # Create resources directory if it doesn't exist
    resources_path = Path(resources_dir)
    resources_path.mkdir(exist_ok=True)

    # Save to file
    output_file = resources_path / filename

    # Sort genes alphabetically and remove any duplicates
    unique_genes = sorted(set(genes))

    with open(output_file, "w", encoding="utf-8") as f:
        for gene in unique_genes:
            f.write(f"{gene}\n")

    logger.info(f"Saved {len(unique_genes)} unique gene symbols to {output_file}")
    return output_file


def load_genes_from_file(
    filename: str = "genes.txt", resources_dir: str = "resources"
) -> List[str]:
    """
    Load gene symbols from the saved txt file.

    Args:
        filename: Name of the file to load
        resources_dir: Directory containing the file

    Returns:
        List of gene symbols
    """
    resources_path = Path(resources_dir)
    input_file = resources_path / filename

    if not input_file.exists():
        logger.error(f"File {input_file} does not exist")
        return []

    with open(input_file, "r", encoding="utf-8") as f:
        genes = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(genes)} gene symbols from {input_file}")
    return genes


def main():
    """Main function to fetch and save gene symbols."""

    logger.info("Starting gene symbol extraction from local dictionary files...")

    try:
        # Fetch genes from the intersection of both dictionaries
        genes = fetch_genes_from_local_dicts()
        filename = "genes.txt"

        if genes:
            # Save to file
            output_file = save_genes_to_file(genes, filename)

            # Show sample results
            logger.info("Sample gene symbols:")
            for gene in genes[:15]:
                logger.info(f"  - {gene}")
            if len(genes) > 15:
                logger.info(f"  ... and {len(genes) - 15} more")

            logger.info(f"\n✅ Successfully saved gene symbols to {output_file}")
            logger.info(
                "You can now use these genes with your term description system!"
            )

            # Show usage example
            logger.info(f"\n{'=' * 50}")
            logger.info("USAGE EXAMPLE")
            logger.info(f"{'=' * 50}")
            logger.info("To load and use these gene symbols:")
            logger.info("""
# Load genes from file
from scripts.fetch_genes import load_genes_from_file
genes = load_genes_from_file()

# Use with your term description system
from src.adata_hf_datasets.config import Config, TermDescriptionConfig
from src.adata_hf_datasets.term_descriptions import gen_term_descriptions

config = Config(data_dir="gene_descriptions")
description_config = TermDescriptionConfig(
    config=config,
    email="your.email@institution.edu",
    genes=genes[:50],  # Use subset for testing
    pull_descriptions=True
)
gen_term_descriptions(description_config)
""")
        else:
            logger.error("Failed to load gene symbols from dictionary files")
            logger.error("Please ensure both files exist in the resources directory:")
            logger.error("  - resources/gene_name_id_dict_gc95M.pkl")
            logger.error("  - resources/vocab.json")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
