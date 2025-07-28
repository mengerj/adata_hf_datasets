#!/usr/bin/env python3
"""
Script to fetch human gene symbols from HGNC (HUGO Gene Nomenclature Committee).

This script uses the HGNC complete set to fetch approved human gene symbols
and saves them to a txt file in the resources folder.
"""

import io
import logging
from typing import Iterable, List, Optional
import pandas as pd
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

HGNC_TSV = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"


def load_hgnc_symbols(locus_types: Optional[Iterable[str]] = None) -> List[str]:
    """Return HGNC approved human gene symbols from the current *hgnc_complete_set*.

    Parameters
    ----------
    locus_types : iterable of str, optional
        Filter by HGNC ``locus_type`` values, e.g. ``["gene with protein product"]``,
        ``["RNA, long non-coding"]``. If ``None``, return all approved symbols.

    Returns
    -------
    list of str
        Unique HGNC-approved symbols.

    Notes
    -----
    Data source
        HGNC "hgnc_complete_set" TSV hosted on Google Cloud Storage.
        Files are updated on Tuesdays and Fridays. See HGNC downloads page
        (new bucket paths) and archive/help.
    """
    logger.info("Downloading HGNC complete set from %s", HGNC_TSV)
    r = requests.get(HGNC_TSV, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str)
    df = df[df["status"] == "Approved"]
    if locus_types:
        df = df[df["locus_type"].isin(list(locus_types))]
    symbols = df["symbol"].dropna().unique().tolist()
    logger.info("Collected %d HGNC symbols", len(symbols))
    return symbols


def fetch_all_genes() -> List[str]:
    """
    Fetch all approved human gene symbols from HGNC.

    Returns:
        List of all approved gene symbols
    """
    logger.info("Fetching all approved human gene symbols from HGNC...")
    return load_hgnc_symbols()


def fetch_protein_coding_genes() -> List[str]:
    """
    Fetch only protein-coding gene symbols from HGNC.

    Returns:
        List of protein-coding gene symbols
    """
    logger.info("Fetching protein-coding gene symbols from HGNC...")
    return load_hgnc_symbols(locus_types=["gene with protein product"])


def fetch_common_gene_types() -> List[str]:
    """
    Fetch commonly used gene types (protein-coding + some RNA types).

    Returns:
        List of gene symbols for common gene types
    """
    logger.info("Fetching common gene types from HGNC...")
    common_types = [
        "gene with protein product",
        "RNA, long non-coding",
        "RNA, micro",
        "RNA, ribosomal",
        "RNA, transfer",
    ]
    return load_hgnc_symbols(locus_types=common_types)


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

    logger.info("Starting gene symbol fetching from HGNC...")

    try:
        # You can choose which gene types to fetch by uncommenting one of these:

        # Option 1: All approved genes (largest set, ~45k genes)
        # genes = fetch_all_genes()
        # filename = "genes_all.txt"

        # Option 2: Only protein-coding genes (most commonly used, ~20k genes)
        genes = fetch_protein_coding_genes()
        filename = "genes.txt"

        # Option 3: Common gene types including some RNA types (~25k genes)
        # genes = fetch_common_gene_types()
        # filename = "genes_common.txt"

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
            logger.error("Failed to fetch gene symbols")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
