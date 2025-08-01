#!/usr/bin/env python3
"""
Script to fetch cell types from the Cell Annotation Platform (CAP).

This script uses the working approach from the notebook to fetch cell types
and saves them to a txt file in the resources folder.
"""

import logging
from typing import List
from pathlib import Path
from cap_client.cap import CapClient as Cap

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cap_human_cell_labels(limit: int = 10000) -> List[str]:
    """Fetch published CAP cell labels for Homo sapiens.

    Parameters
    ----------
    limit : int, default=10000
        Maximum number of labels to retrieve.

    Returns
    -------
    List[str]
        List of unique cell type names from ontology_term field.

    Notes
    -----
    Data source
        Cell Annotation Platform (CAP) GraphQL API via `cap-python-client`.
        CAP publishes HCA community annotations with Cell Ontology IDs, parent
        categories, marker-gene evidence, and downloadable AnnData/CAP-JSON
        files. See CAP docs and client README.
    """
    cap = Cap()  # uses public endpoints; set CAP_LOGIN/CAP_PWD or CAP_TOKEN if needed
    logger.info("Querying CAP for Homo sapiens cell labels")

    resp = cap.search_cell_labels(organism=["Homo sapiens"], limit=limit)
    cell_types = resp["ontology_term"].unique()

    logger.info(f"Found {len(cell_types)} unique cell types")
    return cell_types.tolist()


def save_cell_types_to_file(cell_types: List[str], resources_dir: str = "resources"):
    """
    Save cell types to a txt file in the resources directory.

    Args:
        cell_types: List of cell type names
        resources_dir: Directory to save the file in
    """
    # Create resources directory if it doesn't exist
    resources_path = Path(resources_dir)
    resources_path.mkdir(exist_ok=True)

    # Save to file
    output_file = resources_path / "cell_types.txt"

    # Sort cell types alphabetically and remove any duplicates
    unique_cell_types = sorted(set(cell_types))

    with open(output_file, "w", encoding="utf-8") as f:
        for cell_type in unique_cell_types:
            f.write(f"{cell_type}\n")

    logger.info(f"Saved {len(unique_cell_types)} unique cell types to {output_file}")
    return output_file


def load_cell_types_from_file(resources_dir: str = "resources") -> List[str]:
    """
    Load cell types from the saved txt file.

    Args:
        resources_dir: Directory containing the file

    Returns:
        List of cell type names
    """
    resources_path = Path(resources_dir)
    input_file = resources_path / "cell_types.txt"

    if not input_file.exists():
        logger.error(f"File {input_file} does not exist")
        return []

    with open(input_file, "r", encoding="utf-8") as f:
        cell_types = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(cell_types)} cell types from {input_file}")
    return cell_types


def main():
    """Main function to fetch and save cell types."""

    logger.info("Starting cell type fetching from CAP...")

    try:
        # Fetch cell types using the working approach
        cell_types = cap_human_cell_labels(limit=10000)

        if cell_types:
            # Save to file
            output_file = save_cell_types_to_file(cell_types)

            # Show sample results
            logger.info("Sample cell types:")
            for cell_type in cell_types[:10]:
                logger.info(f"  - {cell_type}")
            if len(cell_types) > 10:
                logger.info(f"  ... and {len(cell_types) - 10} more")

            logger.info(f"\n✅ Successfully saved cell types to {output_file}")
            logger.info(
                "You can now use these cell types with your term description system!"
            )

            # Show usage example
            logger.info(f"\n{'=' * 50}")
            logger.info("USAGE EXAMPLE")
            logger.info(f"{'=' * 50}")
            logger.info("To load and use these cell types:")
            logger.info("""
# Load cell types from file
from scripts.fetch_cell_types import load_cell_types_from_file
cell_types = load_cell_types_from_file()

# Use with your term description system
from src.adata_hf_datasets.config import Config, TermDescriptionConfig
from src.adata_hf_datasets.term_descriptions import gen_term_descriptions

config = Config(data_dir="cell_type_descriptions")
description_config = TermDescriptionConfig(
    config=config,
    email="your.email@institution.edu",
    cell_types=cell_types[:50],  # Use subset for testing
    pull_descriptions=True
)
gen_term_descriptions(description_config)
""")
        else:
            logger.error("Failed to fetch cell types")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
