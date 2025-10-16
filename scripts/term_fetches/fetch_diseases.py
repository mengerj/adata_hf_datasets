#!/usr/bin/env python3
"""
Script to fetch disease names from MONDO Disease Ontology via EBI OLS4.

This script uses the MONDO ontology through the EBI Ontology Lookup Service v4
to fetch comprehensive disease terms and saves them to a txt file in the resources folder.
"""

import logging
from typing import Any, Dict, List, Optional
import requests
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OLS_BASE = "https://www.ebi.ac.uk/ols4/api"


def fetch_mondo_via_ols(size: int = 500, max_pages: Optional[int] = None) -> List[str]:
    """Fetch MONDO disease terms from OLS4 with labels, definitions, synonyms, xrefs.

    Parameters
    ----------
    size : int, default=500
        Page size for pagination. OLS4 supports paging via ``page`` and ``size``;
        clients commonly use 500.
    max_pages : int, optional
        Stop after this many pages (for testing). If ``None``, iterate all.

    Returns
    -------
    List[str]
        List of unique disease labels from MONDO ontology.

    Notes
    -----
    Data source
        EMBL-EBI **Ontology Lookup Service v4 (OLS4)**, ontology ``mondo``.
        OLS4 exposes MONDO with flattened fields:
        - definition text under ``description``
        - synonyms under ``synonyms``
        - db xrefs under ``annotation.hasDbXref``

        See OLS4 help/app note and discussion on field names. Some endpoints
        occasionally return HTTP 500; retry/backoff is recommended.
    """
    logger.info("Fetching MONDO disease terms from OLS4...")

    page = 0
    rows: List[Dict[str, Any]] = []

    while True:
        if max_pages is not None and page >= max_pages:
            break

        url = f"{OLS_BASE}/ontologies/mondo/terms"
        params = {"size": size, "page": page}

        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
        except requests.HTTPError as e:
            logger.warning("OLS4 request failed on page %d: %s", page, e)
            break

        data = r.json()
        terms = data.get("_embedded", {}).get("terms", []) or []

        for t in terms:
            desc = t.get("description")
            if isinstance(desc, list):
                definition = desc[0] if desc else None
            else:
                definition = desc
            xrefs = (t.get("annotation") or {}).get("hasDbXref") or []
            rows.append(
                {
                    "mondo_id": t.get("obo_id"),
                    "label": t.get("label"),
                    "definition": definition,
                    "synonyms": t.get("synonyms") or [],
                    "xrefs": xrefs,
                }
            )

        logger.debug("Fetched page %d with %d terms", page, len(terms))

        # Check if we've reached the last page
        pinfo = data.get("page", {})
        if page >= (pinfo.get("totalPages", 0) - 1):
            break

        page += 1

        # Progress update
        if page % 10 == 0:
            logger.info(
                f"Processed {page} pages, collected {len(rows)} disease terms so far..."
            )

    # Convert to DataFrame and get unique labels
    df = pd.DataFrame(rows).drop_duplicates(subset=["mondo_id"]).reset_index(drop=True)
    logger.info("Collected %d MONDO terms via OLS4", len(df))

    # Return unique labels as list
    list_labels = list(df["label"].unique())
    logger.info(f"Found {len(list_labels)} unique disease labels")

    return list_labels


def fetch_diseases_subset(max_pages: int = 5) -> List[str]:
    """
    Fetch a smaller subset of diseases for testing.

    Args:
        max_pages: Maximum number of pages to fetch (for testing)

    Returns:
        List of disease names
    """
    logger.info(f"Fetching subset of diseases (max {max_pages} pages)...")
    return fetch_mondo_via_ols(size=500, max_pages=max_pages)


def fetch_all_diseases() -> List[str]:
    """
    Fetch all available diseases from MONDO ontology.

    Returns:
        List of all disease names
    """
    logger.info("Fetching all diseases from MONDO ontology...")
    return fetch_mondo_via_ols()


def save_diseases_to_file(
    diseases: List[str],
    filename: str = "diseases.txt",
    resources_dir: str = "resources",
):
    """
    Save disease names to a txt file in the resources directory.

    Args:
        diseases: List of disease names
        filename: Name of the output file
        resources_dir: Directory to save the file in
    """
    # Create resources directory if it doesn't exist
    resources_path = Path(resources_dir)
    resources_path.mkdir(exist_ok=True)

    # Save to file
    output_file = resources_path / filename

    # Sort diseases alphabetically and remove any duplicates
    unique_diseases = sorted(set(diseases))

    with open(output_file, "w", encoding="utf-8") as f:
        for disease in unique_diseases:
            f.write(f"{disease}\n")

    logger.info(f"Saved {len(unique_diseases)} unique diseases to {output_file}")
    return output_file


def load_diseases_from_file(
    filename: str = "diseases.txt", resources_dir: str = "resources"
) -> List[str]:
    """
    Load disease names from the saved txt file.

    Args:
        filename: Name of the file to load
        resources_dir: Directory containing the file

    Returns:
        List of disease names
    """
    resources_path = Path(resources_dir)
    input_file = resources_path / filename

    if not input_file.exists():
        logger.error(f"File {input_file} does not exist")
        return []

    with open(input_file, "r", encoding="utf-8") as f:
        diseases = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(diseases)} diseases from {input_file}")
    return diseases


def main():
    """Main function to fetch and save disease names."""

    logger.info("Starting disease name fetching from MONDO ontology...")

    try:
        # You can choose which approach to use:

        # Option 1: Fetch a subset for testing (faster, ~2,500 diseases)
        # diseases = fetch_diseases_subset(max_pages=5)
        # filename = "diseases_subset.txt"

        # Option 2: Fetch all diseases (slower but comprehensive, ~20,000+ diseases)
        diseases = fetch_all_diseases()
        filename = "diseases.txt"

        if diseases:
            # Save to file
            output_file = save_diseases_to_file(diseases, filename)

            # Show sample results
            logger.info("Sample disease names:")
            for disease in diseases[:15]:
                logger.info(f"  - {disease}")
            if len(diseases) > 15:
                logger.info(f"  ... and {len(diseases) - 15} more")

            logger.info(f"\n✅ Successfully saved diseases to {output_file}")
            logger.info(
                "You can now use these diseases with your term description system!"
            )

            # Show usage example
            logger.info(f"\n{'=' * 50}")
            logger.info("USAGE EXAMPLE")
            logger.info(f"{'=' * 50}")
            logger.info("To load and use these diseases:")
            logger.info("""
# Load diseases from file
from scripts.fetch_diseases import load_diseases_from_file
diseases = load_diseases_from_file()

# Use with your term description system
from src.adata_hf_datasets.config import Config, TermDescriptionConfig
from src.adata_hf_datasets.term_descriptions import gen_term_descriptions

config = Config(data_dir="disease_descriptions")
description_config = TermDescriptionConfig(
    config=config,
    email="your.email@institution.edu",
    diseases=diseases[:50],  # Use subset for testing
    pull_descriptions=True
)
gen_term_descriptions(description_config)
""")
        else:
            logger.error("Failed to fetch diseases")

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
