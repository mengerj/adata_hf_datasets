#!/usr/bin/env python
"""
Process test AnnData files to create and push Hugging Face datasets for each file.

Each test file is processed with the InitialEmbedder methods using a batch key specified
in a JSON file (batch_keys.json) stored in the same directory as the test data. The resulting
datasets are created in "single" mode and pushed to separate repositories.

Data Sources:
    - Test files: All .h5ad files in the designated test folder.

JSON Mapping:
    - A JSON file (batch_keys.json) in the test directory maps each file name to its batch key.
      Example content:
          {
              "human_pancreas_norm_complexBatch.h5ad": "tech",
              "sim1_1_norm.h5ad": "Batch",
              "Immune_ALL_hum_mou.h5ad": "batch",
              "sim2_norm.h5ad": "Batch",
              "Immune_ALL_human.h5ad": "batch",
              "bowel_disease.h5ad": "batch"
          }

References:
    - anndata: https://anndata.readthedocs.io
    - Hugging Face datasets: https://huggingface.co/docs/datasets
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import anndata
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.adata_ref_ds import AnnDataSetConstructor
from adata_hf_datasets.utils import setup_logging, annotate_and_push_dataset
import logging

# Use the predefined logger per instructions
logger = logging.getLogger(__name__)


def process_test_file(
    file_path, batch_key, methods, nextcloud_config, negatives_per_sample
):
    """
    Process a test AnnData file: apply embeddings using the provided batch_key,
    and create a Hugging Face dataset of type "single".

    Parameters
    ----------
    file_path : str or Path
        Path to the test AnnData (.h5ad) file.
    batch_key : str
        The key in the AnnData object to use for batch correction.
    methods : list of str
        List of embedding methods to apply.
    nextcloud_config : dict
        Configuration dictionary for Nextcloud storage.
    negatives_per_sample : int
        Number of negative samples to generate for each positive sample.

    Returns
    -------
    test_name : str
        The stem (file name without extension) of the test file.
    test_dataset : datasets.Dataset
        The constructed Hugging Face dataset for the test file.
    """
    logger.info("Processing test file: %s with batch key: %s", file_path, batch_key)
    adata = anndata.read_h5ad(file_path)

    # Remove unnecessary fields to free up memory
    if "natural_language_annotation_replicates" in adata.obsm:
        del adata.obsm["natural_language_annotation_replicates"]
    if hasattr(adata, "layers"):
        del adata.layers
    adata.layers = {"counts": adata.X.copy()}

    # Apply each embedding method using the provided batch key
    for method in methods:
        logger.info("Applying embedding method '%s' on test file %s", method, file_path)
        embedder = InitialEmbedder(method=method)
        embedder.fit(adata, batch_key=batch_key)
        adata = embedder.embed(adata)
        # The embedder is assumed to store its embedding in adata.obsm (e.g., adata.obsm[f'X_{method}'])

    # Save the processed test file to disk
    project_dir = Path(__file__).resolve().parents[1]
    test_processed_dir = project_dir / "data" / "RNA" / "processed" / "test"
    os.makedirs(test_processed_dir, exist_ok=True)
    test_processed_path = str(test_processed_dir / f"{Path(file_path).stem}.h5ad")
    adata.write_h5ad(test_processed_path)
    logger.info("Saved processed test file to: %s", test_processed_path)

    # Create the dataset for the test file (always in "single" mode)
    nextcloud_config["remote_path"] = f"datasets/test/{Path(file_path).stem}.h5ad"
    constructor = AnnDataSetConstructor(
        store_nextcloud=True,
        nextcloud_config=nextcloud_config,
        negatives_per_sample=negatives_per_sample,
        dataset_format="single",
    )
    constructor.add_anndata(file_path=test_processed_path)
    test_dataset = constructor.get_dataset()
    return Path(file_path).stem, test_dataset


def main():
    """
    Main function to process all test files in the test directory and push each as a separate repository.

    The function loads the batch key mapping from a JSON file (batch_keys.json) located in the test directory.
    It then loops over all .h5ad files, processes each file using its specified batch key, and pushes
    the resulting dataset to its respective repository.
    """
    setup_logging()
    load_dotenv(override=True)

    # Define directories and parameters
    project_dir = Path(__file__).resolve().parents[1]
    test_dir = project_dir / "data" / "RNA" / "raw" / "test"
    processed_test_dir = project_dir / "data" / "RNA" / "processed" / "test"
    os.makedirs(processed_test_dir, exist_ok=True)

    # Path to JSON file containing batch key mappings
    batch_keys_json = test_dir / "batch_keys.json"
    if not batch_keys_json.exists():
        logger.error("Batch keys JSON file not found at: %s", batch_keys_json)
        return

    with open(batch_keys_json, "r") as f:
        batch_keys_mapping = json.load(f)

    methods = ["hvg", "pca", "scvi", "geneformer"]
    # caption_key = "natural_language_annotation"
    negatives_per_sample = 2

    nextcloud_config = {
        "url": "https://nxc-fredato.imbi.uni-freiburg.de",
        "username": "NEXTCLOUD_USER",  # To be obtained from environment variables in practice
        "password": "NEXTCLOUD_PASSWORD",
        "remote_path": "",
    }

    # Process only .h5ad files in the test directory
    test_files = sorted(test_dir.glob("*.h5ad"))
    if not test_files:
        logger.error("No .h5ad test files found in: %s", test_dir)
        return

    for file_path in test_files:
        file_name = file_path.name
        # Get the batch key for this file from the JSON mapping.
        # If not provided, you can choose a default (here, "tech")
        batch_key = batch_keys_mapping.get(file_name, "batch")
        test_name, test_dataset = process_test_file(
            file_path=file_path,
            batch_key=batch_key,
            methods=methods,
            nextcloud_config=nextcloud_config,
            negatives_per_sample=negatives_per_sample,
        )
        logger.info(
            "Processed test file '%s' with batch key '%s'", file_name, batch_key
        )

        embedding_generation = (
            f"Embeddings were generated with the InitialEmbedder class for methods: {methods}. "
            "Each method stored its embeddings in the corresponding adata.obsm key."
        )
        dataset_type_explanation = (
            f"Test dataset for file: {test_name} (dataset_type 'single')."
        )

        # Push the dataset for this test file to a separate repository
        repo_id = f"jo-mengr/{test_name}_single"
        annotate_and_push_dataset(
            dataset=test_dataset,
            embedding_generation=embedding_generation,
            dataset_type_explanation=dataset_type_explanation,
            repo_id=repo_id,
            readme_template_name="test_data",
        )
        logger.info(
            "Test dataset '%s' pushed successfully to repo: %s", test_name, repo_id
        )


if __name__ == "__main__":
    main()
