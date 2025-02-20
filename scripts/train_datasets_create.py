from adata_hf_datasets.utils import setup_logging
import anndata
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.utils import split_anndata
import os
from pathlib import Path
from dotenv import load_dotenv
from adata_hf_datasets.adata_ref_ds import (
    AnnDataSetConstructor,
    SimpleCaptionConstructor,
)
from adata_hf_datasets.utils import (
    annotate_and_push_dataset,
    remove_zero_variance_cells,
    remove_zero_variance_genes,
)
from adata_hf_datasets.sys_monitor import SystemMonitor
from datasets import DatasetDict, concatenate_datasets
import logging
import argparse

# Define project parameters and paths
project_dir = Path(__file__).resolve().parents[1]

methods = ["scvi"]  # , ["hvg", "pca", "scvi", "geneformer"]
dataset_types = ["pairs", "multiplets"]
negatives_per_sample = 2
batch_keys = {"geo": "study", "cellxgene": "assay"}
caption_key = "natural_language_annotation"
push_to_hub = False

nextcloud_config = {
    "url": "https://nxc-fredato.imbi.uni-freiburg.de",
    "username": "NEXTCLOUD_USER",  # Retrieved from environment variables in practice
    "password": "NEXTCLOUD_PASSWORD",
    "remote_path": "",
}

#!/usr/bin/env python
"""
Process multiple AnnData files, create Hugging Face datasets for each split,
and concatenate them across files.

Data Sources:
    - GEO dataset: sourced from raw files like "geo_7k.h5ad"
    - Cellxgene dataset: sourced from raw files like "cellxgene_pseudo_bulk_1_7K.h5ad"

References:
    - anndata: https://anndata.readthedocs.io
    - Hugging Face datasets: https://huggingface.co/docs/datasets
"""

# Use the predefined logger per instructions
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parses command-line arguments for geo_n and cellxgene_n.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing geo_n and cellxgene_n.
    """
    parser = argparse.ArgumentParser(
        description="Generate training datasets with different dataset sizes."
    )

    parser.add_argument(
        "--geo_n",
        type=str,
        default="7k",
        help="Number of samples to take from the GEO dataset (default: '0_2k')",
    )
    parser.add_argument(
        "--cellxgene_n",
        type=str,
        default="3_5k",
        help="Number of samples to take from the Cellxgene dataset (default: '0_2k')",
    )

    return parser.parse_args()


def process_file_to_dataset(
    file_path,
    methods,
    batch_key,
    caption_key,
    processed_paths,
    nextcloud_config,
    dataset_types,
    negatives_per_sample,
    monitor,
):
    """
    Process a single AnnData file: apply embeddings, split the data, and create Hugging Face datasets.

    Parameters
    ----------
    file_path : str or Path
        Path to the AnnData (.h5ad) file.
    methods : list of str
        List of embedding methods to apply (e.g., ["hvg", "pca", "scvi", "geneformer"]).
    batch_key : str
        Key in AnnData used for batch correction by the embedder.
    caption_key : str
        Observation key used for generating captions.
    processed_paths : dict
        Dictionary with keys "train" and "val" for saving processed AnnData files.
        Example: {"train": "/path/to/train.h5ad", "val": "/path/to/val.h5ad"}
    nextcloud_config : dict
        Configuration dictionary for Nextcloud storage.
    dataset_types : list of str
        List of dataset types to create (e.g., ["pairs", "multiplets", "single"]).
    negatives_per_sample : int
        Number of negative samples to generate for each positive sample.
    monitor : SystemMonitor
        SystemMonitor object to monitor memory usage.

    Returns
    -------
    local_dataset : dict
        Dictionary with keys "train" and "val" containing the Hugging Face datasets for each split.
        Data is ultimately sourced from the file at `file_path`.
    """
    logger.info("Processing file: %s", file_path)
    monitor.log_event(f"Processing file: {file_path}")
    adata = anndata.read_h5ad(file_path)
    # Remove zero variance cells and genes
    adata = remove_zero_variance_cells(adata)
    adata = remove_zero_variance_genes(adata)

    # Clean up unnecessary fields to free up memory
    if "natural_language_annotation_replicates" in adata.obsm:
        del adata.obsm["natural_language_annotation_replicates"]
    if hasattr(adata, "layers"):
        del adata.layers

    # Apply each embedding method; embeddings are stored in adata.obsm
    for method in methods:
        monitor.log_event(f"Applying embedding method '{method}' on {file_path}")
        embedder = InitialEmbedder(method=method)
        embedder.fit(adata, batch_key=batch_key)
        adata = embedder.embed(adata)
        # Log the current amount of memory in GB
        monitor.log_event(f"Embedding completed for method {method}")
        # Each embedder is assumed to store its embedding in adata.obsm (e.g., adata.obsm[f'X_{method}'])

    # Split the data into training and validation sets
    train_adata, val_adata = split_anndata(adata, train_size=0.9)
    del adata  # Free up memory
    # Save the processed AnnData objects to disk
    train_path = processed_paths["train"]
    val_path = processed_paths["val"]
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    train_adata.write_h5ad(train_path)
    val_adata.write_h5ad(val_path)
    logger.info("Saved processed files to: %s and %s", train_path, val_path)
    del train_adata, val_adata  # Free up memory
    datasets_all = {}  # These will be pushed as seperate datasets
    if isinstance(dataset_types, str):
        dataset_types = [dataset_types]
    for dataset_type in dataset_types:
        # Create datasets for each split using the AnnDataSetConstructor
        local_dataset = {}
        for split, path in zip(["train", "val"], [train_path, val_path]):
            # Update remote_path in nextcloud_config for the current split (if needed)
            nextcloud_config["remote_path"] = (
                f"datasets/{split}/{Path(file_path).stem}.h5ad"
            )
            caption_constructor = SimpleCaptionConstructor(obs_keys=caption_key)
            constructor = AnnDataSetConstructor(
                caption_constructor=caption_constructor,
                store_nextcloud=True,
                nextcloud_config=nextcloud_config,
                negatives_per_sample=negatives_per_sample,
                dataset_format=dataset_type,
            )
            constructor.add_anndata(
                file_path=path, obsm_keys=[f"X_{method}" for method in methods]
            )
            dataset = constructor.get_dataset()
            local_dataset[split] = dataset
        datasets_all[dataset_type] = local_dataset
    return datasets_all


def main():
    """
    Main function to process multiple AnnData files, concatenate the resulting Hugging Face datasets
    split-wise, and push the final dataset with annotations.

    The data is sourced from raw AnnData files located in the project directory.
    """
    setup_logging()
    load_dotenv(override=True)
    monitor = SystemMonitor(logger=logger)
    monitor.start()
    args = parse_arguments()  # Get arguments from command line
    geo_n = args.geo_n
    cellxgene_n = args.cellxgene_n
    raw_full_data = {
        "geo": f"geo_{geo_n}",
        "cellxgene": f"cellxgene_pseudo_bulk_{cellxgene_n}",
    }
    # Process each raw file and concatenate its dataset with previous ones split-wise
    for key, data_name in raw_full_data.items():
        file_path = project_dir / "data" / "RNA" / "raw" / "train" / f"{data_name}.h5ad"
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
            continue

        # Loop over
        processed_paths = {
            "train": str(
                project_dir / "data" / "RNA" / "processed" / data_name / "train.h5ad"
            ),
            "val": str(
                project_dir / "data" / "RNA" / "processed" / data_name / "val.h5ad"
            ),
        }
        # Process the file and obtain datasets for the train and val splits
        local_ds = process_file_to_dataset(
            file_path=file_path,
            methods=methods,
            batch_key=batch_keys[key],
            caption_key=caption_key,
            processed_paths=processed_paths,
            nextcloud_config=nextcloud_config,
            dataset_types=dataset_types,
            negatives_per_sample=negatives_per_sample,
            monitor=monitor,
        )

    for dataset_type in local_ds.keys():
        # Concatenate the new dataset with any previously processed dataset for each split
        # publish a new dataset for each type ("pairs", "multiplets", "single")
        hf_dataset = DatasetDict()
        type_ds = local_ds[dataset_type]
        for split in ["train", "val"]:
            if split in hf_dataset:
                hf_dataset[split] = concatenate_datasets(
                    [hf_dataset[split], type_ds[split]]
                )
                logger.info(
                    "Concatenated %s - %s dataset with new data from %s",
                    dataset_type,
                    split,
                    file_path,
                )
            else:
                hf_dataset[split] = type_ds[split]
                logger.info(
                    "Initialized %s dataset with data from %s", split, file_path
                )
        # Compose metadata descriptions for dataset annotation
        caption_generation = (
            f"Captions were generated with the SimpleCaptionConstructor class. "
            f"Obs_keys concatenated: {caption_key}."
        )
        embedding_generation = (
            f"Embeddings were generated with the InitialEmbedder class for methods: {methods}. "
            "Each method stored its embeddings in the corresponding adata.obsm key."
        )
        dataset_type_explanation = f"""Dataset type: {dataset_type}. This can be used for several loss functions from the
                                    sentence_transformers library."""

        if push_to_hub is not False:
            # Annotate and push the concatenated dataset
            annotate_and_push_dataset(
                dataset=hf_dataset,
                caption_generation=caption_generation,
                embedding_generation=embedding_generation,
                dataset_type_explanation=dataset_type_explanation,
                repo_id=f"jo-mengr/geo_{geo_n}_cellxgene_{cellxgene_n}_{dataset_type}",
                readme_template_name="cellwhisperer_train",
            )
            logger.info("Final concatenated dataset pushed successfully.")

        monitor.stop()
        metric_dir = "out/training_datasets_create"
        os.makedirs(metric_dir, exist_ok=True)
        monitor.save(metric_dir)
        monitor.plot_metrics(save_dir=metric_dir)


if __name__ == "__main__":
    main()
