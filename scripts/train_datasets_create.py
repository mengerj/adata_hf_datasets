from adata_hf_datasets.utils import setup_logging
import anndata
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.utils import split_anndata
import os
from pathlib import Path
from dotenv import load_dotenv
from adata_hf_datasets.adata_ref_ds import AnnDataSetConstructor
from adata_hf_datasets.adata_ref_ds import SimpleCaptionConstructor
from adata_hf_datasets.utils import annotate_and_push_dataset
from datasets import DatasetDict

methods = ["hvg","pca","scvi","geneformer"]
geo_n = "7k" # amount of samples to take from the GEO dataset
cellxgene_n = "3_5K" # amount of samples to take from the cellxgene dataset
raw_full_data = {"geo": f"geo_{geo_n}", "cellxgene": f"cellxgene_pseudo_bulk_{cellxgene_n}"} #This data will be used to create train and val datasets

project_dir = Path(__file__).resolve().parents[1]
test_dir = f"{project_dir}/data/RNA/raw/test" # All data in this folder will be used to create test datasets
caption_key = "natural_language_annotation"
#caption_key = "cluster_label"
batch_keys = {"geo": "batch","cellxgene": "assay"}

nextcloud_config = {
    "url": "https://nxc-fredato.imbi.uni-freiburg.de",
    "username": "NEXTCLOUD_USER",  # env will we obtained within code
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

import os
from pathlib import Path
from dotenv import load_dotenv
import anndata
from datasets import DatasetDict, concatenate_datasets
from adata_hf_datasets.utils import setup_logging, split_anndata, annotate_and_push_dataset
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.adata_ref_ds import AnnDataSetConstructor, SimpleCaptionConstructor
import logging

# Use the predefined logger per instructions
logger = logging.get_logger(__name__)


def process_file_to_dataset(file_path, methods, batch_key, caption_key, processed_paths, nextcloud_config, dataset_types, negatives_per_sample):
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

    Returns
    -------
    local_dataset : dict
        Dictionary with keys "train" and "val" containing the Hugging Face datasets for each split.
        Data is ultimately sourced from the file at `file_path`.
    """
    logger.info("Processing file: %s", file_path)
    adata = anndata.read_h5ad(file_path)

    # Clean up unnecessary fields to free up memory
    if "natural_language_annotation_replicates" in adata.obsm:
        del adata.obsm["natural_language_annotation_replicates"]
    if hasattr(adata, "layers"):
        del adata.layers
    adata.layers = {"counts": adata.X.copy()}

    # Apply each embedding method; embeddings are stored in adata.obsm
    for method in methods:
        logger.info("Applying embedding method '%s' on %s", method, file_path)
        embedder = InitialEmbedder(method=method)
        embedder.fit(adata, batch_key=batch_key)
        embedder.embed(adata)
        # Each embedder is assumed to store its embedding in adata.obsm (e.g., adata.obsm[f'X_{method}'])

    # Split the data into training and validation sets
    train_adata, val_adata = split_anndata(adata, train_size=0.9)

    # Save the processed AnnData objects to disk
    train_path = processed_paths["train"]
    val_path = processed_paths["val"]
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    train_adata.write_h5ad(train_path)
    val_adata.write_h5ad(val_path)
    logger.info("Saved processed files to: %s and %s", train_path, val_path)

    datasets_all = {} # These will be pushed as seperate datasets
    if isinstance(dataset_types, str):
        dataset_types = [dataset_types]
    for dataset_type in dataset_types:
        # Create datasets for each split using the AnnDataSetConstructor
        local_dataset = {}
        for split, path in zip(["train", "val"], [train_path, val_path]):
            # Update remote_path in nextcloud_config for the current split (if needed)
            nextcloud_config["remote_path"] = f"datasets/{split}/{Path(file_path).stem}.h5ad"
            caption_constructor = SimpleCaptionConstructor(obs_keys=caption_key)
            constructor = AnnDataSetConstructor(
                caption_constructor=caption_constructor,
                store_nextcloud=True,
                nextcloud_config=nextcloud_config,
                negatives_per_sample=negatives_per_sample,
            )
            constructor.dataset_type = dataset_type
            constructor.add_anndata(file_path=path)
            dataset = constructor.get_dataset()
            local_dataset[split] = dataset
        datasets_all[dataset_type] = local_dataset
    return local_dataset


def main():
    """
    Main function to process multiple AnnData files, concatenate the resulting Hugging Face datasets
    split-wise, and push the final dataset with annotations.

    The data is sourced from raw AnnData files located in the project directory.
    """
    setup_logging()
    load_dotenv(override=True)

    # Define project parameters and paths
    project_dir = Path(__file__).resolve().parents[1]
    raw_full_data = {"geo": "geo_7k", "cellxgene": "cellxgene_pseudo_bulk_1_7K"}
    methods = ["hvg", "pca", "scvi", "geneformer"]
    dataset_types = ["pairs","multiplets","single"]
    negatives_per_sample = 2
    batch_keys = {"geo": "batch", "cellxgene": "assay"}
    caption_key = "natural_language_annotation"

    nextcloud_config = {
        "url": "https://nxc-fredato.imbi.uni-freiburg.de",
        "username": "NEXTCLOUD_USER",  # Retrieved from environment variables in practice
        "password": "NEXTCLOUD_PASSWORD",
        "remote_path": "",
    }

    # Initialize an empty DatasetDict to accumulate datasets for each split
    hf_dataset = DatasetDict()

    # Process each raw file and concatenate its dataset with previous ones split-wise
    for key, data_name in raw_full_data.items():
        file_path = project_dir / "data" / "RNA" / "raw" / "train" / f"{data_name}.h5ad"
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            continue

        # Loop over 
        processed_paths = {
            "train": str(project_dir / "data" / "RNA" / "processed" / data_name / "train.h5ad"),
            "val": str(project_dir / "data" / "RNA" / "processed" / data_name / "val.h5ad"),
        }
        # Process the file and obtain datasets for the train and val splits
        local_ds = process_file_to_dataset(
            file_path=file_path,
            methods=methods,
            batch_key=batch_keys[key],
            caption_key=caption_key,
            processed_paths=processed_paths,
            nextcloud_config=nextcloud_config,
        )

        # Concatenate the new dataset with any previously processed dataset for each split
        for split in ["train", "val"]:
            if split in hf_dataset:
                hf_dataset[split] = concatenate_datasets([hf_dataset[split], local_ds[split]])
                logger.info("Concatenated %s dataset with new data from %s", split, file_path)
            else:
                hf_dataset[split] = local_ds[split]
                logger.info("Initialized %s dataset with data from %s", split, file_path)

    for dataset_type in hf_dataset.keys():
        
        # Compose metadata descriptions for dataset annotation
        caption_generation = (
            f"Captions were generated with the SimpleCaptionConstructor class. "
            f"Obs_keys concatenated: {caption_key}."
        )
        embedding_generation = (
            f"Embeddings were generated with the InitialEmbedder class for methods: {methods}. "
            "Each method stored its embeddings in the corresponding adata.obsm key."
        )
        dataset_type_explanation = (f"""Dataset type: {dataset_type}. This can be used for several loss functions from the 
                                    sentence_transformers library.""")
    

        # Annotate and push the concatenated dataset
        annotate_and_push_dataset(
            dataset=hf_dataset[dataset_type],
            caption_generation=caption_generation,
            embedding_generation=embedding_generation,
            dataset_type_explanation=dataset_type_explanation,
            repo_id=f"jo-mengr/geo_{geo_n}_cellxgene_{cellxgene_n}_{dataset_type}",
            readme_template_name="cellwhisperer_train",
        )
        logger.info("Final concatenated dataset pushed successfully.")


if __name__ == "__main__":
    main()