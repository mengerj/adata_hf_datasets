#!/usr/bin/env python
"""
Create Hugging Face datasets from processed AnnData train/val files
or a single processed file (test). Optionally push them to the Hugging Face Hub.

References
----------
- Hydra: https://hydra.cc
- anndata: https://anndata.readthedocs.io
- Hugging Face datasets: https://huggingface.co/docs/datasets
"""

import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from datasets import DatasetDict
from adata_hf_datasets.adata_ref_ds import (
    AnnDataSetConstructor,
    SimpleCaptionConstructor,
)
from adata_hf_datasets.utils import (
    setup_logging,
    annotate_and_push_dataset,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="create_dataset")
def main(cfg: DictConfig):
    """
    Main function to build train/val (split_dataset=true) or single (split_dataset=false)
    huggingface dataset from processed .h5ad files, then optionally push to the Hub.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config containing:
        - split_dataset (bool)
        - processed_paths_train (list[str])  [if split_dataset=true]
        - processed_paths_val (list[str])    [if split_dataset=true]
        - processed_paths_all (list[str])    [if split_dataset=false]
        - caption_key (str)
        - negatives_per_sample (int)
        - dataset_type (str)
        - push_to_hub (bool)
        - repo_id (str)
        - obsm_keys (list[str])
    """
    setup_logging()
    load_dotenv(override=True)

    split_dataset = cfg.split_dataset
    caption_key = cfg.caption_key
    negatives_per_sample = cfg.negatives_per_sample
    dataset_type = cfg.dataset_type
    push_to_hub_flag = cfg.push_to_hub
    repo_id = cfg.repo_id
    obsm_keys = cfg.obsm_keys

    if split_dataset:
        # 1) Construct the train dataset
        logger.info("Building train dataset from multiple .h5ad files.")
        train_dataset = build_hf_dataset(
            processed_paths=cfg.processed_paths_train,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obsm_keys=obsm_keys,
        )

        # 2) Construct the val dataset
        logger.info("Building val dataset from multiple .h5ad files.")
        val_dataset = build_hf_dataset(
            processed_paths=cfg.processed_paths_val,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obsm_keys=obsm_keys,
        )

        # 3) Combine into a DatasetDict
        hf_dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

    else:
        # Single dataset scenario (e.g., for test data).
        logger.info("Building single dataset (no train/val split).")
        single_dataset = build_hf_dataset(
            processed_paths=cfg.processed_paths_all,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obsm_keys=obsm_keys,
        )

        # Use "test" or "all" as the single split name
        # (HuggingFace Datasets can have any split name).
        hf_dataset = DatasetDict({"test": single_dataset})

    logger.info("Constructed a DatasetDict with keys: %s", list(hf_dataset.keys()))

    # (Optional) push to Hub
    if push_to_hub_flag:
        push_dataset_to_hub(
            hf_dataset=hf_dataset,
            repo_id=repo_id,
            caption_key=caption_key,
            obsm_keys=obsm_keys,
            dataset_type=dataset_type,
        )

    logger.info("Dataset creation script completed successfully.")


def build_hf_dataset(
    processed_paths,
    caption_key,
    negatives_per_sample,
    dataset_type,
    obsm_keys,
):
    """
    Build a Hugging Face dataset from one or more processed .h5ad files.

    Parameters
    ----------
    processed_paths : list of str
        Paths to processed AnnData .h5ad files.
    caption_key : str
        Observation key used for generating captions.
    negatives_per_sample : int
        Number of negative samples to generate per positive sample.
    dataset_type : str
        Type of dataset to construct (e.g. "pairs", "multiplets", "single").
    obsm_keys : list of str
        Keys in adata.obsm that contain embeddings.

    Returns
    -------
    dataset : datasets.Dataset
        The combined Hugging Face dataset from all provided .h5ad files.
    """
    logger.info(
        "Building HF dataset (type='%s') from: %s", dataset_type, processed_paths
    )

    caption_constructor = SimpleCaptionConstructor(obs_keys=caption_key)
    constructor = AnnDataSetConstructor(
        caption_constructor=caption_constructor,
        store_nextcloud=False,  # or True if needed
        nextcloud_config={},
        negatives_per_sample=negatives_per_sample,
        dataset_format=dataset_type,
    )

    for fpath in processed_paths:
        if not Path(fpath).is_file():
            logger.error("Processed file not found: %s", fpath)
            raise FileNotFoundError(f"Processed file not found: {fpath}")
        constructor.add_anndata(file_path=fpath, obsm_keys=obsm_keys)

    dataset = constructor.get_dataset()
    return dataset


def push_dataset_to_hub(hf_dataset, repo_id, caption_key, obsm_keys, dataset_type):
    """
    Push a DatasetDict to the Hugging Face Hub, with annotated metadata.

    Parameters
    ----------
    hf_dataset : DatasetDict
        The dataset with one or more splits (train/val or test).
    repo_id : str
        The Hugging Face repository ID (e.g., 'username/my_adata_dataset').
    caption_key : str
        Observation column used to create captions.
    obsm_keys : list of str
        Which embeddings were included (.obsm) for metadata.
    dataset_type : str
        The type of dataset ("pairs", "multiplets", "single", etc.).

    Returns
    -------
    None
    """
    if dataset_type in ["pairs", "multiplets"]:
        if caption_key in ["natural_language_annotation"]:
            caption_generation = "Captions generated by LLMs based on available metadata. See the CellWhisperer paper for details."
        else:
            caption_generation = f"Captions generated via SimpleCaptionConstructor (obs key: '{caption_key}')."
    else:
        caption_generation = None
    embedding_generation = (
        f"Included embeddings: {obsm_keys}, stored in AnnData .obsm fields."
    )
    dataset_type_explanation = f"Dataset type: {dataset_type} (suitable for relevant training or inference tasks)."

    annotate_and_push_dataset(
        dataset=hf_dataset,
        caption_generation=caption_generation,
        embedding_generation=embedding_generation,
        dataset_type_explanation=dataset_type_explanation,
        repo_id=repo_id,
        readme_template_name="cellwhisperer_train",  # or whichever template you have
    )
    logger.info("Dataset pushed to HF Hub at %s", repo_id)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred during dataset creation.")
        sys.exit(1)
