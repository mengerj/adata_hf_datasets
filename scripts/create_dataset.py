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
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="create_dataset")
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
    base_repo_id = cfg.base_repo_id
    obsm_keys = cfg.obsm_keys
    use_nextcloud = cfg.use_nextcloud
    nextcloud_config = cfg.nextcloud_config if use_nextcloud else None

    if split_dataset:
        # 1) Construct the train dataset
        logger.info("Building train dataset from multiple .h5ad files.")
        train_dataset = build_hf_dataset(
            processed_paths=cfg.processed_paths_train,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            nextcloud_config=nextcloud_config,
            use_nextcloud=use_nextcloud,
            obsm_keys=obsm_keys,
        )

        # 2) Construct the val dataset
        logger.info("Building val dataset from multiple .h5ad files.")
        val_dataset = build_hf_dataset(
            processed_paths=cfg.processed_paths_val,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            nextcloud_config=nextcloud_config,
            use_nextcloud=use_nextcloud,
            obsm_keys=obsm_keys,
        )

        # 3) Combine into a DatasetDict
        hf_dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

        # collect the paths for the final repo_id
        naming_paths = cfg.processed_paths_train

    else:
        # Single dataset scenario (e.g., for test data).
        logger.info("Building single dataset (no train/val split).")
        single_dataset = build_hf_dataset(
            processed_paths=cfg.processed_paths_all,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obsm_keys=obsm_keys,
            nextcloud_config=nextcloud_config,
            use_nextcloud=use_nextcloud,
        )

        # Use "test" or "all" as the single split name
        # (HuggingFace Datasets can have any split name).
        hf_dataset = DatasetDict({"test": single_dataset})

        # collect the paths for the final repo_id
        naming_paths = cfg.processed_paths_all

    logger.info("Constructed a DatasetDict with keys: %s", list(hf_dataset.keys()))

    final_repo_id = build_repo_id(
        base_repo_id=base_repo_id,
        file_paths=naming_paths,
        dataset_type=dataset_type,
        caption_key=caption_key,
    )
    logger.info("Final repo_id would be: %s", final_repo_id)

    # (Optional) push to Hub
    if push_to_hub_flag:
        push_dataset_to_hub(
            hf_dataset=hf_dataset,
            repo_id=final_repo_id,
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
    use_nextcloud=False,
    nextcloud_config=None,
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
    use_nextcloud : bool
        Whether to store embeddings and adata in Nextcloud and include a share_link into the dataset.
    nextcloud_config : dict
        Configuration dictionary for Nextcloud storage.

    Returns
    -------
    dataset : datasets.Dataset
        The combined Hugging Face dataset from all provided .h5ad files.
    """
    logger.info(
        "Building HF dataset (type='%s') from: %s", dataset_type, processed_paths
    )
    # If nextcloud is used, set the remote path to the local path
    caption_constructor = SimpleCaptionConstructor(obs_keys=caption_key)
    constructor = AnnDataSetConstructor(
        caption_constructor=caption_constructor,
        store_nextcloud=use_nextcloud,
        nextcloud_config=nextcloud_config,
        negatives_per_sample=negatives_per_sample,
        dataset_format=dataset_type,
    )

    for fpath in processed_paths:
        local_path = to_absolute_path(fpath)
        if not Path(local_path).is_file():
            logger.error("Processed file not found: %s", local_path)
            raise FileNotFoundError(f"Processed file not found: {local_path}")
        if use_nextcloud:
            constructor.nextcloud_config["remote_path"] = fpath
        constructor.add_anndata(file_path=fpath, obsm_keys=obsm_keys)

    dataset = constructor.get_dataset()
    return dataset


def build_repo_id(base_repo_id, file_paths, dataset_type, caption_key):
    """
    Dynamically build the final Hugging Face repo_id by:
    1) Extracting each file's "name" (e.g. parent directory).
    2) Deduplicating and concatenating them with underscores.
    3) Appending dataset_type and caption_key.
    4) Prepending the base_repo_id (e.g. "jo-mengr/").

    Parameters
    ----------
    base_repo_id : str
        Base portion of the HF repo ID, e.g. "jo-mengr/".
    file_paths : list of str
        Paths to processed AnnData .h5ad files (train, val, or test).
    dataset_type : str
        e.g. "pairs", "multiplets", "single".
    caption_key : str
        e.g. "natural_language_annotation".

    Returns
    -------
    str
        e.g.: "jo-mengr/cellxgene_pseudo_bulk_3_5k_pairs_natural_language_annotation"
        or "jo-mengr/cellxgene_pseudo_bulk_3_5k_geo_7k_pairs_natural_language_annotation"
        if multiple datasets.
    """
    # Collect dataset "names" from each path.
    # For instance, if the path is: "data/RNA/processed/train/cellxgene_pseudo_bulk_3_5k/train.h5ad"
    # we can look at parent.name -> "cellxgene_pseudo_bulk_3_5k"
    # Alternatively, use Path(p).stem or another approach.
    names = []
    for p in file_paths:
        p_obj = Path(p)
        dataset_name = p_obj.parent.name  # e.g. "cellxgene_pseudo_bulk_3_5k"
        if dataset_name not in names:
            names.append(dataset_name)

    # Join them with underscores
    joined_names = "_".join(names)

    # Trim trailing slash from base_repo_id if needed
    base_stripped = base_repo_id.rstrip("/")

    # Build final
    final_repo_id = f"{base_stripped}/{joined_names}_{dataset_type}_{caption_key}"
    return final_repo_id


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
