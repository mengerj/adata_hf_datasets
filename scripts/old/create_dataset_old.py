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
import scanpy as sc
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from datasets import DatasetDict
from adata_hf_datasets.ds_constructor import (
    AnnDataSetConstructor,
)
from adata_hf_datasets.utils import (
    setup_logging,
    annotate_and_push_dataset,
)
from adata_hf_datasets.file_utils import save_and_upload_adata
from adata_hf_datasets.cell_sentences import create_cell_sentences
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
        - obsm_key (str)
    """
    setup_logging()
    load_dotenv(override=True)

    split_dataset = cfg.split_dataset
    caption_key = cfg.caption_key
    negatives_per_sample = cfg.negatives_per_sample
    dataset_type = cfg.dataset_type
    push_to_hub_flag = cfg.push_to_hub
    base_repo_id = cfg.base_repo_id
    # obsm_key = cfg.obsm_key if "obsm_key" in cfg else None
    obs_key = cfg.obs_key if "obs_key" in cfg else None
    batch_key = cfg.batch_key
    nextcloud_config = cfg.nextcloud_config
    share_links = {}  # to store share links for each dataset
    # data_rep_tag = obsm_key if obsm_key else obs_key
    # logger.info("Using %s as the data representation.", data_rep_tag)
    if split_dataset:
        # 1) Construct the train dataset
        adata_train = sc.read_h5ad(to_absolute_path(cfg.processed_path_train))
        adata_train = create_cell_sentences(
            adata=adata_train,
            gene_name_column=cfg.gene_name_column,
            annotation_column=cfg.annotation_key,
            cs_length=cfg.cs_length,  # or whatever number of genes you want
        )
        train_dataset = build_hf_dataset(
            adata=adata_train,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obs_key=obs_key,
            batch_key=batch_key,
        )
        # Step 1: Save and upload .h5ad file
        nextcloud_config["remote_path"] = cfg.processed_path_train
        share_links["train"] = save_and_upload_adata(
            to_absolute_path(cfg.processed_path_train),
            nextcloud_config,
            create_share_link=True,
        )

        # 2) Construct the val dataset
        adata_val = sc.read_h5ad(to_absolute_path(cfg.processed_path_val))
        adata_val = create_cell_sentences(
            adata=adata_val,
            gene_name_column=cfg.gene_name_column,
            annotation_column=cfg.annotation_key,
            cs_length=cfg.cs_length,  # or whatever number of genes you want
        )
        val_dataset = build_hf_dataset(
            adata=adata_val,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obs_key=obs_key,
            batch_key=batch_key,
        )
        nextcloud_config["remote_path"] = cfg.processed_path_val
        share_links["val"] = save_and_upload_adata(
            to_absolute_path(cfg.processed_path_val),
            nextcloud_config,
            create_share_link=True,
        )

        # 3) Combine into a DatasetDict
        hf_dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

        # collect the paths for the final repo_id
        naming_path = cfg.processed_path_train

    else:
        # Single dataset scenario (e.g., for test data).
        logger.info("Building single dataset (no train/val split).")
        adata_all = sc.read_h5ad(cfg.processed_path_all)
        single_dataset = build_hf_dataset(
            adata=adata_all,
            caption_key=caption_key,
            negatives_per_sample=negatives_per_sample,
            dataset_type=dataset_type,
            obs_key=obs_key,
        )
        nextcloud_config["remote_path"] = cfg.processed_path_all
        share_links["all"] = save_and_upload_adata(
            to_absolute_path(cfg.processed_path_all),
            nextcloud_config,
            create_share_link=True,
        )

        # Use "test" or "all" as the single split name
        # (HuggingFace Datasets can have any split name).
        hf_dataset = DatasetDict({"test": single_dataset})

        # collect the paths for the final repo_id
        naming_path = cfg.processed_paths_all

    logger.info("Constructed a DatasetDict with keys: %s", list(hf_dataset.keys()))

    final_repo_id = build_repo_id(
        base_repo_id=base_repo_id,
        file_path=naming_path,
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
            dataset_type=dataset_type,
            share_links=share_links,
        )

    logger.info("Dataset creation script completed successfully.")


def build_hf_dataset(
    adata,
    caption_key,
    negatives_per_sample,
    dataset_type,
    obs_key,
    batch_key,
):
    """
    Build a Hugging Face dataset from one or more processed .h5ad files.

    Parameters
    ----------
    caption_key : str
        Observation key used for generating captions.
    negatives_per_sample : int
        Number of negative samples to generate per positive sample.
    dataset_type : str
        Type of dataset to construct (e.g. "pairs", "multiplets", "single").
    obsm_key : list of str
        Keys in adata.obsm that contain embeddings.
    obs_key : str
        Key in adata.obs that contains string representations of the data.
    batch_key : str
        Key in adata.obs that contains batch information. For sampling in batch negatives.

    Returns
    -------
    dataset : datasets.Dataset
        The combined Hugging Face dataset from all provided .h5ad files.
    """
    # If nextcloud is used, set the remote path to the local path
    cons = AnnDataSetConstructor(
        dataset_format=dataset_type, negatives_per_sample=negatives_per_sample
    )

    cons.add_anndata(
        adata,
        obs_key=obs_key,
        caption_key=caption_key,
        batch_key=batch_key,
    )

    dataset = cons.get_dataset()
    return dataset


def build_repo_id(base_repo_id, file_path, dataset_type, caption_key, data_rep_tag):
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
    data_rep_tag : str
        e.g. "obsm_key" or "obs_key" to indicate the type of data representation.

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
    p_obj = Path(file_path)
    if p_obj.parent.name == "joined":
        # Use the directory before "joined" for the dataset name
        dataset_name = p_obj.parent.parent.name
    else:
        # If 'joined' is not in the expected place, log a warning and use the immediate parent
        logger.warning(
            f"Expected 'joined' directory, but found '{p_obj.parent.name}' in path: {file_path}"
        )
        dataset_name = p_obj.parent.name
    if dataset_name not in names:
        names.append(dataset_name)

    # Join them with underscores
    joined_names = "_".join(names)

    # Trim trailing slash from base_repo_id if needed
    base_stripped = base_repo_id.rstrip("/")

    # Build final
    final_repo_id = (
        f"{base_stripped}/{joined_names}_{dataset_type}_{caption_key}_{data_rep_tag}"
    )
    return final_repo_id


def push_dataset_to_hub(
    hf_dataset, repo_id, caption_key, obsm_key, dataset_type, share_links
):
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
    obsm_key : str
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
    embedding_generation = f"Included embeddings: {obsm_key}, used as a numeric representation of the data."
    dataset_type_explanation = f"Dataset type: {dataset_type} (suitable for relevant training or inference tasks)."

    annotate_and_push_dataset(
        dataset=hf_dataset,
        caption_generation=caption_generation,
        embedding_generation=embedding_generation,
        dataset_type_explanation=dataset_type_explanation,
        repo_id=repo_id,
        readme_template_name="cellwhisperer_train",  # or whichever template you have
        metadata={"adata_links": share_links},
        private=True,
    )
    logger.info("Dataset pushed to HF Hub at %s", repo_id)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred during dataset creation.")
        sys.exit(1)
