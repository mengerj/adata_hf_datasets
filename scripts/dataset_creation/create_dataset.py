"""
Create Hugging Face datasets from one or more processed AnnData folders.

A *data directory* must contain either
    data_dir/
        ├── train/   *.h5ad|*.zarr
        └── val/     *.h5ad|*.zarr
or
    data_dir/
        └── all/     *.h5ad|*.zarr

For each split folder we

1. read every file into an AnnData, check embedding keys.
2. create cell sentences, and add anndata to the dataset constructor
3. upload the chunks of h5ad or zarr stores to nextcloud (if use_nextcloud is True)
4. build the Hugging Face dataset split
5. optionally push the resulting DatasetDict to the HF Hub
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List
import hydra
import anndata as ad
from dotenv import load_dotenv
from omegaconf import DictConfig
from datasets import Dataset, DatasetDict
from hydra.core.hydra_config import HydraConfig

from adata_hf_datasets.ds_constructor import AnnDataSetConstructor
from adata_hf_datasets.utils import annotate_and_push_dataset, setup_logging
from adata_hf_datasets.cell_sentences import create_cell_sentences
from adata_hf_datasets.file_utils import upload_folder_to_nextcloud
from adata_hf_datasets.config_utils import apply_all_transformations
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# core helpers
# -----------------------------------------------------------------------------#
def _read_adata(path: Path):
    """Read an AnnData from .h5ad or .zarr."""
    if path.suffix == ".h5ad":
        return ad.read_h5ad(path)
    if path.suffix == ".zarr":
        return ad.read_zarr(path)
    raise ValueError(f"Unsupported file type: {path}")


def _validate_obsm_keys(adata, required_keys: List[str], file_path: Path):
    """Raise if *any* required embedding key is missing."""
    missing = [k for k in required_keys if k not in adata.obsm.keys()]
    if missing:
        raise KeyError(f"File '{file_path}': missing required obsm keys: {missing}")


def _add_sample_id_column(adata, column_name: str = "sample_id_og"):
    """Create a new column in adata.obs with the values stored in obs.index."""
    adata.obs[column_name] = adata.obs.index
    return adata


def build_split_dataset(
    split_dir: Path,
    share_links: Dict[str, str],
    sentence_keys: List[str],
    caption_key: str | None,
    batch_key: str,
    negatives_per_sample: int,
    dataset_format: str,
    gene_name_column: str,
    annotation_key: str | None,
    cs_length: int,
) -> Dataset:
    """
    Build a Hugging Face dataset for **one split** (train / val / all).

    Parameters
    ----------
    split_dir
        Folder with *.h5ad / *.zarr files.
    share_links
        Mapping *filename → Nextcloud share link or local path* returned by
        ``upload_folder_to_nextcloud`` or created locally.
    sentence_keys, caption_key, batch_key
        Column names to feed into `AnnDataSetConstructor`.
    negatives_per_sample, dataset_format
        Passed through unchanged.
    gene_name_column
        Column name in AnnData.obs with gene names.
    annotation_key
        Column name in AnnData.obs with cell type annotations.
        If None, semantic sentences will be skipped.
    cs_length
        Length of the cell sentence to create.

    Returns
    -------
    datasets.Dataset
        All samples from *split_dir* pooled together.
    """
    constructor = AnnDataSetConstructor(
        dataset_format=dataset_format, negatives_per_sample=negatives_per_sample
    )

    # iterate over every file in this split
    for f in sorted(split_dir.glob("*.h5ad")) + sorted(split_dir.glob("*.zarr")):
        logger.info("Reading %s", f.name)
        adata = _read_adata(f)

        _add_sample_id_column(
            adata, column_name="sample_id_og"
        )  # This is to have the sample id in obs, to use it as a cell sentence
        # create / update cell sentences *in place*
        adata = create_cell_sentences(
            adata=adata,
            gene_name_column=gene_name_column,
            annotation_column=annotation_key,
            cs_length=cs_length,
        )

        # Get the share link or local path for this file
        # The key might be f.name + ".zip" (for Nextcloud) or just the path (for local)
        file_reference = share_links.get(f.name + ".zip")
        if file_reference is None:
            # Fallback: try without .zip suffix or use the file path directly
            file_reference = share_links.get(f.name, str(f.resolve()))

        constructor.add_anndata(
            adata=adata,
            sentence_keys=sentence_keys,
            caption_key=caption_key,
            batch_key=batch_key,
            share_link=file_reference,
        )

    return constructor.get_dataset()


def build_repo_id(
    base_repo_id: str,
    dataset_names: List[str],
    dataset_format: str,
    caption_key: str,
) -> str:
    """
    Compose the final HF repo-ID.

    Example:
        >>> build_repo_id("jo-mengr", ["bulk_5k", "geo"], "pairs", "cell_type")
        'jo-mengr/bulk_5k_geo_pairs_cell_type'
    """
    return f"{base_repo_id.rstrip('/')}/{dataset_names}_{dataset_format}_{caption_key}"


def check_and_version_repo_id(base_repo_id: str) -> str:
    """
    Check if a repository exists and add version suffix if needed.

    Parameters
    ----------
    base_repo_id : str
        The base repository ID to check

    Returns
    -------
    str
        The final repository ID with version suffix if needed
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    import os

    api = HfApi()
    token = os.getenv("HF_TOKEN_UPLOAD") or os.getenv("HF_TOKEN")

    # Try the base repo_id first
    try:
        api.repo_info(repo_id=base_repo_id, repo_type="dataset", token=token)
        # If we get here, the repo exists, so we need to version it
        logger.info(f"Repository {base_repo_id} already exists, adding version suffix")

        version = 2
        while True:
            versioned_repo_id = f"{base_repo_id}_v{version}"
            try:
                api.repo_info(
                    repo_id=versioned_repo_id, repo_type="dataset", token=token
                )
                version += 1
            except RepositoryNotFoundError:
                logger.info(f"Using versioned repository ID: {versioned_repo_id}")
                return versioned_repo_id

    except RepositoryNotFoundError:
        # Repository doesn't exist, use the original name
        logger.info(f"Repository {base_repo_id} is available")
        return base_repo_id
    except Exception as e:
        logger.warning(
            f"Could not check repository existence: {e}. Using original name."
        )
        return base_repo_id


def push_dataset_to_hub(
    hf_dataset: DatasetDict,
    repo_id: str,
    caption_key: str,
    embedding_keys: List[str],
    dataset_format: str,
    share_links: Dict[str, Dict[str, str]],
    cs_length: int,
    sentence_keys: List[str],
):
    """
    Push DatasetDict *hf_dataset* to the Hub, writing a rich README.
    """
    # Check for existing repo and add version if needed
    final_repo_id = check_and_version_repo_id(repo_id)

    embedding_generation = (
        f"Each AnnData contained the following embedding keys: {embedding_keys}."
    )
    dataset_type_explanation = (
        f"Dataset type: {dataset_format} (suitable for relevant "
        "contrastive-learning or inference tasks)."
    )

    # Create example data from the first row of the dataset for the description
    example_data = {}
    first_split = list(hf_dataset.keys())[0]
    if len(hf_dataset[first_split]) > 0:
        first_row = hf_dataset[first_split][0]

        # Define column priority order for better presentation
        priority_columns = ["sample_idx"]
        if caption_key and caption_key in first_row:
            priority_columns.append(caption_key)
        priority_columns.extend(sentence_keys)

        # Add priority columns first
        for key in priority_columns:
            if key in first_row:
                value = str(first_row[key])
                if len(value) > 150:  # Increased from 100 for better context
                    value = value[:150] + "..."
                example_data[key] = value

        # Add remaining columns (excluding those already added)
        for key in first_row.keys():
            if key not in example_data:
                value = str(first_row[key])
                if len(value) > 150:
                    value = value[:150] + "..."
                example_data[key] = value

    # Get one example share link (not all of them)
    example_share_link = None
    if share_links:
        for split_links in share_links.values():
            if split_links:
                example_share_link = list(split_links.values())[0]
                break

    metadata = {
        "adata_links": share_links,
        "cs_length": cs_length,
        "sentence_keys": sentence_keys,
        "example_data": example_data,
        "example_share_link": example_share_link,
    }

    annotate_and_push_dataset(
        dataset=hf_dataset,
        embedding_generation=embedding_generation,
        dataset_type_explanation=dataset_type_explanation,
        repo_id=final_repo_id,
        readme_template_name="cellwhisperer_train",
        metadata=metadata,
        private=True,
    )
    logger.info("Dataset pushed to HF Hub at %s", final_repo_id)


# -----------------------------------------------------------------------------#
# main Hydra entry-point
# -----------------------------------------------------------------------------#
@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="dataset_human_disease",
)
def main(cfg: DictConfig):
    """
    Build a Hugging Face dataset (with optional splits) from a *data directory*.
    """
    # Get Hydra run directory for logging
    hydra_run_dir = HydraConfig.get().run.dir
    setup_logging(log_dir=hydra_run_dir)
    load_dotenv(override=True)

    # Apply all transformations to the config (paths, common keys, etc.)
    cfg = apply_all_transformations(cfg)

    # Extract dataset creation specific config
    dataset_cfg = cfg.dataset_creation

    data_dir = Path(to_absolute_path(dataset_cfg.data_dir)).expanduser()
    data_name = data_dir.name
    if not data_dir.exists():
        logger.warning(f"data_dir not found: {data_dir}")
        if len(dataset_cfg.required_obsm_keys) == 0:
            logger.info(
                "No required_obsm_keys specified; attempting to use 'processed' directory as fallback (embedding step was skipped)."
            )
            # TODO: Do not hardcode the path replacement here; make this configurable in the future.
            fallback_data_dir = Path(
                str(data_dir).replace("processed_with_emb", "processed")
            )
            if fallback_data_dir.exists():
                data_dir = fallback_data_dir
                logger.info(
                    f"Using processed data instead of processed_with_emb: {data_dir}"
                )
            else:
                raise FileNotFoundError(
                    f"Neither data_dir '{data_dir}' nor fallback '{fallback_data_dir}' found."
                )
        else:
            raise FileNotFoundError(f"data_dir not found: {data_dir}")

    sentence_keys: List[str] = dataset_cfg.sentence_keys
    caption_key: str | None = (
        dataset_cfg.caption_key if dataset_cfg.dataset_format != "single" else None
    )

    # Get annotation_key early to filter sentence_keys
    annotation_key = getattr(dataset_cfg, "annotation_key", None)

    # Filter sentence_keys to remove semantic columns if annotation_key is None
    if annotation_key is None:
        # Remove semantic sentence keys that won't be created
        sentence_keys = [
            key
            for key in sentence_keys
            if key not in ["semantic_true", "semantic_similar"]
        ]
        logger.warning(
            "annotation_key is None. Removed 'semantic_true' and 'semantic_similar' "
            "from sentence_keys. Final sentence_keys: %s",
            sentence_keys,
        )

    batch_key: str = cfg.batch_key
    negatives_per_sample: int = dataset_cfg.negatives_per_sample
    dataset_format: str = dataset_cfg.dataset_format
    required_obsm_keys: List[str] = dataset_cfg.required_obsm_keys
    base_repo_id: str = dataset_cfg.base_repo_id
    push_to_hub_flag: bool = dataset_cfg.push_to_hub
    use_nextcloud: bool = dataset_cfg.get(
        "use_nextcloud", True
    )  # Default to True for backward compatibility

    nextcloud_cfg = dict(dataset_cfg.nextcloud_config) if use_nextcloud else None

    # ------------------------------------------------------------------ #
    # detect splits
    # ------------------------------------------------------------------ #
    if (data_dir / "train").is_dir() and (data_dir / "val").is_dir():
        split_names = ["train", "val"]
    elif (data_dir / "all").is_dir():
        split_names = ["all"]
    else:
        raise ValueError(
            "data_dir must contain 'train' & 'val' OR a single 'all' folder."
        )

    hf_splits: Dict[str, Dataset] = {}
    share_links_per_split: Dict[str, Dict[str, str]] = {}

    for split in split_names:
        split_dir = data_dir / split
        logger.info("Processing split '%s' (%s)", split, split_dir)

        # ------------------------------------------------------------------ #
        # 1) sanity-check every file before uploading
        # ------------------------------------------------------------------ #
        for f in sorted(split_dir.glob("*.h5ad")) + sorted(split_dir.glob("*.zarr")):
            adata_tmp = _read_adata(f)
            _validate_obsm_keys(adata_tmp, required_obsm_keys, f)
            del adata_tmp

        # ------------------------------------------------------------------ #
        # 2) upload folder or use local paths
        # ------------------------------------------------------------------ #
        if use_nextcloud and nextcloud_cfg:
            nextcloud_cfg["remote_path"] = str(split_dir)
            share_links = upload_folder_to_nextcloud(
                data_folder=str(split_dir),
                nextcloud_config=nextcloud_cfg | {"progress": True},
            )
            logger.info("Uploaded files to Nextcloud for split '%s'", split)
        else:
            # Create a mapping of filename to local path instead of share links
            share_links = {}
            for f in sorted(split_dir.glob("*.h5ad")) + sorted(
                split_dir.glob("*.zarr")
            ):
                # Use absolute path for consistency
                share_links[f.name + ".zip"] = str(f.resolve())
            logger.info("Using local file paths for split '%s'", split)

        share_links_per_split[split] = share_links

        # ------------------------------------------------------------------ #
        # 3) build the HF dataset for this split
        # ------------------------------------------------------------------ #
        hf_splits[split] = build_split_dataset(
            split_dir=split_dir,
            share_links=share_links,
            sentence_keys=sentence_keys,
            caption_key=caption_key,
            batch_key=batch_key,
            negatives_per_sample=negatives_per_sample,
            dataset_format=dataset_format,
            gene_name_column=dataset_cfg.gene_name_column,
            annotation_key=annotation_key,
            cs_length=dataset_cfg.cs_length,
        )
        # if the split is called all, change it to "test" to avoid issue with hf format
        if split == "all":
            hf_splits["test"] = hf_splits.pop(split)

    hf_dataset = DatasetDict(hf_splits)
    logger.info("Built DatasetDict with splits: %s", list(hf_dataset.keys()))

    # ------------------------------------------------------------------ #
    # final repo-ID and optional push
    # ------------------------------------------------------------------ #
    repo_id = build_repo_id(
        base_repo_id=base_repo_id,
        dataset_names=data_name,
        dataset_format=dataset_format,
        caption_key=caption_key or "no_caption",
    )
    logger.info("Final repo_id would be: %s", repo_id)

    # save dataset locally to the hydra run directory
    dataset_save_path = Path(hydra_run_dir) / data_name
    hf_dataset.save_to_disk(str(dataset_save_path))
    logger.info("Dataset saved locally to: %s", dataset_save_path)

    if push_to_hub_flag:
        push_dataset_to_hub(
            hf_dataset=hf_dataset,
            repo_id=repo_id,
            caption_key=caption_key or "",
            embedding_keys=required_obsm_keys,
            dataset_format=dataset_format,
            share_links=share_links_per_split,
            cs_length=dataset_cfg.cs_length,
            sentence_keys=sentence_keys,
        )

    logger.info("Dataset creation script finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Dataset creation failed.")
        sys.exit(1)
