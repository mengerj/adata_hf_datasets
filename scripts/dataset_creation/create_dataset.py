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
2. upload the chunks of h5ad or zarr stores to nextcloud
3. create cell sentences, and add anndata to the dataset constructor
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
    annotation_key: str,
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
    base_repo_id: str, dataset_names: List[str], dataset_format: str, caption_key: str
) -> str:
    """
    Compose the final HF repo-ID.

    Example:
        >>> build_repo_id("jo-mengr", ["bulk_5k", "geo"], "pairs", "cell_type")
        'jo-mengr/bulk_5k_geo_pairs_cell_type'
    """
    return f"{base_repo_id.rstrip('/')}/{dataset_names}_{dataset_format}_{caption_key}"


def push_dataset_to_hub(
    hf_dataset: DatasetDict,
    repo_id: str,
    caption_key: str,
    embedding_keys: List[str],
    dataset_format: str,
    share_links: Dict[str, Dict[str, str]],
):
    """
    Push DatasetDict *hf_dataset* to the Hub, writing a rich README.
    """
    if dataset_format in {"pairs", "multiplets"}:
        if caption_key == "natural_language_annotation":
            caption_generation = (
                "Captions generated by LLMs based on available metadata. "
                "See the CellWhisperer paper for details."
            )
        else:
            caption_generation = f"Captions taken from obs column '{caption_key}'."
    else:
        caption_generation = None

    embedding_generation = (
        f"Each AnnData contained the following embedding keys: {embedding_keys}."
    )
    dataset_type_explanation = (
        f"Dataset type: {dataset_format} (suitable for relevant "
        "contrastive-learning or inference tasks)."
    )

    annotate_and_push_dataset(
        dataset=hf_dataset,
        caption_generation=caption_generation,
        embedding_generation=embedding_generation,
        dataset_type_explanation=dataset_type_explanation,
        repo_id=repo_id,
        readme_template_name="cellwhisperer_train",
        metadata={"adata_links": share_links},
        private=True,
    )
    logger.info("Dataset pushed to HF Hub at %s", repo_id)


# -----------------------------------------------------------------------------#
# main Hydra entry-point
# -----------------------------------------------------------------------------#
@hydra.main(version_base=None, config_path="../../conf", config_name="create_dataset")
def main(cfg: DictConfig):
    """
    Build a Hugging Face dataset (with optional splits) from a *data directory*.
    """
    # Get Hydra run directory for logging
    hydra_run_dir = HydraConfig.get().run.dir
    setup_logging(log_dir=hydra_run_dir)
    load_dotenv(override=True)

    data_dir = Path(to_absolute_path(cfg.data_dir)).expanduser()
    data_name = data_dir.name
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    sentence_keys: List[str] = cfg.sentence_keys
    caption_key: str | None = (
        cfg.caption_key if cfg.dataset_format != "single" else None
    )
    batch_key: str = cfg.batch_key
    negatives_per_sample: int = cfg.negatives_per_sample
    dataset_format: str = cfg.dataset_format
    required_obsm_keys: List[str] = cfg.required_obsm_keys
    base_repo_id: str = cfg.base_repo_id
    push_to_hub_flag: bool = cfg.push_to_hub
    use_nextcloud: bool = cfg.get(
        "use_nextcloud", True
    )  # Default to True for backward compatibility

    nextcloud_cfg = dict(cfg.nextcloud_config) if use_nextcloud else None

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
            gene_name_column=cfg.gene_name_column,
            annotation_key=cfg.annotation_key,
            cs_length=cfg.cs_length,
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
        )

    logger.info("Dataset creation script finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Dataset creation failed.")
        sys.exit(1)
