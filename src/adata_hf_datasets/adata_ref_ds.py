import json
import logging
import random
import pandas as pd
import anndata
from datasets import Dataset
from typing import Optional
import tempfile
from .file_utils import save_and_upload_adata, download_file_from_share_link

logger = logging.getLogger(__name__)


class AnnDataSetConstructor:
    """
    Class to generate a dataset compatible with the SentenceTransformer library from AnnData files.

    Data is sourced from AnnData files (.zarr or .h5ad) that are added via `add_anndata`.
    The generated dataset can be created in one of three formats:
    
    - "pairs": Each record is a pair (positive example and, optionally, separate negative examples).
    - "multiplets": Each record contains an anchor, a positive, and a list of negative examples.
    - "single": Each record contains only a single caption (useful for inference).
    """

    def __init__(
        self,
        caption_constructor,
        negatives_per_sample: int = 1,
        dataset_format: str = "pairs",
        store_nextcloud: bool = False,
        nextcloud_config: Optional[dict] = None,
    ):
        """
        Initialize the AnnDataSetConstructor.

        Parameters
        ----------
        caption_constructor : callable
            Constructor for creating captions from AnnData files.
        negatives_per_sample : int, optional
            Number of negative examples to create per positive example (applies to "pairs" and "multiplets"
            formats). A value of 0 is allowed.
        dataset_format : str, optional
            The format of the dataset to construct. Allowed values are:
              - "pairs": Each record is a pair with one positive example and, per sample,
                         `negatives_per_sample` negative records are generated.
              - "multiplets": Each record is a multiplet consisting of an anchor, a positive, and a list
                              of negatives (the number of negatives equals `negatives_per_sample`).
              - "single": Each record contains only a single caption (suitable for inference).
            Default is "pairs".
        store_nextcloud : bool, optional
            If True, upload AnnData files to Nextcloud and store share links instead of local paths.
        nextcloud_config : dict, optional
            Configuration dictionary for Nextcloud which must contain:
                - 'url' (str): URL to the Nextcloud server.
                - 'username' (str): Username for Nextcloud.
                - 'password' (str): Password for Nextcloud.
                - 'remote_path' (str): Remote path in Nextcloud where the file will be uploaded.
        
        Raises
        ------
        ValueError
            If `dataset_format` is not one of "pairs", "multiplets", or "single".
        
        Notes
        -----
        Data is sourced from AnnData files (either .zarr or .h5ad) that are added via `add_anndata`.
        """
        self.caption_constructor = caption_constructor
        self.negatives_per_sample = negatives_per_sample
        self.dataset_format = dataset_format.lower()
        if self.dataset_format not in ("pairs", "multiplets", "single"):
            error_msg = "dataset_format must be one of 'pairs', 'multiplets', or 'single'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.store_nextcloud = store_nextcloud
        self.nextcloud_config = nextcloud_config if store_nextcloud else None
        self.anndata_files = []
        self.sample_id_keys = {}
        self.dataset = []

    def _check_sample_id_uniqueness(
        self, adata: anndata.AnnData, file_path: str, sample_id_key: str | None
    ) -> None:
        """
        Check if sample IDs are unique for the given AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object to check.
        file_path : str
            Path to the AnnData file (for error messages).
        sample_id_key : str or None
            Key in adata.obs to use for sample IDs; if None, uses adata.obs.index.

        Raises
        ------
        ValueError
            If duplicate sample IDs are found.
        """
        sample_ids = (
            adata.obs.index if sample_id_key is None else adata.obs[sample_id_key]
        )
        n_total = len(sample_ids)
        n_unique = len(set(sample_ids))

        if n_unique < n_total:
            duplicates = sample_ids[sample_ids.duplicated()].unique()
            error_msg = (
                f"Found {n_total - n_unique} duplicate sample IDs in {file_path}.\n"
                f"Example duplicates: {list(duplicates)[:3]}...\n"
                "To fix this, either:\n"
                "1. Provide a different sample_id_key that contains unique identifiers, or\n"
                "2. Remove duplicate samples from your dataset"
            )
            if sample_id_key is None:
                error_msg += "\nCurrently using adata.obs.index as sample IDs."
            else:
                error_msg += (
                    f"\nCurrently using adata.obs['{sample_id_key}'] as sample IDs."
                )

            logger.error(error_msg)
            raise ValueError(error_msg)

    def add_anndata(self, file_path: str, sample_id_key: str | None = None) -> None:
        """
        Add an AnnData file to the constructor.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.
        sample_id_key : str or None, optional
            Optional key in adata.obs to use for sample IDs. If None, uses adata.obs.index.

        Raises
        ------
        ValueError
            If the file format is unsupported or if the file has already been added.
        """
        self.is_zarr = False
        self.is_h5ad = False
        # 1. Check extension
        if file_path.endswith(".zarr") or file_path.endswith(".zarr/"):
            self.is_zarr = True
        elif file_path.endswith(".h5ad"):
            self.is_h5ad = True
        else:
            logger.error("Unsupported AnnData format for file: %s", file_path)
            raise ValueError(
                f"File {file_path} does not appear to be .zarr or .h5ad format."
            )

        # 2. Check for duplicates
        for files in self.anndata_files:
            if file_path in files["local_path"]:
                logger.error(
                    "File %s has already been added to the constructor.", file_path
                )
                raise ValueError(f"File {file_path} has already been added.")

        # 3. Check sample ID uniqueness
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        if self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        self.local_path = file_path  # Store local path for reference
        path_for_dataset = file_path  # Default: store the local path for dataset creation
        self._check_sample_id_uniqueness(adata, file_path, sample_id_key)

        # Upload to Nextcloud if enabled
        if self.store_nextcloud and self.nextcloud_config:
            share_link = save_and_upload_adata(
                adata, file_path, self.nextcloud_config, create_share_link=True
            )
            if share_link:
                path_for_dataset = share_link
                if not self._check_sharelink(share_link):
                    logger.error(f"Nextcloud sharelink {share_link} not working")
                    raise ValueError(f"Nextcloud sharelink {share_link} not working")
            else:
                logger.error("Failed to upload file to Nextcloud: %s", file_path)
                raise ValueError(f"Nextcloud upload failed for {file_path}")

        self.anndata_files.append(
            {"local_path": file_path, "dataset_path": path_for_dataset}
        )
        self.sample_id_keys[file_path] = sample_id_key
        logger.info("Successfully added AnnData file: %s", file_path)

    def buildCaption(self, file_path: str) -> None:
        """
        Build captions for an AnnData file using the provided caption constructor.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.

        Notes
        -----
        Captions are constructed and added to the AnnData object's .obs["caption"] column.
        """
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        if self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        self.caption_constructor.construct_captions(adata)
        if self.is_zarr:
            adata.write_zarr(file_path)
        if self.is_h5ad:
            adata.write_h5ad(file_path)

    def getCaption(self, file_path: str) -> dict[str, str]:
        """
        Get a dictionary mapping sample IDs to captions from an AnnData file.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.

        Returns
        -------
        dict
            Dictionary mapping sample IDs to captions.

        Raises
        ------
        ValueError
            If no "caption" column is found in the AnnData file.
        """
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        if self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        if "caption" not in adata.obs.columns:
            raise ValueError(f"No 'caption' column found in {file_path}")

        sample_id_key = self.sample_id_keys[file_path]
        sample_ids = (
            adata.obs.index if sample_id_key is None else adata.obs[sample_id_key]
        )

        return dict(zip(sample_ids, adata.obs["caption"], strict=False))

    def _create_negative_example(
        self,
        current_dataset_path: str,
        current_sample: str,
        current_caption: str,
        all_captions: dict[str, dict[str, str]],
    ) -> tuple[str, str, float]:
        """
        Create a negative example by finding a caption that does not match the current sample.

        Parameters
        ----------
        current_dataset_path : str
            Path used to reference the current file.
        current_sample : str
            ID of the current sample.
        current_caption : str
            Caption of the current sample.
        all_captions : dict
            Nested dict mapping file paths to {sample_id: caption} dictionaries.

        Returns
        -------
        tuple
            A tuple (sentence_1, sentence_2, label) where:
              - sentence_1 : JSON string containing file_path and sample_id (of the current sample).
              - sentence_2 : The negative caption.
              - label : 0.0 (indicating a negative example).
        """
        while True:
            # Randomly choose a file
            neg_file = random.choice(self.anndata_files)["local_path"]
            # Randomly choose a sample from that file
            neg_sample = random.choice(list(all_captions[neg_file].keys()))
            neg_caption = all_captions[neg_file][neg_sample]

            # Check if this is actually a negative example
            if neg_caption != current_caption:
                sentence_1 = json.dumps(
                    {"file_path": current_dataset_path, "sample_id": current_sample}
                )
                sentence_2 = neg_caption
                label = 0.0
                return (sentence_1, sentence_2, label)

    def _check_sharelink(self, share_link: str) -> bool:
        """
        Validate that the Nextcloud share link is working.

        Parameters
        ----------
        share_link : str
            The share link URL.

        Returns
        -------
        bool
            True if the share link is working; False otherwise.
        """
        with tempfile.NamedTemporaryFile(suffix=".h5ad") as temp_file:
            if download_file_from_share_link(share_link, temp_file.name):
                return True
            else:
                return False

    def get_dataset(self) -> Dataset:
        """
        Create and return a Hugging Face Dataset in the specified format.

        Depending on `self.dataset_format`, the returned dataset structure varies:

        - "pairs": Each record is a pair with keys:
            - anndata_ref: JSON string with file_path and sample_id.
            - caption: Caption text (positive or negative).
            - label: 1.0 for positive examples, 0.0 for negatives.
          Negative examples are generated as separate records per positive sample.
        
        - "multiplets": Each record is a multiplet with keys:
            - anndata_ref: JSON string with file_path and sample_id.
            - anchor: The caption from the current sample.
            - positive: Duplicate of the caption (serving as the positive example).
            - negatives: List of negative captions (length defined by negatives_per_sample).
        
        - "single": Each record contains only the caption (and its reference) suitable for inference.
            - anndata_ref: JSON string with file_path and sample_id.
            - caption: Caption text.

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset constructed from the AnnData files.

        Notes
        -----
        Captions are built using the provided `caption_constructor` and are sourced from AnnData files added via
        `add_anndata`.
        """
        hf_data = []
        all_captions = {}  # Nested dict: {file_path: {sample_id: caption}}

        # Build & retrieve captions for each file
        for files in self.anndata_files:
            file_path = files["local_path"]
            dataset_path = files["dataset_path"]
            self.buildCaption(file_path)
            all_captions[file_path] = self.getCaption(file_path)

        # Build dataset entries based on the selected format
        for files in self.anndata_files:
            file_path = files["local_path"]
            dataset_path = files["dataset_path"]
            caption_dict = all_captions[file_path]

            for sample_id, caption in caption_dict.items():
                ref_json = json.dumps({"file_path": dataset_path, "sample_id": sample_id})
                if self.dataset_format == "pairs":
                    # Positive example
                    hf_data.append(
                        {"anndata_ref": ref_json, "caption": caption, "label": 1.0}
                    )
                    # Negative examples (if any)
                    for _ in range(self.negatives_per_sample):
                        neg_ref, neg_caption, neg_label = self._create_negative_example(
                            dataset_path, sample_id, caption, all_captions
                        )
                        hf_data.append(
                            {"anndata_ref": neg_ref, "caption": neg_caption, "label": neg_label}
                        )
                elif self.dataset_format == "multiplets":
                    entry = {
                        "anndata_ref": ref_json, # The anchor
                        "positive": caption,
                    }
                    for idx in range(1, self.negatives_per_sample + 1):
                        # Generate a negative example and add it as a separate column
                        _, neg_caption, _ = self._create_negative_example(
                            dataset_path, sample_id, caption, all_captions
                        )
                        entry[f"negative_{idx}"] = neg_caption
                    hf_data.append(entry)
                elif self.dataset_format == "single":
                    # Only include the anndata ref
                    hf_data.append({"anndata_ref": ref_json})
                else:
                    error_msg = (
                        "Invalid dataset_format. Choose from 'pairs', 'multiplets', or 'single'."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        hf_dataset = Dataset.from_list(hf_data)
        return hf_dataset

    def get_inference_dataset(
        self,
    ) -> tuple[list[dict[str, str]], list[str], list[str]]:
        """
        Build a dataset from AnnData files suitable for SentenceTransformer.encode.

        The method returns parallel lists: a metadata list (with JSON strings of file_path and sample_id),
        a captions list, and a sample_ids list. This is useful for inference scenarios where the exact order of
        samples must be maintained.

        Returns
        -------
        metadata_list : list of dict
            Each dictionary is a JSON-encoded string with keys {"file_path", "sample_id"}.
        captions_list : list of str
            A list of caption strings corresponding to each sample.
        sample_ids : list of str
            A list of sample IDs in the same order as the captions.

        Notes
        -----
        Captions are built using the provided `caption_constructor` and are sourced from AnnData files added via
        `add_anndata`.
        """
        metadata_list = []
        captions_list = []
        sample_ids = []

        # For each file, ensure captions are built and then retrieved
        for files in self.anndata_files:
            file_path = files["local_path"]
            dataset_path = files["dataset_path"]
            logger.info("Building caption for inference from file: %s", file_path)
            self.buildCaption(file_path)

            logger.info("Retrieving captions for inference from file: %s", file_path)
            caption_dict = self.getCaption(file_path)

            # Gather data into parallel lists
            for sid, caption in caption_dict.items():
                metadata_list.append(
                    json.dumps({"file_path": dataset_path, "sample_id": sid})
                )
                captions_list.append(caption)
                sample_ids.append(sid)

        logger.info("Constructed inference dataset with %d samples.", len(sample_ids))
        return metadata_list, captions_list, sample_ids

    def clear(self) -> None:
        """
        Clear all stored data in the constructor.
        
        This removes all added AnnData files, sample ID keys, and any cached dataset entries.
        """
        self.anndata_files.clear()
        self.sample_id_keys.clear()
        self.dataset.clear()