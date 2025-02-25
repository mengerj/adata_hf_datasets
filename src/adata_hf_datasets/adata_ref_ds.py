import json
import logging
import random
import pandas as pd
import anndata
from datasets import Dataset
from typing import Optional
import tempfile
from .file_utils import (
    save_and_upload_adata,
    download_file_from_share_link,
    save_embedding_data,
)
import os

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
        caption_constructor=None,
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
        if caption_constructor is None and dataset_format != "single":
            error_msg = "caption_constructor must be provided for dataset formats other than 'single'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.caption_constructor = caption_constructor
        self.negatives_per_sample = negatives_per_sample
        self.dataset_format = dataset_format.lower()
        if self.dataset_format not in ("pairs", "multiplets", "single"):
            error_msg = (
                "dataset_format must be one of 'pairs', 'multiplets', or 'single'."
            )
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

    def add_anndata(
        self,
        file_path: str,
        sample_id_key: str | None = None,
        obsm_keys: Optional[list[str]] = None,
    ) -> None:
        """
        Add an AnnData file to the constructor.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.
        sample_id_key : str or None, optional
            Optional key in adata.obs to use for sample IDs. If None, uses adata.obs.index.
        obsm_keys : list of str, optional
            List of .obsm keys to extract from the AnnData object.
            Each extracted layer is stored separately and its reference is added to the dataset record.

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
        path_for_dataset = (
            file_path  # Default: store the local path for dataset creation
        )
        self._check_sample_id_uniqueness(adata, file_path, sample_id_key)

        # Upload to Nextcloud if enabled
        if self.store_nextcloud and self.nextcloud_config:
            share_link = save_and_upload_adata(
                adata, file_path, self.nextcloud_config, create_share_link=True
            )
            if share_link:
                path_for_dataset = share_link
                # if not self._check_sharelink(share_link):
                #    logger.error(f"Nextcloud sharelink {share_link} not working")
                #    raise ValueError(f"Nextcloud sharelink {share_link} not working")
            else:
                logger.error("Failed to upload file to Nextcloud: %s", file_path)
                raise ValueError(f"Nextcloud upload failed for {file_path}")

        # Create a record for this file.
        file_record = {"local_path": file_path, "dataset_path": path_for_dataset}

        # If obsm keys are provided, extract and save the embedding objects.
        if obsm_keys:
            extracted = self.extract_obsm_layers(adata, obsm_keys)
            file_record["embeddings"] = {}
            for key, df in extracted.items():
                # Construct a local file name for the embedding.
                embedding_local_path = (
                    f"{os.path.splitext(file_path)[0]}_{key}_embedding.npz"
                )
                share_link = None
                share_link = save_embedding_data(
                    df,
                    embedding_local_path,
                    self.nextcloud_config if self.store_nextcloud else None,
                    create_share_link=True if self.store_nextcloud else False,
                )
                # Store the share link if available, otherwise the local path.
                file_record["embeddings"][key] = (
                    share_link if share_link else embedding_local_path
                )

        self.anndata_files.append(file_record)
        self.sample_id_keys[file_path] = sample_id_key
        logger.info("Successfully added AnnData file: %s", file_path)

    def extract_obsm_layers(
        self, adata: anndata.AnnData, obsm_keys: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        Extract specified .obsm layers from an AnnData object and return each as a pandas DataFrame.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing .obsm layers.
        obsm_keys : list of str
            List of keys corresponding to the .obsm layers to extract.

        Returns
        -------
        dict of {str: pd.DataFrame}
            Dictionary mapping each obsm key to a DataFrame. Each DataFrame uses adata.obs.index as its index,
            and its rows are the numeric vectors from the corresponding obsm layer.

        Raises
        ------
        KeyError
            If any provided obsm key is not found in adata.obsm.
        """
        extracted = {}
        for key in obsm_keys:
            if key not in adata.obsm.keys():
                error_msg = f"obsm key '{key}' not found in the AnnData object."
                logger.error(error_msg)
                raise KeyError(error_msg)
            # Create a DataFrame: rows are samples (using adata.obs.index) and columns are the embedding dimensions.
            df = pd.DataFrame(adata.obsm[key], index=adata.obs.index)
            extracted[key] = df
        return extracted

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
        current_file_path: str,
        current_file_record: str,
        current_sample: str,
        current_caption: str,
        all_captions: dict[str, dict[str, str]],
    ) -> tuple[str, str, float]:
        """Create a negative example, ensuring it's truly negative."""

        possible_negatives = all_captions[
            current_file_path
        ]  # Get captions from the same file
        possible_negative_ids = list(possible_negatives.keys())
        random.shuffle(
            possible_negative_ids
        )  # Shuffle to avoid always picking the same ones.

        for neg_sample in possible_negative_ids:
            neg_caption = possible_negatives[neg_sample]
            if (
                neg_caption != current_caption and neg_sample != current_sample
            ):  # Check sample id too
                sentence_1 = json.dumps(
                    {"file_record": current_file_record, "sample_id": current_sample}
                )
                sentence_2 = neg_caption
                label = 0.0
                return (sentence_1, sentence_2, label)

        # Handle the case where no true negative is found (rare, but possible)
        # In this case, choose a negative from a different file.
        # This is not ideal, but better than returning a positive example.
        other_files = [
            f for f in self.anndata_files if f["local_path"] != current_file_path
        ]
        if other_files:
            neg_file = random.choice(other_files)["local_path"]
            neg_sample = random.choice(list(all_captions[neg_file].keys()))
            neg_caption = all_captions[neg_file][neg_sample]
            sentence_1 = json.dumps(
                {"file_record": current_file_record, "sample_id": current_sample}
            )
            sentence_2 = neg_caption
            label = 0.0
            return (sentence_1, sentence_2, label)
        else:
            raise ValueError("No true negative example could be found.")

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
            - anndata_ref: JSON string with file_path and sample_id. (anchor)
            - positive: The caption of the current sample (serving as the positive example).
            - negative_1: A negative caption
            - negative_2: ...
            - negative_n: ...

        - "single": Each record contains only the AnnData reference (suitable for inference).
            - anndata_ref: JSON string with file_path and sample_id.

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset constructed from the AnnData files.

        Notes
        -----
        Captions are built using the provided `caption_constructor` and are sourced from AnnData files added via
        `add_anndata`. However, captions are **not generated** for the `"single"` dataset format.
        """
        hf_data = []

        all_captions = {}  # Nested dict: {file_path: {sample_id: caption}}

        # Skip caption construction for "single" dataset format
        if self.dataset_format != "single":
            for files in self.anndata_files:
                file_path = files["local_path"]
                self.buildCaption(file_path)  # Build captions only if needed
                all_captions[file_path] = self.getCaption(file_path)

        # Build dataset entries based on the selected format
        for files in self.anndata_files:
            file_path = files["local_path"]
            file_record = {
                k: v for k, v in files.items() if k != "local_path"
            }  # create a new dict to avoid in place modification

            # No caption retrieval for "single" dataset format
            caption_dict = (
                all_captions.get(file_path, {})
                if self.dataset_format != "single"
                else {}
            )

            for sample_id in (
                caption_dict.keys()
                if self.dataset_format != "single"
                else self._get_sample_ids(file_path)
            ):
                ref_json = json.dumps(
                    {"file_record": file_record, "sample_id": sample_id}
                )

                if self.dataset_format == "pairs":
                    # Positive example
                    hf_data.append(
                        {
                            "anndata_ref": ref_json,
                            "caption": caption_dict[sample_id],
                            "label": 1.0,
                        }
                    )

                    # Negative examples (if any)
                    for _ in range(self.negatives_per_sample):
                        neg_ref, neg_caption, neg_label = self._create_negative_example(
                            file_path,
                            file_record,
                            sample_id,
                            caption_dict,
                            all_captions,
                        )
                        hf_data.append(
                            {
                                "anndata_ref": neg_ref,
                                "caption": neg_caption,
                                "label": neg_label,
                            }
                        )

                elif self.dataset_format == "multiplets":
                    entry = {
                        "anndata_ref": ref_json,
                        "positive": caption_dict[sample_id],
                    }
                    for idx in range(1, self.negatives_per_sample + 1):
                        _, neg_caption, _ = self._create_negative_example(
                            file_path,
                            file_record,
                            sample_id,
                            caption_dict,
                            all_captions,
                        )
                        entry[f"negative_{idx}"] = neg_caption
                    hf_data.append(entry)

                elif self.dataset_format == "single":
                    # Only include the anndata ref, no captions
                    hf_data.append({"anndata_ref": ref_json})

                else:
                    error_msg = "Invalid dataset_format. Choose from 'pairs', 'multiplets', or 'single'."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        hf_dataset = Dataset.from_list(hf_data)
        return hf_dataset

    def _get_sample_ids(self, file_path):
        """
        Retrieve sample IDs from an AnnData file for 'single' dataset format.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.

        Returns
        -------
        list
            List of sample IDs.
        """
        import anndata

        adata = anndata.read_h5ad(file_path)
        return (
            adata.obs.index.tolist()
        )  # Assuming sample IDs are stored in adata.obs.index


class SimpleCaptionConstructor:
    """Construct captions for each sample by concatenating values from specified obs keys"""

    def __init__(self, obs_keys: list[str] | str, separator: str = " "):
        """
        Initialize the SimpleCaptionConstructor.

        Args:
            obs_keys: List of keys from adata.obs to include in the caption
            separator: String to use between concatenated values (default: space)
        """
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]
        self.obs_keys = obs_keys
        self.separator = separator

    def construct_captions(self, adata: anndata.AnnData) -> None:
        """Include captions for each sample

        Construct captions by concatenating values from specified obs keys.
        Adds a 'caption' column to adata.obs.

        Args:
            adata: AnnData object to process

        Raises
        ------
            KeyError: If any of the specified obs_keys is not found in adata.obs
        """
        # Verify all keys exist
        missing_keys = [key for key in self.obs_keys if key not in adata.obs.columns]
        if missing_keys:
            raise KeyError(
                f"The following keys were not found in adata.obs: {missing_keys}"
            )

        # Convert all values to strings and replace NaN with empty string
        str_values = [
            adata.obs[key].astype(str).replace("nan", "") for key in self.obs_keys
        ]

        # Concatenate the values
        adata.obs["caption"] = pd.DataFrame(str_values).T.agg(
            self.separator.join, axis=1
        )
