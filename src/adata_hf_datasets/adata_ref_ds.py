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
    """Class to generate a dataset compatible with the SentenceTransformer library from anndata files."""

    def __init__(
        self,
        caption_constructor,
        negatives_per_sample: int = 1,
        store_nextcloud: bool = False,
        nextcloud_config: Optional[dict] = None,
        push_to_hf: bool = False,
        dataset_name: str | None = None,
        hf_username: str | None = None,
    ):
        """
        Initialize the AnnDataSetConstructor.

        Parameters
        ----------
        caption_constructor
            Constructor for creating captions
        negatives_per_sample
            Number of negative examples to create per positive example
        store_nextcloud
            If True, upload AnnData files to Nextcloud and store share links instead of local paths.
        nextcloud_config (dict, optional):
        Configuration dictionary for Nextcloud which contains:
            'url' (str): URL to the Nextcloud server.
            'username' (str): Username for Nextcloud.
            'password' (str): Password for Nextcloud.
            'remote_path' (str): Remote path in Nextcloud where the file will be uploaded.

        Example
        ---------
        nextcloud_config={
            'url': 'https://nxc-fredato.imbi.uni-freiburg.de',
            'username': 'your_username',
            'password': 'your_password',
            'remote_path': '/path/on/nextcloud/file.h5ad.gz'}

        """
        self.caption_constructor = caption_constructor
        self.negatives_per_sample = negatives_per_sample
        self.store_nextcloud = store_nextcloud
        self.nextcloud_config = nextcloud_config if store_nextcloud else None
        self.push_to_hf = push_to_hf
        if push_to_hf and (dataset_name is None or hf_username is None):
            raise ValueError(
                "Please provide a hf_username and a dataset_name for push_to_hf=True."
            )
        self.dataset_name = dataset_name
        self.anndata_files = []
        self.sample_id_keys = {}
        self.dataset = []

    def _check_sample_id_uniqueness(
        self, adata: anndata.AnnData, file_path: str, sample_id_key: str | None
    ) -> None:
        """
        Check if sample IDs are unique for the given anndata object.

        Parameters
        ----------
        adata
            AnnData object to check
        file_path
            Path to the anndata file (for error message)
        sample_id_key
            Key in adata.obs to use for sample IDs, if None uses adata.obs.index

        Raises
        ------
            ValueError: If sample IDs are not unique
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
        Add an anndata file to the constructor.

        Parameters
        ----------
        file_path
            Path to the anndata file
        sample_id_key
            Optional key in adata.obs to use for sample IDs. If None, uses adata.obs.index
        """
        self.is_zarr = False
        self.is_h5ad = False
        # 1. Check extension
        if file_path.endswith(".zarr") or file_path.endswith(".zarr/"):
            self.is_zarr = True
        elif file_path.endswith(".h5ad"):
            self.is_h5ad = True
        else:
            logger.error("Unsupported anndata format for file: %s", file_path)
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
        path_for_dataset = file_path  # Store path for dataset creation
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
        )  # First is needed to load the data now for captions, second is stored in dataset later download
        self.sample_id_keys[file_path] = sample_id_key
        logger.info("Successfully added anndata file: %s", file_path)

    def buildCaption(self, file_path: str) -> None:
        """
        Build captions for an anndata file using the provided caption constructor.

        Args:
            file_path: Path to the anndata file
            caption_constructor: Instance of a caption constructor class
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
        Get a dictionary mapping sample IDs to captions from an anndata file.

        Args:
            file_path: Path to the anndata file

        Returns
        -------
            Dict mapping sample IDs to captions
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
        Create a negative example by finding a caption that doesn't match the current sample.

        Parameters
        ----------
        current_dataset_path : str
            Path used to reference the current file
        current_sample : str
            ID of the current sample
        current_caption : str
            Caption of the current sample
        all_captions : dict
            Nested dict mapping file paths to {sample_id: caption} dicts

        Returns
        -------
        sentence_1 : str
            JSON string containing file_path and sample_id
        sentence_2 : str
            The negative caption
        label : float
            0.0 for negative
        """
        while True:
            # Randomly choose a file
            neg_file = random.choice(self.anndata_files)
            neg_file = neg_file["local_path"]
            # Randomly choose a sample from that file
            neg_sample = random.choice(list(all_captions[neg_file].keys()))
            neg_caption = all_captions[neg_file][neg_sample]

            # Check if this is actually a negative example
            if neg_caption != current_caption:
                # store metadata in JSON so we keep a single string
                sentence_1 = json.dumps(
                    {"file_path": current_dataset_path, "sample_id": current_sample}
                )
                # sentence_1 = {"file_path": current_file, "sample_id": current_sample}
                sentence_2 = neg_caption
                label = 0.0
                return (sentence_1, sentence_2, label)

    def _check_sharelink(self, share_link: str) -> anndata.AnnData:
        """This is mainly to validate that the share link is working."""

        # create a temporary file to download the file
        with tempfile.NamedTemporaryFile(suffix=".h5ad") as temp_file:
            if download_file_from_share_link(share_link, temp_file.name):
                return True
            else:
                return False

    def get_dataset(self) -> Dataset:
        """
        Create and return a Hugging Face Dataset containing pairs of sentences and a label.

        The resulting dataset has these columns:
        - sentence_1: JSON-serialized string with file_path and sample_id
        - sentence_2: caption string
        - label: float (1.0 or 0.0)

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset with columns [sentence_1, sentence_2, label].

        Notes
        -----
        1. This method automatically calls `buildCaption(...)` on each file
           and ensures the 'caption' column is present.
        2. The data is sourced from .zarr files previously added via `add_anndata`.
        3. The dataset is not tokenized; you can apply tokenization separately.
        """
        # import json

        # Prepare a list of dicts suitable for HF Dataset
        hf_data = []
        all_captions = {}  # Nested dict: {file_path: {sample_id: caption}}

        # Build & retrieve captions for each file
        for files in self.anndata_files:
            file_path = files["local_path"]
            self.buildCaption(file_path)
            all_captions[file_path] = self.getCaption(file_path)

        # Create positive and negative examples
        for files in self.anndata_files:
            file_path = files["local_path"]
            dataset_path = files["dataset_path"]
            caption_dict = all_captions[file_path]

            for sample_id, caption in caption_dict.items():
                # Positive example
                sentence_1 = json.dumps(
                    {"file_path": dataset_path, "sample_id": sample_id}
                )
                # sentence_1 = {"file_path": file_path, "sample_id": sample_id}
                sentence_2 = caption
                label = 1.0

                hf_data.append(
                    {"anndata_ref": sentence_1, "caption": sentence_2, "label": label}
                )

                # Negative examples
                for _ in range(self.negatives_per_sample):
                    neg_sentence_1, neg_sentence_2, neg_label = (
                        self._create_negative_example(
                            dataset_path, sample_id, caption, all_captions
                        )
                    )
                    hf_data.append(
                        {
                            "anndata_ref": neg_sentence_1,
                            "caption": neg_sentence_2,
                            "label": neg_label,
                        }
                    )
        hf_dataset = Dataset.from_list(hf_data)
        if self.push_to_hf:
            hf_dataset.push_to_hub(f"jo-mengr/{self.dataset_name}", private=True)
        logger.info("Created %d examples for Hugging Face dataset.", len(hf_data))
        return hf_dataset

    def get_inference_dataset(
        self,
    ) -> tuple[list[dict[str, str]], list[str], list[str]]:
        """Build a dataset from an anndata file suitable for SentenceTransformer.encode.

        Build and return separate lists for inference: a list of metadata dicts, a list of captions,
        and a parallel list of sample IDs (all in the same order).

        The method reads each .zarr file (adding captions if not already present via `buildCaption`),
        then extracts sample IDs and captions. This is useful for inference scenarios where you
        need to maintain the exact order of samples for external reference.

        Returns
        -------
        metadata_list : list of dict
            Each dictionary contains ``{"file_path": <path_to_zarr>, "sample_id": <sample_id>}``.
            This is useful if you need to retrieve the file path and sample ID for downstream processing.
        captions_list : list of str
            A list of caption strings corresponding to each sample in the dataset.
        sample_ids : list of str
            A list of sample IDs in the same index order as the captions_list.

        Notes
        -----
        - This method internally calls ``buildCaption(file_path)`` to ensure each file
        is annotated with a ``"caption"`` column. If the column already exists, the
        constructor logic may simply overwrite or skip as needed.
        - The data is sourced from the .zarr files previously added via ``add_anndata``.
        - Logging messages are issued at various points to indicate progress.
        """
        metadata_list = []
        captions_list = []
        sample_ids = []

        # For each file, ensure captions are built and then retrieved
        for files in self.anndata_files:
            file_path = files["local_path"]
            dataset_path = files["dataset_path"]
            logger.info("Building caption for inference from file: %s", file_path)
            self.buildCaption(
                file_path
            )  # This will overwrite the .zarr file if new captions were generated

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
        """Clear all stored data in the constructor."""
        self.anndata_files.clear()
        self.sample_id_keys.clear()
        self.dataset.clear()


class SimpleCaptionConstructor:
    """Construct captions for each sample by concatenating values from specified obs keys"""

    def __init__(self, obs_keys: list[str], separator: str = " "):
        """
        Initialize the SimpleCaptionConstructor.

        Args:
            obs_keys: List of keys from adata.obs to include in the caption
            separator: String to use between concatenated values (default: space)
        """
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
