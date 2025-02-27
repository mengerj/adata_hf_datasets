import json
import logging
import random
import pandas as pd
import anndata
from datasets import Dataset
from typing import Optional, Union
import tempfile
import os
from adata_hf_datasets.file_utils import (
    save_and_upload_adata,
    download_file_from_share_link,
    save_embedding_data,
)

logger = logging.getLogger(__name__)


class AnnDataSetConstructor:
    """
    Class to generate a dataset compatible with the SentenceTransformer library from AnnData files.

    Data is sourced from AnnData files (.zarr or .h5ad) that are added via `add_anndata`.
    The generated dataset can be created in one of three formats:

    - "pairs": Each record is a pair containing an anchor and a positive example,
               plus two negative pairs (one with a randomly chosen modality for the negative and
               one with the same modality as the anchor).
    - "multiplets": Each record contains an anchor, a positive, and a list of negative examples.
    - "single": Each record contains only a single caption (useful for inference).
    """

    def __init__(
        self,
        caption_constructor: Optional[object] = None,
        negatives_per_sample: int = 1,
        dataset_format: str = "pairs",
        store_nextcloud: bool = False,
        nextcloud_config: Optional[dict] = None,
    ):
        """
        Initialize the AnnDataSetConstructor.

        Parameters
        ----------
        caption_constructor : callable or None
            Constructor for creating captions from AnnData files. Must be provided for formats other than "single".
        negatives_per_sample : int, optional
            (Unused in the new pairs logic) Number of negative examples per positive sample.
        dataset_format : str, optional
            Format of the dataset to construct. Allowed values are:
              - "pairs": Each record is a pair with one positive example and two negative pairs.
              - "multiplets": Each record is a multiplet consisting of an anchor, a positive, and a list of negatives.
              - "single": Each record contains only an AnnData reference (suitable for inference).
            Default is "pairs".
        store_nextcloud : bool, optional
            If True, upload AnnData files to Nextcloud and store share links instead of local paths.
        nextcloud_config : dict, optional
            Configuration for Nextcloud. Must include keys: 'url', 'username', 'password', 'remote_path'.

        Raises
        ------
        ValueError
            If caption_constructor is missing (when required) or if an invalid dataset_format is provided.
        """
        if caption_constructor is None and dataset_format.lower() != "single":
            error_msg = "caption_constructor must be provided for dataset formats other than 'single'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.caption_constructor = caption_constructor
        self.negatives_per_sample = negatives_per_sample  # kept for compatibility
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
        self, adata: anndata.AnnData, file_path: str, sample_id_key: Optional[str]
    ) -> None:
        """
        Check that sample IDs are unique for the provided AnnData object.

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
        sample_id_key: Optional[str] = None,
        obsm_keys: Optional[list[str]] = None,
    ) -> None:
        """
        Add an AnnData file to the constructor.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.
        sample_id_key : str or None, optional
            Key in adata.obs to use as sample IDs. If None, uses adata.obs.index.
        obsm_keys : list of str, optional
            List of .obsm keys to extract from the AnnData object.

        Raises
        ------
        ValueError
            If the file format is unsupported or if the file has already been added.
        """
        self.is_zarr = False
        self.is_h5ad = False
        if file_path.endswith(".zarr") or file_path.endswith(".zarr/"):
            self.is_zarr = True
        elif file_path.endswith(".h5ad"):
            self.is_h5ad = True
        else:
            logger.error("Unsupported AnnData format for file: %s", file_path)
            raise ValueError(
                f"File {file_path} does not appear to be .zarr or .h5ad format."
            )

        for files in self.anndata_files:
            if file_path in files["local_path"]:
                logger.error(
                    "File %s has already been added to the constructor.", file_path
                )
                raise ValueError(f"File {file_path} has already been added.")

        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        elif self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        self.local_path = file_path
        path_for_dataset = file_path

        self._check_sample_id_uniqueness(adata, file_path, sample_id_key)

        if self.store_nextcloud and self.nextcloud_config:
            share_link = save_and_upload_adata(
                adata, file_path, self.nextcloud_config, create_share_link=True
            )
            if share_link:
                path_for_dataset = share_link
            else:
                logger.error("Failed to upload file to Nextcloud: %s", file_path)
                raise ValueError(f"Nextcloud upload failed for {file_path}")

        file_record = {"local_path": file_path, "dataset_path": path_for_dataset}
        if obsm_keys:
            extracted = self.extract_obsm_layers(adata, obsm_keys)
            file_record["embeddings"] = {}
            for key, df in extracted.items():
                embedding_local_path = (
                    f"{os.path.splitext(file_path)[0]}_{key}_embedding.npz"
                )
                share_link = save_embedding_data(
                    df,
                    embedding_local_path,
                    self.nextcloud_config if self.store_nextcloud else None,
                    create_share_link=True if self.store_nextcloud else False,
                )
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
        Extract specified .obsm layers from an AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing .obsm layers.
        obsm_keys : list of str
            List of keys corresponding to the .obsm layers to extract.

        Returns
        -------
        dict of {str: pd.DataFrame}
            Mapping each obsm key to its corresponding DataFrame.
        """
        extracted = {}
        for key in obsm_keys:
            if key not in adata.obsm.keys():
                error_msg = f"obsm key '{key}' not found in the AnnData object."
                logger.error(error_msg)
                raise KeyError(error_msg)
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
        Captions are constructed and stored in adata.obs["caption"].
        """
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        elif self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        self.caption_constructor.construct_captions(adata)
        if self.is_zarr:
            adata.write_zarr(file_path)
        elif self.is_h5ad:
            adata.write_h5ad(file_path)

    def getCaption(self, file_path: str) -> dict[str, str]:
        """
        Get a mapping from sample IDs to captions from an AnnData file.

        Parameters
        ----------
        file_path : str
            Path to the AnnData file.

        Returns
        -------
        dict
            Mapping from sample ID to caption.

        Raises
        ------
        ValueError
            If the AnnData file lacks a 'caption' column.
        """
        if self.is_zarr:
            adata = anndata.read_zarr(file_path)
        elif self.is_h5ad:
            adata = anndata.read_h5ad(file_path)
        if "caption" not in adata.obs.columns:
            raise ValueError(f"No 'caption' column found in {file_path}")

        sample_id_key = self.sample_id_keys[file_path]
        sample_ids = (
            adata.obs.index if sample_id_key is None else adata.obs[sample_id_key]
        )
        return dict(zip(sample_ids, adata.obs["caption"], strict=False))

    def _get_negative_sample(
        self,
        current_file_path: str,
        current_file_record: dict,
        current_sample: str,
        current_caption: str,
        desired_modality: str,
        all_captions: dict[str, dict[str, str]],
    ) -> str:
        """
        Retrieve a negative sample (from the same file if possible, otherwise from a different file)
        that satisfies: sample ID is different and its caption differs from the current caption.

        Parameters
        ----------
        current_file_path : str
            File path of the current sample.
        current_file_record : dict
            File record for the current file.
        current_sample : str
            Sample ID of the current sample.
        current_caption : str
            Caption of the current sample.
        desired_modality : str
            Negative modality to use: "caption" or "file_record".
        all_captions : dict
            Nested mapping {file_path: {sample_id: caption}} for all files.

        Returns
        -------
        str
            If desired_modality is "caption": the negative sample's caption text.
            If desired_modality is "file_record": a JSON string representing the negative sample's file record and sample ID.

        Raises
        ------
        ValueError
            If no appropriate negative sample can be found.
        """
        # Try candidates from the same file first.
        candidates = [
            sample
            for sample in all_captions[current_file_path]
            if sample != current_sample
            and all_captions[current_file_path][sample] is not None
            and all_captions[current_file_path][sample] != current_caption
        ]
        if candidates:
            neg_sample = random.choice(candidates)
            neg_caption = all_captions[current_file_path][neg_sample]
            neg_file_record = current_file_record
        else:
            # Fallback: search in other files.
            other_files = [
                f for f in self.anndata_files if f["local_path"] != current_file_path
            ]
            for neg_file in other_files:
                neg_file_path = neg_file["local_path"]
                candidates = [
                    sample
                    for sample in all_captions.get(neg_file_path, {})
                    if all_captions[neg_file_path][sample] is not None
                    and all_captions[neg_file_path][sample] != current_caption
                ]
                if candidates:
                    neg_sample = random.choice(candidates)
                    neg_caption = all_captions[neg_file_path][neg_sample]
                    neg_file_record = {
                        k: v for k, v in neg_file.items() if k != "local_path"
                    }
                    break
            else:
                raise ValueError("No true negative example could be found.")

        if desired_modality == "caption":
            return neg_caption
        elif desired_modality == "file_record":
            return json.dumps({"file_record": neg_file_record, "sample_id": neg_sample})
        else:
            raise ValueError(
                "desired_modality must be either 'caption' or 'file_record'."
            )

    def get_dataset(self) -> Dataset:
        """
        Create and return a Hugging Face Dataset in the specified format.

        For "multiplets" format, each record contains an anchor (a JSON file_record reference),
        a positive caption, and a list of negatives. Each negative is selected by randomly choosing
        a modality (either a caption or a file_record) while ensuring that:
        - The negative comes from a different sample.
        - The negative sample’s caption is different from the current caption.

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset constructed from the AnnData files.
        """
        hf_data = []
        all_captions = {}  # Nested dict: {file_path: {sample_id: caption}}

        if self.dataset_format != "single":
            for files in self.anndata_files:
                file_path = files["local_path"]
                self.buildCaption(file_path)
                all_captions[file_path] = self.getCaption(file_path)

        if self.dataset_format == "pairs":
            # ... (pairs branch remains as you previously defined)
            for files in self.anndata_files:
                file_path = files["local_path"]
                file_record = {k: v for k, v in files.items() if k != "local_path"}
                caption_dict = all_captions.get(file_path, {})

                for sample_id, current_caption in caption_dict.items():
                    anchor_modality = random.choice(["file_record", "caption"])
                    if anchor_modality == "file_record":
                        anndata_ref = json.dumps(
                            {"file_record": file_record, "sample_id": sample_id}
                        )
                        positive = current_caption
                    else:
                        anndata_ref = current_caption
                        positive = json.dumps(
                            {"file_record": file_record, "sample_id": sample_id}
                        )

                    hf_data.append(
                        {
                            "anndata_ref": anndata_ref,
                            "caption": positive,
                            "label": 1.0,
                        }
                    )

                    neg_mod_random = random.choice(["caption", "file_record"])
                    neg_candidate_random = self._get_negative_sample(
                        current_file_path=file_path,
                        current_file_record=file_record,
                        current_sample=sample_id,
                        current_caption=current_caption,
                        desired_modality=neg_mod_random,
                        all_captions=all_captions,
                    )
                    hf_data.append(
                        {
                            "anndata_ref": anndata_ref,
                            "caption": neg_candidate_random,
                            "label": 0.0,
                        }
                    )

                    neg_candidate_same = self._get_negative_sample(
                        current_file_path=file_path,
                        current_file_record=file_record,
                        current_sample=sample_id,
                        current_caption=current_caption,
                        desired_modality=anchor_modality,
                        all_captions=all_captions,
                    )
                    hf_data.append(
                        {
                            "anndata_ref": anndata_ref,
                            "caption": neg_candidate_same,
                            "label": 0.0,
                        }
                    )

        elif self.dataset_format == "multiplets":
            for files in self.anndata_files:
                file_path = files["local_path"]
                file_record = {k: v for k, v in files.items() if k != "local_path"}
                caption_dict = all_captions.get(file_path, {})
                for sample_id, current_caption in caption_dict.items():
                    # For multiplets, the anchor is always the JSON file_record reference,
                    # and the positive is the caption.
                    ref_json = json.dumps(
                        {"file_record": file_record, "sample_id": sample_id}
                    )
                    entry = {"anndata_ref": ref_json, "positive": current_caption}
                    # For negatives, randomly select a modality for each negative.
                    for idx in range(1, self.negatives_per_sample + 1):
                        neg_mod = random.choice(["caption", "file_record"])
                        neg_candidate = self._get_negative_sample(
                            current_file_path=file_path,
                            current_file_record=file_record,
                            current_sample=sample_id,
                            current_caption=current_caption,
                            desired_modality=neg_mod,
                            all_captions=all_captions,
                        )
                        entry[f"negative_{idx}"] = neg_candidate
                    hf_data.append(entry)

        elif self.dataset_format == "single":
            for files in self.anndata_files:
                file_path = files["local_path"]
                for sample_id in self._get_sample_ids(file_path):
                    ref_json = json.dumps(
                        {"file_record": files, "sample_id": sample_id}
                    )
                    hf_data.append({"anndata_ref": ref_json})
        else:
            error_msg = "Invalid dataset_format. Choose from 'pairs', 'multiplets', or 'single'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        hf_dataset = Dataset.from_list(hf_data)
        return hf_dataset

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

    def _get_sample_ids(self, file_path: str) -> list:
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
        adata = anndata.read_h5ad(file_path)
        return adata.obs.index.tolist()


'''
    def get_dataset(self) -> Dataset:
        """
        Create and return a Hugging Face Dataset in the specified format.

        For dataset_format "pairs" the following structure is used:
          - Each sample is used as an anchor exactly once.
          - The anchor modality is randomly chosen to be either:
              * A file_record (JSON with file_record and sample_id), or
              * A caption (string).
          - The positive example is the complementary modality.
          - Two negative pairs are generated:
              1. One where the negative modality is randomly chosen.
              2. One where the negative modality matches the anchor modality.
          - Each record in the dataset is a dictionary with keys:
              * "anchor": the anchor value,
              * "positive" or "negative": the corresponding pair value,
              * "label": 1.0 for positive pairs and 0.0 for negatives.

        For "multiplets" and "single" formats the behavior is unchanged from before.

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset constructed from the AnnData files.
        """
        hf_data = []
        all_captions = {}  # {file_path: {sample_id: caption}}

        if self.dataset_format != "single":
            for files in self.anndata_files:
                file_path = files["local_path"]
                self.buildCaption(file_path)
                all_captions[file_path] = self.getCaption(file_path)

        if self.dataset_format == "pairs":
            # For each file and sample, decide on anchor modality and create pairs.
            for files in self.anndata_files:
                file_path = files["local_path"]
                # Build a file-level record (exclude local_path)
                file_record = {k: v for k, v in files.items() if k != "local_path"}
                caption_dict = all_captions.get(file_path, {})

                for sample_id in caption_dict.keys():
                    current_caption = caption_dict[sample_id]
                    # Randomly decide the anchor modality for this sample.
                    anchor_modality = random.choice(["file_record", "caption"])
                    if anchor_modality == "file_record":
                        anchor = json.dumps({"file_record": file_record, "sample_id": sample_id})
                        positive = current_caption
                    else:
                        anchor = current_caption
                        positive = json.dumps({"file_record": file_record, "sample_id": sample_id})

                    # Positive pair record
                    hf_data.append({
                        "anchor": anchor,
                        "positive": positive,
                        "label": 1.0,
                    })

                    # Negative pair 1: negative modality chosen at random.
                    neg_mod_random = random.choice(["caption", "file_record"])
                    neg_negative_random = self._get_negative_sample(
                        current_file_path=file_path,
                        current_file_record=file_record,
                        current_sample=sample_id,
                        current_caption=current_caption,
                        desired_modality=neg_mod_random,
                        all_captions=all_captions,
                    )
                    hf_data.append({
                        "anchor": anchor,
                        "negative": neg_negative_random,
                        "label": 0.0,
                    })

                    # Negative pair 2: negative modality same as the anchor modality.
                    neg_negative_same = self._get_negative_sample(
                        current_file_path=file_path,
                        current_file_record=file_record,
                        current_sample=sample_id,
                        current_caption=current_caption,
                        desired_modality=anchor_modality,
                        all_captions=all_captions,
                    )
                    hf_data.append({
                        "anchor": anchor,
                        "negative": neg_negative_same,
                        "label": 0.0,
                    })

        elif self.dataset_format == "multiplets":
            # Keep multiplets behavior unchanged.
            for files in self.anndata_files:
                file_path = files["local_path"]
                file_record = {k: v for k, v in files.items() if k != "local_path"}
                caption_dict = all_captions.get(file_path, {})
                for sample_id in caption_dict.keys():
                    ref_json = json.dumps({"file_record": file_record, "sample_id": sample_id})
                    entry = {"anndata_ref": ref_json, "positive": caption_dict[sample_id]}
                    for idx in range(1, self.negatives_per_sample + 1):
                        _, neg_caption, _ = self._create_negative_example(
                            file_path,
                            file_record,
                            sample_id,
                            caption_dict[sample_id],
                            all_captions,
                        )
                        entry[f"negative_{idx}"] = neg_caption
                    hf_data.append(entry)

        elif self.dataset_format == "single":
            for files in self.anndata_files:
                file_path = files["local_path"]
                caption_dict = {}  # not needed for single format
                for sample_id in self._get_sample_ids(file_path):
                    ref_json = json.dumps({"file_record": files, "sample_id": sample_id})
                    hf_data.append({"anndata_ref": ref_json})
        else:
            error_msg = "Invalid dataset_format. Choose from 'pairs', 'multiplets', or 'single'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        hf_dataset = Dataset.from_list(hf_data)
        return hf_dataset
'''


class SimpleCaptionConstructor:
    """
    Construct captions for each sample by concatenating values from specified obs keys.
    """

    def __init__(self, obs_keys: Union[list[str], str], separator: str = " "):
        """
        Initialize the SimpleCaptionConstructor.

        Parameters
        ----------
        obs_keys : list of str or str
            Keys from adata.obs to include in the caption.
        separator : str, optional
            Separator used between values (default is a space).
        """
        if isinstance(obs_keys, str):
            obs_keys = [obs_keys]
        self.obs_keys = obs_keys
        self.separator = separator

    def construct_captions(self, adata: anndata.AnnData) -> None:
        """
        Construct captions by concatenating values from specified obs keys and add them as a new column.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object to process.

        Raises
        ------
        KeyError
            If any specified key is not found in adata.obs.
        """
        missing_keys = [key for key in self.obs_keys if key not in adata.obs.columns]
        if missing_keys:
            raise KeyError(
                f"The following keys were not found in adata.obs: {missing_keys}"
            )

        str_values = [
            adata.obs[key].astype(str).replace("nan", "") for key in self.obs_keys
        ]
        adata.obs["caption"] = pd.DataFrame(str_values).T.agg(
            self.separator.join, axis=1
        )

        '''
    def _get_negative_sample(
        self,
        current_file_path: str,
        current_file_record: dict,
        current_sample: str,
        current_caption: str,
        desired_modality: str,
        all_captions: dict[str, dict[str, str]],
    ) -> str:
        """
        Retrieve a negative sample (from the same file if possible, otherwise from a different file)
        that satisfies: sample ID is different and its caption differs from the current caption.

        Parameters
        ----------
        current_file_path : str
            File path of the current sample.
        current_file_record : dict
            File record (e.g. embeddings, paths) for the current file.
        current_sample : str
            Sample ID of the current sample.
        current_caption : str
            Caption of the current sample.
        desired_modality : str
            Negative modality to use: "caption" or "file_record".
        all_captions : dict
            Nested mapping {file_path: {sample_id: caption}} for all files.

        Returns
        -------
        str
            If desired_modality is "caption": the negative sample's caption text.
            If desired_modality is "file_record": a JSON string representing the negative sample's file record and sample ID.

        Raises
        ------
        ValueError
            If no appropriate negative sample can be found.
        """
        # First try the same file.
        candidates = [
            sample for sample in all_captions[current_file_path]
            if sample != current_sample and all_captions[current_file_path][sample] != current_caption
        ]
        if candidates:
            neg_sample = random.choice(candidates)
            neg_caption = all_captions[current_file_path][neg_sample]
            neg_file_record = current_file_record
        else:
            # Fallback: search in other files.
            other_files = [
                f for f in self.anndata_files if f["local_path"] != current_file_path
            ]
            if not other_files:
                logger.error("No other files found to search for negative examples.")
                raise ValueError("No true negative example could be found.")
            neg_file = random.choice(other_files)
            neg_file_path = neg_file["local_path"]
            candidates = [
                sample for sample in all_captions[neg_file_path]
                if all_captions[neg_file_path][sample] != current_caption
            ]
            if not candidates:
                logger.error("No negative examples found in other files.")
                raise ValueError("No true negative example could be found in other files.")
            neg_sample = random.choice(candidates)
            neg_caption = all_captions[neg_file_path][neg_sample]
            neg_file_record = {k: v for k, v in neg_file.items() if k != "local_path"}

        if desired_modality == "caption":
            return neg_caption
        elif desired_modality == "file_record":
            return json.dumps({"file_record": neg_file_record, "sample_id": neg_sample})
        else:
            raise ValueError("desired_modality must be either 'caption' or 'file_record'.")
'''
