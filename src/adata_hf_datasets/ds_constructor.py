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
               plus one negative pair (randomly chosen).
    - "multiplets": Each record contains an anchor, a positive, and a list of negative examples.
    - "single": Each record contains only a single caption (useful for inference).

    Parameters
    ----------
    caption_constructor : callable or None
        Constructor for creating captions from AnnData files. Must be provided for formats other than "single".
    negatives_per_sample : int, optional
        (Unused in the new pairs logic) Number of negative examples per positive sample.
    dataset_format : str, optional
        Format of the dataset to construct. Must be one of "pairs", "multiplets", or "single".
    store_nextcloud : bool, optional
        If True, upload AnnData files to Nextcloud and store share links instead of local paths.
    nextcloud_config : dict, optional
        Configuration for Nextcloud. Must include keys: 'url', 'username', 'password', 'remote_path'.
    """

    def __init__(
        self,
        caption_constructor: Optional[object] = None,
        negatives_per_sample: int = 1,
        dataset_format: str = "pairs",
        store_nextcloud: bool = False,
        nextcloud_config: Optional[dict] = None,
    ):
        if caption_constructor is None and dataset_format.lower() != "single":
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

        #: list of dict
        #  Each dict = {
        #      "local_path": str (the local anndata file path),
        #      "dataset_path": str (path or share link for huggingface dataset),
        #      "embeddings": optional dict of obsm key to path/link
        #  }
        self.anndata_files = []

        #: dict of file_path -> sample_id_key
        self.sample_id_keys = {}

        #: dict of file_path -> dict of sample_id -> batch_value
        self.batch_dicts = {}

        #: eventually constructed dataset
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
            duplicates = (
                sample_ids[sample_ids.duplicated()].unique()
                if hasattr(sample_ids, "duplicated")
                else []
            )
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
        batch_key: Optional[str] = None,
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
        batch_key : str or None, optional
            Key in adata.obs to use when choosing negatives. Prioritizes samples from the same batch as negatives, to mitigate batch effects.

        Raises
        ------
        ValueError
            If the file format is unsupported or if the file has already been added or Nextcloud upload fails.
        """
        self.batch_key = batch_key
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

        # check uniqueness of sample IDs
        self._check_sample_id_uniqueness(adata, file_path, sample_id_key)

        # store share link if Nextcloud is used
        if self.store_nextcloud and self.nextcloud_config:
            share_link = save_and_upload_adata(
                adata, file_path, self.nextcloud_config, create_share_link=True
            )
            if share_link:
                path_for_dataset = share_link
            else:
                logger.error("Failed to upload file to Nextcloud: %s", file_path)
                raise ValueError(f"Nextcloud upload failed for {file_path}")

        # create record
        file_record = {"local_path": file_path, "dataset_path": path_for_dataset}
        # optionally extract embeddings
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
                self._check_sharelink(share_link, suffix=".npz")

        self.anndata_files.append(file_record)
        self.sample_id_keys[file_path] = sample_id_key

        # if batch_key was given, store the batch for each sample
        if self.batch_key is not None:
            if self.batch_key not in adata.obs.columns:
                msg = f"batch_key '{self.batch_key}' not found in adata.obs columns for file: {file_path}"
                logger.error(msg)
                raise ValueError(msg)

            # sample IDs
            if sample_id_key is None:
                sample_ids = adata.obs.index
            else:
                sample_ids = adata.obs[sample_id_key]

            # build dictionary of sample_id -> batch_value
            batch_values = adata.obs[self.batch_key]
            self.batch_dicts[file_path] = dict(zip(sample_ids, batch_values))

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

        # The user must have provided a constructor for non-single formats
        if self.caption_constructor is not None:
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
        # For Python 3.8+, you can omit 'strict=False' if not needed
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
        Retrieve a negative sample (from the same batch if batch_key is used) that
        satisfies:
        - sample ID is different
        - its caption differs from the current caption

        Parameters
        ----------
        current_file_path : str
            File path of the current sample.
        current_file_record : dict
            File record for the current file (minus the "local_path" key).
        current_sample : str
            Sample ID of the current sample.
        current_caption : str
            Caption of the current sample.
        desired_modality : str
            Negative modality to use: "caption" or "file_record".
        all_captions : dict of dict
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
        # Figure out the "batch" of the current sample if a batch key is used
        current_batch = None
        if self.batch_key is not None and current_file_path in self.batch_dicts:
            current_batch = self.batch_dicts[current_file_path].get(current_sample)

        # Helper function to confirm a candidate is valid
        def is_valid_negative(fp: str, sid: str) -> bool:
            # must have a non-None, different caption
            if all_captions[fp][sid] is None:
                return False
            if all_captions[fp][sid] == current_caption:
                return False
            if sid == current_sample:
                return False

            # check batch if relevant
            if self.batch_key is not None:
                if fp not in self.batch_dicts:
                    return False
                candidate_batch = self.batch_dicts[fp].get(sid)
                # must match current_batch
                if candidate_batch != current_batch:
                    return False
            return True

        # 1. Try within the same file (and same batch if relevant)
        same_file_candidates = [
            sid
            for sid in all_captions[current_file_path]
            if is_valid_negative(current_file_path, sid)
        ]

        if same_file_candidates:
            neg_sample = random.choice(same_file_candidates)
            neg_caption = all_captions[current_file_path][neg_sample]
            neg_file_record = current_file_record
        else:
            # 2. Fallback: search in other files that match the same batch (if batch_key is used),
            #    or in general if batch_key is not set
            fallback_found = False
            for neg_file in self.anndata_files:
                neg_file_path = neg_file["local_path"]
                if neg_file_path == current_file_path:
                    continue  # skip the same file if no valid candidate
                candidates = [
                    sid
                    for sid in all_captions.get(neg_file_path, {})
                    if is_valid_negative(neg_file_path, sid)
                ]
                if candidates:
                    neg_sample = random.choice(candidates)
                    neg_caption = all_captions[neg_file_path][neg_sample]
                    neg_file_record = {
                        k: v for k, v in neg_file.items() if k != "local_path"
                    }
                    fallback_found = True
                    break

            if not fallback_found:
                raise ValueError("No suitable negative example could be found.")

        # Return desired modality
        if desired_modality == "caption":
            return neg_caption
        elif desired_modality == "file_record":
            return {
                "file_record": neg_file_record,
                "sample_id": neg_sample,
            }  # json.dumps
        else:
            raise ValueError(
                "desired_modality must be either 'caption' or 'file_record'."
            )

    def get_dataset(self) -> Dataset:
        """
        Create and return a Hugging Face Dataset in the specified format.

        For "multiplets" format, each record contains:
            - 'anndata_ref': a JSON file_record reference to the anchor
            - 'positive': the anchor's caption
            - 'negative_i': negative samples from the same batch if batch_key is specified

        For "pairs" format, each record is two columns ('anndata_ref', 'caption') plus a 'label' (1 or 0).

        For "single" format, each record just has the 'anndata_ref'.

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset constructed from the AnnData files.
        """
        hf_data = []
        all_captions = {}  # Nested dict: {file_path: {sample_id: caption}}

        # Build or check captions
        if self.dataset_format != "single":
            for files in self.anndata_files:
                file_path = files["local_path"]
                self.buildCaption(file_path)
                all_captions[file_path] = self.getCaption(file_path)

        # Generate dataset
        if self.dataset_format == "pairs":
            for files in self.anndata_files:
                file_path = files["local_path"]
                file_record = {k: v for k, v in files.items() if k != "local_path"}
                caption_dict = all_captions.get(file_path, {})

                for sample_id, current_caption in caption_dict.items():
                    # anchor modality is always the file_record
                    anndata_ref = {"file_record": file_record, "sample_id": sample_id}
                    positive = current_caption

                    hf_data.append(
                        {
                            "anndata_ref": anndata_ref,
                            "caption": positive,
                            "label": 1.0,
                        }
                    )

                    # Add a negative
                    neg_mod_random = "caption"  # e.g. always text or random.choice(["caption", "file_record"])
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

        elif self.dataset_format == "multiplets":
            for files in self.anndata_files:
                file_path = files["local_path"]
                file_record = {k: v for k, v in files.items() if k != "local_path"}
                caption_dict = all_captions.get(file_path, {})
                for sample_id, current_caption in caption_dict.items():
                    # anchor is the JSON file_record reference, positive is the caption
                    ref_json = {
                        "file_record": file_record,
                        "sample_id": sample_id,
                    }  # json.dumps

                    entry = {"anndata_ref": ref_json, "positive": current_caption}

                    # For negatives, randomly select a modality for each negative
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
            # no caption constructor needed
            for files in self.anndata_files:
                file_path = files["local_path"]
                adata = (
                    anndata.read_zarr(file_path)
                    if file_path.endswith(".zarr")
                    else anndata.read_h5ad(file_path)
                )

                # sample_id_key
                sample_id_key = self.sample_id_keys[file_path]
                if sample_id_key is None:
                    sample_ids = adata.obs.index
                else:
                    sample_ids = adata.obs[sample_id_key]

                for sample_id in sample_ids:
                    ref_json = {
                        "file_record": files,
                        "sample_id": sample_id,
                    }  # json.dumps

                    hf_data.append({"anndata_ref": ref_json})
        else:
            error_msg = "Invalid dataset_format. Choose from 'pairs', 'multiplets', or 'single'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        hf_dataset = Dataset.from_list(hf_data)
        return hf_dataset

    def _check_sharelink(self, share_link: str, suffix=".h5ad") -> bool:
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
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            if download_file_from_share_link(share_link, temp_file.name):
                return True
            else:
                return False


class SimpleCaptionConstructor:
    """
    Construct captions for each sample by concatenating values from specified obs keys.
    Data is sourced from AnnData.obs based on the provided obs_keys.
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
