import logging
import numpy as np
import random
from datasets import Dataset
from collections import defaultdict
from tqdm import tqdm

# Use the predefined logger per best practices.
logger = logging.getLogger(__name__)  # Assumes a logger with this API is available


class AnnDataSetConstructor:
    """
    Class to generate a dataset compatible with the SentenceTransformer library from AnnData files.

    Data is sourced from AnnData files (e.g. .zarr or .h5ad) that are added via `add_anndata`.
    The generated dataset can be created in one of three formats:

    - "pairs": Each record is a pair containing an anchor and a positive example,
               plus one negative pair (randomly chosen). Each record has:
               "data_representation" (from obsm), "caption" (from obs) and "label"
               indicating positive (1.0) or negative (0.0) pair.
    - "multiplets": Each record contains an anchor (data_representation), a positive (caption),
                    and a list of negative examples. Negative examples are selected
                    from samples with different `sample_id`.
    - "single": Each record contains only a single example, which is useful for inference.
                If an obsm key is provided, the record contains a numeric representation;
                otherwise, if an obs key is provided, the record contains a caption.

    Parameters
    ----------
    obsm_key : str, optional
        Key in `adata.obsm` to use for the numeric data representation (e.g., embeddings).
    obs_key : str, optional
        Key in `adata.obs` to use for the string data representation (e.g., cell sentences).
    float16 : bool, optional
        If True, convert numeric data to float16 for reduced memory usage.
    negatives_per_sample : int, optional
        Number of negative examples to include for each anchor in multiplets.
    dataset_format : str, optional
        Format of the dataset to construct. Must be one of "pairs", "multiplets", or "single".
        For "pairs" and "multiplets", both `obsm_key` and `obs_key` must be provided.
        For "single", exactly one of them should be provided.

    Methods
    -------
    add_anndata(adata)
        Adds an AnnData object to the constructor. Checks that `adata` contains the required keys
        and that a "sample_id" column exists in `adata.obs`.
    _get_numeric_representation(adata, obsm_key, float16)
        Retrieves the numeric representation from `adata.obsm`.
    _get_string_representation(adata, obs_key)
        Retrieves the string representation from `adata.obs`.
    _get_caption(adata, obs_key)
        Retrieves the caption from `adata.obs`.
    _get_positive_example(index)
        Retrieves the index of a positive example for a given anchor.
    get_dataset()
        Constructs the dataset based on the specified format ("pairs", "multiplets", "single")
        and returns a list of dictionary records.
    """

    def __init__(
        self,
        float16: bool = True,
        negatives_per_sample: int = 1,
        dataset_format: str = "pairs",
    ):
        """
        Parameters
        ----------
        float16 : bool, optional
            If True, numeric data will be converted to float16.
        negatives_per_sample : int, optional
            Number of negative examples (for multiplets) per anchor sample.
        dataset_format : str, optional
            Format of dataset to construct. Must be one of "pairs", "multiplets", or "single".

        Raises
        ------
        ValueError
            If the provided keys do not match the requirements of the selected dataset_format.
        """
        self.float16 = float16
        self.negatives_per_sample = negatives_per_sample
        self.dataset_format = dataset_format
        self.adata = None  # Will hold the added AnnData object

        # Internal caches for faster lookups
        self._batch_caption_map = None
        self._caption_map = None
        self._numeric_data = None

    def add_anndata(
        self,
        adata,
        batch_key: str = "batch",
        obsm_key: str | None = None,
        obs_key: str | None = None,
        caption_key: str | None = None,
    ):
        """
        Adds an AnnData object to the constructor.

        Checks that the required keys exist in the AnnData object and that 'sample_id' is present in `adata.obs`.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the data sources.
            Expected to have attributes:
              - adata.obsm: a dictionary-like object for numeric representations.
              - adata.obs: a pandas DataFrame that must include a 'sample_id' column.
        batch_key : str, optional
            Key in `adata.obs` for batch information (default is "batch").
        obsm_key : str, optional
            Key in `adata.obsm` for the numeric representation (e.g., X_geneformer).
        obs_key : str, optional
            Key in `adata.obs` for the string representation (e.g., "cell_sentence").
        caption_key : str, optional
            Key in `adata.obs` for the captions (e.g., "cell_type" or "natural_language_annotation").

        Raises
        ------
        """
        self.obsm_key = obsm_key
        self.obs_key = obs_key
        self.caption_key = caption_key
        self.batch_key = batch_key

        if self.obsm_key and self.obs_key:
            raise ValueError(
                "Only one of obsm_key or obs_key can be chosen, as these will be used to construct the data representation"
            )
        if self.dataset_format in ["pairs", "multiplets"]:
            if (
                self.obsm_key is None and self.obs_key is None
            ) or self.caption_key is None:
                raise ValueError(
                    "For 'pairs' or 'multiplets' format, caption_key and either obsm_key or obs_key must be provided."
                )
        elif self.dataset_format == "single":
            # For a single dataset, exactly one key should be provided.
            if self.caption_key:
                logger.warning("For 'single' format, caption key will not be used.")
        else:
            raise ValueError(
                "dataset_format must be one of 'pairs', 'multiplets', or 'single'."
            )
        # Validate that adata.obs has the required column.
        if self.obsm_key is not None:
            if self.obsm_key not in adata.obsm.keys():
                raise ValueError(f"obsm_key '{self.obsm_key}' not found in adata.obsm.")
        if self.obs_key is not None:
            if self.obs_key not in adata.obs.columns:
                raise ValueError(f"obs_key '{self.obs_key}' not found in adata.obs.")
        if self.caption_key is not None:
            if self.caption_key not in adata.obs.columns:
                raise ValueError(
                    f"caption_key '{self.caption_key}' not found in adata.obs."
                )
        if self.batch_key is not None:
            if self.batch_key not in adata.obs.columns:
                raise ValueError(
                    f"batch_key '{self.batch_key}' not found in adata.obs."
                )

        if self.obsm_key:
            self.numeric = True
            arr = adata.obsm[self.obsm_key]
            if self.float16:
                arr = arr.astype(np.float16)
            self._numeric_data = arr
        elif self.obs_key:
            self.numeric = False
        else:
            raise ValueError("Either obsm_key or obs_key must be provided.")
            # Build dictionaries to speed up negative lookups
        if (
            self.batch_key in adata.obs.columns
            and self.caption_key in adata.obs.columns
        ):
            # (batch_val, caption_val) -> list of indices
            self._batch_caption_map = defaultdict(list)
            # caption_val -> list of indices (for global fallback)
            self._caption_map = defaultdict(list)

            for i, idx in enumerate(adata.obs.index):
                batch_val = adata.obs.loc[idx, batch_key]
                cap_val = adata.obs.loc[idx, caption_key]

                self._batch_caption_map[(batch_val, cap_val)].append(idx)
                self._caption_map[cap_val].append(idx)

        self.adata = adata
        logger.info("Added AnnData with %d samples.", self.adata.obs.shape[0])

    def _get_numeric_representation(self, adata, obsm_key, index, float16):
        """
        Retrieves the numeric representation from adata.obsm.

        Parameters
        ----------
        adata : AnnData
            AnnData object from which numeric data is sourced.
        obsm_key : str
            Key in adata.obsm for the numeric representation.
        float16 : bool
            If True, converts the data to float16.

        Returns
        -------
        np.ndarray
            Numeric representation as a NumPy array.

        Reference
        ---------
        Data is sourced from adata.obsm.
        """
        pos = self.adata.obs.index.get_loc(index)
        return self._numeric_data[pos]

    def _get_string_representation(self, adata, obs_key, index):
        """
        Retrieves the string representation from adata.obs.

        Parameters
        ----------
        adata : AnnData
            AnnData object from which string data is sourced.
        obs_key : str
            Key in adata.obs for the string representation.

        Returns
        -------
        list
            List of strings extracted from adata.obs.

        Reference
        ---------
        Data is sourced from adata.obs.
        """
        # Assuming adata.obs[obs_key] is a pandas Series.
        return adata.obs.loc[index, obs_key]

    def get_data_representation(self, index):
        """
        Retrieves the data representation for a sample.

        Depending on which key was provided (obsm_key or obs_key), the numeric or string representation is returned.

        Parameters
        ----------
        index : any
            Sample index (from adata.obs.index).

        Returns
        -------
        np.ndarray or str
            Data representation for the sample.
        """
        if self.numeric:
            return self._get_numeric_representation(
                self.adata, self.obsm_key, index, self.float16
            )
        else:
            return self._get_string_representation(self.adata, self.obs_key, index)

    def _get_caption(self, index):
        """
        Retrieves the caption from adata.obs.

        Parameters
        ----------
        adata : AnnData
            AnnData object from which captions are sourced.
        obs_key : str
            Key in adata.obs for the captions.

        Returns
        -------
        list
            List of captions.

        Reference
        ---------
        Data is sourced from adata.obs.
        """
        return self.adata.obs.loc[index, self.caption_key]

    def _get_negative_idx(self, anchor_idx, pos_caption):
        """
        Efficient negative example lookup using pre-built dictionaries.

        Parameters
        ----------
        anchor_idx : any
            Index of the anchor sample in adata.obs.
        pos_caption : str
            Positive caption of the anchor sample.

        Returns
        -------
        any or None
            Negative index label if found, else None.
        """
        # Try same-batch candidates first
        anchor_batch = self.adata.obs.loc[anchor_idx, self.batch_key]
        in_batch_candidates = []
        for (b_val, c_val), idx_list in self._batch_caption_map.items():
            if b_val == anchor_batch and c_val != pos_caption:
                in_batch_candidates.extend(idx_list)

        # Remove the anchor if it slipped in
        if anchor_idx in in_batch_candidates:
            in_batch_candidates.remove(anchor_idx)

        # If we have candidates in-batch, sample from them
        if in_batch_candidates:
            return random.choice(in_batch_candidates)

        # Fallback: choose from all samples with different caption
        out_batch_candidates = []
        for c_val, idx_list in self._caption_map.items():
            if c_val != pos_caption:
                out_batch_candidates.extend(idx_list)
        if anchor_idx in out_batch_candidates:
            out_batch_candidates.remove(anchor_idx)
        if not out_batch_candidates:
            return None

        return random.choice(out_batch_candidates)

    def get_dataset(self):
        """
        Constructs and returns the dataset records based on the selected format.

        For each anchor sample (indexed in adata.obs):
        - In "pairs" format:
            Two records are generated. Both share the anchor's data representation.
            One record uses the anchor's caption (label 1.0) and the other uses a negative caption (label 0.0)
            obtained via _get_negative_idx.
        - In "multiplets" format:
            A single record is generated with:
                "anchor": the data representation of the anchor,
                "positive": the anchor's caption,
                "negatives": a list of negative values. For each negative,
                            a negative index is retrieved via _get_negative_idx and then the value alternates
                            between the caption (if even) and the data representation (if odd).
        - In "single" format:
            A single record is generated with only the data representation for inference.

        Returns
        -------
        list of dict
            The constructed dataset as a list of records.
        """
        if self.adata is None:
            raise ValueError("No AnnData object added. Call add_anndata() first.")

        records = []
        for idx in tqdm(self.adata.obs.index):
            anchor_rep = self.get_data_representation(idx)
            pos_caption = (
                self._get_caption(idx) if self.caption_key is not None else None
            )

            if self.dataset_format == "pairs":
                neg_idx = self._get_negative_idx(idx, pos_caption)
                if neg_idx is None:
                    continue
                neg_caption = self.adata.obs.loc[neg_idx, self.caption_key]
                record_pos = {
                    "sample_idx": idx,
                    "data_representation": anchor_rep,
                    "caption": pos_caption,
                    "label": 1.0,
                }
                record_neg = {
                    "sample_idx": idx,
                    "data_representation": anchor_rep,
                    "caption": neg_caption,
                    "label": 0.0,
                }
                records.extend([record_pos, record_neg])

            elif self.dataset_format == "multiplets":
                # Loop for the required number of negatives.
                negative_dict = {}
                for i in range(self.negatives_per_sample):
                    neg_idx = self._get_negative_idx(idx, pos_caption)
                    if neg_idx is None:
                        continue
                    # Alternate between obtaining the negative caption and its data representation.
                    if i % 2 == 0:
                        neg_value = self._get_caption(neg_idx)
                    else:
                        neg_value = self.get_data_representation(neg_idx)

                    negative_key = f"negative_{i + 1}"
                    negative_dict[negative_key] = neg_value

                if not negative_dict:
                    continue

                record = {
                    "sample_idx": idx,
                    "anchor": anchor_rep,
                    "positive": pos_caption,
                    **negative_dict,  # unpack negatives as separate keys
                }
                records.append(record)

            elif self.dataset_format == "single":
                record = {"sample_idx": idx, "data_representation": anchor_rep}
                records.append(record)
            else:
                raise ValueError(
                    "Unrecognized dataset_format. Must be 'pairs', 'multiplets', or 'single'."
                )

        # Log example details.
        if self.obsm_key is not None:
            example_numeric = None
            for rec in records:
                # Look for a record whose data_representation is numeric.
                if "data_representation" in rec and isinstance(
                    rec["data_representation"], np.ndarray
                ):
                    example_numeric = rec["data_representation"]
                    break
            if example_numeric is not None:
                example_numeric = np.array(example_numeric)
                mean_val = float(np.mean(example_numeric))
                std_val = float(np.std(example_numeric))
                logger.info(
                    "Using obsm_key: %s. Example numeric vector: shape=%s, mean=%.3f, std=%.3f",
                    self.obsm_key,
                    example_numeric.shape,
                    mean_val,
                    std_val,
                )
        else:
            logger.info("Using obs_key for data representation.")
        logger.info(
            "Constructed dataset with %d records in '%s' format.",
            len(records),
            self.dataset_format,
        )
        hf_dataset = Dataset.from_list(records)
        return hf_dataset
