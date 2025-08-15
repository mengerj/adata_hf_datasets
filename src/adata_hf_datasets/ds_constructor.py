import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd  # new: for explicit DataFrame input
from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)  # best–practice logger


class AnnDataSetConstructor:
    """
    Create a Hugging Face :class:`datasets.Dataset` out of single-cell-style
    data without retaining large objects in memory.

    Parameters
    ----------
    negatives_per_sample : int, default 1
        Number of negatives per anchor in *multiplets* or *pairs* format.
    dataset_format : {'pairs', 'multiplets', 'single'}, default 'pairs'
        Output layout.

    Notes
    -----
    * Numeric ``obsm`` data are **ignored** – only text columns are kept.
    * You may call :py:meth:`add_anndata` or :py:meth:`add_df` as many times
      as you wish; all data are concatenated in RAM-light dicts.
    * For 'multiplets' format, negative samples are now stored as sample indices
      rather than actual content, allowing the model to choose which modality
      to use at training time.
    """

    # ------------------------------------------------------------------ #
    # construction / initialisation
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        negatives_per_sample: int = 1,
        dataset_format: str = "pairs",
    ) -> None:
        if dataset_format not in {"pairs", "multiplets", "single"}:
            raise ValueError(
                "dataset_format must be 'pairs', 'multiplets', or 'single'."
            )
        if negatives_per_sample < 1:
            raise ValueError("negatives_per_sample must be ≥ 1.")

        self.negatives_per_sample = negatives_per_sample
        self.dataset_format = dataset_format

        # lightweight per-sample caches
        self._index_to_sentences: Dict[Any, List[str]] = {}
        self._index_to_caption: Dict[Any, Optional[str]] = {}
        self._index_to_batch: Dict[Any, Any] = {}
        self._index_to_share: Dict[Any, Optional[str]] = {}

        # optimized data structures for fast negative sampling
        self._batch_caption_map: defaultdict = defaultdict(list)  # (batch, cap) → idxs
        self._caption_map: defaultdict = defaultdict(list)  # cap → idxs

        # lightweight caching - only cache what we actually need
        self._batch_indices: Dict[Any, List] = {}  # batch → all indices in that batch
        self._caption_indices: Dict[
            str, List
        ] = {}  # caption → all indices with that caption
        self._all_indices: List = []  # all sample indices
        self._pools_built = False

    # ------------------------------------------------------------------ #
    # public ingest methods
    # ------------------------------------------------------------------ #
    def add_anndata(
        self,
        adata,
        sentence_keys: List[str],
        caption_key: Optional[str] = None,
        batch_key: str = "batch",
        share_link: Optional[str] = None,
    ) -> None:
        """
        Extract required columns from an :class:`~anndata.AnnData` object
        and immediately free the heavy matrix.
        """
        self._ingest_obs_df(
            obs_df=adata.obs,
            source_name="AnnData",
            sentence_keys=sentence_keys,
            caption_key=caption_key,
            batch_key=batch_key,
            share_link=share_link,
        )
        del adata  # allow GC on the big object

    def add_df(
        self,
        df: pd.DataFrame,
        sentence_keys: List[str],
        caption_key: Optional[str] = None,
        batch_key: str = "batch",
        share_link: Optional[str] = None,
    ) -> None:
        """
        Register a plain DataFrame that fulfils the same column requirements
        as ``adata.obs``.

        Parameters
        ----------
        df : pandas.DataFrame
            Must have ``sentence_keys``, ``batch_key`` and (if required)
            ``caption_key`` columns.  Its index becomes ``sample_idx``.
        sentence_keys, caption_key, batch_key, share_link
            Same semantics as in :py:meth:`add_anndata`.
        """
        self._ingest_obs_df(
            obs_df=df,
            source_name="DataFrame",
            sentence_keys=sentence_keys,
            caption_key=caption_key,
            batch_key=batch_key,
            share_link=share_link,
        )

    # ------------------------------------------------------------------ #
    # optimization methods
    # ------------------------------------------------------------------ #
    def _build_simple_pools(self) -> None:
        """Build minimal pools for fast negative sampling."""
        if self._pools_built:
            return

        # Build simple lookup structures from existing data
        for batch, indices_list in self._batch_caption_map.keys():
            if batch not in self._batch_indices:
                self._batch_indices[batch] = []

        for (batch, caption), indices in self._batch_caption_map.items():
            self._batch_indices[batch].extend(indices)
            if caption not in self._caption_indices:
                self._caption_indices[caption] = []
            self._caption_indices[caption].extend(indices)

        self._all_indices = list(self._index_to_sentences.keys())
        self._pools_built = True

    # ------------------------------------------------------------------ #
    # dataset construction
    # ------------------------------------------------------------------ #
    def get_dataset(self) -> Dataset:
        """Assemble the Hugging Face dataset according to ``dataset_format``."""
        if not self._index_to_sentences:
            raise ValueError("No data present – call add_anndata/add_df first.")

        # Build minimal lookup structures once
        self._build_simple_pools()

        # Use generator to avoid building massive list in memory
        def record_generator():
            for idx in tqdm(
                self._index_to_sentences.keys(), desc="Building dataset", leave=False
            ):
                sent_vals = self._index_to_sentences[idx]
                sentences = {
                    f"cell_sentence_{i + 1}": s for i, s in enumerate(sent_vals)
                }
                share = self._index_to_share[idx]
                pos_cap = self._index_to_caption[idx]

                if self.dataset_format == "pairs":
                    # unchanged
                    neg_idx = self._get_negative_idx(idx, pos_cap)
                    if neg_idx is None:
                        continue
                    neg_cap = self._index_to_caption[neg_idx]
                    for cap, label in ((pos_cap, 1.0), (neg_cap, 0.0)):
                        rec = {
                            "sample_idx": idx,
                            **sentences,
                            "caption": cap,
                            "label": label,
                        }
                        if share:
                            rec["share_link"] = share
                        yield rec

                elif self.dataset_format == "multiplets":
                    neg_indices: Dict[str, Any] = {}
                    seen_idxs = set()
                    for i in range(self.negatives_per_sample):
                        # even i (0-based) → caption negative (different caption)
                        if i % 2 == 0:
                            neg_idx = self._get_negative_idx(idx, pos_cap)
                            if neg_idx is None or neg_idx in seen_idxs:
                                # if no valid caption negative is available, skip
                                continue
                            seen_idxs.add(neg_idx)
                            neg_indices[f"negative_{i + 1}_idx"] = neg_idx
                        # odd i → sentence negative (different sample, any caption)
                        else:
                            neg_idx = self._get_sentence_negative_idx(idx, seen_idxs)
                            if neg_idx is None or neg_idx in seen_idxs:
                                # if no valid sentence negative is available, skip
                                continue
                            seen_idxs.add(neg_idx)
                            neg_indices[f"negative_{i + 1}_idx"] = neg_idx

                    if not neg_indices:
                        # very rare corner-case: no negatives could be drawn
                        continue

                    rec = {
                        "sample_idx": idx,
                        **sentences,
                        "positive": pos_cap,
                        **neg_indices,
                    }
                    if share:
                        rec["share_link"] = share
                    yield rec

                else:  # 'single'
                    rec = {"sample_idx": idx, **sentences}
                    if share:
                        rec["share_link"] = share
                    yield rec

        # Use Dataset.from_generator for memory efficiency
        logger.info("Building dataset from generator...")

        try:
            # HuggingFace datasets supports from_generator for memory-efficient creation
            dataset = Dataset.from_generator(record_generator)
            logger.info(
                "Constructed dataset with %d records in '%s' format.",
                len(dataset),
                self.dataset_format,
            )
            return dataset
        except Exception as e:
            logger.warning(f"from_generator failed ({e}), falling back to from_list...")
            # Fallback to chunked processing if from_generator fails
            records = []
            chunk_size = 5000  # Smaller chunks to reduce memory pressure
            for i, record in enumerate(record_generator()):
                records.append(record)
                if i % chunk_size == 0 and i > 0:
                    logger.info(f"Processed {i} records...")

            logger.info(
                "Constructed dataset with %d records in '%s' format.",
                len(records),
                self.dataset_format,
            )
            return Dataset.from_list(records)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _ingest_obs_df(
        self,
        obs_df: pd.DataFrame,
        source_name: str,
        sentence_keys: List[str],
        caption_key: Optional[str],
        batch_key: str,
        share_link: Optional[str],
    ) -> None:
        """Common ingestion routine for AnnData.obs or any standalone DataFrame."""
        if self.dataset_format in {"pairs", "multiplets"} and caption_key is None:
            raise ValueError("caption_key must be supplied for this dataset_format.")

        missing_sent = [k for k in sentence_keys if k not in obs_df.columns]
        if missing_sent:
            raise ValueError(f"{source_name}: sentence_keys not found: {missing_sent}")
        if caption_key and caption_key not in obs_df.columns:
            raise ValueError(f"{source_name}: caption_key '{caption_key}' missing.")
        if batch_key not in obs_df.columns:
            raise ValueError(f"{source_name}: batch_key '{batch_key}' missing.")

        for idx in obs_df.index:
            self._index_to_sentences[idx] = [obs_df.at[idx, k] for k in sentence_keys]
            batch = obs_df.at[idx, batch_key]
            self._index_to_batch[idx] = batch
            self._index_to_share[idx] = share_link

            cap = obs_df.at[idx, caption_key] if caption_key else None
            self._index_to_caption[idx] = cap
            if cap is not None:
                self._batch_caption_map[(batch, cap)].append(idx)
                self._caption_map[cap].append(idx)

        logger.info("%s ingested (%d rows).", source_name, obs_df.shape[0])

    def _get_negative_idx(self, anchor_idx: Any, pos_caption: str) -> Optional[Any]:
        """Return an obs index with a *different* caption (preferring same batch)."""
        anchor_batch = self._index_to_batch[anchor_idx]

        # same-batch but different caption - iterate through batch_caption_map efficiently
        for (batch, caption), indices in self._batch_caption_map.items():
            if batch == anchor_batch and caption != pos_caption:
                # Try to find a candidate that's not the anchor
                for idx in indices:
                    if idx != anchor_idx:
                        return idx

        # cross-batch fallback - iterate through caption_map efficiently
        for caption, indices in self._caption_map.items():
            if caption != pos_caption:
                # Try to find a candidate that's not the anchor
                for idx in indices:
                    if idx != anchor_idx:
                        return idx

        return None

    def _get_sentence_negative_idx(
        self, anchor_idx: Any, seen_idxs: set
    ) -> Optional[Any]:
        """Return an obs index for sentence negatives (different sample, any caption)."""
        anchor_batch = self._index_to_batch[anchor_idx]

        # same-batch but different sample - iterate through batch efficiently
        if anchor_batch in self._batch_indices:
            for idx in self._batch_indices[anchor_batch]:
                if idx != anchor_idx and idx not in seen_idxs:
                    return idx

        # cross-batch fallback - iterate through all indices
        for idx in self._all_indices:
            if idx != anchor_idx and idx not in seen_idxs:
                return idx

        return None
