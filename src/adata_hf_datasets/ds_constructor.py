import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd  # new: for explicit DataFrame input
from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)  # best–practice logger


class AnnDataSetConstructor:
    """
    Create a Hugging Face :class:`datasets.Dataset` out of one or more data
    sources without retaining large objects in memory.

    You may add data via

    * :py:meth:`add_anndata` – takes an AnnData, internally uses ``adata.obs``
    * :py:meth:`add_df` – takes any :class:`pandas.DataFrame` with the same
      column requirements

    After extraction only lightweight Python dicts are kept.

    Parameters
    ----------
    negatives_per_sample : int, default 1
        Number of negatives per anchor in *multiplets* format.
    dataset_format : {'pairs', 'multiplets', 'single'}, default 'pairs'
        Dataset layout.

    Notes
    -----
    * Numeric ``obsm`` data have been **removed** – only text columns are used.
    * Unlimited sequential calls to `add_anndata` and/or `add_df` are allowed.
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
        self.negatives_per_sample = negatives_per_sample
        self.dataset_format = dataset_format

        # lightweight per-sample caches
        self._index_to_sentences: Dict[Any, List[str]] = {}
        self._index_to_caption: Dict[Any, Optional[str]] = {}
        self._index_to_batch: Dict[Any, Any] = {}
        self._index_to_share: Dict[Any, Optional[str]] = {}

        # global maps for fast negative sampling
        self._batch_caption_map: defaultdict = defaultdict(
            list
        )  # (batch, caption) → [idx …]
        self._caption_map: defaultdict = defaultdict(list)  # caption → [idx …]

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
    # dataset construction
    # ------------------------------------------------------------------ #
    def get_dataset(self) -> Dataset:
        """Assemble the Hugging Face dataset according to ``dataset_format``."""
        if not self._index_to_sentences:
            raise ValueError("No data present – call add_anndata/add_df first.")

        records: List[Dict[str, Any]] = []
        for idx in tqdm(
            self._index_to_sentences.keys(), desc="Building dataset", leave=False
        ):
            sent_vals = self._index_to_sentences[idx]
            sentences = {f"cell_sentence_{i + 1}": s for i, s in enumerate(sent_vals)}
            share = self._index_to_share[idx]
            pos_cap = self._index_to_caption[idx]

            if self.dataset_format == "pairs":
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
                    records.append(rec)

            elif self.dataset_format == "multiplets":
                negs: Dict[str, str] = {}
                seen = set()
                for i in range(self.negatives_per_sample):
                    neg_idx = self._get_negative_idx(idx, pos_cap)
                    if neg_idx is None or neg_idx in seen:
                        continue
                    seen.add(neg_idx)
                    negs[f"negative_{i + 1}"] = self._index_to_caption[neg_idx]
                if not negs:
                    continue
                rec = {"sample_idx": idx, **sentences, "positive": pos_cap, **negs}
                if share:
                    rec["share_link"] = share
                records.append(rec)

            else:  # 'single'
                rec = {"sample_idx": idx, **sentences}
                if share:
                    rec["share_link"] = share
                records.append(rec)

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

        # same-batch but different caption
        in_batch = [
            i
            for (b, c), idxs in self._batch_caption_map.items()
            if b == anchor_batch and c != pos_caption
            for i in idxs
        ]
        in_batch = [i for i in in_batch if i != anchor_idx]
        if in_batch:
            return random.choice(in_batch)

        # cross-batch fallback
        cross = [
            i
            for cap, idxs in self._caption_map.items()
            if cap != pos_caption
            for i in idxs
        ]
        cross = [i for i in cross if i != anchor_idx]
        return random.choice(cross) if cross else None
