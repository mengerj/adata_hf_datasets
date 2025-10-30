import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd  # new: for explicit DataFrame input
from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)  # best–practice logger


class AnnDataSetConstructor:
    """
    Create a Hugging Face :class:`datasets.Dataset` from single-cell-style
    data without retaining large objects in memory.

    This class is designed to construct datasets for **individual splits** (e.g., train, val, test).
    After creating datasets for each split, combine them using :class:`datasets.DatasetDict`:

    .. code-block:: python

        from datasets import DatasetDict
        hf_dataset = DatasetDict()
        for split in ["train", "val", "test"]:
            constructor = AnnDataSetConstructor(...)
            constructor.add_anndata(adata, ...)
            hf_dataset[split] = constructor.get_dataset()

    Parameters
    ----------
    negatives_per_sample : int, default 1
        Number of negatives per anchor in *multiplets* or *pairs* format.
    dataset_format : {'pairs', 'multiplets', 'single'}, default 'multiplets'
        Output layout:

        - **'multiplets'**: Training format with positive caption and multiple negative sample
          indices. Negatives alternate between caption negatives (different caption, even indices)
          and sentence negatives (different sample, odd indices). Negatives are drawn from
          in-batch samples based on ``batch_key``.
        - **'pairs'**: Training format that creates individual records with a binary ``label``
          column (1.0 for positive pairs, 0.0 for negative pairs). Each anchor generates
          one positive and one negative pair.
        - **'single'**: Test/inference format containing only cell sentences and ``adata_link``.
          No caption or negative sampling. Use this for test datasets where you only need
          the omics data representation.
    resolve_negatives : bool, default False
        If True and only one ``sentence_key`` is provided, resolves negative indices to their
        actual content. This creates additional columns ``negative_1``, ``negative_2``, etc.
        where odd-numbered negatives (1, 3, 5, ...) contain captions and even-numbered negatives
        (2, 4, 6, ...) contain cell sentences. By default, negatives are stored as indices to
        allow flexibility in choosing which cell sentence representation to use at training time.
        Only applicable for 'multiplets' format.

    Notes
    -----
    **Cell Sentences**
        A "cell sentence" is a string representation of a single cell. The ``sentence_keys``
        parameter specifies which columns from ``adata.obs`` to include. In the output dataset,
        these are named ``cell_sentence_1``, ``cell_sentence_2``, etc., in order.

        For text-based methods, these are typically space-separated lists of gene names like
        "CD4 CD8A IL7R CCR7 FOXP3 ..." representing the expressed genes in that cell.
        For numeric methods, a common approach is to use the cell ID (e.g., barcode), which a
        tokenizer will later use to extract the corresponding numeric embedding from the AnnData object.

    **Data Storage**
        This class does **not** store AnnData objects directly. Instead, it stores an access path
        or download link (``adata_link``) for each sample's source AnnData file. This allows
        downstream models to load the actual numeric data on-demand.

    **Caption and Negative Sampling**
        The ``caption_key`` defines the positive label for each sample (stored as ``positive`` in
        multiplets format, or used for pair generation in pairs format). Captions are typically
        natural language descriptions of the sample metadata, such as "This sample was obtained
        from lung tissue and was annotated as a T cell. It's from a healthy 50 year old...".

        Negative samples are drawn from in-batch samples that have different captions, where
        "batch" is defined by ``batch_key``. This ensures contrastive learning happens within
        meaningful groups.

        By default, negatives are stored as sample indices rather than resolved content. This
        design allows users to provide multiple cell sentence columns (e.g., both for textual and numeric representations) and have downstream methods choose which representation
        to use at training time. The same dataset can thus serve multiple modeling approaches without
        rebuilding.

    **Memory Efficiency**
        Numeric ``obsm`` data are **ignored** – only text columns from ``obs`` are kept.
        You may call :py:meth:`add_anndata` or :py:meth:`add_df` multiple times; all data
        are concatenated in RAM-light dictionaries.
    """

    # ------------------------------------------------------------------ #
    # construction / initialisation
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        negatives_per_sample: int = 1,
        dataset_format: str = "multiplets",
        resolve_negatives: bool = False,
    ) -> None:
        if dataset_format not in {"pairs", "multiplets", "single"}:
            raise ValueError(
                "dataset_format must be 'pairs', 'multiplets', or 'single'."
            )
        if negatives_per_sample < 1:
            raise ValueError("negatives_per_sample must be ≥ 1.")

        self.negatives_per_sample = negatives_per_sample
        self.dataset_format = dataset_format
        self.resolve_negatives = resolve_negatives

        # lightweight per-sample caches
        self._index_to_sentences: Dict[Any, List[str]] = {}
        self._index_to_caption: Dict[Any, Optional[str]] = {}
        self._index_to_batch: Dict[Any, Any] = {}
        self._index_to_adata_link: Dict[Any, str] = {}
        self._num_sentence_keys: Optional[int] = (
            None  # track number of sentence columns
        )

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
        adata_link: str,
        caption_key: Optional[str] = None,
        batch_key: str = "batch",
    ) -> None:
        """
        Extract required columns from an :class:`~anndata.AnnData` object
        and immediately free the heavy matrix.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object to extract metadata from. The actual matrix data
            is not stored; only the ``obs`` columns are retained.
        sentence_keys : list of str
            Column names from ``adata.obs`` to use as cell sentences. These will
            appear in the output dataset as ``cell_sentence_1``, ``cell_sentence_2``, etc.
            For text-based methods, these typically contain space-separated gene names
            (e.g., "CD4 CD8A IL7R CCR7 ..."). For numeric methods, use the cell ID/barcode column.
        adata_link : str
            Path or URL to access this AnnData file. Can be a local file path
            (e.g., "/data/experiment1.h5ad") or a download link
            (e.g., "https://example.com/data.h5ad"). This allows downstream models
            to load the numeric data on-demand.
        caption_key : str, optional
            Column name from ``adata.obs`` to use as the positive caption for
            contrastive learning. Captions are natural language descriptions of sample
            metadata (e.g., "This sample was obtained from lung tissue and was annotated
            as a T cell..."). Required for 'pairs' and 'multiplets' formats, ignored for
            'single' format.
        batch_key : str, default "batch"
            Column name from ``adata.obs`` that defines batches for negative sampling.
            Negatives are preferentially drawn from the same batch to ensure meaningful
            contrastive pairs.
        """
        self._ingest_obs_df(
            obs_df=adata.obs,
            source_name="AnnData",
            sentence_keys=sentence_keys,
            caption_key=caption_key,
            batch_key=batch_key,
            adata_link=adata_link,
        )
        del adata  # allow GC on the big object

    def add_df(
        self,
        df: pd.DataFrame,
        sentence_keys: List[str],
        adata_link: str,
        caption_key: Optional[str] = None,
        batch_key: str = "batch",
    ) -> None:
        """
        Register a plain DataFrame that fulfils the same column requirements
        as ``adata.obs``.

        Parameters
        ----------
        df : pandas.DataFrame
            Must have ``sentence_keys``, ``batch_key`` and (if required)
            ``caption_key`` columns. Its index becomes ``sample_idx``.
        sentence_keys : list of str
            Column names to use as cell sentences. See :py:meth:`add_anndata` for details.
        adata_link : str
            Path or URL to access the corresponding AnnData file. See :py:meth:`add_anndata` for details.
        caption_key : str, optional
            Column name to use as positive caption. See :py:meth:`add_anndata` for details.
        batch_key : str, default "batch"
            Column name that defines batches. See :py:meth:`add_anndata` for details.
        """
        self._ingest_obs_df(
            obs_df=df,
            source_name="DataFrame",
            sentence_keys=sentence_keys,
            caption_key=caption_key,
            batch_key=batch_key,
            adata_link=adata_link,
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
        """
        Assemble the Hugging Face dataset according to ``dataset_format``.

        Returns
        -------
        datasets.Dataset
            A Hugging Face Dataset with the following structure depending on format:

            **'multiplets' format:**
                - ``sample_idx``: Original index from the source data
                - ``cell_sentence_1``, ``cell_sentence_2``, ...: Cell sentences in order
                - ``positive``: The positive caption from ``caption_key``
                - ``negative_1_idx``, ``negative_2_idx``, ...: Sample indices for negatives
                - ``adata_link``: Path or URL to the source AnnData file

                If ``resolve_negatives=True`` and only one sentence_key was provided:
                - ``negative_1``, ``negative_3``, ...: Resolved negative captions (odd numbers)
                - ``negative_2``, ``negative_4``, ...: Resolved negative cell sentences (even numbers)

            **'pairs' format:**
                - ``sample_idx``: Original index from the source data
                - ``cell_sentence_1``, ``cell_sentence_2``, ...: Cell sentences in order
                - ``caption``: Either positive or negative caption
                - ``label``: 1.0 for positive pairs, 0.0 for negative pairs
                - ``adata_link``: Path or URL to the source AnnData file

            **'single' format:**
                - ``sample_idx``: Original index from the source data
                - ``cell_sentence_1``, ``cell_sentence_2``, ...: Cell sentences in order
                - ``adata_link``: Path or URL to the source AnnData file
        """
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
                adata_link = self._index_to_adata_link[idx]
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
                            "adata_link": adata_link,
                        }
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
                        "adata_link": adata_link,
                    }

                    # Optionally resolve negatives to their content
                    if self.resolve_negatives and self._num_sentence_keys == 1:
                        for neg_key, neg_idx in neg_indices.items():
                            # Extract the number from the key (e.g., "negative_1_idx" -> 1)
                            neg_num = int(neg_key.split("_")[1])
                            # Odd numbers (1,3,5...) are caption negatives → resolve to caption
                            # Even numbers (2,4,6...) are sentence negatives → resolve to cell sentence
                            if neg_num % 2 == 1:
                                rec[f"negative_{neg_num}"] = self._index_to_caption[
                                    neg_idx
                                ]
                            else:
                                rec[f"negative_{neg_num}"] = self._index_to_sentences[
                                    neg_idx
                                ][0]

                    yield rec

                else:  # 'single'
                    rec = {
                        "sample_idx": idx,
                        **sentences,
                        "adata_link": adata_link,
                    }
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
        adata_link: str,
    ) -> None:
        """Common ingestion routine for AnnData.obs or any standalone DataFrame."""
        if self.dataset_format in {"pairs", "multiplets"} and caption_key is None:
            raise ValueError("caption_key must be supplied for this dataset_format.")

        # Track and validate number of sentence keys
        if self._num_sentence_keys is None:
            self._num_sentence_keys = len(sentence_keys)
        elif self._num_sentence_keys != len(sentence_keys):
            raise ValueError(
                f"All calls to add_anndata/add_df must use the same number of sentence_keys. "
                f"Expected {self._num_sentence_keys}, got {len(sentence_keys)}."
            )

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
            self._index_to_adata_link[idx] = adata_link

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
