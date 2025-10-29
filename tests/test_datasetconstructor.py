"""
Tests for the AnnDataSetConstructor.

The suite covers

* both ingestion paths: ``add_anndata`` **and** ``add_df``
* all three ``dataset_format`` modes: *pairs*, *multiplets*, *single*
* correctness of negative–caption sampling
* handling of several sources (AnnData **plus** DataFrame)
* preservation of unique ``sample_idx`` values
* presence / absence of expected columns per format
* failure cases: missing columns

A minimal in-memory *AnnData* stand-in is used so the tests do not depend
on the *anndata* package.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
import pytest
from datasets import Dataset

from adata_hf_datasets.dataset import AnnDataSetConstructor

logger = logging.getLogger(__name__)  # predefined logger per guidelines


# ---------------------------------------------------------------------- #
# helpers / fixtures
# ---------------------------------------------------------------------- #
class DummyAnnData:
    """Tiny stand-in that exposes only the attributes the constructor needs."""

    def __init__(self, obs: pd.DataFrame):
        self.obs = obs
        self.n_obs = obs.shape[0]  # used only for an info-log line


@pytest.fixture
def make_obs_df() -> callable:
    """
    Factory that returns a function for building a *fresh* obs DataFrame.

    Returns
    -------
    Callable[[str, int], pd.DataFrame]
        ``f(batch_prefix, n_samples)`` → DataFrame with the required columns.
    """

    def _factory(batch_prefix: str, n: int) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "sentence_short": [f"short_{i}" for i in range(n)],
                "sentence_long": [f"long_{i}" for i in range(n)],
                "cell_type": ["T" if i % 2 else "B" for i in range(n)],
                "batch": [f"{batch_prefix}{i // 3}" for i in range(n)],
            },
            index=[f"{batch_prefix}{i}" for i in range(n)],
        )
        return df

    return _factory


def _check_unique_sample_ids(ds: Dataset, expected: int, dataset_format: str):
    """Assert that every sample_idx appears exactly once in the HF dataset."""
    idxs: List[str] = list(ds.to_pandas()["sample_idx"])
    assert len(idxs) == expected
    if dataset_format != "pairs":
        assert len(set(idxs)) == expected, "duplicate sample_idx detected"
    else:
        # 'pairs' format: each sample_idx appears twice
        assert len(set(idxs)) == expected // 2, "duplicate sample_idx detected"
        assert all(idxs.count(idx) == 2 for idx in set(idxs)), (
            "sample_idx count mismatch"
        )


# ---------------------------------------------------------------------- #
# parametrised *happy-path* tests
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dataset_format,negatives",
    [
        ("pairs", 1),
        ("multiplets", 2),
        ("single", 0),
    ],
)
def test_full_pipeline(dataset_format, negatives, make_obs_df):
    """
    End-to-end verification of each ``dataset_format``.

    Parameters
    ----------
    dataset_format : str
        One of 'pairs', 'multiplets', 'single'.
    negatives : int
        Requested negatives per anchor (only for multiplets).
    """
    # ---------------- prepare two input sources ------------------------ #
    df1 = make_obs_df("A", 8)
    df2 = make_obs_df("B", 5)

    adata = DummyAnnData(df1)

    builder = AnnDataSetConstructor(
        dataset_format=dataset_format,
        negatives_per_sample=negatives or 1,  # value ignored for pairs/single
    )

    # add AnnData + DataFrame
    builder.add_anndata(
        adata,
        sentence_keys=["sentence_short", "sentence_long"],
        caption_key="cell_type" if dataset_format != "single" else None,
        adata_link="link_A",
    )
    builder.add_df(
        df2,
        sentence_keys=["sentence_short", "sentence_long"],
        caption_key="cell_type" if dataset_format != "single" else None,
        batch_key="batch",
        adata_link="link_B",
    )

    ds = builder.get_dataset()

    # ---------------- generic checks ----------------------------------- #
    total_unique = df1.shape[0] + df2.shape[0]
    _check_unique_sample_ids(
        ds,
        expected=total_unique if dataset_format != "pairs" else total_unique * 2,
        dataset_format=dataset_format,
    )

    df_hf = ds.to_pandas()

    # check share_link column behaviour
    if dataset_format != "single":
        assert set(df_hf.query("share_link == 'link_A'")["sample_idx"]).issubset(
            set(df1.index)
        )
        assert set(df_hf.query("share_link == 'link_B'")["sample_idx"]).issubset(
            set(df2.index)
        )

    # ---------------- format-specific checks --------------------------- #
    if dataset_format == "pairs":
        # sanity: captions with label 1.0 match the ground truth,
        #         captions with 0.0 do *not*.
        truth = {idx: cap for idx, cap in df1._append(df2)["cell_type"].items()}

        for _, row in df_hf.iterrows():
            if row["label"] == 1.0:
                assert row["caption"] == truth[row["sample_idx"]]
            else:
                assert row["caption"] != truth[row["sample_idx"]]

    elif dataset_format == "multiplets":
        # all negatives differ from positive
        neg_cols = [c for c in df_hf.columns if c.startswith("negative_")]
        for _, row in df_hf.iterrows():
            for c in neg_cols:
                assert row[c] != row["positive"]

    elif dataset_format == "single":
        # no caption/label columns present
        assert "caption" not in df_hf.columns
        assert "label" not in df_hf.columns


# ---------------------------------------------------------------------- #
# failure / edge-case tests
# ---------------------------------------------------------------------- #
def test_missing_caption_key_raises(make_obs_df):
    """Constructor must reject a 'pairs' dataset without caption_key."""
    df = make_obs_df("C", 3)
    builder = AnnDataSetConstructor(dataset_format="pairs")

    with pytest.raises(ValueError):
        builder.add_df(
            df,
            sentence_keys=["sentence_short", "sentence_long"],
            # caption_key intentionally omitted
        )


def test_sentence_key_missing_raises(make_obs_df):
    """Missing sentence_keys should raise a ValueError."""
    df = make_obs_df("D", 3).drop(columns=["sentence_long"])
    builder = AnnDataSetConstructor(dataset_format="single")

    with pytest.raises(ValueError):
        builder.add_df(
            df,
            sentence_keys=["sentence_short", "sentence_long"],  # 'sentence_long' absent
        )
