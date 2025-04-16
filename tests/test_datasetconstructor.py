import pytest
import numpy as np
import pandas as pd

# Suppose this is how you import your final class.
from adata_hf_datasets.ds_constructor import AnnDataSetConstructor

###############################################################################
# Helper to simulate an AnnData object
###############################################################################


class SimulatedAnnData:
    """A minimal simulation of an AnnData object for testing."""

    def __init__(self, obs: pd.DataFrame, obsm: dict):
        self.obs = obs
        self.obsm = obsm


def create_simulated_adata(n_samples: int = 5) -> SimulatedAnnData:
    """
    Create a simulated AnnData object.

    Each sample gets:
      - a unique sample_id,
      - a column "cell_sentence" with a list of ten gene names,
      - a column "cell_type" (the caption),
      - a "batch" column.
    Also, obsm contains a key "X_test" with a numeric vector for each sample.
    """
    sample_ids = [f"s{i}" for i in range(n_samples)]
    # Each sample: list of 10 gene names (for example purposes).
    cell_sentence = [[f"gene{j}" for j in range(1, 11)] for _ in range(n_samples)]
    # Let the caption be a simple cell type that switches halfway.
    cell_type = ["Type_A" if i < n_samples // 2 else "Type_B" for i in range(n_samples)]
    # Batch info: first three samples in one batch, the rest in another.
    batch = ["batch1" if i < 3 else "batch2" for i in range(n_samples)]

    # Build the obs DataFrame. Use sample_ids as the index.
    df_obs = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cell_sentence": cell_sentence,
            "cell_type": cell_type,
            "batch": batch,
        },
        index=sample_ids,
    )

    # Create obsm numeric representations, for example a vector of length 5.
    X_test = np.random.rand(n_samples, 5)
    obsm = {"X_test": X_test}

    return SimulatedAnnData(df_obs, obsm)


###############################################################################
# Tests for error/edge cases in add_anndata.
###############################################################################


def test_add_anndata_no_keys():
    """
    When no keys are provided, add_anndata should fail for dataset formats
    that require either an obsm_key or obs_key (here, for "pairs").
    """
    adata = create_simulated_adata()
    cons = AnnDataSetConstructor(dataset_format="pairs")
    # Expect a ValueError because required keys are missing.
    with pytest.raises(ValueError):
        cons.add_anndata(adata)


def test_add_anndata_both_keys():
    """
    Providing both obsm_key and obs_key should raise an error.
    """
    adata = create_simulated_adata()
    cons = AnnDataSetConstructor(dataset_format="pairs")
    with pytest.raises(ValueError):
        cons.add_anndata(
            adata, obsm_key="X_test", obs_key="cell_sentence", caption_key="cell_type"
        )


def test_add_anndata_wrong_keys():
    """
    Providing keys that are not present in the simulated AnnData should raise errors.
    """
    adata = create_simulated_adata()
    cons = AnnDataSetConstructor(dataset_format="pairs")
    # Wrong obsm_key
    with pytest.raises(ValueError):
        cons.add_anndata(adata, obsm_key="wrong_key", caption_key="cell_type")
    # Wrong obs_key
    with pytest.raises(ValueError):
        cons.add_anndata(adata, obs_key="wrong_col", caption_key="cell_type")
    # Wrong caption_key
    with pytest.raises(ValueError):
        cons.add_anndata(adata, obs_key="cell_sentence", caption_key="wrong_caption")
    # Wrong batch_key
    with pytest.raises(ValueError):
        cons.add_anndata(
            adata, obsm_key="X_test", caption_key="cell_type", batch_key="wrong_batch"
        )


###############################################################################
# Tests for valid dataset construction
###############################################################################


def test_pairs_dataset_numeric():
    """
    For a pairs dataset using a numeric data representation.
    We provide obsm_key for the data rep and caption_key for the captions.
    The returned records should be pairs (two records per anchor), and the positive
    record should have a caption that matches the original cell_type.
    """
    adata = create_simulated_adata()
    cons = AnnDataSetConstructor(dataset_format="pairs", negatives_per_sample=1)
    cons.add_anndata(
        adata, obsm_key="X_test", caption_key="cell_type", batch_key="batch"
    )
    dataset = cons.get_dataset()

    # Check that each record has the required keys.
    for rec in dataset:
        assert "sample_idx" in rec
        assert "data_representation" in rec
        assert "caption" in rec
        assert "label" in rec
        sample_idx = rec["sample_idx"]
        obsm_sample_idx = adata.obs.index.get_loc(sample_idx)
        # For positive records (label 1.0), the caption should be one of the expected cell types.
        if rec["label"] == 1.0:
            true_caption = adata.obs.loc[sample_idx, "cell_type"]
            assert rec["caption"] == true_caption
        # For negative records, the caption should differ from the anchor's caption.
        if rec["label"] == 0.0:
            # Since our negative selection excludes matching caption, verify that.
            assert rec["caption"] != true_caption
        true_data_rep = adata.obsm["X_test"][obsm_sample_idx]
        # Check that the data representation matches the original.
        assert np.allclose(
            rec["data_representation"], true_data_rep, rtol=1e-3, atol=1e-5
        )
        #


def test_pairs_dataset_string():
    """
    For a pairs dataset using a string representation from obs.
    The obs_key is used to retrieve the representation and caption_key for captions.
    """
    adata = create_simulated_adata()
    cons = AnnDataSetConstructor(dataset_format="pairs", negatives_per_sample=1)
    cons.add_anndata(
        adata, obs_key="cell_sentence", caption_key="cell_type", batch_key="batch"
    )
    dataset = cons.get_dataset()

    for rec in dataset:
        assert "data_representation" in rec
        assert "caption" in rec
        assert "label" in rec
        # Since cell_sentence is a list of genes, check that it is indeed a list of length 10.
        if isinstance(rec["data_representation"], list):
            assert len(rec["data_representation"]) == 10
        # Check valid caption values.
        if rec["label"] == 1.0:
            sample_idx = rec["sample_idx"]
            true_caption = adata.obs.loc[sample_idx, "cell_type"]
            assert rec["caption"] == true_caption
        # For negative records, the caption should differ from the anchor's caption.
        if rec["label"] == 0.0:
            # Since our negative selection excludes matching caption, verify that.
            assert rec["caption"] != true_caption
        true_data_rep = adata.obs["cell_sentence"][sample_idx]
        # Check that the data representation matches the original.
        assert rec["data_representation"] == true_data_rep


def test_multiplets_dataset():
    """
    For a multiplets dataset using numeric representation.
    Each record should contain 'anchor', 'positive', and 'negatives'.
    We also test that the positive caption equals the original value in adata.obs.
    """
    adata = create_simulated_adata()
    cons = AnnDataSetConstructor(dataset_format="multiplets", negatives_per_sample=2)
    cons.add_anndata(
        adata, obsm_key="X_test", caption_key="cell_type", batch_key="batch"
    )
    dataset = cons.get_dataset()

    for rec in dataset:
        assert "anchor" in rec
        assert "positive" in rec
        assert "negative_1" in rec
        assert "negative_2" in rec
        sample_idx = rec["sample_idx"]
        obsm_sample_idx = adata.obs.index.get_loc(sample_idx)
        # Check that the positive caption matches the original cell_type.
        true_caption = adata.obs.loc[sample_idx, "cell_type"]
        assert rec["positive"] == true_caption
        # Check that the anchor and positive data representations match the original.
        true_data_rep = adata.obsm["X_test"][obsm_sample_idx]
        assert np.allclose(rec["anchor"], true_data_rep, rtol=1e-3, atol=1e-5)
        # Check that the negative_1 is a caption and negative_2 is a data representation.
        assert isinstance(rec["negative_1"], str)
        assert isinstance(rec["negative_2"][0], np.float16)
        # Check that the negative_1 caption is different from the positive caption.
        assert rec["negative_1"] != true_caption
        # Check that the negative_2 data representation is different from the anchor.
        assert not np.allclose(rec["negative_2"], true_data_rep, rtol=1e-3, atol=1e-5)


def test_single_dataset():
    """
    For a single dataset, only the representation is returned.
    We test both numeric and string representations.
    """
    adata = create_simulated_adata()
    # Test with numeric (obsm) representation.
    cons_numeric = AnnDataSetConstructor(dataset_format="single")
    cons_numeric.add_anndata(adata, obsm_key="X_test", batch_key="batch")
    dataset_numeric = cons_numeric.get_dataset()
    for rec in dataset_numeric:
        # In the 'single' format using numeric rep, record contains key 'data_representation'
        assert "data_representation" in rec
        sample_idx = rec["sample_idx"]
        obsm_sample_idx = adata.obs.index.get_loc(sample_idx)
        true_data_rep = adata.obsm["X_test"][obsm_sample_idx]
        # Check that the data representation matches the original.
        assert np.allclose(
            rec["data_representation"], true_data_rep, rtol=1e-3, atol=1e-5
        )

        # It should not include a caption.

    # Test with string (obs) representation.
    cons_string = AnnDataSetConstructor(dataset_format="single")
    cons_string.add_anndata(adata, obs_key="cell_sentence", batch_key="batch")
    dataset_string = cons_string.get_dataset()
    for rec in dataset_string:
        # In the 'single' format using string rep, record contains key 'caption'
        assert "data_representation" in rec
        sample_idx = rec["sample_idx"]
        true_data_rep = adata.obs["cell_sentence"][sample_idx]
        # Check that the data representation matches the original.
        assert rec["data_representation"] == true_data_rep
