import json
import logging
from unittest.mock import MagicMock

import anndata
import numpy as np
import pandas as pd
import pytest

# If your class is in a separate module, you'd import from there, e.g.:
# from my_package.my_module import AnnDataSetConstructor

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_caption_constructor():
    """
    Pytest fixture that returns a mock caption constructor.

    The mock simply sets adata.obs["caption"] to be a string derived
    from the obs_names, for demonstration.
    """
    constructor = MagicMock()

    def construct_captions(adata):
        """Simulated construction of captions, stored in adata.obs['caption'].
        For demonstration, each sample gets a caption 'caption_{sample_id}'."""
        captions = [f"caption_{idx}" for idx in adata.obs_names]
        adata.obs["caption"] = captions

    constructor.construct_captions.side_effect = construct_captions
    return constructor


@pytest.fixture
def ann_data_file_1(tmp_path):
    """
    Create a small anndata object and write it to a .h5ad file for testing.

    Returns
    -------
    str
        The path to the .h5ad file containing the test anndata object.
    """
    # Create a small adata
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(3)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(5)])
    X = np.random.rand(3, 5)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test1.h5ad")
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def ann_data_file_2(tmp_path):
    """
    Create another small anndata object and write it to a .h5ad file for testing.

    Returns
    -------
    str
        The path to the second .h5ad file containing the test anndata object.
    """
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(2)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(4)])
    X = np.random.rand(2, 4)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test2.h5ad")
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def ann_data_file_h5ad(tmp_path):
    """
    Create a small anndata object and write it to an .h5ad file for testing.

    Returns
    -------
    str
        The path to the .h5ad file containing the test anndata object.
    """
    # Create a small adata
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(3)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(5)])
    X = np.random.rand(3, 5)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test_h5ad_input.h5ad")
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def ann_data_file_with_duplicates(tmp_path):
    """
    Create an anndata object with duplicate sample IDs and an alternative unique ID column.

    Returns
    -------
    str
        The path to the .h5ad file containing the test anndata object with duplicates.
    """
    # Create data with duplicate indices
    obs_data = pd.DataFrame(
        {"unique_id": [f"unique_{i}" for i in range(4)], "batch": ["A", "A", "B", "B"]},
        index=["S1", "S1", "S2", "S2"],  # Duplicate indices
    )
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(5)])
    X = np.random.rand(4, 5)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test_duplicates.h5ad")
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def dataset_constructor(mock_caption_constructor):
    """
    Fixture to instantiate the AnnDataSetConstructor with a mocked caption constructor.

    Returns
    -------
    AnnDataSetConstructor
        A fresh instance for testing.
    """
    from adata_hf_datasets.adata_ref_ds import AnnDataSetConstructor

    return AnnDataSetConstructor(
        caption_constructor=mock_caption_constructor, negatives_per_sample=1
    )


def test_add_anndata_success(dataset_constructor, ann_data_file_1, ann_data_file_2):
    """
    Test that we can successfully add distinct anndata files without error.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' and 'ann_data_file_2' fixtures.
    """
    logger.info("Testing adding anndata files to the constructor.")

    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.add_anndata(ann_data_file_2)

    assert len(dataset_constructor.anndata_files) == 2
    files_list = []
    for files in dataset_constructor.anndata_files:
        files_list.append(files["local_path"])
    assert ann_data_file_1 in files_list
    assert ann_data_file_2 in files_list


def test_add_anndata_duplicate(dataset_constructor, ann_data_file_1):
    """
    Test that adding the same file twice raises a ValueError.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing duplicate anndata file addition.")

    dataset_constructor.add_anndata(ann_data_file_1)
    with pytest.raises(ValueError) as excinfo:
        dataset_constructor.add_anndata(ann_data_file_1)
    assert "has already been added" in str(excinfo.value)


def test_add_anndata_nonexistent_file(dataset_constructor):
    """
    Test that adding a non-existent file path raises an error when building the dataset.

    We do NOT raise the error in add_anndata itself, but rather rely on buildCaption failing
    or anndata.read_h5ad failing. If you'd like to fail earlier, you could check existence
    in add_anndata.

    References
    ----------
    No real file provided, a dummy path is used to test error handling.
    """

    logger.info("Testing adding a non-existent anndata file path.")

    fake_path = "some/non_existent_file.h5ad"
    with pytest.raises(FileNotFoundError):
        dataset_constructor.add_anndata(fake_path)


def test_no_caption_constructor(ann_data_file_1):
    """
    Test that if no caption constructor is provided, buildCaption might fail or we skip building captions.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing behavior with no caption constructor provided.")

    from adata_hf_datasets.adata_ref_ds import AnnDataSetConstructor

    # Create constructor without a caption_constructor
    constructor = AnnDataSetConstructor(
        caption_constructor=None, negatives_per_sample=1
    )
    constructor.add_anndata(ann_data_file_1)

    # If your code does not handle 'None' and tries to call 'construct_captions', it should fail
    with pytest.raises(AttributeError):
        constructor.get_dataset()


def test_caption_constructor_fail(dataset_constructor, ann_data_file_1):
    """
    Test that if the provided caption constructor fails, we catch and raise the exception.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing failing caption constructor scenario.")

    # Force the mock to raise an exception
    def fail_construct_captions(adata):
        raise RuntimeError("Caption constructor error.")

    dataset_constructor.caption_constructor.construct_captions.side_effect = (
        fail_construct_captions
    )

    dataset_constructor.add_anndata(ann_data_file_1)

    with pytest.raises(RuntimeError) as excinfo:
        dataset_constructor.get_dataset()
    assert "Caption constructor error" in str(excinfo.value)


def separate_pos_neg_examples(ds_list):
    """
    Convert the ds_list dictionary into a list of row dictionaries,
    then split them into positives (label=1.0) and negatives (label=0.0).

    ds_list structure:
    {
       "anndata_ref": [str, str, ...],  # JSON-encoded metadata
       "caption": [str, str, ...],
       "label": [float, float, ...]
    }

    Returns
    -------
    pos_examples : list of dict
    neg_examples : list of dict

    where each dict has keys "anndata_ref", "caption", "label"
    and "metadata" (the parsed JSON from anndata_ref).
    """
    # Convert column-based ds_list to row-based
    num_rows = len(ds_list["label"])
    row_list = []
    for i in range(num_rows):
        row_list.append(
            {
                # We can keep the raw JSON string in "anndata_ref",
                # or parse it right away
                "anndata_ref": ds_list["anndata_ref"][i],
                "caption": ds_list["caption"][i],
                "label": ds_list["label"][i],
            }
        )

    # Now separate positives & negatives
    pos_examples = []
    neg_examples = []
    for row in row_list:
        if row["label"] == 1.0:
            pos_examples.append(row)
        else:
            neg_examples.append(row)

    return pos_examples, neg_examples


def test_get_dataset_positive_and_negative(
    dataset_constructor, ann_data_file_1, ann_data_file_2
):
    """
    Demonstration of using the separation logic in a typical test scenario.
    """
    logger.info(
        "Testing construction of positive and negative examples in the dataset."
    )

    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.add_anndata(ann_data_file_2)

    # ds is your huggingface Dataset
    ds = dataset_constructor.get_dataset()

    # ds_list is a dict of columns -> lists
    ds_list = ds[:]

    # Separate pos/neg
    pos_examples, neg_examples = separate_pos_neg_examples(ds_list)

    # Basic checks
    assert len(pos_examples) == 5, f"Expected 5 positives, got {len(pos_examples)}"
    assert len(neg_examples) == 5, f"Expected 5 negatives, got {len(neg_examples)}"

    # Now parse each row and verify
    for ex in pos_examples:
        # parse the JSON string
        metadata = json.loads(ex["anndata_ref"])
        caption = ex["caption"]
        file_path = metadata["file_path"]
        sample_id = metadata["sample_id"]

        adata = anndata.read_h5ad(file_path)
        original_caption = adata.obs.loc[sample_id, "caption"]
        assert caption == original_caption

    for ex in neg_examples:
        metadata = json.loads(ex["anndata_ref"])
        caption = ex["caption"]
        file_path = metadata["file_path"]
        sample_id = metadata["sample_id"]

        adata = anndata.read_h5ad(file_path)
        original_caption = adata.obs.loc[sample_id, "caption"]
        assert caption != original_caption


def test_clear_method(dataset_constructor, ann_data_file_1):
    """
    Test that the clear method resets all internal data structures.

    References
    ----------
    Simulated anndata from the 'ann_data_file_1' fixture.
    """
    logger.info("Testing the clear method of the constructor.")

    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.clear()

    assert len(dataset_constructor.anndata_files) == 0
    assert len(dataset_constructor.sample_id_keys) == 0
    assert len(dataset_constructor.dataset) == 0


def test_duplicate_sample_ids(dataset_constructor, ann_data_file_with_duplicates):
    """
    Test that adding an anndata file with duplicate sample IDs:
    1. Raises an error when using default index
    2. Works when using a unique sample_id_key
    """
    logger.info("Testing behavior with duplicate sample IDs.")

    # Should raise error when using default index (which has duplicates)
    with pytest.raises(ValueError) as excinfo:
        dataset_constructor.add_anndata(ann_data_file_with_duplicates)

    err_msg = str(excinfo.value)
    assert "duplicate sample IDs" in err_msg
    assert "Currently using adata.obs.index as sample IDs" in err_msg
    assert "Example duplicates: ['S1', 'S2']" in err_msg

    # Clear the constructor
    dataset_constructor.clear()

    # Should work when using the unique_id column
    dataset_constructor.add_anndata(
        ann_data_file_with_duplicates, sample_id_key="unique_id"
    )

    # Verify that the file was added successfully
    assert len(dataset_constructor.anndata_files) == 1
    for files in dataset_constructor.anndata_files:
        assert ann_data_file_with_duplicates in files["local_path"]

    # Verify that we can build and get the dataset without errors
    dataset = dataset_constructor.get_dataset()
    assert len(dataset) > 0  # Should have both positive and negative examples


def test_duplicate_sample_ids_in_custom_key(dataset_constructor, tmp_path):
    """
    Test that using a sample_id_key that contains duplicates also raises an error.
    """
    # Create data with duplicates in the custom key
    obs_data = pd.DataFrame(
        {
            "custom_id": ["ID1", "ID1", "ID2", "ID2"],  # Duplicate IDs
            "batch": ["A", "A", "B", "B"],
        },
        index=[f"S{i}" for i in range(4)],  # Unique indices
    )
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(5)])
    X = np.random.rand(4, 5)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    file_path = str(tmp_path / "test_duplicate_custom_ids.h5ad")
    adata.write_h5ad(file_path)

    # Should raise error when using custom_id which has duplicates
    with pytest.raises(ValueError) as excinfo:
        dataset_constructor.add_anndata(file_path, sample_id_key="custom_id")

    err_msg = str(excinfo.value)
    assert "duplicate sample IDs" in err_msg
    assert "Currently using adata.obs['custom_id']" in err_msg
    assert "Example duplicates: ['ID1', 'ID2']" in err_msg
