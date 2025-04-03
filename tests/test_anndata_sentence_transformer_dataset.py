import logging
from unittest.mock import MagicMock

import anndata
import numpy as np
import pandas as pd
import pytest
import os
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
    obs_data = pd.DataFrame(index=[f"S1{i}" for i in range(3)])
    var_data = pd.DataFrame(index=[f"G1{i}" for i in range(5)])
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
    obs_data = pd.DataFrame(index=[f"S2{i}" for i in range(2)])
    var_data = pd.DataFrame(index=[f"G2{i}" for i in range(4)])
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
    from adata_hf_datasets.ds_constructor import AnnDataSetConstructor

    return AnnDataSetConstructor(
        caption_constructor=mock_caption_constructor, negatives_per_sample=1
    )


@pytest.fixture
def ann_data_file_with_obsm(tmp_path):
    """
    Create an AnnData object with a defined .obsm layer ("X_emb") and write it to a .h5ad file.

    The "X_emb" layer will be a random numeric matrix of shape (3, 4) and the sample IDs will be the index.

    Returns
    -------
    str
        The path to the .h5ad file.
    """
    # Create a small AnnData with an obsm embedding layer
    obs_data = pd.DataFrame(index=[f"S{i}" for i in range(3)])
    var_data = pd.DataFrame(index=[f"G{i}" for i in range(6)])
    X = np.random.rand(3, 6)
    adata = anndata.AnnData(X=X, obs=obs_data, var=var_data)

    # Create an embedding layer "X_emb" with shape (3, 4)
    adata.obsm["X_emb"] = np.random.rand(3, 4)

    file_path = str(tmp_path / "test_with_obsm.h5ad")
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def ann_data_file_with_batch(tmp_path):
    """
    Create a small anndata object with a 'batch' column in obs for testing batch-specific negative sampling.
    Each sample is assigned a unique 'caption' in obs.

    Returns
    -------
    str
        Path to the .h5ad file containing the test AnnData object with a 'batch' column.

    Notes
    -----
    This fixture explicitly creates two batches 'batchA' and 'batchB'. We will confirm that
    when this data is used in pairs or multiplets mode with a specified batch_key, all
    negative samples come from the same batch as the anchor.
    """
    # We'll create 6 samples: 3 in batchA, 3 in batchB
    n_samples_per_batch = 3
    sample_ids_batchA = [f"sampleA_{i}" for i in range(n_samples_per_batch)]
    sample_ids_batchB = [f"sampleB_{i}" for i in range(n_samples_per_batch)]

    obs_index = sample_ids_batchA + sample_ids_batchB
    obs_data = pd.DataFrame(index=obs_index)
    # Fill 'batch' column
    obs_data["batch"] = ["batchA"] * n_samples_per_batch + [
        "batchB"
    ] * n_samples_per_batch
    # Create some random data for X
    n_vars = 5
    X = np.random.randn(len(obs_index), n_vars)

    # Create the AnnData object
    adata = anndata.AnnData(X=X, obs=obs_data)

    # We store a unique column for demonstration if needed
    # But if you have a mock_caption_constructor, it typically populates 'caption' automatically
    # or you can store a placeholder column in obs and rely on the real constructor logic
    adata.obs["my_unique_metadata"] = [
        f"unique_meta_{i}" for i in range(len(obs_index))
    ]

    # Write to .h5ad
    file_path = str(tmp_path / "test_with_batch.h5ad")
    adata.write_h5ad(file_path)
    return file_path


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

    from adata_hf_datasets.ds_constructor import AnnDataSetConstructor

    # Create constructor without a caption_constructor
    with pytest.raises(ValueError) as excinfo:
        AnnDataSetConstructor(caption_constructor=None, negatives_per_sample=1)
    assert "caption_constructor must be provided" in str(excinfo.value)


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
        metadata = ex["anndata_ref"]
        caption = ex["caption"]
        file_path = metadata["file_record"]["dataset_path"]
        sample_id = metadata["sample_id"]

        adata = anndata.read_h5ad(file_path)
        original_caption = adata.obs.loc[sample_id, "caption"]
        assert caption == original_caption

    for ex in neg_examples:
        metadata = ex["anndata_ref"]
        caption = ex["caption"]
        file_path = metadata["file_record"]["dataset_path"]
        sample_id = metadata["sample_id"]

        adata = anndata.read_h5ad(file_path)
        original_caption = adata.obs.loc[sample_id, "caption"]
        assert caption != original_caption


def test_batch_key_negatives_remain_within_same_batch(
    mock_caption_constructor, ann_data_file_with_batch
):
    """
    Test that when a batch_key is specified in the AnnDataSetConstructor, all negative samples
    come from the same batch as the anchor sample.

    References
    ----------
    Simulated data from the 'ann_data_file_with_batch' fixture, which contains two batches
    ('batchA' and 'batchB') in adata.obs['batch'].

    This test:
    1. Instantiates a constructor with batch_key='batch'.
    2. Adds the ann_data_file_with_batch.
    3. Generates the dataset in 'pairs' format (could also be 'multiplets').
    4. Parses each record:
       - For the positive/anchor sample, we get the anchor's batch.
       - For the negative sample, confirm that it has the same batch as the anchor.
    """
    from adata_hf_datasets.ds_constructor import AnnDataSetConstructor

    # Instantiate the constructor, specifying batch_key='batch'
    constructor = AnnDataSetConstructor(
        caption_constructor=mock_caption_constructor,
        negatives_per_sample=1,
        dataset_format="pairs",
    )

    # Add the file that contains multiple batches
    constructor.add_anndata(ann_data_file_with_batch, batch_key="batch")

    # Build the dataset
    ds = constructor.get_dataset()
    ds_list = ds[:]  # Convert to dict of columns -> lists

    # We expect pairs => each row has 'anndata_ref', 'caption', 'label'
    # The anchor + positive sample (label=1) is a separate record from the anchor + negative sample (label=0).
    # We'll pair them up or just check negative rows as a group.
    for anndata_ref, caption, label in zip(
        ds_list["anndata_ref"], ds_list["caption"], ds_list["label"]
    ):
        if label == 0.0:
            # This is a negative row
            # 1. Parse the anchor sample info from anndata_ref
            metadata = anndata_ref
            file_record = metadata["file_record"]
            sample_id = metadata["sample_id"]

            # 2. Read the AnnData from local path (or share link, if that is how you stored it)
            file_path = file_record["dataset_path"]
            adata_neg_check = anndata.read_h5ad(file_path)

            # 3. Anchor sample's batch
            anchor_batch = adata_neg_check.obs.loc[sample_id, "batch"]

            # 4. The negative's caption is purely text, so let's figure out which sample ID
            #    has that 'caption' in adata. We must find the row in adata that has the same text in adata.obs["caption"].
            #    Because the test uses mock_caption_constructor, the caption is "caption_{obs_name}" by default.
            #    So we can map back from the text to the sample. For example, if caption == 'caption_sampleA_1',
            #    then the negative sample is 'sampleA_1'.

            # The mock constructor sets "caption_{sample_id}" in adata.obs["caption"].
            # We can strip off the prefix "caption_" to find the sample ID.
            if not caption.startswith("caption_"):
                raise ValueError(f"Unexpected negative caption format: {caption}")
            negative_sample_id = caption.split("caption_", 1)[1]

            # 5. Check the negative sample's batch
            negative_batch = adata_neg_check.obs.loc[negative_sample_id, "batch"]
            # 6. Assert they match
            assert anchor_batch == negative_batch, (
                f"Expected negative to come from same batch. Anchor batch={anchor_batch}, "
                f"Negative batch={negative_batch} for anchor={sample_id}, negative={negative_sample_id}"
            )


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


def test_get_dataset_multiplets(dataset_constructor, ann_data_file_1, ann_data_file_2):
    """
    Test that get_dataset returns records with separate negative columns
    when dataset_format is set to "multiplets".
    """
    # Configure the constructor for multiplets with 2 negatives per sample
    dataset_constructor.dataset_format = "multiplets"
    dataset_constructor.negatives_per_sample = 2

    dataset_constructor.add_anndata(ann_data_file_1)
    dataset_constructor.add_anndata(ann_data_file_2)

    ds = dataset_constructor.get_dataset()

    # Check required columns
    assert "anndata_ref" in ds.features.keys()
    assert "positive" in ds.features.keys()
    # The negatives should appear as separate columns: negative_1 and negative_2
    assert "negative_1" in ds.features.keys()
    assert "negative_2" in ds.features.keys()
    # Optionally, check that anchor equals positive (if that is the intended behavior)


def test_get_dataset_single(dataset_constructor, ann_data_file_1):
    """
    Test that get_dataset returns records with only 'anndata_ref'
    when dataset_format is set to "single".
    """
    dataset_constructor.dataset_format = "single"
    dataset_constructor.add_anndata(ann_data_file_1)

    ds = dataset_constructor.get_dataset()
    # Only 'anndata_ref' and 'caption' should be present
    assert "anndata_ref" in ds.features.keys()
    # There should be no extra keys like 'anchor', 'positive', or negatives
    assert len(ds.features.keys()) == 1


def test_invalid_dataset_format(mock_caption_constructor):
    """
    Test that providing an invalid dataset_format during construction raises a ValueError.
    """
    from adata_hf_datasets.ds_constructor import AnnDataSetConstructor

    with pytest.raises(ValueError) as excinfo:
        AnnDataSetConstructor(
            caption_constructor=mock_caption_constructor,
            negatives_per_sample=1,
            dataset_format="invalid_format",
        )
    assert "dataset_format must be one of" in str(excinfo.value)


def test_obsm_extraction_and_storage(
    dataset_constructor, ann_data_file_with_obsm, tmp_path
):
    """
    Test that when providing obsm_keys to add_anndata:

    - The file record contains an "embeddings" key with a mapping for each obsm key.
    - The stored embedding file is saved locally.
    - Loading the embedding file returns an npz file containing "data" and "sample_ids"
      with expected shape and type.
    """
    logger.info("Testing extraction and storage of obsm layers.")

    # Use a temporary directory to store embedding files
    # (nextcloud is not configured in this test so local saving is used)
    obsm_keys = ["X_emb"]
    dataset_constructor.add_anndata(ann_data_file_with_obsm, obsm_keys=obsm_keys)

    # Retrieve the file record for the added anndata file.
    file_record = dataset_constructor.anndata_files[0]
    assert "embeddings" in file_record, (
        "Expected an 'embeddings' key in the file record."
    )
    assert "X_emb" in file_record["embeddings"], (
        "Expected key 'X_emb' in embeddings record."
    )

    embedding_path = file_record["embeddings"]["X_emb"]
    # Check that the file exists locally.
    assert os.path.exists(embedding_path), (
        f"Embedding file {embedding_path} does not exist."
    )

    # Load the saved npz file.
    loaded = np.load(embedding_path, allow_pickle=True)
    # The saved file should have arrays 'data' and 'sample_ids'
    assert "data" in loaded, "Expected 'data' key in saved embedding file."
    assert "sample_ids" in loaded, "Expected 'sample_ids' key in saved embedding file."

    # Verify the shape of the embedding matrix.
    # Our obsm["X_emb"] was of shape (3, 4)
    data_array = loaded["data"]
    sample_ids = loaded["sample_ids"]
    assert data_array.shape == (3, 4), f"Expected shape (3, 4), got {data_array.shape}."

    # Verify that sample_ids match the AnnData obs index
    adata = anndata.read_h5ad(ann_data_file_with_obsm)
    expected_ids = adata.obs.index.astype(str).to_numpy()
    # In case the sample_ids were saved as bytes (depending on numpy version), decode if needed.
    if sample_ids.dtype.type is np.bytes_:
        sample_ids = np.array([s.decode("utf-8") for s in sample_ids])
    np.testing.assert_array_equal(sample_ids, expected_ids)


def test_dataset_includes_embedding_reference(
    dataset_constructor, ann_data_file_with_obsm
):
    """
    Test that get_dataset returns records where the anndata_ref JSON includes the original file path.

    Although the embeddings are stored separately, the dataset record should still reference the
    original AnnData file (or its share link) so downstream tasks can retrieve full objects if needed.
    """
    logger.info(
        "Testing dataset records include the full AnnData reference even when embeddings are extracted."
    )

    # Add anndata with obsm extraction.
    obsm_keys = ["X_emb"]
    dataset_constructor.add_anndata(ann_data_file_with_obsm, obsm_keys=obsm_keys)

    # Build the dataset.
    ds = dataset_constructor.get_dataset()

    # Check each record for the "anndata_ref" key and that it contains a valid JSON with "file_path"
    for record in ds:
        assert "anndata_ref" in record, "Dataset record missing 'anndata_ref'."
        metadata = record["anndata_ref"]
        assert "file_record" in metadata, "Metadata missing 'file_record'."
        # Optionally, check that the file_path is the one we added.
        assert metadata["file_record"]["dataset_path"] == ann_data_file_with_obsm, (
            f"Expected file_path to be {ann_data_file_with_obsm}, got {metadata['file_record']}."
        )
        assert metadata["file_record"]["embeddings"]["X_emb"] is not None, (
            "Expected 'X_emb' key in embeddings record."
        )
