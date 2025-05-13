#!/usr/bin/env python
"""
Script to verify that the Hugging Face datasets (owned by "jo-mengr") contain valid .h5ad
files with matching sample IDs and correct embedding matrices in .obsm for each row.

Data Sources
------------
- Hugging Face Datasets owned by user "jo-mengr".
- Each row in the dataset references:
    1) A "file_record" with a "dataset_path" (Nextcloud share link to an .h5ad).
    2) A "sample_id" (IDs stored in adata.obs).
    3) An "embeddings" dict with share links to .npz files for various methods.
    4) For pair datasets:
       - 'caption' (str)
       - 'label' (int: 0 or 1) indicating negative or positive pair
    5) For multiplet datasets:
       - 'positive' (str) caption for positive
       - 'negative_1', 'negative_2', ... each referencing anndata_ref + caption entries

References
----------
- Nextcloud link usage
- anndata: https://anndata.readthedocs.io
- h5py: https://docs.h5py.org
- huggingface_hub: https://github.com/huggingface/huggingface_hub
- datasets: https://huggingface.co/docs/datasets

Example
-------
python verify_jo_mengr_datasets.py
"""

import os
import json
import tempfile
import requests
import numpy as np
import anndata
from huggingface_hub import HfApi
from datasets import load_dataset
from adata_hf_datasets.file_utils import download_file_from_share_link
from adata_hf_datasets.utils import setup_logging

logger = setup_logging()


def verify_h5ad_and_embeddings(dataset_split, split_name):
    """
    Verify the .h5ad + embeddings for a single split of a dataset.

    Parameters
    ----------
    dataset_split : datasets.Dataset
        The dataset split object or list of rows.
    split_name : str
        Name of the split (e.g. 'train', 'val', 'test').

    Returns
    -------
    valid : bool
        True if all checks in this split pass, False otherwise.
    fail_reasons : list of str
        List of textual descriptions for any failures encountered.
    """
    if "label" in dataset_split.column_names:
        dataset_split = dataset_split.filter(lambda x: x["label"] == 1)
    rows = list(dataset_split)
    if not rows:
        # If empty, treat as valid but note it (or adjust as needed).
        return True, [f"Split '{split_name}' is empty."]

    fail_reasons = []
    logger.info(f"Verifying split: {split_name}, Number of rows: {len(rows)}")

    # Basic key presence checks (only on the first row for speed).
    first_anndata = rows[0].get("anndata_ref", None)
    if not first_anndata:
        fail_reasons.append("Dataset missing 'anndata_ref' key.")
        return False, fail_reasons

    if not isinstance(first_anndata, dict):
        fail_reasons.append("anndata_ref is not a dictionary.")
        return False, fail_reasons

    if "sample_id" not in first_anndata:
        fail_reasons.append("anndata_ref missing 'sample_id' key.")
        return False, fail_reasons
    if "file_record" not in first_anndata:
        fail_reasons.append("anndata_ref missing 'file_record' key.")
        return False, fail_reasons
    if "dataset_path" not in first_anndata["file_record"]:
        fail_reasons.append("file_record missing 'dataset_path' key.")
        return False, fail_reasons

    # Group rows by share link to download .h5ad only once per link.
    adata_map = {}
    for i, row in enumerate(rows):
        # parse the 'anndata_ref' JSON
        row_info = row.get("anndata_ref", None)
        if not row_info:
            fail_reasons.append(f"Row {i} is missing 'anndata_ref'.")
            continue

        share_link = row_info["file_record"]["dataset_path"]
        sample_id = row_info["sample_id"]
        embeddings_dict = row_info.get("embeddings", {})

        # Store all row info so we can do positivity checks later
        if share_link not in adata_map:
            adata_map[share_link] = {
                "rows": [],
            }
        adata_map[share_link]["rows"].append((i, sample_id, embeddings_dict, row_info))

    # Now, for each share link, download once, load anndata, then do checks.
    with tempfile.TemporaryDirectory() as tmpdir:
        for adata_link, data_dict in adata_map.items():
            local_adata_path = os.path.join(tmpdir, "temp_adata.h5ad")
            success = download_file_from_share_link(adata_link, local_adata_path)
            if not success:
                fail_reasons.append(
                    f"Failed to download or validate .h5ad from link={adata_link}"
                )
                continue

            # Load the adata object (in read-only backed mode).
            try:
                adata = anndata.read_h5ad(local_adata_path, backed="r")
            except Exception as e:
                fail_reasons.append(f"Cannot read .h5ad for link={adata_link}: {e}")
                continue

            # Check sample IDs from the dataset vs. file
            dataset_sample_ids = [r[1] for r in data_dict["rows"]]
            file_sample_ids = list(adata.obs.index)
            if dataset_sample_ids != file_sample_ids:
                fail_reasons.append(
                    "Mismatch in sample IDs or ordering for ={}. "
                    "First few dataset IDs: {}, first file IDs: {}".format(
                        adata_link, dataset_sample_ids[:1], file_sample_ids[:1]
                    )
                )

            # Optional: check if 'caption' is in adata.obs columns before verifying positive captions
            obs_cols = adata.obs.columns
            has_caption_col = "caption" in obs_cols

            # For each row referencing this share link, do positivity checks
            # before or after embedding checks as you prefer:
            for row_idx, samp_id, embed_dict, row_info in data_dict["rows"]:
                # 1) If this row is a pair with label=1, ensure row_info["caption"] matches adata.obs["caption"][samp_id].
                if "caption" in row_info and "label" in row_info:
                    if row_info["label"] == 1:
                        if not has_caption_col:
                            fail_reasons.append(
                                f"Row {row_idx}: 'caption' column not found in adata.obs. Cannot verify positive pair."
                            )
                        else:
                            actual_caption = adata.obs["caption"][samp_id]
                            expected_caption = row_info["caption"]
                            if actual_caption != expected_caption:
                                fail_reasons.append(
                                    f"Row {row_idx}: Positive pair mismatch. "
                                    f"adata.obs['caption'][{samp_id}] = '{actual_caption}' "
                                    f"!= '{expected_caption}' in row_info."
                                )

                # 2) If this row is a multiplet with "positive", ensure row_info["positive"] == adata.obs["caption"][samp_id].
                if "positive" in row_info:
                    if not has_caption_col:
                        fail_reasons.append(
                            f"Row {row_idx}: 'caption' column not found in adata.obs. Cannot verify positive caption."
                        )
                    else:
                        actual_caption = adata.obs["caption"][samp_id]
                        expected_caption = row_info["positive"]
                        if actual_caption != expected_caption:
                            fail_reasons.append(
                                f"Row {row_idx}: Positive mismatch. "
                                f"adata.obs['caption'][{samp_id}] = '{actual_caption}' "
                                f"!= '{expected_caption}' in row_info."
                            )

            # ------------------------------------------
            # EMBEDDING CHECKS (same as your original logic)
            # ------------------------------------------
            # Collect unique embeddings across all rows referencing this link
            embedding_links_by_method = {}
            for row_idx, samp_id, embed_dict, _ in data_dict["rows"]:
                for method, emb_link in embed_dict.items():
                    embedding_links_by_method.setdefault(method, set()).add(emb_link)

            # Verify each embedding method's .obsm key and compare with .npz
            for method_name, emb_links in embedding_links_by_method.items():
                obsm_key = f"X_{method_name}"
                if obsm_key not in adata.obsm:
                    fail_reasons.append(
                        f"Missing .obsm key={obsm_key} in adata for link={adata_link}."
                    )
                    continue

                for link in emb_links:
                    local_npz_path = os.path.join(tmpdir, f"{method_name}.npz")
                    resp = requests.get(link)
                    if resp.status_code != 200:
                        fail_reasons.append(
                            f"Failed to download .npz for method={method_name}, link={link}"
                        )
                        continue
                    with open(local_npz_path, "wb") as f:
                        f.write(resp.content)

                    try:
                        npz_data = np.load(local_npz_path, allow_pickle=True)
                    except Exception as e:
                        fail_reasons.append(
                            f"Could not load .npz for method={method_name}, link={link}: {e}"
                        )
                        continue

                    if "arr_0" not in npz_data:
                        fail_reasons.append(
                            f"Key 'arr_0' missing in npz for method={method_name}, link={link}"
                        )
                        continue

                    adata_emb = adata.obsm[obsm_key]
                    npz_emb = npz_data["arr_0"]

                    if adata_emb.shape != npz_emb.shape:
                        fail_reasons.append(
                            f"Shape mismatch for method={method_name}, link={link}, "
                            f"adata shape={adata_emb.shape}, npz shape={npz_emb.shape}"
                        )
                    else:
                        # Check numeric equivalence
                        if not np.allclose(adata_emb, npz_emb):
                            fail_reasons.append(
                                f"Value mismatch in embeddings for method={method_name}, link={link}"
                            )

    valid = len(fail_reasons) == 0
    return valid, fail_reasons


def verify_dataset(dataset_id):
    """
    Verifies a dataset by checking each split's .h5ad and embedding references.

    Parameters
    ----------
    dataset_id : str
        The full HF dataset ID, e.g. 'jo-mengr/my_dataset'.

    Returns
    -------
    valid : bool
        True if all splits are valid, False otherwise.
    fail_messages : list of str
        List of textual fail messages if invalid.
    """
    setup_logging()
    logger.info(f"Verifying dataset: {dataset_id}")
    try:
        hf_ds = load_dataset(dataset_id, download_mode="force_redownload")
    except Exception as e:
        logger.error(f"❌ Cannot load dataset {dataset_id}: {e}")
        return False, [f"Cannot load dataset: {e}"]

    all_fail_msgs = []
    for split_name in hf_ds.keys():
        logger.info(f"Checking split: {split_name}")
        valid_split, fail_reasons = verify_h5ad_and_embeddings(
            hf_ds[split_name], split_name
        )
        if not valid_split:
            for reason in fail_reasons:
                logger.error(f"❌ {reason}")
            all_fail_msgs.extend(fail_reasons)

    dataset_valid = len(all_fail_msgs) == 0
    if dataset_valid:
        logger.info(f"✅ All checks passed for dataset: {dataset_id}")
    else:
        logger.error(
            f"❌ Dataset {dataset_id} is invalid with {len(all_fail_msgs)} issues."
        )

    return dataset_valid, all_fail_msgs


def main():
    """
    Main entry point to iterate over all HF datasets owned by 'jo-mengr'
    and validate their .h5ad and embedding references.

    If any are invalid, writes a JSON file with details.

    Returns
    -------
    None
    """
    # If desired, store invalid results in a structure
    invalid_datasets = []
    # load_dotenv(override=True)
    api = HfApi()
    user_datasets = api.list_datasets(author="jo-mengr", limit=None)
    user_datasets_list = list(user_datasets)
    # user_datasets_gen = api.list_datasets(
    #    author="jo-mengr", limit=None, token=os.getenv("HF_TOKEN")
    # )
    logger.info(f"Found {len(user_datasets_list)} datasets in user 'jo-mengr' space.")

    for ds_info in user_datasets_list:
        ds_id = ds_info.id  # e.g. "jo-mengr/my_dataset_name"
        valid, fail_msgs = verify_dataset(ds_id)
        if not valid:
            invalid_entry = {"dataset_id": ds_id, "reasons": fail_msgs}
            invalid_datasets.append(invalid_entry)

    if invalid_datasets:
        # Save them to a JSON file
        invalid_file = "invalid_datasets.json"
        with open(invalid_file, "w") as f:
            json.dump(invalid_datasets, f, indent=2)
        logger.error(
            f"❌ There are {len(invalid_datasets)} invalid datasets. "
            f"Details saved to '{invalid_file}'."
        )
        logger.info(
            "Run python scripts/delete_invalid_ds.py to remove these from your HF account, "
            "and then optionally delete the JSON file."
        )
    else:
        logger.info("✅ All datasets are valid! No invalid list needed.")


if __name__ == "__main__":
    main()
