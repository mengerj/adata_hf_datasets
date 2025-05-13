#!/usr/bin/env python
"""
Turn all Hugging Face models and datasets under a given user into private repositories.

Data Source
-----------
- Hugging Face Hub: https://huggingface.co

References
----------
- huggingface_hub: https://github.com/huggingface/huggingface_hub

Example
-------
python make_all_hf_repos_private.py --user my_hf_username
"""

import os
import argparse
import logging
from huggingface_hub import HfApi
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)


def make_all_hf_repos_private(username, token=None):
    """
    Make all models and datasets owned by a given Hugging Face user private.

    Parameters
    ----------
    username : str
        The Hugging Face username (e.g., "myuser") for which you want to set all repositories to private.
    token : str, optional
        A valid Hugging Face token with write access. If not provided, the script attempts
        to read the token from the HF_TOKEN environment variable.

    Returns
    -------
    None
        This function does not return any value; it updates repository visibility to private.

    Raises
    ------
    ValueError
        If no valid token is found (either from the parameter or environment).
    """
    if not token:
        token = os.getenv("HF_TOKEN", None)

    if not token:
        raise ValueError(
            "No Hugging Face token provided. Please pass `token` or set `HF_TOKEN` env variable."
        )

    hf_api = HfApi()

    # Fetch all models for the user and set them to private
    models = hf_api.list_models(author=username, token=token)
    models_list = list(models)
    logger.info(f"Found {len(models_list)} model(s) under user '{username}'.")
    for model_info in models_list:
        repo_id = model_info.modelId  # e.g., "username/model_name"
        logger.info(f"Setting model '{repo_id}' to private.")
        hf_api.update_repo_settings(
            repo_id=repo_id, private=True, token=token, repo_type="model"
        )

    # Fetch all datasets for the user and set them to private
    datasets = hf_api.list_datasets(author=username, token=token)
    datasets_list = list(datasets)
    logger.info(f"Found {len(datasets_list)} dataset(s) under user '{username}'.")
    for ds_info in datasets_list:
        repo_id = ds_info.id  # e.g., "username/dataset_name"
        logger.info(f"Setting dataset '{repo_id}' to private.")
        hf_api.update_repo_settings(
            repo_id=repo_id, private=True, token=token, repo_type="dataset"
        )

    logger.info("✅ All Hugging Face repositories have been set to private.")


def main():
    """
    Main entry point for the script. Parses command-line arguments and calls
    make_all_hf_repos_private.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Turn all Hugging Face models and datasets under a given username private."
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Hugging Face username (e.g., 'jo-mengr')",
        default="jo-mengr",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token with write permission. If not provided, reads HF_TOKEN env var.",
    )
    args = parser.parse_args()
    load_dotenv()
    make_all_hf_repos_private(username=args.user, token=args.token)


if __name__ == "__main__":
    main()
