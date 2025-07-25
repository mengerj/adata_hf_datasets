#!/usr/bin/env python
"""
delete_invalid_ds.py

Reads a file of invalid datasets and attempts to delete them
from Hugging Face Hub using HfApi.

Dependencies
------------
- huggingface_hub

Usage
-----
python delete_invalid_ds.py --file invalid_datasets.json
"""

import os
import sys
import json
import argparse
import logging
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete invalid datasets from Hugging Face Hub"
    )
    parser.add_argument(
        "--file",
        default="invalid_datasets.json",
        help="Path to the JSON file listing invalid datasets.",
    )
    parser.add_argument(
        "--yes", action="store_true", help="If set, do not prompt for confirmation."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    invalid_file = args.file
    if not os.path.exists(invalid_file):
        logger.error(f"File not found: {invalid_file}")
        return 1

    with open(invalid_file, "r") as f:
        invalid_datasets = json.load(f)

    if not invalid_datasets:
        logger.info("No invalid datasets found in the file.")
        return 0

    logger.info(f"Found {len(invalid_datasets)} invalid datasets in {invalid_file}:")
    for entry in invalid_datasets:
        logger.info(f"- {entry['dataset_id']}")
    logger.info("")

    if not args.yes:
        confirm = input(
            "Are you sure you want to DELETE these datasets from Hugging Face? [y/N]: "
        )
        if confirm.lower() not in ("y", "yes"):
            logger.info("Aborted.")
            return 0

    api = HfApi()

    # Attempt to delete each dataset
    for entry in invalid_datasets:
        ds_id = entry["dataset_id"]
        logger.info(f"Deleting dataset: {ds_id}")
        try:
            api.delete_repo(repo_id=ds_id, repo_type="dataset")
            logger.info(f"Deleted {ds_id} successfully.")
        except Exception as e:
            logger.error(f"Failed to delete {ds_id}: {e}")

    # Optionally remove the JSON file
    os.remove(invalid_file)
    logger.info(f"Removed {invalid_file}. Cleanup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
