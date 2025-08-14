import argparse
import logging
import time
from pathlib import Path

import anndata as ad

from adata_hf_datasets.ds_constructor import AnnDataSetConstructor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AnnDataSetConstructor speed on a Zarr-backed AnnData file."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path(
            "/Users/mengerj/repos/adata_hf_datasets/data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_10k/train/chunk_0.zarr"
        ),
        help="Path to the .zarr directory to read with anndata.read_zarr",
    )
    parser.add_argument(
        "--negatives-per-sample",
        type=int,
        default=2,
        help="Number of negatives per sample (default: 2)",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="multiplets",
        choices=["pairs", "multiplets", "single"],
        help="Dataset output format (default: multiplets)",
    )
    parser.add_argument(
        "--caption-key",
        type=str,
        default="natural_language_annotation",
        help="Column in .obs to use as caption",
    )
    parser.add_argument(
        "--batch-key",
        type=str,
        default="dataset_title",
        help="Column in .obs to use as batch",
    )
    parser.add_argument(
        "--share-link",
        type=str,
        default="https://example.org/dummy_share_link",
        help="Dummy share link to include in records",
    )
    parser.add_argument(
        "--use-add-df",
        action="store_true",
        help=(
            "If set, build the dataset using only .obs via add_df instead of add_anndata."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {args.zarr_path}")

    print(f"Reading AnnData from: {args.zarr_path}")
    t_read_start = time.perf_counter()
    adata = ad.read_zarr(str(args.zarr_path))
    t_read_end = time.perf_counter()

    n_obs = adata.n_obs
    print(f"Loaded AnnData with {n_obs} observations and {adata.n_vars} variables")

    # Validate required columns
    for required_col in (args.caption_key, args.batch_key):
        if required_col not in adata.obs.columns:
            raise KeyError(
                f"Required obs column '{required_col}' is missing. Present columns: {list(adata.obs.columns)}"
            )

    # Build dataset
    constructor = AnnDataSetConstructor(
        negatives_per_sample=args.negatives_per_sample,
        dataset_format=args.dataset_format,
    )

    if args.use_add_df:
        # Create a standalone DataFrame from obs and drop AnnData immediately
        t_add_start = time.perf_counter()
        obs_df = adata.obs.copy(deep=True)
        # Add dummy sentence column with enumerated sentences
        obs_df["sentence_1"] = [f"sentence {i}" for i in range(n_obs)]
        # Free AnnData as early as possible to avoid touching X/var/etc.
        del adata
        constructor.add_df(
            df=obs_df,
            sentence_keys=["sentence_1"],
            caption_key=args.caption_key,
            batch_key=args.batch_key,
            share_link=args.share_link,
        )
        t_add_end = time.perf_counter()
    else:
        # Add dummy sentence column with enumerated sentences on AnnData.obs
        t_add_start = time.perf_counter()
        adata.obs["sentence_1"] = [f"sentence {i}" for i in range(n_obs)]
        constructor.add_anndata(
            adata=adata,
            sentence_keys=["sentence_1"],
            caption_key=args.caption_key,
            batch_key=args.batch_key,
            share_link=args.share_link,
        )
        t_add_end = time.perf_counter()

    t_build_start = time.perf_counter()
    hf_ds = constructor.get_dataset()
    t_build_end = time.perf_counter()

    elapsed_read = t_read_end - t_read_start
    elapsed_add = t_add_end - t_add_start
    elapsed_build = t_build_end - t_build_start
    elapsed_total = elapsed_read + elapsed_add + elapsed_build
    num_records = len(hf_ds)
    # In 'multiplets' format, one record corresponds to one anchor sample (if negatives were found)
    samples_per_second = (
        num_records / elapsed_total if elapsed_total > 0 else float("inf")
    )

    print("\n=== Benchmark Results ===")
    print(f"Records constructed: {num_records}")
    print(f"Read .zarr time: {elapsed_read:.2f} s")
    print(f"{'add_df' if args.use_add_df else 'add_anndata'} time: {elapsed_add:.2f} s")
    print(f"get_dataset time: {elapsed_build:.2f} s")
    print(f"Total elapsed: {elapsed_total:.2f} s")
    print(f"Throughput: {samples_per_second:.2f} samples/s")


if __name__ == "__main__":
    main()
