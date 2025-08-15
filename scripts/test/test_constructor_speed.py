import argparse
import logging
import time
from pathlib import Path

import anndata as ad
import pandas as pd

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
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=4,
        help="Number of sentence_* columns to include (default: 1)",
    )
    parser.add_argument(
        "--long-sentence-cols",
        type=str,
        default="2,3",
        help=(
            "Comma-separated 1-based indices of sentence columns to fill with a long string (e.g. '2,3')."
        ),
    )
    parser.add_argument(
        "--long-sentence-words",
        type=int,
        default=5000,
        help="Number of space-separated tokens to include in the long sentence string (default: 5000)",
    )
    parser.add_argument(
        "--replicate-rows",
        type=int,
        default=10,
        help="Replicate the dataset rows this many times to artificially increase size (default: 1)",
    )
    parser.add_argument(
        "--add-multiple-times",
        type=int,
        default=5,
        help="Call add_anndata/add_df this many times with the same data (default: 1)",
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

    # determine sentence keys and prepare long sentence payload, if any
    sentence_keys = [f"sentence_{i}" for i in range(1, args.num_sentences + 1)]
    long_cols = set()
    if args.long_sentence_cols.strip():
        try:
            long_cols = {
                int(tok.strip())
                for tok in args.long_sentence_cols.split(",")
                if tok.strip()
            }
        except ValueError:
            raise ValueError(
                "--long-sentence-cols must be a comma-separated list of integers"
            )
    # Precompute a constant long sentence to avoid generating per-row strings
    long_sentence_value = (
        " ".join(f"GENE{i}" for i in range(1, args.long_sentence_words + 1))
        if long_cols
        else None
    )

    if args.use_add_df:
        # Create a standalone DataFrame from obs and drop AnnData immediately
        t_add_start = time.perf_counter()
        obs_df = adata.obs.copy(deep=True)
        # Add sentence columns (some potentially very long)
        for i in range(1, args.num_sentences + 1):
            col = f"sentence_{i}"
            if i in long_cols and long_sentence_value is not None:
                obs_df[col] = long_sentence_value
            else:
                obs_df[col] = [f"row {r} sentence_col {i}" for r in range(n_obs)]

        # Replicate rows if requested
        if args.replicate_rows > 1:
            print(f"Replicating DataFrame {args.replicate_rows} times...")
            replicated_dfs = []
            for rep in range(args.replicate_rows):
                df_copy = obs_df.copy()
                # Modify index to avoid duplicates
                df_copy.index = [f"{idx}_rep{rep}" for idx in df_copy.index]
                replicated_dfs.append(df_copy)
            obs_df = pd.concat(replicated_dfs, ignore_index=False)
            print(f"DataFrame now has {len(obs_df)} rows after replication")

        # Free AnnData as early as possible to avoid touching X/var/etc.
        del adata

        # Add the DataFrame multiple times if requested
        print(f"Adding DataFrame {args.add_multiple_times} times...")
        for add_idx in range(args.add_multiple_times):
            # Create unique indices for each addition to avoid conflicts
            df_to_add = obs_df.copy()
            if args.add_multiple_times > 1:
                df_to_add.index = [f"{idx}_add{add_idx}" for idx in df_to_add.index]

            constructor.add_df(
                df=df_to_add,
                sentence_keys=sentence_keys,
                caption_key=args.caption_key,
                batch_key=args.batch_key,
                share_link=args.share_link,
            )
        t_add_end = time.perf_counter()
    else:
        # Add dummy sentence column with enumerated sentences on AnnData.obs
        t_add_start = time.perf_counter()
        for i in range(1, args.num_sentences + 1):
            col = f"sentence_{i}"
            if i in long_cols and long_sentence_value is not None:
                adata.obs[col] = long_sentence_value
            else:
                adata.obs[col] = [f"row {r} sentence_col {i}" for r in range(n_obs)]

        # For replication with add_anndata, we need to use the DataFrame approach
        # because AnnData enforces obs/X size consistency
        if args.replicate_rows > 1:
            print("Replicating via DataFrame (add_anndata path with replication)...")
            obs_df = adata.obs.copy()
            replicated_dfs = []
            for rep in range(args.replicate_rows):
                df_copy = obs_df.copy()
                # Modify index to avoid duplicates
                df_copy.index = [f"{idx}_rep{rep}" for idx in df_copy.index]
                replicated_dfs.append(df_copy)
            obs_df = pd.concat(replicated_dfs, ignore_index=False)
            print(f"DataFrame now has {len(obs_df)} rows after replication")
            # Free AnnData and use DataFrame approach
            del adata

            # Add the DataFrame multiple times if requested
            print(f"Adding DataFrame {args.add_multiple_times} times...")
            for add_idx in range(args.add_multiple_times):
                # Create unique indices for each addition to avoid conflicts
                df_to_add = obs_df.copy()
                if args.add_multiple_times > 1:
                    df_to_add.index = [f"{idx}_add{add_idx}" for idx in df_to_add.index]

                constructor.add_df(
                    df=df_to_add,
                    sentence_keys=sentence_keys,
                    caption_key=args.caption_key,
                    batch_key=args.batch_key,
                    share_link=args.share_link,
                )
        else:
            # Add AnnData multiple times if requested
            print(f"Adding AnnData {args.add_multiple_times} times...")
            for add_idx in range(args.add_multiple_times):
                # For multiple additions, we need to create copies with unique indices
                if args.add_multiple_times > 1:
                    adata_copy = adata.copy()
                    adata_copy.obs.index = [
                        f"{idx}_add{add_idx}" for idx in adata_copy.obs.index
                    ]
                    constructor.add_anndata(
                        adata=adata_copy,
                        sentence_keys=sentence_keys,
                        caption_key=args.caption_key,
                        batch_key=args.batch_key,
                        share_link=args.share_link,
                    )
                else:
                    constructor.add_anndata(
                        adata=adata,
                        sentence_keys=sentence_keys,
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
    print(
        "Config: keys="
        + ", ".join(sentence_keys)
        + f" | long_cols={sorted(list(long_cols)) if long_cols else []}"
        + (f" | long_words={args.long_sentence_words}" if long_cols else "")
        + f" | replicate_rows={args.replicate_rows}"
        + f" | add_multiple_times={args.add_multiple_times}"
    )


if __name__ == "__main__":
    main()
