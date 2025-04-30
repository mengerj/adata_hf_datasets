#!/usr/bin/env bash
set -euo pipefail

# Adjust these as needed
DATANAME="cellxgene_pseudo_bulk_35k"
BATCH_KEY="dataset_title"
BATCH_SIZE=32
METHODS="hvg pca scvi_fm"   # space‐separated list
SCRIPT="embed_array.slurm"

# function to submit one array job
submit_array() {
    local label=$1
    local dir=$2

    # count chunks
    local n=$(ls -1 "${dir}"/*.h5ad | wc -l)
    if (( n == 0 )); then
        echo "No chunks in $dir, skipping $label"
        return
    fi

    sbatch \
      --job-name="embed_${label}" \
      --array=0-$((n-1)) \
      --export=INPUT_DIR="${dir}",\
METHODS="${METHODS}",\
BATCH_KEY="${BATCH_KEY}",\
BATCH_SIZE="${BATCH_SIZE}" \
      "${SCRIPT}"
}

TRAIN_DIR="data/RNA/processed/train/${DATANAME}/train_chunks"
VAL_DIR="data/RNA/processed/train/${DATANAME}/val_chunks"

submit_array train "$TRAIN_DIR"
submit_array val   "$VAL_DIR"
