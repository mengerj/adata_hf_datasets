#!/usr/bin/env bash
set -euo pipefail


# === User‐configurable section ===
MODE="gpu"         # "cpu" or "gpu"
GPU_COUNT="1"      # how many GPUs if MODE=gpu
DATANAME="cellxgene_pseudo_bulk_3_5k"
BATCH_KEY="dataset_title"
BATCH_SIZE=32
METHODS="geneformer"
SCRIPT="scripts/embed_chunks.slurm"
# =================================

# build extra sbatch flags based on MODE
SBATCH_EXTRA=()
if [[ "$MODE" == "gpu" ]]; then
    SBATCH_EXTRA+=( --partition=gpu --gres=gpu:"$GPU_COUNT" )
    JOB_SUFFIX="gpu"
else
    JOB_SUFFIX="cpu"
fi

TRAIN_DIR="data/RNA/processed/train/${DATANAME}/train"
VAL_DIR="data/RNA/processed/train/${DATANAME}/val"

# function to submit one array job
submit_array() {
    local label=$1
    local dir=$2
    local n
    n=$(ls -1 "${dir}"/*.h5ad | wc -l)
    if (( n == 0 )); then
        echo "No chunks in $dir → skipping"
        return
    fi

    sbatch \
      --job-name="embed_${label}_${JOB_SUFFIX}" \
      --array=0-$((n-1)) \
      "${SBATCH_EXTRA[@]}" \
      --export=INPUT_DIR="${dir}",\
METHODS="${METHODS}",\
BATCH_KEY="${BATCH_KEY}",\
BATCH_SIZE="${BATCH_SIZE}" \
      "$SCRIPT"
}

submit_array train "$TRAIN_DIR"
submit_array val   "$VAL_DIR"
