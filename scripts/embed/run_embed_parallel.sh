#!/usr/bin/env bash
set -euo pipefail


# === User‐configurable section ===
MODE="cpu"         # "cpu" or "gpu"
GPU_COUNT="1"      # how many GPUs if MODE=gpu
DATANAME="cellxgene_pseudo_bulk_3_5k"
BATCH_KEY="dataset_title"
BATCH_SIZE=128
METHODS="pca" #"scvi_fm pca"  # space‐separated list - eg one string with spaces
SCRIPT="scripts/embed/embed_chunks_parallel.slurm"
MAX_PROCS=2
#SCRIPT="scripts/embed/prepare_embed_chunks_parallel.slurm"
TRAIN_OR_TEST="train"
#DATA_BASE_DIR="/scratch/global/menger/data/RNA/processed"
DATA_BASE_DIR="data/RNA/processed/"
# =================================
# build extra sbatch flags based on MODE
SBATCH_EXTRA=()
if [[ "$MODE" == "gpu" ]]; then
    SBATCH_EXTRA+=( --partition=gpu --gres=gpu:"$GPU_COUNT" )
    JOB_SUFFIX="gpu"
else
    JOB_SUFFIX="cpu"
fi

# function to submit one array job
submit_array() {
    local label=$1
    local dir=$2

    # count chunks
    local n
    n=$(ls -1d "${dir}"/*.zarr 2>/dev/null | wc -l)
    if (( n == 0 )); then
        echo "No chunks in $dir → skipping"
        return
    fi
    echo "Submitting $n chunks for '$label' in parallel"
    if command -v sbatch &>/dev/null; then
        # — under SLURM, submit an array job —
        sbatch \
          --array=0-$((n-1)) \
          "${SBATCH_EXTRA[@]}" \
          --export=ALL,INPUT_DIR="${dir}",\
METHODS="${METHODS}",\
BATCH_KEY="${BATCH_KEY}",\
BATCH_SIZE="${BATCH_SIZE}",\
TRAIN_OR_TEST="${TRAIN_OR_TEST}" \
          "$SCRIPT"
    else
        # — locally, just run the script (it will detect LOCAL MODE) —
        echo "[LOCAL] Running all $n chunks for '$label' in parallel"
        INPUT_DIR="$dir" METHODS="$METHODS" MAX_PROCS="$MAX_PROCS" \
BATCH_KEY="$BATCH_KEY" BATCH_SIZE="$BATCH_SIZE" \
TRAIN_OR_TEST="$TRAIN_OR_TEST" \
bash "$SCRIPT"
    fi
}

if [[ "$TRAIN_OR_TEST" == "test" ]]; then
    DATA_DIR="${DATA_BASE_DIR}/test/${DATANAME}/all"
    submit_array test "$DATA_DIR"
else
    TRAIN_DIR="${DATA_BASE_DIR}/train/${DATANAME}/train"
    VAL_DIR="${DATA_BASE_DIR}/train/${DATANAME}/val"
    submit_array train "$TRAIN_DIR"
    submit_array val   "$VAL_DIR"
fi
