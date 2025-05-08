#!/usr/bin/env bash
set -euo pipefail


# === User‐configurable section ===
MODE="cpu"         # "cpu" or "gpu"
GPU_COUNT="1"      # how many GPUs if MODE=gpu
DATANAME="geo_700k"
BATCH_KEY="study"
BATCH_SIZE=32
METHODS="hvg pca scvi_fm"  #"geneformer"  # space‐separated list
SCRIPT="scripts/embed_chunks_parallel.slurm"
TRAIN_OR_TEST="train"
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
    n=$(ls -1 "${dir}"/*.h5ad | wc -l)
    if (( n == 0 )); then
        echo "No chunks in $dir → skipping"
        return
    fi

    if command -v sbatch &>/dev/null; then
        # — under SLURM, submit an array job —
        sbatch \
          --job-name="embed_${label}_${JOB_SUFFIX}" \
          --array=0-$((n-1)) \
          "${SBATCH_EXTRA[@]}" \
          --export=INPUT_DIR="${dir}",\
METHODS="${METHODS}",\
BATCH_KEY="${BATCH_KEY}",\
BATCH_SIZE="${BATCH_SIZE}",\
TRAIN_OR_TEST="${TRAIN_OR_TEST}" \
          "$SCRIPT"
    else
        # — locally, just run the script (it will detect LOCAL MODE) —
        echo "[LOCAL] Running all $n chunks for '$label' in parallel"
        INPUT_DIR="$dir" METHODS="$METHODS" \
BATCH_KEY="$BATCH_KEY" BATCH_SIZE="$BATCH_SIZE" \
TRAIN_OR_TEST="$TRAIN_OR_TEST" \
bash "$SCRIPT"
    fi
}

if [[ "$TRAIN_OR_TEST" == "test" ]]; then
    DATA_DIR="data/RNA/processed/test/${DATANAME}/all"
    submit_array test "$DATA_DIR"
else
    TRAIN_DIR="data/RNA/processed/train/${DATANAME}/train"
    VAL_DIR="data/RNA/processed/train/${DATANAME}/val"
    submit_array train "$TRAIN_DIR"
    submit_array val   "$VAL_DIR"
fi
