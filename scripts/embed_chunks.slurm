#!/usr/bin/env bash
#SBATCH --job-name=embed               # overridden by sbatch --job-name
#SBATCH --output=logs/embed/%x_%A_%a.out
#SBATCH --error =logs/embed/%x_%A_%a.err
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# === Required via sbatch --export ===
#   INPUT_DIR   e.g. data/.../train_chunks
#   METHODS     e.g. "hvg pca scvi_fm"
#   BATCH_KEY   e.g. dataset_title
#   BATCH_SIZE  e.g. 32 #used for geneformer only

source .venv/bin/activate

# Gather the Nth file for this task
files=( "${INPUT_DIR}"/*.h5ad )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"

echo "[$SLURM_ARRAY_JOB_ID:$SLURM_ARRAY_TASK_ID] Processing $this_file"

# Base output dir for this task
BASE_OUT="outputs/$(date +%Y-%m-%d)/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE_OUT"

# Turn the space‐separated METHODS string into an array
IFS=' ' read -r -a method_list <<< "$METHODS"

for METHOD in "${method_list[@]}"; do
    # Per‐method subdirectory
    OUTDIR="${BASE_OUT}/${METHOD}"
    mkdir -p "$OUTDIR"
    echo " → Running embedder method: $METHOD"

    python3 scripts/embed_adata.py \
        ++method="$METHOD" \
        ++batch_key="$BATCH_KEY" \
        ++batch_size="$BATCH_SIZE" \
        ++input_files="[\"$this_file\"]" \
        ++hydra.run.dir="$OUTDIR"

    echo " ← Completed method: $METHOD"
done

echo "All methods done for $this_file"
