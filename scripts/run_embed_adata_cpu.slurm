#!/bin/bash
#SBATCH --job-name=cpu_initial_emb
#SBATCH --output=cpu_initial_emb.out
#SBATCH --error=cpu_initial_emb.err
#SBATCH --mem=60G
#SBATCH --time=12:00:00   # Adjust if needed

batch_key="dataset_title"
batch_size=32

INPUT_FILES='["data/RNA/processed/train/cellxgene_pseudo_bulk_35k/all_train.h5ad","data/RNA/processed/train/cellxgene_pseudo_bulk_35k/all_val.h5ad"]'

METHODS=("hvg" "pca" "scvi_fm")

source .venv/bin/activate
echo "Environment activated."

for METHOD in "${METHODS[@]}"
do
    echo "Running embedder method: $METHOD"

    python3 scripts/embed_adata.py \
        ++method=$METHOD \
        ++batch_key=$batch_key \
        ++batch_size=$batch_size \
        ++input_files="$INPUT_FILES"

    echo "Completed embedder method: $METHOD"
done
