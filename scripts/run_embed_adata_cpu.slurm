#!/bin/bash
#SBATCH --job-name=cpu_initial_emb
#SBATCH --output=logs/embed/cpu_emb_%j.out
#SBATCH --error=logs/embed/cpu_emb_%j.err
#SBATCH --mem=400G
#SBATCH --time=24:00:00   # Adjust if needed

batch_key="sample_id"
batch_size=32

#INPUT_FILES='["data/RNA/processed/train/cellxgene_pseudo_bulk_35k/all_train.h5ad","data/RNA/processed/train/cellxgene_pseudo_bulk_35k/all_val.h5ad"]'
INPUT_FILES='["data/RNA/processed/test/tabula_sapiens_bone_marrow/all_all.h5ad"]'

METHODS=("geneformer")

source .venv/bin/activate
echo "Environment activated."

for METHOD in "${METHODS[@]}"
do
    echo "Running embedder method: $METHOD"

    python3 scripts/embed_adata.py \
        ++method=$METHOD \
        ++batch_key=$batch_key \
        ++batch_size=$batch_size \
        ++input_files=$INPUT_FILES

    echo "Completed embedder method: $METHOD"
done
