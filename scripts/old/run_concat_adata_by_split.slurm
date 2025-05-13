#!/bin/bash
#SBATCH --job-name=concat_h5ad
#SBATCH --output=logs/concat/concat_h5ad_%j.out
#SBATCH --error=logs/concat/concat_h5ad_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=320G
#SBATCH --cpus-per-task=4

source .venv/bin/activate

python scripts/concat_adata_by_split.py \
    --data-dir data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k \
    --output-dir data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k/joined
