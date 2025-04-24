#!/bin/bash
#SBATCH --job-name=initial_emb
#SBATCH --output=logs/embed/gpu_emb_%j.out
#SBATCH --error=logs/embed/gpu_emb_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --mem=60G
#SBATCH --time=12:00:00

METHOD="geneformer"
batch_key="dataset_title"
batch_size=32
# _%a is the array index
#INPUT_FILES=(
#    "data/RNA/processed/train/cellxgene_pseudo_bulk_350k/all_train.h5ad"
#    "data/RNA/processed/train/cellxgene_pseudo_bulk_350k/all_val.h5ad"
#)
input_files='["data/RNA/processed/train/cellxgene_pseudo_bulk_35k/train.h5ad","data/RNA/processed/train/cellxgene_pseudo_bulk_35k/val.h5ad"]'
#input_files='["data/RNA/processed/test/immgen/all_all.h5ad"]'

# Create a custom output directory name with date and SLURM job ID
CUSTOM_DIR="outputs/$(date +%Y-%m-%d)/${SLURM_JOB_ID}"
# Activate environment
source .venv/bin/activate

#echo "Processing $input_file, outputting to $output_dir"

python3 scripts/embed_adata.py \
    ++method=$METHOD \
    ++batch_key=$batch_key \
    ++batch_size=$batch_size \
    ++input_files=$input_files \
    ++hydra.run.dir=$CUSTOM_DIR \
#    ++output_dir="$output_dir"
