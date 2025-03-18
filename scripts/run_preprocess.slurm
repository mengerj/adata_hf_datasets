#!/bin/bash
#SBATCH --job-name=pp_adata
#SBATCH --output=pp_adata.out
#SBATCH --error=pp_adata.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=60G               # Request memory
#SBATCH --time=8:00:00         # Max job time

FILES='["data/RNA/raw/train/cellxgene_pseudo_bulk_35k.h5ad"]'
OUTPUT_DIR="data/RNA/processed/train"
BATCH_KEYS='["dataset_title"]'
METHODS='["hvg","pca","scvi","geneformer"]'
TRAIN_SPLIT=0.9
SPLIT_DATASET=true
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.
echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/preprocess_adata.py \
    ++files=$FILES \
    ++output_dir=$OUTPUT_DIR \
    ++batch_keys=$BATCH_KEYS \
    ++methods=$METHODS \
    ++split_dataset=$SPLIT_DATASET \
    ++train_split=$TRAIN_SPLIT
