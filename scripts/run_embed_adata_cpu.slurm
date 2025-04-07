#!/bin/bash
#SBATCH --job-name=cpu_initial_emb
#SBATCH --output=cpu_initial_emb.out
#SBATCH --error=cpu_initial_emb.err
#SBATCH --mem=60G               # Request memory
#SBATCH --time=1:00:00         # Max job time

INPUT_FILES='["data/RNA/processed/train/cellxgene_pseudo_bulk_3_5k/train.h5ad",
             "data/RNA/processed/train/cellxgene_pseudo_bulk_3_5k/val.h5ad"]'
METHOD="pca"
batch_key="dataset_title" #only relevant for scvi
batch_size=32 #batch sized used by geneformer
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.
echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
#python3 scripts/verify_gpu.py
# Now run your Python script
python3 scripts/embed_adata.py \
    ++method=$METHOD \
    ++batch_key=$batch_key \
    ++batch_size=$batch_size \
    ++input_files=$INPUT_FILES \
