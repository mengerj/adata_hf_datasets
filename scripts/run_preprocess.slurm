#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --output=logs/pp/pp_%j.out
#SBATCH --error=logs/pp/pp_%j.err
#SBATCH --mem=100G               # Request memory
#SBATCH --time=12:00:00         # Max job time

INPUT_FILE="data/RNA/raw/train/cellxgene_pseudo_bulk_350k.h5ad"
OUTPUT_DIR="data/RNA/processed/train/"
SPLIT_DATASET=true
CHUNK_SIZE=70000
BATCH_KEY=dataset_title #still need to change other fields in the config
CATEGORY_THRESHOLD=5
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.
echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/preprocess.py \
    --config-name preprocess_adata_cellxgene \
    ++input_file=$INPUT_FILE \
    ++chunk_size=$CHUNK_SIZE \
    ++batch_key=$BATCH_KEY \
    ++output_dir=$OUTPUT_DIR \
    ++split_dataset=$SPLIT_DATASET \
    ++category_threshold=$CATEGORY_THRESHOLD \
