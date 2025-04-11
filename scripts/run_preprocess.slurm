#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --output=pp_adata.out
#SBATCH --error=pp_adata.err
#SBATCH --mem=30G               # Request memory
#SBATCH --time=24:00:00         # Max job time

INPUT_FILE="data/RNA/raw/train/geo_70k.h5ad"
OUTPUT_DIR="data/RNA/processed/train"
SPLIT_DATASET=true
CATEGORY_THRESHOLD=10
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.
echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
python3 scripts/preprocess_adata.py \
    --config-name preprocess_adata_geo \
    ++input_file=$INPUT_FILE \
    ++output_dir=$OUTPUT_DIR \
    ++split_dataset=$SPLIT_DATASET \
    ++category_threshold=$CATEGORY_THRESHOLD \
