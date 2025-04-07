#!/bin/bash
#SBATCH --job-name=geo_700k
#SBATCH --output=pp_adata700k.out
#SBATCH --error=pp_adata700k.err
#SBATCH --mem=400G               # Request memory
#SBATCH --time=48:00:00         # Max job time

INPUT_FILE="data/RNA/raw/train/geo_700k.h5ad"
OUTPUT_DIR="data/RNA/processed/train"
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
    ++input_file=$INPUT_FILE \
    ++output_dir=$OUTPUT_DIR \
    ++split_dataset=$SPLIT_DATASET \
