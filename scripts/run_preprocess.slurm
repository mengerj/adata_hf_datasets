#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --output=logs/pp/pp_%j.out
#SBATCH --error=logs/pp/pp_%j.err
#SBATCH --mem=50G               # Request memory
#SBATCH --time=12:00:00         # Max job time

INPUT_FILE="data/RNA/raw/test/bowel_disease.h5ad"
OUTPUT_DIR="data/RNA/processed/test/"
SPLIT_DATASET=false
CHUNK_SIZE=70000
BATCH_KEY=sample_id
CATEGORY_THRESHOLD=3
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.

# Create a custom output directory name with date and SLURM job ID
CUSTOM_DIR="outputs/$(date +%Y-%m-%d)/${SLURM_JOB_ID}"

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script with the custom output directory
python3 scripts/preprocess.py \
    --config-name preprocess_adata_test \
    ++input_file=$INPUT_FILE \
    ++chunk_size=$CHUNK_SIZE \
    ++batch_key=$BATCH_KEY \
    ++output_dir=$OUTPUT_DIR \
    ++split_dataset=$SPLIT_DATASET \
    ++category_threshold=$CATEGORY_THRESHOLD \
    ++hydra.run.dir=$CUSTOM_DIR
