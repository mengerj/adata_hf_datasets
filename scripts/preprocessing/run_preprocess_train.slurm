#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --output=logs/pp/pp_%j.out
#SBATCH --error=logs/pp/pp_%j.err
#SBATCH --mem=35G               # Request memory
#SBATCH --time=24:00:00         # Max job time

INPUT_FILE="data/RNA/raw/train/geo_70k.h5ad"
OUTPUT_DIR="data/RNA/processed/train"
SPLIT_DATASET=true
CHUNK_SIZE=2000
BATCH_KEY=study
COUNT_LAYER_KEY=counts
CATEGORY_THRESHOLD=5
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.

# SRA options - set to true to avoid connection issues
SKIP_SRA_FETCH=false
SRA_CONTINUE_ON_FAIL=false
SRA_MAX_RETRIES=5

# Create a custom output directory name with date and SLURM job ID
CUSTOM_DIR="outputs/$(date +%Y-%m-%d)/${SLURM_JOB_ID}"

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script with the custom output directory
python3 scripts/preprocess.py \
    --config-name preprocess_adata_geo \
    ++input_file=$INPUT_FILE \
    ++chunk_size=$CHUNK_SIZE \
    ++batch_key=$BATCH_KEY \
    ++count_layer_key=$COUNT_LAYER_KEY \
    ++output_dir=$OUTPUT_DIR \
    ++split_dataset=$SPLIT_DATASET \
    ++category_threshold=$CATEGORY_THRESHOLD \
    ++skip_sra_fetch=$SKIP_SRA_FETCH \
    ++sra_continue_on_fail=$SRA_CONTINUE_ON_FAIL \
    ++sra_max_retries=$SRA_MAX_RETRIES \
    ++hydra.run.dir=$CUSTOM_DIR
