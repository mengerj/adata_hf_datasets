#!/bin/bash
#SBATCH --job-name=initial_emb
#SBATCH --output=initial_emb.out
#SBATCH --error=initial_emb.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --mem=60G               # Request memory
#SBATCH --time=1:00:00         # Max job time

INPUT_FILES='["data/RNA/processed/train/cellxgene_pseudo_bulk_3_5k/train.h5ad",
             "data/RNA/processed/train/cellxgene_pseudo_bulk_3_5k/val.h5ad"]'
METHOD="geneformer"
batch_key="dataset_title" #only relevant for scvi
batch_size=32 #batch sized used by geneformer
# If false, the script will NOT split into train/val but instead
# output a single 'all.h5ad' file. This can be used for test data or
# "single" data scenario.
echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
echo "==== Debugging Python Environment ===="
which python
which python3
which pip
python --version
python -c "import sys; print('sys.path:', sys.path)"
python -c "import transformers; print('Transformers path:', transformers.__file__)"
python -c "import transformers; print(transformers.__version__)"
python -c "from transformers import AdamW; print('AdamW imported successfully')"
echo "======================================"
#python3 scripts/verify_gpu.py
# Now run your Python script
python3 scripts/embed_adata.py \
    ++input_file=$INPUT_FILE \
    ++output_dir=$OUTPUT_DIR \
    ++split_dataset=$SPLIT_DATASET \
