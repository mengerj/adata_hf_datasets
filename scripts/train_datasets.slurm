#!/bin/bash
#SBATCH --job-name=train_dataset_70K
#SBATCH --output=train_dataset_70K.out
#SBATCH --error=train_dataset_70K.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # Request amount of GPUs
#SBATCH --mem=200G               # Request amount of memory
#SBATCH --time=12:00:00         # Max job time of 1 hour

echo "Starting job"
# Source the setup script to ensure the environment is ready
source .venv/bin/activate
echo "venv activated"
# Now run your Python script
#python3 scripts/train_datasets_create.py --files data/RNA/raw/train/cellxgene_pseudo_bulk_350k.h5ad --batch_keys assay --caption_key natural_language_annotation
python3 scripts/train_datasets_create.py --files data/RNA/raw/train/geo_70k.h5ad --batch_keys study --caption_key natural_language_annotation
