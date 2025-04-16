#!/bin/bash
#SBATCH --job-name=create_ds
#SBATCH --output=logs/create_ds/create_ds_%j.out
#SBATCH --error=logs/create_ds/create_ds_%j.err
#SBATCH --time=12:00:00

###
# 1. Set default overrides:
#    These match the current YAML defaults in conf/create_dataset.yaml
###
SPLIT_DATASET=true

PROCESSED_PATHS_TRAIN='["data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k/joined/all_train.h5ad"]'
PROCESSED_PATHS_VAL='["data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_35k/joined/all_val.h5ad"]'
PROCESSED_PATHS_ALL='["data/RNA/processed/test/tabula_sapiens_bone_marrow/all.h5ad"]' #only used if SPLIT_DATASET=false

CAPTION_KEY="natural_language_annotation"
DATASET_TYPE="multiplets"

###
# 2. Activate environment, etc.
###
echo "Starting job"
source cpu_venv/bin/activate
echo "venv activated"

###
# 3. Run your Hydra-powered script with overrides
#    Note: The syntax create_dataset.split_dataset=... etc. assumes
#          your config group or config file is "create_dataset".
###
python3 scripts/create_dataset.py \
    ++split_dataset=$SPLIT_DATASET \
    ++processed_paths_train=$PROCESSED_PATHS_TRAIN \
    ++processed_paths_val=$PROCESSED_PATHS_VAL \
    ++processed_paths_all=$PROCESSED_PATHS_ALL \
    ++caption_key=$CAPTION_KEY \
    ++dataset_type=$DATASET_TYPE
