#!/bin/bash
#SBATCH --job-name=clean_layers_obsm
#SBATCH --output=clean_layers_obsm_%j.out
#SBATCH --error=clean_layers_obsm_%j.err
#SBATCH --time=8:00:00         # Max job time - adjust as needed
#SBATCH --mem=32G              # Memory request - adjust based on file sizes
#SBATCH --cpus-per-task=4      # Number of CPU cores

# Usage: sbatch run_clean_layers_obsm.slurm [input_path] [output_path] [layers] [obsm_keys] [hvg] [n_top_genes] [batch_key] [min_genes_per_cell] [min_cells_per_gene] [rename_obs_from] [rename_obs_to]
# Examples:
#   sbatch run_clean_layers_obsm.slurm data/ "" "layer1,layer2" "X_pca,X_umap" false 2000 "" 200 3 "" ""
#   sbatch run_clean_layers_obsm.slurm file.h5ad processed/ "counts" "X_pca" true 3000 batch 100 5 "" ""
#   sbatch run_clean_layers_obsm.slurm file.h5ad "" "" "" false 2000 "" 200 3 "old_col1,old_col2" "new_col1,new_col2"

echo "=========================================="
echo "Starting clean_layers_obsm job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Parse arguments
INPUT_PATH=${1:-"/scratch/global/menger/data/RNA/raw/train/cellxgene_pseudo_bulk_350k.h5ad"}
OUTPUT_PATH=${2:-"/scratch/global/scIMBI/krissmer-menger/data/RNA/raw/train/cellxgene_pseudo_bulk_350k_cleaned.h5ad"}
LAYERS=${3:-"replicate_1,replicate_2,replicate_3,replicate_4,replicate_5"}
OBSM_KEYS=${4:-"natural_language_annotation_replicates"}
HVG=${5:-"true"}
N_TOP_GENES=${6:-"2000"}
BATCH_KEY=${7:-"dataset_title"}
MIN_GENES_PER_CELL=${8:-"200"}
MIN_CELLS_PER_GENE=${9:-"3"}
RENAME_OBS_FROM=${10:-"cell_type"}
RENAME_OBS_TO=${11:-"celltype"}

# Check if input path is provided
if [ -z "$INPUT_PATH" ]; then
    echo "Error: Input path is required"
    echo "Usage: sbatch run_clean_layers_obsm.slurm [input_path] [output_path] [layers] [obsm_keys] [hvg] [n_top_genes] [batch_key] [min_genes_per_cell] [min_cells_per_gene] [rename_obs_from] [rename_obs_to]"
    exit 1
fi

# Check if at least one operation is specified
if [ -z "$LAYERS" ] && [ -z "$OBSM_KEYS" ] && [ "$HVG" != "true" ] && [ -z "$RENAME_OBS_FROM" ]; then
    echo "Error: At least one of layers, obsm_keys, HVG selection, or column renaming must be specified"
    echo "Usage: sbatch run_clean_layers_obsm.slurm [input_path] [output_path] [layers] [obsm_keys] [hvg] [n_top_genes] [batch_key] [min_genes_per_cell] [min_cells_per_gene] [rename_obs_from] [rename_obs_to]"
    exit 1
fi

# Validate rename arguments
if [ ! -z "$RENAME_OBS_FROM" ] && [ -z "$RENAME_OBS_TO" ]; then
    echo "Error: rename_obs_to must be provided when rename_obs_from is used"
    exit 1
fi

if [ ! -z "$RENAME_OBS_TO" ] && [ -z "$RENAME_OBS_FROM" ]; then
    echo "Error: rename_obs_from must be provided when rename_obs_to is used"
    exit 1
fi

echo "Input path: $INPUT_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Layers to remove: $LAYERS"
echo "Obsm keys to remove: $OBSM_KEYS"
echo "HVG selection: $HVG"
echo "N top genes: $N_TOP_GENES"
echo "Batch key: $BATCH_KEY"
echo "Min genes per cell: $MIN_GENES_PER_CELL"
echo "Min cells per gene: $MIN_CELLS_PER_GENE"
echo "Rename obs from: $RENAME_OBS_FROM"
echo "Rename obs to: $RENAME_OBS_TO"
echo "=========================================="

# Source the setup script to ensure the environment is ready
source cpu_venv/bin/activate
echo "Virtual environment activated"

# Build the command
CMD="python3 scripts/util/clean_layers_obsm.py --input $INPUT_PATH"

if [ ! -z "$OUTPUT_PATH" ]; then
    CMD="$CMD --output $OUTPUT_PATH"
fi

if [ ! -z "$LAYERS" ]; then
    CMD="$CMD --layers $LAYERS"
fi

if [ ! -z "$OBSM_KEYS" ]; then
    CMD="$CMD --obsm $OBSM_KEYS"
fi

if [ "$HVG" = "true" ]; then
    CMD="$CMD --hvg"
fi

if [ ! -z "$N_TOP_GENES" ] && [ "$N_TOP_GENES" != "2000" ]; then
    CMD="$CMD --n-top-genes $N_TOP_GENES"
fi

if [ ! -z "$BATCH_KEY" ]; then
    CMD="$CMD --batch-key $BATCH_KEY"
fi

if [ ! -z "$MIN_GENES_PER_CELL" ]; then
    CMD="$CMD --min-genes-per-cell $MIN_GENES_PER_CELL"
fi

if [ ! -z "$MIN_CELLS_PER_GENE" ]; then
    CMD="$CMD --min-cells-per-gene $MIN_CELLS_PER_GENE"
fi

if [ ! -z "$RENAME_OBS_FROM" ]; then
    CMD="$CMD --rename-obs-from $RENAME_OBS_FROM"
fi

if [ ! -z "$RENAME_OBS_TO" ]; then
    CMD="$CMD --rename-obs-to $RENAME_OBS_TO"
fi

echo "Running command: $CMD"
echo "=========================================="

# Run the Python script
eval $CMD

EXIT_CODE=$?

echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
