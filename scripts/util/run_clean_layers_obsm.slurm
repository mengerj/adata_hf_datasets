#!/bin/bash
#SBATCH --job-name=clean_layers_obsm
#SBATCH --output=clean_layers_obsm_%j.out
#SBATCH --error=clean_layers_obsm_%j.err
#SBATCH --time=8:00:00         # Max job time - adjust as needed
#SBATCH --mem=32G              # Memory request - adjust based on file sizes
#SBATCH --cpus-per-task=4      # Number of CPU cores

# Usage: sbatch run_clean_layers_obsm.slurm [input_path] [output_path] [layers] [obsm_keys] [hvg] [n_top_genes] [batch_key] [min_genes_per_cell] [min_cells_per_gene] [dry_run]
# Examples:
#   sbatch run_clean_layers_obsm.slurm data/ "" "layer1,layer2" "X_pca,X_umap" false 2000 "" 200 3 false
#   sbatch run_clean_layers_obsm.slurm file.h5ad processed/ "counts" "X_pca" true 3000 batch 100 5 false
#   sbatch run_clean_layers_obsm.slurm data/ "" "" "" true 2000 "" 200 3 true

echo "=========================================="
echo "Starting clean_layers_obsm job"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=========================================="

# Parse arguments
INPUT_PATH=${1:-""}
OUTPUT_PATH=${2:-""}
LAYERS=${3:-""}
OBSM_KEYS=${4:-""}
HVG=${5:-"false"}
N_TOP_GENES=${6:-"2000"}
BATCH_KEY=${7:-""}
MIN_GENES_PER_CELL=${8:-"200"}
MIN_CELLS_PER_GENE=${9:-"3"}
DRY_RUN=${10:-"false"}

# Check if input path is provided
if [ -z "$INPUT_PATH" ]; then
    echo "Error: Input path is required"
    echo "Usage: sbatch run_clean_layers_obsm.slurm [input_path] [output_path] [layers] [obsm_keys] [hvg] [n_top_genes] [batch_key] [min_genes_per_cell] [min_cells_per_gene] [dry_run]"
    exit 1
fi

# Check if at least one operation is specified
if [ -z "$LAYERS" ] && [ -z "$OBSM_KEYS" ] && [ "$HVG" != "true" ]; then
    echo "Error: At least one of layers, obsm_keys, or HVG selection must be specified"
    echo "Usage: sbatch run_clean_layers_obsm.slurm [input_path] [output_path] [layers] [obsm_keys] [hvg] [n_top_genes] [batch_key] [min_genes_per_cell] [min_cells_per_gene] [dry_run]"
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
echo "Dry run: $DRY_RUN"
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

if [ "$DRY_RUN" = "true" ]; then
    CMD="$CMD --dry-run"
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
