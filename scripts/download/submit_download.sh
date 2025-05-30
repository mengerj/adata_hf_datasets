#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --account=menger     # Update this to your account
#SBATCH --time=6:00:00       # Maximum runtime (6 hours)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G            # Memory allocation
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Started at: $(date)"
echo "===================="

# Configuration variables with defaults
# Override these when submitting the job
URL=${URL:-"https://datasets.cellxgene.cziscience.com/f886c7d9-1392-4f09-9e10-31b953afa2da.h5ad"}
OUTPUT_PATH=${OUTPUT_PATH:-"data/downloaded_dataset.h5ad"}
SUBSET_SIZE=${SUBSET_SIZE:-""}
SEED=${SEED:-42}
STRATIFY_BY=${STRATIFY_BY:-""}
PRESERVE_PROPORTIONS=${PRESERVE_PROPORTIONS:-false}
TEMP_DIR=${TEMP_DIR:-"/tmp/download_$SLURM_JOB_ID"}
VALIDATE=${VALIDATE:-true}
KEEP_TEMP=${KEEP_TEMP:-false}

# Print configuration
echo "Configuration:"
echo "  URL: $URL"
echo "  Output path: $OUTPUT_PATH"
echo "  Subset size: $SUBSET_SIZE"
echo "  Random seed: $SEED"
echo "  Stratify by: $STRATIFY_BY"
echo "  Preserve proportions: $PRESERVE_PROPORTIONS"
echo "  Temp directory: $TEMP_DIR"
echo "  Validate: $VALIDATE"
echo "  Keep temp: $KEEP_TEMP"
echo "===================="

# Load required modules (adjust based on your cluster)
# module load python/3.11
# module load gcc/9.3.0

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Create temporary directory
mkdir -p "$TEMP_DIR"

# Build the command
CMD="python scripts/download/download_dataset.py --url \"$URL\" --output \"$OUTPUT_PATH\" --seed $SEED"

# Add optional arguments
if [ -n "$SUBSET_SIZE" ]; then
    CMD="$CMD --subset-size $SUBSET_SIZE"
fi

if [ -n "$STRATIFY_BY" ]; then
    CMD="$CMD --stratify-by \"$STRATIFY_BY\""
fi

if [ "$PRESERVE_PROPORTIONS" = "true" ]; then
    CMD="$CMD --preserve-proportions"
fi

if [ -n "$TEMP_DIR" ]; then
    CMD="$CMD --temp-dir \"$TEMP_DIR\""
fi

if [ "$VALIDATE" = "true" ]; then
    CMD="$CMD --validate"
fi

if [ "$KEEP_TEMP" = "true" ]; then
    CMD="$CMD --keep-temp"
fi

echo "Executing command:"
echo "$CMD"
echo "===================="

# Execute the download
eval $CMD
EXIT_CODE=$?

# Cleanup
if [ "$KEEP_TEMP" != "true" ] && [ -d "$TEMP_DIR" ]; then
    echo "Cleaning up temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
fi

echo "===================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

# Print disk usage of output file if successful
if [ $EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_PATH" ]; then
    echo "Output file size: $(du -h "$OUTPUT_PATH" | cut -f1)"
    echo "Output file location: $(realpath "$OUTPUT_PATH")"
fi

exit $EXIT_CODE
