#!/bin/bash

# Helper script for submitting download jobs with different configurations
# This script provides easy presets for common download scenarios

set -e  # Exit on any error

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/submit_download.sh"

# Check if Slurm script exists
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: Slurm script not found at $SLURM_SCRIPT"
    exit 1
fi

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [PRESET] [OPTIONS]

PRESETS:
    cellxgene       Download from CellxGene (default example dataset)
    immune-atlas    Download full immune health atlas
    immune-subset   Download immune health atlas with 10k subset
    custom          Custom download (requires --url)

GENERAL OPTIONS:
    --url URL                   URL to download from (required for custom)
    --output PATH               Output file path
    --subset-size N             Create random subset of N observations
    --seed N                    Random seed (default: 42)
    --stratify-by COLUMN        Column to stratify subset by
    --preserve-proportions      Preserve proportions when stratifying
    --temp-dir DIR              Temporary directory
    --validate                  Validate downloaded file
    --keep-temp                 Keep temporary files
    --dry-run                   Show command that would be executed

SLURM OPTIONS:
    --time TIME                 Job time limit (default: 6:00:00)
    --mem MEMORY                Memory allocation (default: 32G)
    --cpus N                    Number of CPUs (default: 4)
    --account ACCOUNT          Slurm account (default: menger)

EXAMPLES:
    # Download CellxGene dataset
    $0 cellxgene

    # Download immune atlas with 5k subset
    $0 immune-subset --subset-size 5000

    # Custom download with subsetting
    $0 custom --url "https://example.com/data.h5ad" --output "my_data.h5ad" --subset-size 1000

    # Download with specific Slurm resources
    $0 immune-atlas --mem 64G --time 12:00:00 --cpus 8

EOF
}

# Parse arguments
PRESET=""
URL=""
OUTPUT_PATH=""
SUBSET_SIZE=""
SEED="42"
STRATIFY_BY=""
PRESERVE_PROPORTIONS="false"
TEMP_DIR=""
VALIDATE="true"
KEEP_TEMP="false"
DRY_RUN="false"

# Slurm options
TIME="6:00:00"
MEMORY="32G"
CPUS="4"
ACCOUNT="menger"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        cellxgene|immune-atlas|immune-subset|custom)
            PRESET="$1"
            shift
            ;;
        --url)
            URL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --subset-size)
            SUBSET_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --stratify-by)
            STRATIFY_BY="$2"
            shift 2
            ;;
        --preserve-proportions)
            PRESERVE_PROPORTIONS="true"
            shift
            ;;
        --temp-dir)
            TEMP_DIR="$2"
            shift 2
            ;;
        --validate)
            VALIDATE="true"
            shift
            ;;
        --no-validate)
            VALIDATE="false"
            shift
            ;;
        --keep-temp)
            KEEP_TEMP="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set defaults based on preset
case "$PRESET" in
    "cellxgene")
        URL="https://datasets.cellxgene.cziscience.com/f886c7d9-1392-4f09-9e10-31b953afa2da.h5ad"
        OUTPUT_PATH="${OUTPUT_PATH:-data/cellxgene_dataset.h5ad}"
        ;;
    "immune-atlas")
        URL="https://allenimmunology.org/public/publication/download/84792154-cdfb-42d0-8e42-39e210e980b4/filesets/3a6afb68-0379-4afa-838a-c0b7f222b517/immune_health_atlas_full.h5ad"
        OUTPUT_PATH="${OUTPUT_PATH:-data/immune_health_atlas_full.h5ad}"
        MEMORY="64G"  # Larger file needs more memory
        TIME="12:00:00"
        ;;
    "immune-subset")
        URL="https://allenimmunology.org/public/publication/download/84792154-cdfb-42d0-8e42-39e210e980b4/filesets/3a6afb68-0379-4afa-838a-c0b7f222b517/immune_health_atlas_full.h5ad"
        OUTPUT_PATH="${OUTPUT_PATH:-data/immune_health_atlas_subset.h5ad}"
        SUBSET_SIZE="${SUBSET_SIZE:-10000}"
        MEMORY="64G"  # Need memory to load full file first
        ;;
    "custom")
        if [ -z "$URL" ]; then
            echo "Error: --url is required for custom preset"
            exit 1
        fi
        OUTPUT_PATH="${OUTPUT_PATH:-data/custom_dataset.h5ad}"
        ;;
    "")
        echo "Error: Please specify a preset or use --help for usage information"
        exit 1
        ;;
    *)
        echo "Error: Unknown preset '$PRESET'"
        exit 1
        ;;
esac

# Validate required parameters
if [ -z "$URL" ]; then
    echo "Error: URL not specified"
    exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: Output path not specified"
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Build the sbatch command
SBATCH_CMD="sbatch"
SBATCH_CMD="$SBATCH_CMD --time=$TIME"
SBATCH_CMD="$SBATCH_CMD --mem=$MEMORY"
SBATCH_CMD="$SBATCH_CMD --cpus-per-task=$CPUS"
SBATCH_CMD="$SBATCH_CMD --account=$ACCOUNT"

# Set environment variables for the Slurm script
export URL="$URL"
export OUTPUT_PATH="$OUTPUT_PATH"
export SUBSET_SIZE="$SUBSET_SIZE"
export SEED="$SEED"
export STRATIFY_BY="$STRATIFY_BY"
export PRESERVE_PROPORTIONS="$PRESERVE_PROPORTIONS"
export TEMP_DIR="$TEMP_DIR"
export VALIDATE="$VALIDATE"
export KEEP_TEMP="$KEEP_TEMP"

# Print configuration
echo "=== Download Job Configuration ==="
echo "Preset: $PRESET"
echo "URL: $URL"
echo "Output: $OUTPUT_PATH"
echo "Subset size: ${SUBSET_SIZE:-"No subsetting"}"
echo "Random seed: $SEED"
echo "Stratify by: ${STRATIFY_BY:-"None"}"
echo "Preserve proportions: $PRESERVE_PROPORTIONS"
echo "Validate: $VALIDATE"
echo "Keep temp: $KEEP_TEMP"
echo ""
echo "=== Slurm Configuration ==="
echo "Time limit: $TIME"
echo "Memory: $MEMORY"
echo "CPUs: $CPUS"
echo "Account: $ACCOUNT"
echo ""

# Show the command that will be executed
FULL_CMD="$SBATCH_CMD $SLURM_SCRIPT"
echo "Command to execute:"
echo "$FULL_CMD"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "Dry run - not submitting job"
    exit 0
fi

# Submit the job
echo "Submitting job..."
JOB_ID=$($FULL_CMD | grep -o '[0-9]\+$')

if [ $? -eq 0 ]; then
    echo "Job submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo "Monitor with: squeue -j $JOB_ID"
    echo "Cancel with: scancel $JOB_ID"
    echo "View output: tail -f logs/download_${JOB_ID}.out"
    echo "View errors: tail -f logs/download_${JOB_ID}.err"
else
    echo "Failed to submit job"
    exit 1
fi
