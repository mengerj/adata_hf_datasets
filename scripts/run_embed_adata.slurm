#!/bin/bash
#SBATCH --job-name=initial_emb
#SBATCH --output=initial_emb_%a.out
#SBATCH --error=initial_emb_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Request 1 GPU
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-1

METHOD="geneformer"
batch_key="dataset_title"
batch_size=32

INPUT_FILES=(
    "data/RNA/processed/train/cellxgene_pseudo_bulk_350k/train.h5ad"
    "data/RNA/processed/train/cellxgene_pseudo_bulk_350k/val.h5ad"
)

input_file=${INPUT_FILES[$SLURM_ARRAY_TASK_ID]}
# Create temporary directory for this job
#tmp_dir=$(mktemp -d /tmp/emb_job_${SLURM_ARRAY_TASK_ID}_XXXX)

# Copy input file preserving its original name
#cp "$input_file" "$tmp_dir/$(basename "$input_file")"

#tmp_input_file="$tmp_dir/$(basename "$input_file")"

# Determine output directory based on original input file path
#output_dir=$(dirname "$input_file" | sed 's/processed/processed_with_emb/')

# Activate environment
source .venv/bin/activate

echo "Processing $input_file, outputting to $output_dir"

python3 scripts/embed_adata.py \
    ++method=$METHOD \
    ++batch_key=$batch_key \
    ++batch_size=$batch_size \
    ++input_files='["'"$input_file"'"]'
#    ++output_dir="$output_dir"

# Cleanup temporary input file
rm "$tmp_input_file"
