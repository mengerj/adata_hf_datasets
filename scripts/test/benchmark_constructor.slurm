#!/bin/bash
#SBATCH --job-name=bench_constructor
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Optional: provide VENV_PATH env var to activate a specific venv
if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  source "${VENV_PATH}/bin/activate"
elif [[ -f "./.venv/bin/activate" ]]; then
  source ./.venv/bin/activate
elif [[ -f "./venv/bin/activate" ]]; then
  source ./venv/bin/activate
else
  echo "Warning: no virtualenv activated. Set VENV_PATH to your venv if needed." >&2
fi

ZARR_PATH_OVERRIDE="${1:-}"

DEFAULT_ZARR="/scratch/global/menger/data/RNA/processed_with_emb/train/cellxgene_pseudo_bulk_10k/train/chunk_0.zarr"
ZARR_PATH="${ZARR_PATH_OVERRIDE:-$DEFAULT_ZARR}"

echo "Using ZARR_PATH=${ZARR_PATH}"

SCRIPT="/home/menger/git/adata_hf_datasets/scripts/test/test_constructor_speed.py"

echo "=== Run 1: add_anndata path ==="
date
srun -c ${SLURM_CPUS_PER_TASK:-4} python "$SCRIPT" \
  --zarr-path "$ZARR_PATH" \
  --negatives-per-sample 2 \
  --dataset-format multiplets \
  --caption-key natural_language_annotation \
  --batch-key dataset_title \
  --num-sentences 4 \
  --long-sentence-cols 2,3 \
  --long-sentence-words 5000 | cat
date

echo
echo "=== Run 2: add_df path (obs-only) ==="
date
srun -c ${SLURM_CPUS_PER_TASK:-4} python "$SCRIPT" \
  --zarr-path "$ZARR_PATH" \
  --negatives-per-sample 2 \
  --dataset-format multiplets \
  --caption-key natural_language_annotation \
  --batch-key dataset_title \
  --num-sentences 4 \
  --long-sentence-cols 2,3 \
  --long-sentence-words 5000 \
  --use-add-df | cat
date

echo "Done. Outputs above include detailed timings for read_zarr, add_* and get_dataset."
