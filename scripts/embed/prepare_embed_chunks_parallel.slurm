#!/usr/bin/env bash
#SBATCH --job-name=prepare_embed
#SBATCH --mem=60G
#SBATCH --time=24:00:00
set -euo pipefail
LOG_DIR="logs/embed/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$LOG_DIR"
# now redirect stdout+stderr into per-task files
exec 1>logs/embed/"${SLURM_ARRAY_JOB_ID}"/"${SLURM_ARRAY_TASK_ID}".out
exec 2>logs/embed/"${SLURM_ARRAY_JOB_ID}"/"${SLURM_ARRAY_TASK_ID}".err

# 1) CONFIGURATION (override via sbatch --export or ENV)
TRAIN_OR_TEST="${TRAIN_OR_TEST:-train}"
DATANAME="${DATANAME:-cellxgene_pseudo_bulk_3_5k}"
SPLIT="${SPLIT:-train}"
INPUT_DIR="${INPUT_DIR:-data/RNA/processed/${TRAIN_OR_TEST}/${DATANAME}/${SPLIT}}"
METHODS="${METHODS:-hvg pca scvi_fm}"
BATCH_KEY="${BATCH_KEY:-dataset_title}"
BATCH_SIZE="${BATCH_SIZE:-32}"
##SBATCH --output=logs/prepare_embed/%A_%a.out
##SBATCH --error=logs/prepare_embed/%A_%a.err
# Build JSON list from space-separated methods
METHODS_JSON="[$(printf '"%s",' $METHODS | sed 's/,$//')]"

# Determine CPU count for local parallelism
if command -v nproc &>/dev/null; then
  NCPU=$(nproc)
elif command -v getconf &>/dev/null; then
  NCPU=$(getconf _NPROCESSORS_ONLN)
else
  NCPU=1
fi

SCRIPT="python3 scripts/embed/prepare_embed_adata.py"

# LOCAL MODE (no SLURM_ARRAY_JOB_ID)
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  echo "[LOCAL MODE] Running prepare on all files with $NCPU workers"
  files=( "$INPUT_DIR"/*.h5ad )
  printf "%s\n" "${files[@]}" \
    | xargs -n1 -P"$NCPU" bash -c '
        f="$0"; base=$(basename "$f" .h5ad)
        OUTDIR="outputs/$(date +%Y-%m-%d)/local/${base}"
        mkdir -p "$OUTDIR"
        echo "[LOCAL][$base] Preparing methods: '"$METHODS_JSON"'"
        '"$SCRIPT"' \
          ++methods="'"$METHODS_JSON"'" \
          ++batch_key="'"$BATCH_KEY"'" \
          ++batch_size="'"$BATCH_SIZE"'" \
          ++input_files="[\"$f\"]" \
          ++hydra.run.dir="$OUTDIR"
      '
  echo "[LOCAL MODE] All preparations done."
  exit 0
fi

# SLURM ARRAY MODE
source .venv/bin/activate

files=( "${INPUT_DIR}"/*.h5ad )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"

echo "[${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Preparing $this_file"
BASE_OUT="outputs/$(date +%Y-%m-%d)/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE_OUT"

echo "[${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Methods: $METHODS_JSON"
$SCRIPT \
  ++methods="$METHODS_JSON" \
  ++batch_key="$BATCH_KEY" \
  ++batch_size="$BATCH_SIZE" \
  ++input_files="[\"$this_file\"]" \
  ++hydra.run.dir="$BASE_OUT"

echo "[${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Done."
