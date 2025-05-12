#!/usr/bin/env bash
#SBATCH --job-name=embed
#SBATCH --output=logs/embed/%x_%A_%a.out
#SBATCH --error=logs/embed/%x_%A_%a.err
#SBATCH --mem=60G
#SBATCH --time=24:00:00

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION (override via sbatch --export or ENV)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_OR_TEST="${TRAIN_OR_TEST:-train}"
DATANAME="${DATANAME:-cellxgene_pseudo_bulk_3_5k}"
SPLIT="${SPLIT:-train}"
INPUT_DIR="${INPUT_DIR:-data/RNA/processed/${TRAIN_OR_TEST}/${DATANAME}/${SPLIT}}"
METHODS="${METHODS:-hvg pca scvi_fm}"
BATCH_KEY="${BATCH_KEY:-dataset_title}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# ─────────────────────────────────────────────────────────────────────────────
# 2) Build a Hydra‐style list literal from our space‑separated METHODS
# e.g. "hvg pca" → ["hvg","pca"]
METHODS_JSON="[$(printf '"%s",' $METHODS | sed 's/,$//')]"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Determine number of CPU cores (local mode only)
# ─────────────────────────────────────────────────────────────────────────────
if command -v nproc &>/dev/null; then
  LOCAL_CONCURRENCY=$(nproc)
elif command -v getconf &>/dev/null; then
  LOCAL_CONCURRENCY=$(getconf _NPROCESSORS_ONLN)
elif [[ "$(uname)" == "Darwin" ]] && command -v sysctl &>/dev/null; then
  LOCAL_CONCURRENCY=$(sysctl -n hw.ncpu)
else
  LOCAL_CONCURRENCY=1
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4) LOCAL MODE (no SLURM_JOB_ID)
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[LOCAL MODE] Running all files in parallel with $LOCAL_CONCURRENCY workers"
  files=( "$INPUT_DIR"/*.h5ad )
  printf "%s\n" "${files[@]}" \
    | xargs -n1 -P"$LOCAL_CONCURRENCY" bash -c '
        this_file="$0"
        base=$(basename "$this_file" .h5ad)
        OUTDIR="outputs/$(date +%Y-%m-%d)/local/${base}"
        mkdir -p "$OUTDIR"
        echo "[LOCAL][$base] Running methods: '"$METHODS_JSON"'"
        python3 scripts/embed_adata.py \
          ++methods="'"$METHODS_JSON"'" \
          ++batch_key="'"$BATCH_KEY"'" \
          ++batch_size="'"$BATCH_SIZE"'" \
          ++input_files="[\"$this_file\"]" \
          ++hydra.run.dir="$OUTDIR"
      '
  echo "[LOCAL MODE] All done."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5) SLURM ARRAY MODE
# ─────────────────────────────────────────────────────────────────────────────
source .venv/bin/activate

files=( "${INPUT_DIR}"/*.h5ad )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"
echo "[${SLURM_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Processing $this_file"

BASE_OUT="outputs/$(date +%Y-%m-%d)/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE_OUT"

echo "[${SLURM_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Running methods: $METHODS_JSON"
python3 scripts/embed_adata.py \
    ++methods="$METHODS_JSON" \
    ++batch_key="$BATCH_KEY" \
    ++batch_size="$BATCH_SIZE" \
    ++input_files="[\"$this_file\"]" \
    ++hydra.run.dir="$BASE_OUT"

echo "[${SLURM_JOB_ID}:${SLURM_ARRAY_TASK_ID}] All done."
