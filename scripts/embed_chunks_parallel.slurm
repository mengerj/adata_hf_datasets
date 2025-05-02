#!/usr/bin/env bash
#SBATCH --job-name=embed               # If under Slurm, this is parsed
#SBATCH --output=logs/embed/%x_%A_%a.out
#SBATCH --error=logs/embed/%x_%A_%a.err
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION (override via sbatch --export or ENV)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_OR_TEST="${TRAIN_OR_TEST:-train}"
DATANAME="${DATANAME:-cellxgene_pseudo_bulk_35k}"
SPLIT="${SPLIT:-train}"
INPUT_DIR="${INPUT_DIR:-data/RNA/processed/${TRAIN_OR_TEST}/${DATANAME}/${SPLIT}}"
METHODS="${METHODS:-hvg pca scvi_fm}"
BATCH_KEY="${BATCH_KEY:-dataset_title}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# ─────────────────────────────────────────────────────────────────────────────

# 2.1) Determine number of CPU cores, cross‐platform
if command -v nproc &>/dev/null; then
  LOCAL_CONCURRENCY=$(nproc)
elif command -v getconf &>/dev/null; then
  LOCAL_CONCURRENCY=$(getconf _NPROCESSORS_ONLN)
elif [[ "$(uname)" == "Darwin" ]] && command -v sysctl &>/dev/null; then
  LOCAL_CONCURRENCY=$(sysctl -n hw.ncpu)
else
  LOCAL_CONCURRENCY=1
fi
# 2.2) Detect Slurm vs Local
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[LOCAL MODE] No \$SLURM_JOB_ID found, running all chunks in parallel with $LOCAL_CONCURRENCY workers"

  files=( "$INPUT_DIR"/*.h5ad )
  printf "%s\n" "${files[@]}" \
    | xargs -n1 -P"$LOCAL_CONCURRENCY" bash -c '
        this_file="$0"
        # derive a nice output dir per chunk
        base=$(basename "$this_file" .h5ad)
        OUTDIR="outputs//$(date +%Y-%m-%d)/local/${base}"
        mkdir -p "$OUTDIR"
        IFS=" " read -r -a list <<< "'"$METHODS"'"
        for METHOD in "${list[@]}"; do
          echo "[LOCAL][$base] Running $METHOD"
          python3 scripts/embed_adata.py \
            ++method="$METHOD" \
            ++batch_key="'"$BATCH_KEY"'" \
            ++batch_size="'"$BATCH_SIZE"'" \
            ++input_files="[\"$this_file\"]" \
            ++hydra.run.dir="$OUTDIR/$METHOD"
        done
      '

  echo "[LOCAL MODE] All done."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3) SLURM‐ARRAY MODE: each task picks one file based on $SLURM_ARRAY_TASK_ID
# ─────────────────────────────────────────────────────────────────────────────
source .venv/bin/activate
files=( "${INPUT_DIR}"/*.h5ad )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"
echo "[${SLURM_JOB_ID}:${SLURM_ARRAY_TASK_ID}] Processing $this_file"

BASE_OUT="outputs/$(date +%Y-%m-%d)/${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE_OUT"

IFS=' ' read -r -a method_list <<< "$METHODS"
for METHOD in "${method_list[@]}"; do
    echo " → Method $METHOD"
    python3 scripts/embed_adata.py \
        ++method="$METHOD" \
        ++batch_key="$BATCH_KEY" \
        ++batch_size="$BATCH_SIZE" \
        ++input_files="[\"$this_file\"]" \
        ++hydra.run.dir="$BASE_OUT/$METHOD"
    echo " ← Done $METHOD"
done

echo "[${SLURM_JOB_ID}:${SLURM_ARRAY_TASK_ID}] All methods complete."
