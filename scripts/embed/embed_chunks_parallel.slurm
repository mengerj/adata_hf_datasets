#!/usr/bin/env bash
#SBATCH --job-name=embed
#SBATCH --mem=60G
#SBATCH --time=3:00:00
##SBATCH --array=0-$(($(ls "${INPUT_DIR}"/*.zarr | wc -l)-1))
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real array job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_ARRAY_JOB_ID}"
else
  # local run: use date+seconds to make it unique
  RUN_ID="embed_$(date +%Y%m%d_%H%M%S)"
  # simulate an array task index if unset (so your redirects still work)
  SLURM_ARRAY_TASK_ID=0
fi
# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare log directory and redirect
# ─────────────────────────────────────────────────────────────────────────────
LOG_DIR="logs/embed/${RUN_ID}"
mkdir -p "$LOG_DIR"
# now redirect stdout+stderr into per-task files
exec 1>logs/embed/"${RUN_ID}"/"${SLURM_ARRAY_TASK_ID}".out
exec 2>logs/embed/"${RUN_ID}"/"${SLURM_ARRAY_TASK_ID}".err

# record start time
start_ts=$(date +%s)
# ─────────────────────────────────────────────────────────────────────────────
# 2) CONFIGURATION (override via sbatch --export or ENV)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_OR_TEST="${TRAIN_OR_TEST:-train}"
DATANAME="${DATANAME:-cellxgene_pseudo_bulk_3_5k}"
SPLIT="${SPLIT:-train}"
INPUT_DIR="${INPUT_DIR:-data/RNA/processed/${TRAIN_OR_TEST}/${DATANAME}/${SPLIT}}"
#INPUT_DIR="${INPUT_DIR:-/scratch/local/menger/data/RNA/processed/${TRAIN_OR_TEST}/${DATANAME}/${SPLIT}}"
METHODS="${METHODS:-hvg pca scvi_fm}"
BATCH_KEY="${BATCH_KEY:-dataset_title}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_PROCS="${MAX_PROCS:-8}"
# ─────────────────────────────────────────────────────────────────────────────
# 3) Build a Hydra‐style list literal from our space‑separated METHODS
# e.g. "hvg pca" → ["hvg","pca"]
METHODS_JSON="[$(printf '"%s",' $METHODS | sed 's/,$//')]"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Determine number of CPU cores (local mode only)
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
# LOCAL MODE with capped parallelism via xargs -P
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  echo "[LOCAL MODE] Running all files with up to ${MAX_PROCS:-auto} parallel jobs"

  # 1) Detect total CPUs and set MAX_PROCS if unset
  LOCAL_CPUS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
  MAX_PROCS="${MAX_PROCS:-$LOCAL_CPUS}"
  echo "[LOCAL MODE] CPUs detected: $LOCAL_CPUS; using MAX_PROCS=$MAX_PROCS"

  # 2) Build the file list and hand off to xargs
  printf "%s\n" "${INPUT_DIR}"/*.zarr \
    | xargs -n1 -P"$MAX_PROCS" -I{} bash -c '
        f="$0"
        idx=$(basename "$f" .zarr)
        OUTDIR="outputs/$(date +%Y-%m-%d)/'"${RUN_ID}"'/local/${idx}"
        mkdir -p "$OUTDIR"
        echo "[LOCAL][${idx}] → $OUTDIR"
        python3 scripts/embed/embed_adata.py \
          ++methods="'"$METHODS_JSON"'" \
          ++batch_key="'"$BATCH_KEY"'" \
          ++batch_size="'"$BATCH_SIZE"'" \
          ++input_files="[\"$f\"]" \
          ++hydra.run.dir="$OUTDIR"
      ' {}

  echo "[LOCAL MODE] All done."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6) SLURM ARRAY MODE
# ─────────────────────────────────────────────────────────────────────────────
source .venv/bin/activate

files=( "${INPUT_DIR}"/*.zarr )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"
echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Processing $this_file"

BASE_OUT="outputs/$(date +%Y-%m-%d)/${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
mkdir -p "$BASE_OUT"

echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Running methods: $METHODS_JSON"
python3 scripts/embed/embed_adata.py \
    ++methods="$METHODS_JSON" \
    ++batch_key="$BATCH_KEY" \
    ++batch_size="$BATCH_SIZE" \
    ++input_files="[\"$this_file\"]" \
    ++hydra.run.dir="$BASE_OUT"
# ─────────────────────────────────────────────────────────────────────────────

# record end time and compute elapsed
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))

# format as H:M:S
hours=$(( elapsed/3600 ))
mins=$(( (elapsed%3600)/60 ))
secs=$(( elapsed%60 ))

echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Done. Elapsed time: ${hours}h${mins}m${secs}s"
