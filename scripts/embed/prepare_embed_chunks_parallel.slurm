#!/usr/bin/env bash
#SBATCH --job-name=prepare_embed
#SBATCH --mem=60G
#SBATCH --time=6:00:00
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0) Define a unified RUN_ID: real array job ID under SLURM, or a timestamp locally
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  RUN_ID="${SLURM_ARRAY_JOB_ID}"
  # Use workflow directory if set, otherwise use default logs directory
  if [[ -n "${WORKFLOW_DIR:-}" ]]; then
    LOG_DIR="${WORKFLOW_DIR}/embedding_prepare/array_${RUN_ID}"
    BASE_OUT="${WORKFLOW_DIR}/embedding_prepare/array_${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
  else
    LOG_DIR="logs/embed/${RUN_ID}"
    BASE_OUT="outputs/$(date +%Y-%m-%d)/${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
  fi
  mkdir -p "$LOG_DIR"
  mkdir -p "$BASE_OUT"
  # Redirect logs to the log directory
  exec 1>"${LOG_DIR}/${SLURM_ARRAY_TASK_ID}.out"
  exec 2>"${LOG_DIR}/${SLURM_ARRAY_TASK_ID}.err"
else
  # local run: use date+seconds to make it unique
  RUN_ID="prepare_$(date +%Y%m%d_%H%M%S)"
  # simulate an array task index if unset (so your redirects still work)
  SLURM_ARRAY_TASK_ID=0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare log directory and redirect
# ─────────────────────────────────────────────────────────────────────────────
# Log redirection is now handled above

# ─────────────────────────────────────────────────────────────────────────────
# 2) CONFIGURATION (override via sbatch --export or ENV)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_OR_TEST="${TRAIN_OR_TEST:-train}"
DATANAME="${DATANAME:-cellxgene_pseudo_bulk_3_5k}"
SPLIT="${SPLIT:-train}"
INPUT_DIR="${INPUT_DIR:-data/RNA/processed/${TRAIN_OR_TEST}/${DATANAME}/${SPLIT}}"
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

SCRIPT="python3 scripts/embed/embed_adata.py"

# ─────────────────────────────────────────────────────────────────────────────
# 5) LOCAL MODE with capped parallelism via xargs -P
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  echo "[LOCAL MODE] Running prepare on all files with up to ${MAX_PROCS:-auto} parallel jobs"

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
          ++prepare_only=true \
          ++embedding.methods="'"$METHODS_JSON"'" \
          ++batch_key="'"$BATCH_KEY"'" \
          ++batch_size="'"$BATCH_SIZE"'" \
          ++embedding.input_files="[\"$f\"]" \
          ++hydra.run.dir="$OUTDIR"
      ' {}

  echo "[LOCAL MODE] All preparations done."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6) SLURM ARRAY MODE
# ─────────────────────────────────────────────────────────────────────────────
source .venv/bin/activate

files=( "${INPUT_DIR}"/*.zarr )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"

echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Preparing $this_file"

echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Methods: $METHODS_JSON"
python3 scripts/embed/embed_adata.py \
  ++prepare_only=true \
  ++embedding.methods="'"$METHODS_JSON"'" \
  ++batch_key="'"$BATCH_KEY"'" \
  ++batch_size="'"$BATCH_SIZE"'" \
  ++embedding.input_files="[\"$this_file\"]" \
  ++hydra.run.dir="$BASE_OUT"

echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Done."
