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
  # Use workflow directory if set, otherwise use default logs directory
  if [[ -n "${WORKFLOW_DIR:-}" ]]; then
    # Determine output directory based on PREPARE_ONLY parameter
    if [[ "$PREPARE_ONLY" == "true" ]]; then
      LOG_DIR="${WORKFLOW_DIR}/embedding_prepare/array_${RUN_ID}"
      BASE_OUT="${WORKFLOW_DIR}/embedding_prepare/array_${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
    else
      LOG_DIR="${WORKFLOW_DIR}/embedding/array_${RUN_ID}"
      BASE_OUT="${WORKFLOW_DIR}/embedding/array_${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
    fi
  else
    # Determine output directory based on PREPARE_ONLY parameter
    if [[ "$PREPARE_ONLY" == "true" ]]; then
      LOG_DIR="logs/embed_prepare/${RUN_ID}"
      BASE_OUT="outputs/$(date +%Y-%m-%d)/embedding_prepare/${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
    else
      LOG_DIR="logs/embed/${RUN_ID}"
      BASE_OUT="outputs/$(date +%Y-%m-%d)/embed/${RUN_ID}/${SLURM_ARRAY_TASK_ID}"
    fi
  fi
  mkdir -p "$LOG_DIR"
  mkdir -p "$BASE_OUT"
  # Redirect logs to the log directory
  exec 1>"${LOG_DIR}/${SLURM_ARRAY_TASK_ID}.out"
  exec 2>"${LOG_DIR}/${SLURM_ARRAY_TASK_ID}.err"
else
  # local run: use date+seconds to make it unique
  if [[ "$PREPARE_ONLY" == "true" ]]; then
    RUN_ID="prepare_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_ID="embed_$(date +%Y%m%d_%H%M%S)"
  fi
  # simulate an array task index if unset (so your redirects still work)
  SLURM_ARRAY_TASK_ID=0
fi
# ─────────────────────────────────────────────────────────────────────────────
# 1) Prepare log directory and redirect
# ─────────────────────────────────────────────────────────────────────────────
# Log redirection is now handled above

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
PREPARE_ONLY="${PREPARE_ONLY:-false}"  # New parameter to control mode
# ─────────────────────────────────────────────────────────────────────────────
# 3) Build a Hydra‐style list literal from our space‑separated METHODS
# e.g. "hvg pca" → ["hvg","pca"]
METHODS_JSON="[$(printf '"%s",' $METHODS | sed 's/,$//')]"

# ─────────────────────────────────────────────────────────────────────────────
# 4) Determine embedding section based on MODE
# ─────────────────────────────────────────────────────────────────────────────
# Determine which embedding section to use based on MODE
if [[ "$MODE" == "cpu" ]]; then
    EMBEDDING_SECTION="embedding_cpu"
elif [[ "$MODE" == "gpu" ]]; then
    EMBEDDING_SECTION="embedding_gpu"
else
    # Fallback to old structure for backward compatibility
    EMBEDDING_SECTION="embedding"
fi
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
# ─────────────────────────────────────────────────────────────────────────────
# 5) Determine number of CPU cores (local mode only)
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
echo "About to check SLURM_ARRAY_JOB_ID: '${SLURM_ARRAY_JOB_ID:-}'"
echo "Is SLURM_ARRAY_JOB_ID empty? $([[ -z "${SLURM_ARRAY_JOB_ID:-}" ]] && echo "YES" || echo "NO")"

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
  if [[ "$PREPARE_ONLY" == "true" ]]; then
    MODE_STR="prepare-only"
  else
    MODE_STR="full-pipeline"
  fi
  echo "[LOCAL MODE] Running $MODE_STR on all files with up to ${MAX_PROCS:-auto} parallel jobs"

  # 1) Detect total CPUs and set MAX_PROCS if unset
  LOCAL_CPUS=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
  MAX_PROCS="${MAX_PROCS:-$LOCAL_CPUS}"
  echo "[LOCAL MODE] CPUs detected: $LOCAL_CPUS; using MAX_PROCS=$MAX_PROCS"

  # 2) Build the file list and hand off to xargs
  CONFIG_PATH="${PROJECT_DIR:-$(pwd)}/conf"
  printf "%s\n" "${INPUT_DIR}"/*.zarr \
    | xargs -n1 -P"$MAX_PROCS" -I{} bash -c '
        f="$0"
        idx=$(basename "$f" .zarr)
        if [[ "'"$PREPARE_ONLY"'" == "true" ]]; then
          OUTDIR="outputs/$(date +%Y-%m-%d)/embedding_prepare/'"${RUN_ID}"'/local/${idx}"
        else
          OUTDIR="outputs/$(date +%Y-%m-%d)/embed/'"${RUN_ID}"'/local/${idx}"
        fi
        mkdir -p "$OUTDIR"
        echo "[LOCAL][${idx}] → $OUTDIR"
        python3 scripts/embed/embed_core.py \
          --config-path="'"$CONFIG_PATH"'" \
          --config-name="${DATASET_CONFIG:-dataset_cellxgene_pseudo_bulk_3_5k}" \
          ++embedding_config_section="'"$EMBEDDING_SECTION"'" \
          ++prepare_only='"$PREPARE_ONLY"' \
          ++"'"$EMBEDDING_SECTION"'".methods="'"$METHODS_JSON"'" \
          ++batch_key="'"$BATCH_KEY"'" \
          ++batch_size="'"$BATCH_SIZE"'" \
          ++"'"$EMBEDDING_SECTION"'".input_files="[\"$f\"]" \
          ++hydra.run.dir="$OUTDIR"
      ' {}

  echo "[LOCAL MODE] All done."
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6) SLURM ARRAY MODE
# ─────────────────────────────────────────────────────────────────────────────
# Activate virtual environment (use VENV_PATH from environment or default to .venv)
VENV_PATH="${VENV_PATH:-.venv}"
echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Activating venv: $VENV_PATH"
source "$VENV_PATH/bin/activate"

files=( "${INPUT_DIR}"/*.zarr )
this_file="${files[$SLURM_ARRAY_TASK_ID]}"
if [[ "$PREPARE_ONLY" == "true" ]]; then
  MODE_STR="prepare-only"
else
  MODE_STR="full-pipeline"
fi
echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Processing $this_file ($MODE_STR)"

echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Running methods: $METHODS_JSON"

# Determine config path (relative to project directory)
CONFIG_PATH="${PROJECT_DIR:-$(pwd)}/conf"
echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Config path: $CONFIG_PATH"
echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Config name: ${DATASET_CONFIG:-dataset_cellxgene_pseudo_bulk_3_5k}"
echo "[${RUN_ID}:${SLURM_ARRAY_TASK_ID}] Embedding section: $EMBEDDING_SECTION"

python3 scripts/embed/embed_core.py \
    --config-path="$CONFIG_PATH" \
    --config-name="${DATASET_CONFIG:-dataset_cellxgene_pseudo_bulk_3_5k}" \
    ++embedding_config_section="$EMBEDDING_SECTION" \
    ++prepare_only="$PREPARE_ONLY" \
    ++"$EMBEDDING_SECTION".methods="$METHODS_JSON" \
    ++batch_key="$BATCH_KEY" \
    ++batch_size="$BATCH_SIZE" \
    ++"$EMBEDDING_SECTION".input_files="[\"$this_file\"]" \
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
