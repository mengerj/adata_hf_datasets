#!/usr/bin/env bash
#SBATCH --job-name=test_array

set -euo pipefail

# create the parent dir before redirection happens
mkdir -p logs/test_array/"${SLURM_ARRAY_JOB_ID}"

# redirect all output into our per-task files
exec 1>logs/test_array/"${SLURM_ARRAY_JOB_ID}"/"${SLURM_ARRAY_TASK_ID}".out
exec 2>logs/test_array/"${SLURM_ARRAY_JOB_ID}"/"${SLURM_ARRAY_TASK_ID}".err

echo "========== environment =========="
echo "SLURM_JOB_ID         = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID   = ${SLURM_ARRAY_JOB_ID:-<unset>}"
echo "SLURM_ARRAY_TASK_ID  = ${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "SLURM_STEP_ID        = ${SLURM_STEP_ID:-<unset>}"
echo "SLURM_STEP_NODELIST  = ${SLURM_STEP_NODELIST:-<unset>}"
echo "SLURM_CPUS_PER_TASK  = ${SLURM_CPUS_PER_TASK:-<unset>}"
echo

echo "========== debugging date =========="
date
echo

echo "========== done =========="
