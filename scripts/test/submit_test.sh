#!/usr/bin/env bash
# submit_test.sh
# Calls sbatch once, launching an array of 5 tasks

set -euo pipefail

# choose how many tasks to launch
ARRAY_SIZE=5

echo "Submitting array of size $ARRAY_SIZE..."
sbatch \
  --job-name=test_array \
  --mem=1G \
  --time=00:05:00 \
  --array=0-$((ARRAY_SIZE-1)) \
  scripts/test/test_array.slurm
