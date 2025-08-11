#!/usr/bin/env bash
#SBATCH --job-name=fetch_descr
#SBATCH --mem=10G
#SBATCH --time=48:00:00

# Activate environment
source .venv/bin/activate

python scripts/fetch_term_descriptions.py
