#!/usr/bin/env python3
"""
Script to submit the master workflow job to the CPU cluster.

Usage:
    python scripts/workflow/submit_workflow.py --config-name dataset_cellxgene_pseudo_bulk_3_5k
"""

import argparse
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def submit_master_workflow(dataset_config_name: str, cpu_host: str = "imbi13") -> int:
    """Submit the master workflow job to the CPU cluster."""
    project_dir = "/home/menger/git/adata_hf_datasets"
    script_path = "scripts/workflow/run_workflow_master.slurm"

    # Build the sbatch command
    cmd = [
        "ssh",
        cpu_host,
        f"cd {project_dir} && sbatch",
        "--partition",
        "slurm",
        "--export",
        f"ALL,DATASET_CONFIG={dataset_config_name}",
        script_path,
    ]

    logger.info(f"Submitting master workflow job ➜ {' '.join(cmd)}")

    # Execute the command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit master workflow job: {result.stderr}")

    # Parse job ID from output
    import re

    output = result.stdout.strip()
    job_id_match = re.search(r"Submitted batch job (\d+)", output)
    if not job_id_match:
        raise RuntimeError(f"Could not parse job ID from output: {output}")

    job_id = int(job_id_match.group(1))
    return job_id


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Submit master workflow job")
    parser.add_argument("--config-name", required=True, help="Dataset config name")
    parser.add_argument("--cpu-host", default="imbi13", help="CPU cluster host")
    args = parser.parse_args()

    try:
        job_id = submit_master_workflow(args.config_name, args.cpu_host)
        logger.info(
            f"✓ Master workflow job {job_id} submitted to cluster ({args.cpu_host})"
        )
        logger.info("The workflow will run automatically on the cluster.")
        logger.info("You can monitor progress using 'squeue' on the CPU cluster.")
    except Exception as e:
        logger.error(f"Failed to submit master workflow job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
