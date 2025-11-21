#!/usr/bin/env python3
"""
Submit a local (macOS) master workflow process in the background.

This sets execution_mode=local and spawns run_workflow_master.py with nohup, printing
the output directory where logs will be collected.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
import shutil


def main():
    parser = argparse.ArgumentParser(description="Submit local workflow master")
    parser.add_argument("--config", required=True, help="Dataset config name or path")
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground (do not detach)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    project_dir = Path(__file__).resolve().parents[2]
    conf_dir = project_dir / "conf"

    # Load workflow config to get output_directory
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="workflow_orchestrator")
    workflow = cfg.workflow

    # Ensure local execution
    os.environ["HYDRA_FULL_ERROR"] = "1"
    # run_workflow_master.py will read config and respect execution_mode
    # We don't modify files on disk; instead rely on the config's default or user-edited value.

    # Compute output dir path for info printing
    output_dir = workflow.get("output_directory", str(project_dir / "outputs"))
    run_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_out = Path(output_dir) / date_str / f"workflow_{run_id}"

    # Build command
    cmd = [
        sys.executable,
        "scripts/workflow/run_workflow_master.py",
        args.config,
    ]

    env = os.environ.copy()
    env["SLURM_JOB_ID"] = run_id
    env["EXECUTION_MODE"] = "local"

    if args.foreground:
        logging.info("Running local master in foreground...")
        subprocess.run(cmd, cwd=str(project_dir), env=env, check=False)
    else:
        logging.info(
            "Submitting local master in background (nohup + caffeinate if available)..."
        )
        log_dir = base_out / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "workflow_master.out"
        stderr_path = log_dir / "workflow_master.err"
        # Build background command with caffeinate if available (prevents sleep)
        bg_cmd = ["nohup"]
        if shutil.which("caffeinate"):
            bg_cmd += ["caffeinate", "-dimsu"]
        else:
            logging.warning(
                "caffeinate not found; background job will not prevent system sleep"
            )
        bg_cmd += cmd
        with open(stdout_path, "a") as out_f, open(stderr_path, "a") as err_f:
            subprocess.Popen(
                bg_cmd,
                cwd=str(project_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
        logging.info(f"Logs: {str(log_dir)}")

    logging.info(f"Outputs will be under: {str(base_out)}")


if __name__ == "__main__":
    main()
