#!/usr/bin/env python3
"""
Unified workflow submission script.

This script runs workflows using the unified WorkflowRunner that respects
per-step execution locations configured in your dataset config.

Each step can specify where it runs via the `execution_location` field:
- local: Run on local machine via subprocess
- cpu: Run on CPU cluster via SSH + SLURM
- gpu: Run on GPU cluster via SSH + SLURM

Data is automatically transferred between locations when consecutive steps
run on different machines.

Usage:
    # Submit workflow (runs in background by default)
    python scripts/workflow/submit_workflow.py --config my_dataset

    # Run in foreground to see all output
    python scripts/workflow/submit_workflow.py --config my_dataset --foreground

    # Skip config synchronization check
    python scripts/workflow/submit_workflow.py --config my_dataset --force

Examples:
    # Run a workflow where steps are distributed across locations
    # (based on execution_location in dataset config)
    python scripts/workflow/submit_workflow.py --config dataset_cellxgene_pseudo_bulk_10k

    # Run with verbose output
    python scripts/workflow/submit_workflow.py --config human_pancreas --foreground
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from adata_hf_datasets.workflow import (
    ensure_config_sync,
    resolve_workflow_config,
    run_workflow_unified,
)

logger = logging.getLogger(__name__)


def load_workflow_config() -> DictConfig:
    """Load the workflow orchestrator configuration."""
    config_path = Path(__file__).parent.parent.parent / "conf"

    # Clear any existing Hydra initialization
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="workflow_orchestrator")

    return cfg


def validate_config_sync(
    dataset_config_name: str,
    workflow_config: DictConfig,
    resolved_config: DictConfig,
    remote_locations_used: list,
    force: bool = False,
) -> None:
    """Validate config synchronization with remote locations that are actually used."""
    if force:
        logger.warning("Skipping config synchronization check (force=True)")
        return

    # Only check if remote locations are actually used
    if not remote_locations_used:
        logger.info("All steps run locally - skipping remote config sync check")
        return

    # Get locations config
    locations = workflow_config.get("locations", {})

    # Check config sync for each remote location that's used
    for loc_name in remote_locations_used:
        loc_config = locations.get(loc_name, {})
        ssh_host = loc_config.get("ssh_host")
        project_dir = loc_config.get("project_directory")

        if not ssh_host or not project_dir:
            logger.warning(
                f"Remote location '{loc_name}' missing ssh_host or project_directory - skipping sync check"
            )
            continue

        logger.info(
            f"Validating config synchronization with {loc_name} ({ssh_host})..."
        )

        try:
            ensure_config_sync(
                config_name=dataset_config_name,
                remote_host=ssh_host,
                remote_project_dir=project_dir,
                force=force,
            )
            logger.info(f"✓ Config sync with {loc_name} validated")
        except Exception as e:
            logger.error(f"Config sync with {loc_name} failed: {e}")
            raise


def run_workflow(
    dataset_config_name: str,
    resolved_config: DictConfig,
    foreground: bool = False,
    force: bool = False,
) -> None:
    """
    Run the unified workflow.

    Each step runs on its configured execution_location (local/cpu/gpu).
    Data is automatically transferred between locations as needed.
    All logs are centralized locally.
    """
    logger.info(f"Submitting workflow for dataset: {dataset_config_name}")

    # Set BASE_FILE_PATH for config transformations in child processes
    os.environ["BASE_FILE_PATH"] = resolved_config.get("base_file_path", "./data/RNA")

    project_dir = Path(__file__).resolve().parents[2]

    # Generate workflow ID that will be used by WorkflowRunner
    workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Compute output dir path (this is where WorkflowRunner will create its directory)
    output_dir = resolved_config.get("output_directory", "./outputs")
    if not Path(output_dir).is_absolute():
        output_dir = project_dir / output_dir
    date_str = datetime.now().strftime("%Y-%m-%d")
    workflow_dir = Path(output_dir) / date_str / workflow_id

    if foreground:
        # Run directly in this process
        logger.info("Running workflow in foreground...")
        logger.info("Press Ctrl+C to stop the workflow")
        logger.info(f"Workflow directory: {workflow_dir}")
        logger.info("")

        try:
            success = run_workflow_unified(
                dataset_config_name_or_path=dataset_config_name,
                workflow_config=resolved_config,
                force=force,
                workflow_id=workflow_id,
            )

            if success:
                logger.info("✓ Workflow completed successfully")
            else:
                logger.error("✗ Workflow completed with errors")
                sys.exit(1)

        except KeyboardInterrupt:
            logger.warning("Workflow interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

    else:
        # Run in background
        logger.info("Submitting workflow in background...")

        # Create a minimal log directory for the launcher process
        # (The WorkflowRunner will create the full workflow directory structure)
        launcher_log_dir = workflow_dir / "logs"
        launcher_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = launcher_log_dir / "launcher.out"
        stderr_path = launcher_log_dir / "launcher.err"
        pid_file = launcher_log_dir / "workflow.pid"

        # Build command to run the unified workflow with the same workflow_id
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from omegaconf import OmegaConf
from adata_hf_datasets.workflow import run_workflow_unified, resolve_workflow_config

# Load workflow config
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path

config_path = Path("{project_dir}") / "conf"
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

with initialize_config_dir(config_dir=str(config_path), version_base=None):
    cfg = compose(config_name="workflow_orchestrator")

resolved = resolve_workflow_config(cfg.workflow)
success = run_workflow_unified("{dataset_config_name}", resolved, force={force}, workflow_id="{workflow_id}")
sys.exit(0 if success else 1)
""",
        ]

        # Build background command with caffeinate if available (prevents sleep on macOS)
        bg_cmd = ["nohup"]
        if shutil.which("caffeinate"):
            bg_cmd += ["caffeinate", "-dimsu"]
        else:
            logger.warning(
                "caffeinate not found; background job will not prevent system sleep"
            )
        bg_cmd += cmd

        env = os.environ.copy()
        env["WORKFLOW_RUN_ID"] = workflow_id

        with open(stdout_path, "a") as out_f, open(stderr_path, "a") as err_f:
            process = subprocess.Popen(
                bg_cmd,
                cwd=str(project_dir),
                env=env,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )

        # Save PID to file
        pid = process.pid
        with open(pid_file, "w") as f:
            f.write(f"{pid}\n")

        # Log info
        logger.info("")
        logger.info("=" * 80)
        logger.info("WORKFLOW RUNNING IN BACKGROUND")
        logger.info("=" * 80)
        logger.info(f"Process ID (PID): {pid}")
        logger.info(f"Workflow directory: {workflow_dir}")
        logger.info(f"Logs: {workflow_dir}/logs/")
        logger.info("")
        logger.info(f"To stop this workflow: kill {pid}")
        logger.info(f"Or use: kill $(cat {pid_file})")
        logger.info("")
        logger.info("Note: Killing the process will also terminate any running steps")
        logger.info("      (including remote SLURM jobs if applicable)")
        logger.info("=" * 80)


def show_execution_plan(
    dataset_config_name: str,
    workflow_config: DictConfig,
    resolved_config: DictConfig,
) -> list:
    """
    Show where each step will run based on config.

    Returns
    -------
    list
        List of remote location names that are used by enabled steps
    """
    from adata_hf_datasets.workflow import (
        get_step_execution_location,
        WORKFLOW_STEPS,
        apply_all_transformations,
    )

    remote_locations_used = set()
    locations = workflow_config.get("locations", {})

    # Try to load dataset config to show execution plan
    try:
        config_path = Path(__file__).parent.parent.parent / "conf"
        config_file = config_path / f"{dataset_config_name}.yaml"

        if config_file.exists():
            # Set BASE_FILE_PATH for config transformations
            os.environ["BASE_FILE_PATH"] = resolved_config.get(
                "base_file_path", "./data/RNA"
            )

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

            with initialize_config_dir(config_dir=str(config_path), version_base=None):
                dataset_config = compose(config_name=dataset_config_name)

            dataset_config = apply_all_transformations(dataset_config)

            default_loc = resolved_config.get("default_execution_location", "local")

            logger.info("")
            logger.info("Execution plan:")
            logger.info("-" * 40)
            for step in WORKFLOW_STEPS:
                loc = get_step_execution_location(step, dataset_config, default_loc)
                section = getattr(dataset_config, step, None)
                enabled = getattr(section, "enabled", True) if section else True
                status = f"→ {loc}" if enabled else "(disabled)"
                logger.info(f"  {step:25} {status}")

                # Track remote locations used by enabled steps
                if enabled and loc != "local":
                    loc_config = locations.get(loc, {})
                    if loc_config.get("ssh_host"):
                        remote_locations_used.add(loc)

            logger.info("-" * 40)

    except Exception as e:
        logger.debug(f"Could not show execution plan: {e}")

    return list(remote_locations_used)


def main():
    """Main function to submit workflow."""
    parser = argparse.ArgumentParser(
        description="Submit workflow with per-step execution locations"
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Dataset config name (without .yaml extension) or path to config file "
            "(e.g., 'human_pancreas' or 'conf/my_dataset.yaml')"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip config synchronization check with remote hosts",
    )
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground (don't detach to background)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        logger.info("=" * 80)
        logger.info("WORKFLOW SUBMISSION")
        logger.info("=" * 80)
        logger.info(f"Dataset config: {args.config}")
        logger.info(f"Force mode: {args.force}")
        logger.info(f"Foreground: {args.foreground}")

        # Load workflow configuration
        logger.info("")
        logger.info("Loading workflow configuration...")
        workflow_config_full = load_workflow_config()
        workflow_section = workflow_config_full.get("workflow", {})
        logger.info("✓ Workflow configuration loaded")

        # Resolve workflow config
        resolved_config = resolve_workflow_config(workflow_section)

        # Show configured locations
        locations = workflow_section.get("locations", {})
        logger.info("")
        logger.info("Configured locations:")
        for name, loc in locations.items():
            is_remote = loc.get("ssh_host") is not None
            if is_remote:
                logger.info(f"  {name}: {loc.get('ssh_user')}@{loc.get('ssh_host')}")
            else:
                logger.info(f"  {name}: local")

        # Show execution plan and get list of remote locations used
        remote_locations_used = show_execution_plan(
            args.config, workflow_section, resolved_config
        )

        # Validate config sync only for remote hosts that are actually used
        validate_config_sync(
            args.config,
            workflow_section,
            resolved_config,
            remote_locations_used,
            force=args.force,
        )

        # Run workflow
        run_workflow(
            dataset_config_name=args.config,
            resolved_config=resolved_config,
            foreground=args.foreground,
            force=args.force,
        )

        if not args.foreground:
            logger.info("")
            logger.info("=" * 80)
            logger.info("WORKFLOW SUBMISSION COMPLETED")
            logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("WORKFLOW SUBMISSION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
