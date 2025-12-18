#!/usr/bin/env python3
"""
Step Executors for Workflow Orchestration

This module provides executor abstractions for running workflow steps
on different execution locations (local, cpu cluster, gpu cluster).

Each executor handles:
- Command execution (subprocess or SSH+SLURM)
- Output capture and logging
- Error handling and reporting
"""

import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .data_transfer import LocationConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .embedding_submitter import ArrayJobInfo

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a step execution."""

    success: bool
    return_code: int
    step_name: str
    location: str
    execution_time_seconds: float
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
    job_id: Optional[str] = None  # SLURM job ID for remote execution
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "success": self.success,
            "return_code": self.return_code,
            "step_name": self.step_name,
            "location": self.location,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "stdout_path": str(self.stdout_path) if self.stdout_path else None,
            "stderr_path": str(self.stderr_path) if self.stderr_path else None,
            "job_id": self.job_id,
            "error_message": self.error_message,
            "warnings": self.warnings,
        }


class StepExecutor(ABC):
    """
    Abstract base class for step execution.

    Each executor implementation handles running workflow steps
    on a specific type of execution location.
    """

    def __init__(self, location_config: LocationConfig, workflow_dir: Path):
        """
        Initialize the executor.

        Parameters
        ----------
        location_config : LocationConfig
            Configuration for the execution location
        workflow_dir : Path
            Local workflow directory for log collection
        """
        self.location_config = location_config
        self.workflow_dir = workflow_dir
        self.location_name = location_config.name

    @abstractmethod
    def execute(
        self,
        cmd: List[str],
        step_name: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a workflow step.

        Parameters
        ----------
        cmd : List[str]
            Command and arguments to execute
        step_name : str
            Name of the workflow step (for logging)
        env : Optional[Dict[str, str]]
            Environment variables for the command
        cwd : Optional[str]
            Working directory for execution
        timeout : Optional[int]
            Timeout in seconds (None = no timeout)

        Returns
        -------
        ExecutionResult
            Result of the execution including success status,
            output paths, and any error information
        """
        pass

    def terminate(self) -> None:
        """
        Terminate any currently running job/process.

        Override in subclasses to provide cleanup logic.
        """
        pass

    def _get_step_log_dir(self, step_name: str) -> Path:
        """Get the log directory for a step."""
        step_dir = self.workflow_dir / step_name.replace(" ", "_").lower()
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir


class LocalExecutor(StepExecutor):
    """
    Execute workflow steps via local subprocess.

    This executor runs commands directly on the local machine
    using subprocess.Popen, with output captured to log files.
    """

    def __init__(self, location_config: LocationConfig, workflow_dir: Path):
        super().__init__(location_config, workflow_dir)
        self._current_process: Optional[subprocess.Popen] = None

    def terminate(self) -> None:
        """Terminate the currently running process if any."""
        if self._current_process is not None and self._current_process.poll() is None:
            logger.info(f"Terminating local process {self._current_process.pid}")
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing process {self._current_process.pid}")
                self._current_process.kill()
                self._current_process.wait()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
            finally:
                self._current_process = None

    def execute(
        self,
        cmd: List[str],
        step_name: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute step locally via subprocess."""
        start_time = time.time()

        # Prepare log directory and files
        log_dir = self._get_step_log_dir(step_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_path = log_dir / f"{step_name}_{timestamp}.out"
        stderr_path = log_dir / f"{step_name}_{timestamp}.err"

        # Merge environment
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        # Default to project directory if no cwd specified
        if cwd is None:
            cwd = self.location_config.project_directory

        logger.info(f"Executing locally: {' '.join(cmd)}")
        logger.info(f"Working directory: {cwd}")
        logger.info(f"Logs: {stdout_path}, {stderr_path}")

        try:
            with open(stdout_path, "w") as stdout_f, open(stderr_path, "w") as stderr_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    env=full_env,
                    cwd=cwd,
                    text=True,
                )
                # Track the process so it can be terminated
                self._current_process = process

                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    execution_time = time.time() - start_time
                    return ExecutionResult(
                        success=False,
                        return_code=-1,
                        step_name=step_name,
                        location=self.location_name,
                        execution_time_seconds=execution_time,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        error_message=f"Step timed out after {timeout} seconds",
                    )
                finally:
                    self._current_process = None

            execution_time = time.time() - start_time

            # Check for errors in stderr
            warnings = []
            if stderr_path.exists() and stderr_path.stat().st_size > 0:
                with open(stderr_path) as f:
                    stderr_content = f.read()
                    if "warning" in stderr_content.lower():
                        warnings.append("Warnings found in stderr")

            return ExecutionResult(
                success=return_code == 0,
                return_code=return_code,
                step_name=step_name,
                location=self.location_name,
                execution_time_seconds=execution_time,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                error_message=f"Exit code {return_code}" if return_code != 0 else None,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed to execute step {step_name}: {e}")
            return ExecutionResult(
                success=False,
                return_code=-1,
                step_name=step_name,
                location=self.location_name,
                execution_time_seconds=execution_time,
                stdout_path=stdout_path if stdout_path.exists() else None,
                stderr_path=stderr_path if stderr_path.exists() else None,
                error_message=str(e),
            )


class RemoteExecutor(StepExecutor):
    """
    Execute workflow steps on a remote cluster via SSH + SLURM.

    This executor:
    1. Generates a SLURM script for the step
    2. Submits it via SSH
    3. Polls for completion
    4. Retrieves logs back to local workflow directory
    """

    def __init__(
        self,
        location_config: LocationConfig,
        workflow_dir: Path,
        poll_interval: int = 60,
        job_timeout: int = 0,
    ):
        """
        Initialize the remote executor.

        Parameters
        ----------
        location_config : LocationConfig
            Configuration for the remote location
        workflow_dir : Path
            Local workflow directory for log collection
        poll_interval : int
            Seconds between job status checks
        job_timeout : int
            Maximum time to wait for job completion (0 = no timeout)
        """
        super().__init__(location_config, workflow_dir)

        if not location_config.is_remote:
            raise ValueError(
                f"RemoteExecutor requires a remote location config, "
                f"got local config: {location_config.name}"
            )

        self.ssh_target = location_config.ssh_target
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout

        # Track current job for cancellation
        self._current_job_id: Optional[str] = None
        # Track array jobs for cancellation
        self._current_array_job_ids: List[str] = []

    def terminate(self) -> None:
        """Cancel the currently running SLURM job(s) if any."""
        # Cancel regular jobs
        if self._current_job_id is not None:
            logger.info(
                f"Cancelling SLURM job {self._current_job_id} on {self.location_name}"
            )
            try:
                code, stdout, stderr = self._run_ssh_command(
                    f"scancel {self._current_job_id}", timeout=30
                )
                if code == 0:
                    logger.info(f"Successfully cancelled job {self._current_job_id}")
                else:
                    logger.warning(
                        f"Failed to cancel job {self._current_job_id}: {stderr}"
                    )
            except Exception as e:
                logger.error(f"Error cancelling job {self._current_job_id}: {e}")
            finally:
                self._current_job_id = None

        # Cancel array jobs
        self.terminate_array_jobs()

    def _run_ssh_command(
        self,
        command: str,
        timeout: int = 300,
    ) -> Tuple[int, str, str]:
        """
        Run a command on the remote host via SSH.

        Returns (return_code, stdout, stderr).
        """
        ssh_cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "BatchMode=yes",
            self.ssh_target,
            command,
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"SSH command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    def _generate_slurm_script(
        self,
        cmd: List[str],
        step_name: str,
        env: Optional[Dict[str, str]],
        remote_stdout: str,
        remote_stderr: str,
    ) -> str:
        """Generate a SLURM batch script for the step."""
        # Clean step name for SLURM job name
        job_name = step_name.replace(" ", "_").lower()

        # Build environment exports
        env_exports = ""
        if env:
            for key, value in env.items():
                # Escape single quotes in values
                escaped_value = value.replace("'", "'\\''")
                env_exports += f"export {key}='{escaped_value}'\n"

        # Build the command string
        cmd_str = " ".join(f"'{arg}'" if " " in arg else arg for arg in cmd)

        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={remote_stdout}
#SBATCH --error={remote_stderr}
#SBATCH --partition={self.location_config.slurm_partition or "slurm"}
"""

        # Add node constraint if specified
        if self.location_config.node:
            script += f"#SBATCH --nodelist={self.location_config.node}\n"

        script += f"""
# Auto-generated SLURM script for workflow step: {step_name}
# Generated at: {datetime.now().isoformat()}

echo "Starting step: {step_name}"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working directory: {self.location_config.project_directory}"

cd {self.location_config.project_directory}

# Activate virtual environment if specified
if [ -n "{self.location_config.venv_path}" ] && [ -d "{self.location_config.venv_path}" ]; then
    source {self.location_config.venv_path}/bin/activate
    echo "Activated venv: {self.location_config.venv_path}"
fi

# Set environment variables
{env_exports}

# Create WORKFLOW_DIR if it doesn't exist (for remote log collection)
if [ -n "$WORKFLOW_DIR" ]; then
    mkdir -p "$WORKFLOW_DIR/logs" 2>/dev/null || true
fi

# Set up error logging to WORKFLOW_DIR if accessible
if [ -n "$WORKFLOW_DIR" ] && [ -d "$WORKFLOW_DIR/logs" ]; then
    ERROR_LOG="$WORKFLOW_DIR/logs/errors_consolidated.log"
else
    echo "Warning: Could not set up error logging to $WORKFLOW_DIR/logs/errors_consolidated.log"
    echo "Continuing without centralized error logging..."
fi

# Execute the step command
{cmd_str}
exit_code=$?

echo "Step completed with exit code: $exit_code"
echo "End time: $(date)"

exit $exit_code
"""
        return script

    def _submit_slurm_job(
        self, script_content: str, remote_workflow_dir: str
    ) -> Optional[str]:
        """
        Submit a SLURM job and return the job ID.

        Args:
            script_content: The SLURM script content
            remote_workflow_dir: Remote workflow directory (from WORKFLOW_DIR env var)

        Returns None if submission failed.
        """
        # Create remote directories for logs and slurm scripts
        remote_slurm_dir = f"{remote_workflow_dir}/slurm"
        mkdir_code, _, mkdir_err = self._run_ssh_command(f"mkdir -p {remote_slurm_dir}")
        if mkdir_code != 0:
            logger.error(f"Failed to create remote slurm dir: {mkdir_err}")
            return None

        # Write script to remote slurm directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_script = f"{remote_slurm_dir}/step_{timestamp}.slurm"

        write_code, _, write_err = self._run_ssh_command(
            f"cat > {remote_script} << 'SLURM_SCRIPT_EOF'\n{script_content}\nSLURM_SCRIPT_EOF"
        )
        if write_code != 0:
            logger.error(f"Failed to write SLURM script: {write_err}")
            return None

        # Submit the job
        submit_code, submit_out, submit_err = self._run_ssh_command(
            f"sbatch {remote_script}"
        )

        if submit_code != 0:
            logger.error(f"SLURM submission failed: {submit_err}")
            return None

        # Parse job ID from output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", submit_out)
        if match:
            job_id = match.group(1)
            logger.info(f"Submitted SLURM job: {job_id}")
            return job_id
        else:
            logger.error(f"Could not parse job ID from: {submit_out}")
            return None

    def _wait_for_job(self, job_id: str) -> Tuple[bool, str]:
        """
        Wait for a SLURM job to complete.

        Returns (success, final_state).
        """
        start_time = time.time()

        while True:
            # Check timeout
            if self.job_timeout > 0:
                elapsed = time.time() - start_time
                if elapsed > self.job_timeout:
                    logger.warning(f"Job {job_id} timed out after {elapsed:.0f}s")
                    # Try to cancel the job
                    self._run_ssh_command(f"scancel {job_id}")
                    return False, "TIMEOUT"

            # Check job status
            code, stdout, stderr = self._run_ssh_command(
                f"sacct -j {job_id} --format=JobID,State,ExitCode --noheader --parsable2"
            )

            if code != 0:
                logger.warning(f"sacct failed: {stderr}")
                # Fall back to squeue
                code, stdout, stderr = self._run_ssh_command(
                    f"squeue -j {job_id} --noheader"
                )
                if code == 0 and stdout.strip():
                    # Job still in queue
                    logger.info(f"Job {job_id} still running...")
                    time.sleep(self.poll_interval)
                    continue
                else:
                    # Job not in queue, assume completed
                    return True, "UNKNOWN"

            # Parse sacct output
            lines = stdout.strip().split("\n")
            main_job_state = None
            main_job_exit_code = None
            failed_array_jobs = []

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    job_id_part = parts[0]
                    state = parts[1]
                    exit_code = parts[2] if len(parts) >= 3 else None

                    # Look for main job (not .batch or .extern)
                    if job_id_part == job_id or job_id_part == f"{job_id}.batch":
                        main_job_state = state
                        main_job_exit_code = exit_code
                    # Check for failed array jobs (job_id_[array_index])
                    elif f"{job_id}_" in job_id_part and state not in [
                        "COMPLETED",
                        "COMPLETED+",
                        "PENDING",
                        "RUNNING",
                    ]:
                        failed_array_jobs.append((job_id_part, state, exit_code))

            if main_job_state:
                if main_job_state in ["COMPLETED", "COMPLETED+"]:
                    # Check if main job succeeded but array jobs failed
                    if failed_array_jobs:
                        error_msg = (
                            f"Job {job_id} main task completed but "
                            f"{len(failed_array_jobs)} array task(s) failed:"
                        )
                        for (
                            array_job_id,
                            array_state,
                            array_exit_code,
                        ) in failed_array_jobs:
                            error_msg += (
                                f"\n  - Array job {array_job_id}: {array_state}"
                                f" (exit code: {array_exit_code})"
                            )
                        logger.error(error_msg)
                        return False, "ARRAY_FAILED"

                    logger.info(f"Job {job_id} completed successfully")
                    return True, main_job_state
                elif main_job_state in [
                    "FAILED",
                    "CANCELLED",
                    "TIMEOUT",
                    "OUT_OF_MEMORY",
                    "NODE_FAIL",
                ]:
                    error_msg = f"Job {job_id} failed with state: {main_job_state}"
                    if main_job_exit_code and main_job_exit_code != "0:0":
                        error_msg += f" (exit code: {main_job_exit_code})"
                    logger.error(error_msg)
                    return False, main_job_state
                elif main_job_state in ["PENDING", "RUNNING", "COMPLETING"]:
                    logger.info(f"Job {job_id} state: {main_job_state}")
                    time.sleep(self.poll_interval)
                    continue
                else:
                    logger.warning(f"Job {job_id} unknown state: {main_job_state}")
                    time.sleep(self.poll_interval)
                    continue
            else:
                # No state found in sacct - check squeue before assuming completed
                code, stdout, stderr = self._run_ssh_command(
                    f"squeue -j {job_id} --noheader"
                )
                if code == 0 and stdout.strip():
                    # Job still in queue (not yet in sacct accounting)
                    logger.info(f"Job {job_id} still running (not yet in sacct)")
                    time.sleep(self.poll_interval)
                    continue
                else:
                    # Job not in queue AND not in sacct - likely completed
                    logger.info(f"Job {job_id} no longer in sacct or squeue")
                    return True, "COMPLETED"

    def _retrieve_logs(
        self,
        remote_stdout: str,
        remote_stderr: str,
        local_log_dir: Path,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Retrieve log files from remote to local.

        Returns (local_stdout_path, local_stderr_path).
        """
        local_stdout = local_log_dir / Path(remote_stdout).name
        local_stderr = local_log_dir / Path(remote_stderr).name

        # Retrieve stdout
        rsync_cmd = [
            "rsync",
            "-avz",
            "--partial",
            f"{self.ssh_target}:{remote_stdout}",
            str(local_stdout),
        ]

        try:
            result = subprocess.run(
                rsync_cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"Failed to retrieve stdout: {result.stderr}")
                local_stdout = None
        except Exception as e:
            logger.warning(f"Failed to retrieve stdout: {e}")
            local_stdout = None

        # Retrieve stderr
        rsync_cmd = [
            "rsync",
            "-avz",
            "--partial",
            f"{self.ssh_target}:{remote_stderr}",
            str(local_stderr),
        ]

        try:
            result = subprocess.run(
                rsync_cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"Failed to retrieve stderr: {result.stderr}")
                local_stderr = None
        except Exception as e:
            logger.warning(f"Failed to retrieve stderr: {e}")
            local_stderr = None

        return local_stdout, local_stderr

    def execute(
        self,
        cmd: List[str],
        step_name: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """Execute step on remote cluster via SSH + SLURM."""
        start_time = time.time()

        # Determine remote workflow directory from env
        # Fall back to project_directory/outputs if WORKFLOW_DIR not set
        remote_workflow_dir = (
            env.get("WORKFLOW_DIR")
            if env
            else f"{self.location_config.project_directory}/outputs"
        )
        if not remote_workflow_dir:
            remote_workflow_dir = f"{self.location_config.project_directory}/outputs"

        # Ensure remote log directory exists
        remote_log_dir = f"{remote_workflow_dir}/{step_name}"
        mkdir_code, _, _ = self._run_ssh_command(f"mkdir -p {remote_log_dir}")
        if mkdir_code != 0:
            logger.warning(f"Could not create remote log dir: {remote_log_dir}")

        # Prepare log paths on remote (in workflow dir, not /tmp)
        log_dir = self._get_step_log_dir(step_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_stdout = f"{remote_log_dir}/{step_name}_{timestamp}.out"
        remote_stderr = f"{remote_log_dir}/{step_name}_{timestamp}.err"

        logger.info(f"Executing remotely on {self.location_name}: {' '.join(cmd)}")
        logger.info(f"SSH target: {self.ssh_target}")

        # Generate and submit SLURM script
        script = self._generate_slurm_script(
            cmd, step_name, env, remote_stdout, remote_stderr
        )
        job_id = self._submit_slurm_job(script, remote_workflow_dir)

        if not job_id:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                return_code=-1,
                step_name=step_name,
                location=self.location_name,
                execution_time_seconds=execution_time,
                error_message="Failed to submit SLURM job",
            )

        # Track the job so it can be cancelled
        self._current_job_id = job_id

        # Wait for job completion
        try:
            success, final_state = self._wait_for_job(job_id)
        finally:
            self._current_job_id = None

        # Retrieve logs to local
        local_stdout, local_stderr = self._retrieve_logs(
            remote_stdout, remote_stderr, log_dir
        )

        execution_time = time.time() - start_time

        # Parse exit code from logs if available
        return_code = 0 if success else 1
        if local_stdout and local_stdout.exists():
            with open(local_stdout) as f:
                content = f.read()
                match = re.search(r"exit code: (\d+)", content.lower())
                if match:
                    return_code = int(match.group(1))

        # Check for warnings
        warnings = []
        if local_stderr and local_stderr.exists() and local_stderr.stat().st_size > 0:
            with open(local_stderr) as f:
                stderr_content = f.read()
                if "warning" in stderr_content.lower():
                    warnings.append("Warnings found in stderr")

        return ExecutionResult(
            success=success and return_code == 0,
            return_code=return_code,
            step_name=step_name,
            location=self.location_name,
            execution_time_seconds=execution_time,
            stdout_path=local_stdout,
            stderr_path=local_stderr,
            job_id=job_id,
            error_message=f"Job ended with state: {final_state}"
            if not success
            else None,
            warnings=warnings,
        )

    def execute_embedding_array_jobs(
        self,
        step_name: str,
        dataset_config: "DictConfig",
        dataset_config_name: str,
        mode: str,
        prepare_only: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """
        Execute embedding step by directly submitting and tracking array jobs.

        This bypasses the master job pattern and directly submits SLURM array jobs,
        then tracks them until all tasks complete.

        Parameters
        ----------
        step_name : str
            Name of the step (e.g., "embedding_gpu")
        dataset_config : DictConfig
            The resolved dataset configuration
        dataset_config_name : str
            Name of the dataset config file
        mode : str
            Processing mode: "cpu" or "gpu"
        prepare_only : bool
            Whether to run only the prepare step
        env : Dict[str, str], optional
            Environment variables to pass to jobs

        Returns
        -------
        ExecutionResult
            Result of the execution
        """
        from .embedding_submitter import EmbeddingArraySubmitter

        start_time = time.time()

        logger.info(
            f"Executing {step_name} via direct array submission on {self.location_name}"
        )

        # Create the submitter
        submitter = EmbeddingArraySubmitter(
            location_config=self.location_config,
            dataset_config=dataset_config,
            workflow_dir=self.workflow_dir,
            dataset_config_name=dataset_config_name,
        )

        # Submit array jobs
        try:
            submitted_jobs = submitter.submit_array_jobs(
                mode=mode,
                prepare_only=prepare_only,
                env=env,
            )
        except Exception as e:
            logger.error(f"Failed to submit array jobs: {e}")
            return ExecutionResult(
                success=False,
                return_code=-1,
                step_name=step_name,
                location=self.location_name,
                execution_time_seconds=time.time() - start_time,
                error_message=f"Failed to submit array jobs: {e}",
            )

        if not submitted_jobs:
            logger.error("No array jobs were submitted")
            return ExecutionResult(
                success=False,
                return_code=-1,
                step_name=step_name,
                location=self.location_name,
                execution_time_seconds=time.time() - start_time,
                error_message="No array jobs were submitted",
            )

        # Track all job IDs for cancellation
        self._current_array_job_ids = [job.job_id for job in submitted_jobs]

        logger.info(
            f"Submitted {len(submitted_jobs)} array jobs: "
            f"{[j.job_id for j in submitted_jobs]}"
        )

        # Wait for all array jobs to complete
        all_success = True
        failed_jobs = []

        try:
            for job_info in submitted_jobs:
                logger.info(
                    f"Waiting for array job {job_info.job_id} ({job_info.label}, "
                    f"{job_info.task_count} tasks)..."
                )
                success, final_state = self._wait_for_array_job(job_info)

                if not success:
                    all_success = False
                    failed_jobs.append((job_info.job_id, job_info.label, final_state))
                    logger.error(
                        f"Array job {job_info.job_id} ({job_info.label}) failed: {final_state}"
                    )
                else:
                    logger.info(
                        f"Array job {job_info.job_id} ({job_info.label}) completed successfully"
                    )
        finally:
            self._current_array_job_ids = []

        execution_time = time.time() - start_time

        if not all_success:
            error_msg = f"Array job(s) failed: {failed_jobs}"
            return ExecutionResult(
                success=False,
                return_code=1,
                step_name=step_name,
                location=self.location_name,
                execution_time_seconds=execution_time,
                error_message=error_msg,
            )

        return ExecutionResult(
            success=True,
            return_code=0,
            step_name=step_name,
            location=self.location_name,
            execution_time_seconds=execution_time,
            job_id=",".join([j.job_id for j in submitted_jobs]),
        )

    def _wait_for_array_job(self, job_info: "ArrayJobInfo") -> Tuple[bool, str]:
        """
        Wait for a SLURM array job to complete, tracking all tasks.

        Returns (success, final_state).
        """
        job_id = job_info.job_id
        start_time = time.time()

        while True:
            # Check timeout
            if self.job_timeout > 0:
                elapsed = time.time() - start_time
                if elapsed > self.job_timeout:
                    logger.warning(f"Array job {job_id} timed out after {elapsed:.0f}s")
                    self._run_ssh_command(f"scancel {job_id}")
                    return False, "TIMEOUT"

            # Check job status using sacct to see all array tasks
            code, stdout, stderr = self._run_ssh_command(
                f"sacct -j {job_id} --format=JobID,State,ExitCode --noheader --parsable2"
            )

            if code != 0:
                logger.warning(f"sacct failed: {stderr}")
                # Fall back to squeue
                code, stdout, stderr = self._run_ssh_command(
                    f"squeue -j {job_id} --noheader"
                )
                if code == 0 and stdout.strip():
                    logger.info(f"Array job {job_id} still has tasks running...")
                    time.sleep(self.poll_interval)
                    continue
                else:
                    # Job not in queue, check sacct again
                    time.sleep(5)
                    continue

            # Parse sacct output
            lines = stdout.strip().split("\n")
            pending_or_running = 0
            completed_tasks = 0
            failed_tasks = []

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue

                job_id_part = parts[0]
                state = parts[1]
                exit_code = parts[2] if len(parts) >= 3 else None

                # Skip main job entry, we only care about array tasks
                if job_id_part == job_id:
                    continue
                # Check for array tasks (job_id_[task_index])
                elif f"{job_id}_" in job_id_part and ".batch" not in job_id_part:
                    if state in ["PENDING", "RUNNING"]:
                        pending_or_running += 1
                    elif state in ["COMPLETED", "COMPLETED+"]:
                        completed_tasks += 1
                    elif state in [
                        "FAILED",
                        "CANCELLED",
                        "TIMEOUT",
                        "OUT_OF_MEMORY",
                        "NODE_FAIL",
                    ]:
                        failed_tasks.append((job_id_part, state, exit_code))

            # Log progress
            total_expected = job_info.task_count
            logger.info(
                f"Array job {job_id}: {completed_tasks}/{total_expected} completed, "
                f"{pending_or_running} running/pending, {len(failed_tasks)} failed"
            )

            # Check if all tasks are done
            if pending_or_running == 0 and (completed_tasks > 0 or failed_tasks):
                if failed_tasks:
                    logger.error(
                        f"Array job {job_id} has {len(failed_tasks)} failed tasks"
                    )
                    for task_id, state, exit_code in failed_tasks[:5]:
                        logger.error(f"  - {task_id}: {state} (exit: {exit_code})")
                    return False, "ARRAY_TASKS_FAILED"
                else:
                    return True, "COMPLETED"

            # Still running - check squeue as backup
            code, stdout, stderr = self._run_ssh_command(
                f"squeue -j {job_id} --noheader"
            )
            if code == 0 and stdout.strip():
                # Jobs still in queue
                time.sleep(self.poll_interval)
                continue
            elif completed_tasks > 0 or failed_tasks:
                # No jobs in queue and we have results
                if failed_tasks:
                    return False, "ARRAY_TASKS_FAILED"
                return True, "COMPLETED"
            else:
                # No jobs in queue but no results yet - wait a bit
                time.sleep(self.poll_interval)
                continue

    def terminate_array_jobs(self) -> None:
        """Cancel all currently tracked array jobs."""
        if hasattr(self, "_current_array_job_ids") and self._current_array_job_ids:
            for job_id in self._current_array_job_ids:
                logger.info(f"Cancelling array job {job_id} on {self.location_name}")
                try:
                    code, stdout, stderr = self._run_ssh_command(
                        f"scancel {job_id}", timeout=30
                    )
                    if code == 0:
                        logger.info(f"Successfully cancelled array job {job_id}")
                    else:
                        logger.warning(f"Failed to cancel array job {job_id}: {stderr}")
                except Exception as e:
                    logger.error(f"Error cancelling array job {job_id}: {e}")
            self._current_array_job_ids = []


def create_executor(
    location_name: str,
    location_config: LocationConfig,
    workflow_dir: Path,
    poll_interval: int = 60,
    job_timeout: int = 0,
) -> StepExecutor:
    """
    Factory function to create the appropriate executor for a location.

    Parameters
    ----------
    location_name : str
        Name of the location ("local", "cpu", "gpu")
    location_config : LocationConfig
        Configuration for the location
    workflow_dir : Path
        Local workflow directory for log collection
    poll_interval : int
        Seconds between job status checks (for remote)
    job_timeout : int
        Maximum time to wait for job completion (for remote)

    Returns
    -------
    StepExecutor
        Appropriate executor instance
    """
    if location_config.is_remote:
        return RemoteExecutor(
            location_config,
            workflow_dir,
            poll_interval=poll_interval,
            job_timeout=job_timeout,
        )
    else:
        return LocalExecutor(location_config, workflow_dir)
