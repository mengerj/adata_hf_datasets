#!/usr/bin/env python3
"""
Log Retrieval Utilities for Remote Workflow Execution

This module provides utilities for retrieving and consolidating logs
from remote execution locations to a local workflow directory.

Features:
- rsync-based log retrieval with progress tracking
- Log parsing for errors and warnings
- Consolidated log summary generation
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .data_transfer import LocationConfig

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A single log entry extracted from logs."""

    timestamp: Optional[str]
    level: str  # "INFO", "WARNING", "ERROR"
    message: str
    source_file: str
    line_number: Optional[int] = None


@dataclass
class LogSummary:
    """Summary of logs from a workflow step."""

    step_name: str
    location: str
    success: bool
    total_lines: int
    error_count: int
    warning_count: int
    errors: List[LogEntry] = field(default_factory=list)
    warnings: List[LogEntry] = field(default_factory=list)
    execution_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_name": self.step_name,
            "location": self.location,
            "success": self.success,
            "total_lines": self.total_lines,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [{"level": e.level, "message": e.message} for e in self.errors],
            "warnings": [
                {"level": w.level, "message": w.message} for w in self.warnings
            ],
            "execution_time": self.execution_time,
        }


class LogRetriever:
    """
    Handles retrieving logs from remote locations.

    This class provides methods to:
    - Retrieve log files from remote hosts via rsync
    - Parse logs for errors and warnings
    - Generate consolidated log summaries
    """

    def __init__(self, local_workflow_dir: Path):
        """
        Initialize the log retriever.

        Parameters
        ----------
        local_workflow_dir : Path
            Local directory where logs should be collected
        """
        self.local_workflow_dir = local_workflow_dir
        self.local_workflow_dir.mkdir(parents=True, exist_ok=True)

    def retrieve_file(
        self,
        location_config: LocationConfig,
        remote_path: str,
        local_subdir: Optional[str] = None,
        preserve_name: bool = True,
    ) -> Optional[Path]:
        """
        Retrieve a single file from a remote location.

        Parameters
        ----------
        location_config : LocationConfig
            Configuration for the remote location
        remote_path : str
            Path to the file on the remote host
        local_subdir : Optional[str]
            Subdirectory within workflow_dir to store the file
        preserve_name : bool
            Whether to preserve the original filename

        Returns
        -------
        Optional[Path]
            Path to the retrieved file, or None if retrieval failed
        """
        if not location_config.is_remote:
            logger.warning(
                f"Cannot retrieve from local location: {location_config.name}"
            )
            return None

        # Prepare local destination
        if local_subdir:
            local_dir = self.local_workflow_dir / local_subdir
        else:
            local_dir = self.local_workflow_dir

        local_dir.mkdir(parents=True, exist_ok=True)

        if preserve_name:
            local_path = local_dir / Path(remote_path).name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_path = local_dir / f"retrieved_{timestamp}_{Path(remote_path).name}"

        ssh_target = location_config.ssh_target
        rsync_cmd = [
            "rsync",
            "-avz",
            "--partial",
            "--progress",
            f"{ssh_target}:{remote_path}",
            str(local_path),
        ]

        logger.info(f"Retrieving {remote_path} from {location_config.name}...")

        try:
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Retrieved: {local_path}")
                return local_path
            else:
                logger.warning(f"Failed to retrieve {remote_path}: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout retrieving {remote_path}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving {remote_path}: {e}")
            return None

    def retrieve_directory(
        self,
        location_config: LocationConfig,
        remote_dir: str,
        local_subdir: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Optional[Path]:
        """
        Retrieve an entire directory from a remote location.

        Parameters
        ----------
        location_config : LocationConfig
            Configuration for the remote location
        remote_dir : str
            Path to the directory on the remote host
        local_subdir : Optional[str]
            Subdirectory within workflow_dir to store the files
        exclude_patterns : Optional[List[str]]
            Patterns to exclude from retrieval

        Returns
        -------
        Optional[Path]
            Path to the local directory, or None if retrieval failed
        """
        if not location_config.is_remote:
            logger.warning(
                f"Cannot retrieve from local location: {location_config.name}"
            )
            return None

        # Prepare local destination
        if local_subdir:
            local_dir = self.local_workflow_dir / local_subdir
        else:
            local_dir = self.local_workflow_dir / Path(remote_dir).name

        local_dir.mkdir(parents=True, exist_ok=True)

        ssh_target = location_config.ssh_target
        rsync_cmd = [
            "rsync",
            "-avz",
            "--partial",
            "--progress",
            f"{ssh_target}:{remote_dir}/",
            str(local_dir) + "/",
        ]

        if exclude_patterns:
            for pattern in exclude_patterns:
                rsync_cmd.extend(["--exclude", pattern])

        logger.info(f"Retrieving directory {remote_dir} from {location_config.name}...")

        try:
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info(f"Retrieved directory: {local_dir}")
                return local_dir
            else:
                logger.warning(f"Failed to retrieve {remote_dir}: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout retrieving {remote_dir}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving {remote_dir}: {e}")
            return None

    def retrieve_step_logs(
        self,
        location_config: LocationConfig,
        step_name: str,
        remote_stdout: str,
        remote_stderr: str,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Retrieve stdout and stderr logs for a workflow step.

        Parameters
        ----------
        location_config : LocationConfig
            Configuration for the remote location
        step_name : str
            Name of the workflow step
        remote_stdout : str
            Path to stdout file on remote
        remote_stderr : str
            Path to stderr file on remote

        Returns
        -------
        Tuple[Optional[Path], Optional[Path]]
            Paths to local stdout and stderr files
        """
        step_subdir = step_name.replace(" ", "_").lower()

        stdout_path = self.retrieve_file(
            location_config, remote_stdout, local_subdir=step_subdir
        )
        stderr_path = self.retrieve_file(
            location_config, remote_stderr, local_subdir=step_subdir
        )

        return stdout_path, stderr_path


class LogParser:
    """
    Parse log files to extract errors, warnings, and summaries.
    """

    # Common patterns for log levels (case-insensitive flag applied at compile time)
    ERROR_PATTERNS = [
        r"\b(error|exception|failed|failure|fatal)\b",
        r"traceback \(most recent call last\)",
        r"^\s*(File|Traceback|Error|Exception)",
    ]

    WARNING_PATTERNS = [
        r"\b(warning|warn|deprecated)\b",
    ]

    def __init__(self):
        """Initialize the log parser."""
        self.error_regex = re.compile("|".join(self.ERROR_PATTERNS), re.IGNORECASE)
        self.warning_regex = re.compile("|".join(self.WARNING_PATTERNS), re.IGNORECASE)

    def parse_log_file(self, log_path: Path) -> Tuple[List[LogEntry], List[LogEntry]]:
        """
        Parse a log file for errors and warnings.

        Parameters
        ----------
        log_path : Path
            Path to the log file

        Returns
        -------
        Tuple[List[LogEntry], List[LogEntry]]
            Lists of error and warning entries
        """
        errors = []
        warnings = []

        if not log_path.exists():
            return errors, warnings

        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                if self.error_regex.search(line):
                    entry = LogEntry(
                        timestamp=None,
                        level="ERROR",
                        message=line[:500],  # Truncate long lines
                        source_file=str(log_path),
                        line_number=i,
                    )
                    errors.append(entry)
                elif self.warning_regex.search(line):
                    entry = LogEntry(
                        timestamp=None,
                        level="WARNING",
                        message=line[:500],
                        source_file=str(log_path),
                        line_number=i,
                    )
                    warnings.append(entry)

        except Exception as e:
            logger.error(f"Error parsing log file {log_path}: {e}")

        return errors, warnings

    def create_summary(
        self,
        step_name: str,
        location: str,
        stdout_path: Optional[Path],
        stderr_path: Optional[Path],
        success: bool,
    ) -> LogSummary:
        """
        Create a summary of logs from a workflow step.

        Parameters
        ----------
        step_name : str
            Name of the workflow step
        location : str
            Execution location
        stdout_path : Optional[Path]
            Path to stdout log
        stderr_path : Optional[Path]
            Path to stderr log
        success : bool
            Whether the step succeeded

        Returns
        -------
        LogSummary
            Summary of the logs
        """
        all_errors = []
        all_warnings = []
        total_lines = 0

        # Parse stdout
        if stdout_path and stdout_path.exists():
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                total_lines += len(f.readlines())
            errors, warnings = self.parse_log_file(stdout_path)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

        # Parse stderr
        if stderr_path and stderr_path.exists():
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                total_lines += len(f.readlines())
            errors, warnings = self.parse_log_file(stderr_path)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

        return LogSummary(
            step_name=step_name,
            location=location,
            success=success,
            total_lines=total_lines,
            error_count=len(all_errors),
            warning_count=len(all_warnings),
            errors=all_errors[:10],  # Limit to first 10 errors
            warnings=all_warnings[:10],  # Limit to first 10 warnings
        )


class ConsolidatedLogWriter:
    """
    Write consolidated log summaries to a file.
    """

    def __init__(self, workflow_dir: Path):
        """
        Initialize the consolidated log writer.

        Parameters
        ----------
        workflow_dir : Path
            Workflow directory for log output
        """
        self.workflow_dir = workflow_dir
        self.logs_dir = workflow_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.logs_dir / "workflow_summary.log"
        self.errors_path = self.logs_dir / "errors_consolidated.log"

    def write_step_summary(self, summary: LogSummary) -> None:
        """
        Write a step's log summary to the consolidated log.

        Parameters
        ----------
        summary : LogSummary
            Summary to write
        """
        timestamp = datetime.now().isoformat()

        with open(self.summary_path, "a") as f:
            status = "SUCCESS" if summary.success else "FAILED"
            f.write(f"\n{'=' * 60}\n")
            f.write(f"[{timestamp}] Step: {summary.step_name}\n")
            f.write(f"Location: {summary.location}\n")
            f.write(f"Status: {status}\n")
            f.write(
                f"Errors: {summary.error_count}, Warnings: {summary.warning_count}\n"
            )

            if summary.errors:
                f.write("\nFirst few errors:\n")
                for err in summary.errors[:5]:
                    f.write(
                        f"  - {err.message[:200]}...\n"
                        if len(err.message) > 200
                        else f"  - {err.message}\n"
                    )

            f.write(f"{'=' * 60}\n")

        # Write errors to consolidated error log
        if summary.errors:
            with open(self.errors_path, "a") as f:
                f.write(f"\n[{timestamp}] Step: {summary.step_name}\n")
                for err in summary.errors:
                    f.write(f"  [{err.level}] {err.message}\n")
                    if err.source_file and err.line_number:
                        f.write(f"    Source: {err.source_file}:{err.line_number}\n")

    def write_workflow_start(self, dataset_name: str, workflow_id: str) -> None:
        """Write workflow start marker."""
        timestamp = datetime.now().isoformat()
        with open(self.summary_path, "w") as f:
            f.write(f"{'#' * 60}\n")
            f.write("# WORKFLOW STARTED\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Workflow ID: {workflow_id}\n")
            f.write(f"# Dataset: {dataset_name}\n")
            f.write(f"{'#' * 60}\n\n")

    def write_workflow_complete(self, success: bool, total_time_seconds: float) -> None:
        """Write workflow completion marker."""
        timestamp = datetime.now().isoformat()
        status = "COMPLETED SUCCESSFULLY" if success else "COMPLETED WITH ERRORS"

        with open(self.summary_path, "a") as f:
            f.write(f"\n{'#' * 60}\n")
            f.write(f"# WORKFLOW {status}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Total time: {total_time_seconds:.2f} seconds\n")
            f.write(f"{'#' * 60}\n")


def retrieve_and_summarize_step_logs(
    location_config: LocationConfig,
    workflow_dir: Path,
    step_name: str,
    remote_stdout: str,
    remote_stderr: str,
    success: bool,
) -> LogSummary:
    """
    Retrieve logs from remote and create a summary.

    This is a convenience function that combines log retrieval and parsing.

    Parameters
    ----------
    location_config : LocationConfig
        Configuration for the remote location
    workflow_dir : Path
        Local workflow directory
    step_name : str
        Name of the workflow step
    remote_stdout : str
        Path to stdout on remote
    remote_stderr : str
        Path to stderr on remote
    success : bool
        Whether the step succeeded

    Returns
    -------
    LogSummary
        Summary of the step's logs
    """
    retriever = LogRetriever(workflow_dir)
    parser = LogParser()

    # Retrieve logs
    local_stdout, local_stderr = retriever.retrieve_step_logs(
        location_config, step_name, remote_stdout, remote_stderr
    )

    # Create summary
    summary = parser.create_summary(
        step_name=step_name,
        location=location_config.name,
        stdout_path=local_stdout,
        stderr_path=local_stderr,
        success=success,
    )

    return summary
