#!/usr/bin/env python3
"""
Data Transfer Module for Multi-Location Workflow

This module provides robust data transfer functionality between execution locations
(local, cpu cluster, gpu cluster) with support for:
- Automatic compression of zarr directories (many small files)
- Progress tracking via rsync
- Remote-to-remote transfers via SSH agent forwarding or local intermediary
- Integrity verification using checksums
"""

import hashlib
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DataTransferError(Exception):
    """Custom exception for data transfer operations."""

    pass


@dataclass
class TransferStats:
    """Statistics from a data transfer operation."""

    source_location: str
    target_location: str
    source_path: str
    target_path: str
    total_time_seconds: float
    compression_used: bool = False
    compression_time_seconds: float = 0.0
    transfer_time_seconds: float = 0.0
    decompression_time_seconds: float = 0.0
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    compression_ratio: float = 1.0
    throughput_mbps: float = 0.0
    verified: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "source_location": self.source_location,
            "target_location": self.target_location,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "compression_used": self.compression_used,
            "compression_time_seconds": round(self.compression_time_seconds, 2),
            "transfer_time_seconds": round(self.transfer_time_seconds, 2),
            "decompression_time_seconds": round(self.decompression_time_seconds, 2),
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": round(self.compression_ratio, 3),
            "throughput_mbps": round(self.throughput_mbps, 2),
            "verified": self.verified,
            "errors": self.errors,
        }


@dataclass
class LocationConfig:
    """Configuration for a single execution location."""

    name: str  # "local", "cpu", or "gpu"
    base_file_path: str
    project_directory: str
    venv_path: str
    output_directory: str
    # Remote-specific fields (None for local)
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    slurm_partition: Optional[str] = None
    node: Optional[str] = None

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote location (requires SSH)."""
        return self.ssh_host is not None

    @property
    def ssh_target(self) -> Optional[str]:
        """Get the SSH target string (user@host) or None for local."""
        if self.is_remote:
            return f"{self.ssh_user}@{self.ssh_host}"
        return None

    @classmethod
    def from_config(cls, name: str, config: DictConfig) -> "LocationConfig":
        """Create LocationConfig from OmegaConf config."""
        return cls(
            name=name,
            base_file_path=str(config.get("base_file_path", "")),
            project_directory=str(config.get("project_directory", ".")),
            venv_path=str(config.get("venv_path", ".venv")),
            output_directory=str(config.get("output_directory", "./outputs")),
            ssh_host=config.get("ssh_host"),
            ssh_user=config.get("ssh_user"),
            slurm_partition=config.get("slurm_partition"),
            node=config.get("node"),
        )


class DataTransfer:
    """
    Handles data transfer between execution locations with progress tracking.

    Supports:
    - Local to remote (local -> cpu, local -> gpu)
    - Remote to local (cpu -> local, gpu -> local)
    - Remote to remote (cpu -> gpu, gpu -> cpu)
    """

    def __init__(
        self,
        locations: Dict[str, LocationConfig],
        transfer_config: Optional[DictConfig] = None,
    ):
        """
        Initialize the data transfer handler.

        Parameters
        ----------
        locations : Dict[str, LocationConfig]
            Dictionary mapping location names to their configurations
        transfer_config : Optional[DictConfig]
            Transfer-specific configuration (compression, temp_dir, etc.)
        """
        self.locations = locations
        self.transfer_config = transfer_config or {}

        # Extract transfer settings with defaults
        self.compression_enabled = self.transfer_config.get("compression", True)
        self.compression_level = self.transfer_config.get("compression_level", 6)
        self.verify_integrity = self.transfer_config.get("verify_integrity", True)
        self.remote_to_remote_via_local = self.transfer_config.get(
            "remote_to_remote_via_local", False
        )
        self.temp_dir = self.transfer_config.get("temp_dir", "/tmp/workflow_transfer")
        self.cleanup_temp = self.transfer_config.get("cleanup_temp", True)
        self.rsync_options = self.transfer_config.get(
            "rsync_options",
            ["--archive", "--compress", "--partial", "--progress", "--human-readable"],
        )

        # Validate SSH availability for remote locations
        self._validate_ssh()

    def _validate_ssh(self) -> None:
        """Validate that SSH is available for remote transfers."""
        try:
            subprocess.run(["ssh", "-V"], capture_output=True, check=True, timeout=5)
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.warning("SSH command not available. Remote transfers will not work.")

    def _run_local_command(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        timeout: int = 3600,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command locally."""
        logger.debug(f"Running local command: {' '.join(command)}")
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )
            return result
        except subprocess.TimeoutExpired:
            raise DataTransferError(
                f"Command timed out after {timeout}s: {' '.join(command)}"
            )

    def _run_ssh_command(
        self,
        location: LocationConfig,
        command: str,
        timeout: int = 3600,
    ) -> subprocess.CompletedProcess:
        """Run a command on a remote location via SSH."""
        if not location.is_remote:
            raise DataTransferError(
                f"Cannot run SSH command on local location: {location.name}"
            )

        ssh_cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            location.ssh_target,
            command,
        ]

        logger.debug(f"Running SSH command on {location.name}: {command}")
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result
        except subprocess.TimeoutExpired:
            raise DataTransferError(
                f"SSH command timed out after {timeout}s on {location.name}: {command}"
            )

    def _get_path_size(self, location: LocationConfig, path: str) -> int:
        """Get the size of a file or directory at a location."""
        if location.is_remote:
            # Use du for directories, stat for files
            cmd = f"if [ -d '{path}' ]; then du -sb '{path}' | cut -f1; else stat -c %s '{path}' 2>/dev/null || echo 0; fi"
            result = self._run_ssh_command(location, cmd, timeout=300)
            if result.returncode != 0:
                logger.warning(f"Failed to get size of {path}: {result.stderr}")
                return 0
            try:
                return int(result.stdout.strip())
            except ValueError:
                return 0
        else:
            path_obj = Path(path)
            if path_obj.is_dir():
                total = 0
                for f in path_obj.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
                return total
            elif path_obj.is_file():
                return path_obj.stat().st_size
            return 0

    def _path_exists(self, location: LocationConfig, path: str) -> bool:
        """Check if a path exists at a location."""
        if location.is_remote:
            result = self._run_ssh_command(location, f"test -e '{path}'", timeout=30)
            return result.returncode == 0
        else:
            return Path(path).exists()

    def _is_zarr_directory(self, location: LocationConfig, path: str) -> bool:
        """Check if a path is a zarr directory."""
        if location.is_remote:
            # Check for .zarray or .zattrs files which indicate zarr format
            result = self._run_ssh_command(
                location,
                f"test -f '{path}/.zarray' -o -f '{path}/.zattrs' -o -f '{path}/.zgroup'",
                timeout=30,
            )
            return result.returncode == 0
        else:
            path_obj = Path(path)
            return path_obj.is_dir() and (
                (path_obj / ".zarray").exists()
                or (path_obj / ".zattrs").exists()
                or (path_obj / ".zgroup").exists()
            )

    def _ensure_directory(self, location: LocationConfig, path: str) -> None:
        """Ensure a directory exists at a location."""
        if location.is_remote:
            result = self._run_ssh_command(location, f"mkdir -p '{path}'", timeout=60)
            if result.returncode != 0:
                raise DataTransferError(
                    f"Failed to create directory {path} on {location.name}: {result.stderr}"
                )
        else:
            Path(path).mkdir(parents=True, exist_ok=True)

    def _compress_directory(
        self,
        location: LocationConfig,
        source_path: str,
        archive_path: str,
    ) -> float:
        """
        Compress a directory to a tar.gz archive.

        Returns the compression time in seconds.
        """
        start_time = time.time()
        source_dir = Path(source_path)
        parent_dir = source_dir.parent
        dir_name = source_dir.name

        # Use pigz for parallel compression if available, fallback to gzip
        if location.is_remote:
            # Check if pigz is available
            pigz_check = self._run_ssh_command(location, "which pigz", timeout=10)
            use_pigz = pigz_check.returncode == 0

            if use_pigz:
                compress_cmd = f"cd '{parent_dir}' && tar -cf - '{dir_name}' | pigz -{self.compression_level} > '{archive_path}'"
            else:
                compress_cmd = f"cd '{parent_dir}' && tar -czf '{archive_path}' --warning=no-file-changed '{dir_name}'"

            result = self._run_ssh_command(
                location, compress_cmd, timeout=7200
            )  # 2 hours
            if (
                result.returncode != 0
                and "file changed as we read it" not in result.stderr
            ):
                raise DataTransferError(
                    f"Failed to compress {source_path} on {location.name}: {result.stderr}"
                )
        else:
            # Local compression
            import tarfile

            with tarfile.open(
                archive_path, "w:gz", compresslevel=self.compression_level
            ) as tar:
                tar.add(source_path, arcname=dir_name)

        return time.time() - start_time

    def _decompress_archive(
        self,
        location: LocationConfig,
        archive_path: str,
        target_dir: str,
    ) -> float:
        """
        Decompress a tar.gz archive to a directory.

        Returns the decompression time in seconds.
        """
        start_time = time.time()

        if location.is_remote:
            # Check if pigz is available for faster decompression
            pigz_check = self._run_ssh_command(location, "which pigz", timeout=10)
            use_pigz = pigz_check.returncode == 0

            if use_pigz:
                decompress_cmd = (
                    f"cd '{target_dir}' && pigz -dc '{archive_path}' | tar -xf -"
                )
            else:
                decompress_cmd = f"cd '{target_dir}' && tar -xzf '{archive_path}'"

            result = self._run_ssh_command(location, decompress_cmd, timeout=7200)
            if result.returncode != 0:
                raise DataTransferError(
                    f"Failed to decompress {archive_path} on {location.name}: {result.stderr}"
                )
        else:
            import tarfile

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir)

        return time.time() - start_time

    def _rsync_transfer(
        self,
        source_location: LocationConfig,
        source_path: str,
        target_location: LocationConfig,
        target_path: str,
    ) -> Tuple[float, str]:
        """
        Transfer using rsync.

        Returns (transfer_time, rsync_output).
        """
        start_time = time.time()

        # Build rsync command
        rsync_cmd = ["rsync"] + list(self.rsync_options)

        # Determine source and target strings
        if source_location.is_remote and target_location.is_remote:
            # Remote to remote - need special handling
            return self._rsync_remote_to_remote(
                source_location, source_path, target_location, target_path
            )

        if source_location.is_remote:
            source_str = f"{source_location.ssh_target}:{source_path}"
        else:
            source_str = source_path

        if target_location.is_remote:
            target_str = f"{target_location.ssh_target}:{target_path}"
        else:
            target_str = target_path

        rsync_cmd.extend([source_str, target_str])

        logger.info(f"Running rsync: {' '.join(rsync_cmd)}")

        # Run rsync with real-time output
        process = subprocess.Popen(
            rsync_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout_lines = []
        # Stream output for progress tracking
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                stdout_lines.append(line)
                # Log progress lines
                if "%" in line or "to-check" in line:
                    logger.info(f"rsync progress: {line.strip()}")

        stderr = process.stderr.read()
        return_code = process.poll()

        if return_code != 0:
            raise DataTransferError(f"rsync failed with code {return_code}: {stderr}")

        transfer_time = time.time() - start_time
        return transfer_time, "".join(stdout_lines)

    def _rsync_remote_to_remote(
        self,
        source_location: LocationConfig,
        source_path: str,
        target_location: LocationConfig,
        target_path: str,
    ) -> Tuple[float, str]:
        """
        Handle remote-to-remote transfers.

        Either via SSH agent forwarding or through local as intermediary.
        """
        start_time = time.time()

        if self.remote_to_remote_via_local:
            # Transfer via local intermediary
            logger.info(
                f"Remote-to-remote transfer via local: {source_location.name} -> local -> {target_location.name}"
            )

            # Create local temp directory
            local_temp = (
                Path(self.temp_dir) / f"r2r_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            local_temp.mkdir(parents=True, exist_ok=True)

            try:
                # Download from source
                logger.info(f"Downloading from {source_location.name}...")
                download_cmd = (
                    ["rsync"]
                    + list(self.rsync_options)
                    + [
                        f"{source_location.ssh_target}:{source_path}",
                        str(local_temp) + "/",
                    ]
                )
                result = subprocess.run(download_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise DataTransferError(f"Download failed: {result.stderr}")

                # Upload to target
                logger.info(f"Uploading to {target_location.name}...")
                upload_cmd = (
                    ["rsync"]
                    + list(self.rsync_options)
                    + [
                        str(local_temp) + "/",
                        f"{target_location.ssh_target}:{target_path}",
                    ]
                )
                result = subprocess.run(upload_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise DataTransferError(f"Upload failed: {result.stderr}")

                output = "Remote-to-remote transfer via local completed"
            finally:
                # Cleanup local temp
                if self.cleanup_temp and local_temp.exists():
                    shutil.rmtree(local_temp)
        else:
            # Direct transfer using SSH agent forwarding
            logger.info(
                f"Remote-to-remote transfer via SSH agent: {source_location.name} -> {target_location.name}"
            )

            # Run rsync on source host to push to target
            rsync_opts = " ".join(self.rsync_options)
            ssh_cmd = (
                f"rsync {rsync_opts} '{source_path}' "
                f"'{target_location.ssh_target}:{target_path}'"
            )

            # Use SSH with agent forwarding
            full_cmd = [
                "ssh",
                "-A",  # Enable agent forwarding
                "-o",
                "ConnectTimeout=30",
                source_location.ssh_target,
                ssh_cmd,
            ]

            logger.info(f"Running remote rsync: {' '.join(full_cmd)}")
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=7200
            )

            if result.returncode != 0:
                raise DataTransferError(
                    f"Remote-to-remote rsync failed: {result.stderr}"
                )
            output = result.stdout

        transfer_time = time.time() - start_time
        return transfer_time, output

    def _compute_checksum(self, location: LocationConfig, path: str) -> Optional[str]:
        """Compute MD5 checksum of a file."""
        if location.is_remote:
            result = self._run_ssh_command(
                location, f"md5sum '{path}' | cut -d' ' -f1", timeout=600
            )
            if result.returncode == 0:
                return result.stdout.strip()
        else:
            if Path(path).exists():
                hash_md5 = hashlib.md5()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                return hash_md5.hexdigest()
        return None

    def _cleanup_file(self, location: LocationConfig, path: str) -> None:
        """Remove a file at a location."""
        if location.is_remote:
            self._run_ssh_command(location, f"rm -f '{path}'", timeout=60)
        else:
            path_obj = Path(path)
            if path_obj.exists():
                path_obj.unlink()

    def translate_path(
        self,
        path: str,
        source_location: str,
        target_location: str,
    ) -> str:
        """
        Translate a path from one location's base to another's.

        Parameters
        ----------
        path : str
            The path to translate
        source_location : str
            Name of the source location
        target_location : str
            Name of the target location

        Returns
        -------
        str
            The translated path for the target location
        """
        source_config = self.locations[source_location]
        target_config = self.locations[target_location]

        source_base = Path(source_config.base_file_path)
        target_base = Path(target_config.base_file_path)

        path_obj = Path(path)

        # Try to make the path relative to the source base
        try:
            relative = path_obj.relative_to(source_base)
            return str(target_base / relative)
        except ValueError:
            # Path is not under source base, return as-is with target base prefix
            logger.warning(
                f"Path {path} is not under source base {source_base}, "
                f"returning original path under target base"
            )
            return str(target_base / path_obj.name)

    def transfer(
        self,
        source_location: str,
        target_location: str,
        source_path: Union[str, Path],
        target_path: Optional[Union[str, Path]] = None,
        use_compression: Optional[bool] = None,
    ) -> TransferStats:
        """
        Transfer data between locations.

        Parameters
        ----------
        source_location : str
            Name of the source location ("local", "cpu", or "gpu")
        target_location : str
            Name of the target location ("local", "cpu", or "gpu")
        source_path : Union[str, Path]
            Path to the source file or directory
        target_path : Optional[Union[str, Path]]
            Path for the target. If None, auto-translated from source_path
        use_compression : Optional[bool]
            Whether to use compression. If None, uses config default
            (compression is recommended for zarr directories)

        Returns
        -------
        TransferStats
            Statistics about the transfer operation
        """
        start_time = time.time()

        # Validate locations
        if source_location not in self.locations:
            raise DataTransferError(f"Unknown source location: {source_location}")
        if target_location not in self.locations:
            raise DataTransferError(f"Unknown target location: {target_location}")
        if source_location == target_location:
            raise DataTransferError(
                f"Source and target locations are the same: {source_location}"
            )

        source_config = self.locations[source_location]
        target_config = self.locations[target_location]

        source_path = str(source_path)
        if target_path is None:
            target_path = self.translate_path(
                source_path, source_location, target_location
            )
        else:
            target_path = str(target_path)

        logger.info(
            f"Starting transfer: {source_location}:{source_path} -> {target_location}:{target_path}"
        )

        # Initialize stats
        stats = TransferStats(
            source_location=source_location,
            target_location=target_location,
            source_path=source_path,
            target_path=target_path,
            total_time_seconds=0,
        )

        # Check source exists
        if not self._path_exists(source_config, source_path):
            raise DataTransferError(f"Source path does not exist: {source_path}")

        # Get original size
        stats.original_size_bytes = self._get_path_size(source_config, source_path)
        logger.info(f"Source size: {stats.original_size_bytes / (1024**2):.2f} MB")

        # Determine if compression should be used
        if use_compression is None:
            # Auto-detect: use compression for zarr directories
            use_compression = self.compression_enabled and self._is_zarr_directory(
                source_config, source_path
            )

        # Ensure target parent directory exists
        target_parent = str(Path(target_path).parent)
        self._ensure_directory(target_config, target_parent)

        if use_compression:
            stats.compression_used = True
            logger.info("Using compression for transfer (zarr directory detected)")

            # Generate archive names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"transfer_{Path(source_path).name}_{timestamp}.tar.gz"

            # Determine temp directories
            if source_config.is_remote:
                source_temp_dir = "/tmp/workflow_transfer"
                self._run_ssh_command(
                    source_config, f"mkdir -p '{source_temp_dir}'", timeout=30
                )
            else:
                source_temp_dir = self.temp_dir
                Path(source_temp_dir).mkdir(parents=True, exist_ok=True)

            if target_config.is_remote:
                target_temp_dir = "/tmp/workflow_transfer"
                self._run_ssh_command(
                    target_config, f"mkdir -p '{target_temp_dir}'", timeout=30
                )
            else:
                target_temp_dir = self.temp_dir
                Path(target_temp_dir).mkdir(parents=True, exist_ok=True)

            source_archive = f"{source_temp_dir}/{archive_name}"
            target_archive = f"{target_temp_dir}/{archive_name}"

            try:
                # Step 1: Compress on source
                logger.info(f"Compressing {source_path} on {source_location}...")
                stats.compression_time_seconds = self._compress_directory(
                    source_config, source_path, source_archive
                )
                logger.info(f"Compression took {stats.compression_time_seconds:.2f}s")

                # Get compressed size
                stats.compressed_size_bytes = self._get_path_size(
                    source_config, source_archive
                )
                if stats.original_size_bytes > 0:
                    stats.compression_ratio = (
                        stats.compressed_size_bytes / stats.original_size_bytes
                    )
                logger.info(
                    f"Compressed size: {stats.compressed_size_bytes / (1024**2):.2f} MB "
                    f"(ratio: {stats.compression_ratio:.2%})"
                )

                # Step 2: Transfer archive
                logger.info(f"Transferring archive to {target_location}...")
                stats.transfer_time_seconds, _ = self._rsync_transfer(
                    source_config, source_archive, target_config, target_archive
                )
                logger.info(f"Transfer took {stats.transfer_time_seconds:.2f}s")

                # Step 3: Decompress on target
                logger.info(f"Decompressing on {target_location}...")
                stats.decompression_time_seconds = self._decompress_archive(
                    target_config, target_archive, target_parent
                )
                logger.info(
                    f"Decompression took {stats.decompression_time_seconds:.2f}s"
                )

                # Handle potential name mismatch after extraction
                extracted_path = str(Path(target_parent) / Path(source_path).name)
                if extracted_path != target_path:
                    # Rename to target path
                    if target_config.is_remote:
                        self._run_ssh_command(
                            target_config,
                            f"mv '{extracted_path}' '{target_path}'",
                            timeout=300,
                        )
                    else:
                        Path(extracted_path).rename(target_path)

                # Verify integrity if enabled
                if self.verify_integrity:
                    logger.info("Verifying transfer integrity...")
                    # Compare sizes as a basic check
                    target_size = self._get_path_size(target_config, target_path)
                    if (
                        abs(target_size - stats.original_size_bytes)
                        / max(stats.original_size_bytes, 1)
                        < 0.01
                    ):
                        stats.verified = True
                        logger.info("Transfer verified successfully")
                    else:
                        stats.errors.append(
                            f"Size mismatch: source={stats.original_size_bytes}, target={target_size}"
                        )
                        logger.warning("Transfer verification failed: size mismatch")

            finally:
                # Cleanup temp files
                if self.cleanup_temp:
                    logger.info("Cleaning up temporary files...")
                    self._cleanup_file(source_config, source_archive)
                    self._cleanup_file(target_config, target_archive)

        else:
            # Direct rsync without compression
            logger.info("Using direct rsync transfer")
            stats.transfer_time_seconds, _ = self._rsync_transfer(
                source_config, source_path, target_config, target_path
            )

            if self.verify_integrity:
                target_size = self._get_path_size(target_config, target_path)
                if (
                    abs(target_size - stats.original_size_bytes)
                    / max(stats.original_size_bytes, 1)
                    < 0.01
                ):
                    stats.verified = True

        # Calculate final stats
        stats.total_time_seconds = time.time() - start_time
        if stats.total_time_seconds > 0 and stats.original_size_bytes > 0:
            stats.throughput_mbps = (
                stats.original_size_bytes / (1024 * 1024)
            ) / stats.total_time_seconds

        logger.info(
            f"Transfer completed in {stats.total_time_seconds:.2f}s "
            f"({stats.throughput_mbps:.2f} MB/s)"
        )

        return stats

    def test_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to all configured locations.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping location names to connectivity status
        """
        results = {}

        for name, config in self.locations.items():
            if config.is_remote:
                try:
                    result = self._run_ssh_command(
                        config, "echo 'connectivity test'", timeout=30
                    )
                    results[name] = result.returncode == 0
                    if results[name]:
                        logger.info(
                            f"✓ Connectivity to {name} ({config.ssh_target}) OK"
                        )
                    else:
                        logger.warning(
                            f"✗ Connectivity to {name} ({config.ssh_target}) FAILED: {result.stderr}"
                        )
                except Exception as e:
                    results[name] = False
                    logger.warning(f"✗ Connectivity to {name} FAILED: {e}")
            else:
                # Local is always "connected"
                results[name] = True
                logger.info(f"✓ Local location '{name}' OK")

        return results


def create_data_transfer_from_config(workflow_config: DictConfig) -> DataTransfer:
    """
    Create a DataTransfer instance from workflow configuration.

    Parameters
    ----------
    workflow_config : DictConfig
        The workflow configuration containing locations and transfer settings

    Returns
    -------
    DataTransfer
        Configured DataTransfer instance
    """
    locations_config = workflow_config.get("locations", {})
    transfer_config = workflow_config.get("transfer", {})

    locations = {}
    for name, loc_config in locations_config.items():
        locations[name] = LocationConfig.from_config(name, loc_config)

    return DataTransfer(locations, transfer_config)
