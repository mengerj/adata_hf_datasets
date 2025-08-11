#!/usr/bin/env python3
"""
SSH Transfer Utilities for Inter-Cluster File Operations

This module provides utility functions for efficiently transferring files
between /scratch/local directories on different SSH-connected clusters.
"""

import logging
import subprocess
import time
import os
from pathlib import Path
from typing import Dict, Optional, List, Union
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)


class SSHTransferError(Exception):
    """Custom exception for SSH transfer operations."""

    pass


class SSHTransferUtils:
    """
    Utility class for SSH-based file transfers between clusters.
    """

    def __init__(self, cpu_login: Dict[str, str], gpu_login: Dict[str, str]):
        """
        Initialize SSH transfer utilities.

        Parameters
        ----------
        cpu_login : Dict[str, str]
            Dictionary with 'host' and 'user' keys for CPU cluster
        gpu_login : Dict[str, str]
            Dictionary with 'host' and 'user' keys for GPU cluster
        """
        self.cpu_login = cpu_login
        self.gpu_login = gpu_login

        # Validate login configurations
        for name, login in [("cpu_login", cpu_login), ("gpu_login", gpu_login)]:
            if (
                not isinstance(login, dict)
                or "host" not in login
                or "user" not in login
            ):
                raise ValueError(
                    f"{name} must be a dictionary with 'host' and 'user' keys"
                )

    def _get_ssh_prefix(self, login: Dict[str, str]) -> str:
        """Get SSH connection prefix for a login configuration."""
        return f"{login['user']}@{login['host']}"

    def _run_ssh_command(
        self, host: str, command: str, timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """
        Run a command on a remote host via SSH.

        Parameters
        ----------
        host : str
            SSH host (in format user@hostname)
        command : str
            Command to execute
        timeout : int
            Command timeout in seconds

        Returns
        -------
        subprocess.CompletedProcess
            Command result
        """
        cmd = ["ssh", "-o", "ConnectTimeout=10", host, command]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
            return result
        except subprocess.TimeoutExpired:
            raise SSHTransferError(
                f"SSH command timed out after {timeout} seconds: {command}"
            )
        except Exception as e:
            raise SSHTransferError(f"SSH command failed: {e}")

    def test_ssh_connectivity(self) -> Dict[str, bool]:
        """
        Test SSH connectivity to both clusters.

        Returns
        -------
        Dict[str, bool]
            Connectivity status for each cluster
        """
        results = {}

        for cluster_name, login in [("cpu", self.cpu_login), ("gpu", self.gpu_login)]:
            try:
                host = self._get_ssh_prefix(login)
                result = self._run_ssh_command(
                    host, "echo 'SSH test successful'", timeout=10
                )
                results[cluster_name] = result.returncode == 0

                if result.returncode == 0:
                    logger.info(
                        f"✓ SSH connectivity to {cluster_name} cluster ({host}) successful"
                    )
                else:
                    logger.warning(
                        f"✗ SSH connectivity to {cluster_name} cluster ({host}) failed: {result.stderr}"
                    )

            except Exception as e:
                logger.warning(
                    f"✗ SSH connectivity to {cluster_name} cluster failed: {e}"
                )
                results[cluster_name] = False

        return results

    def get_file_size(self, cluster: str, file_path: str) -> int:
        """
        Get the size of a file on a remote cluster.

        Parameters
        ----------
        cluster : str
            Cluster name ('cpu' or 'gpu')
        file_path : str
            Path to the file on the remote cluster

        Returns
        -------
        int
            File size in bytes
        """
        login = self.cpu_login if cluster == "cpu" else self.gpu_login
        host = self._get_ssh_prefix(login)

        # Use stat command to get file size
        command = f"stat -c %s '{file_path}'"
        result = self._run_ssh_command(host, command)

        if result.returncode != 0:
            raise SSHTransferError(
                f"Failed to get file size for {file_path}: {result.stderr}"
            )

        try:
            return int(result.stdout.strip())
        except ValueError:
            raise SSHTransferError(f"Invalid file size output: {result.stdout.strip()}")

    def check_file_exists(self, cluster: str, file_path: str) -> bool:
        """
        Check if a file exists on a remote cluster.

        Parameters
        ----------
        cluster : str
            Cluster name ('cpu' or 'gpu')
        file_path : str
            Path to check

        Returns
        -------
        bool
            True if file exists, False otherwise
        """
        login = self.cpu_login if cluster == "cpu" else self.gpu_login
        host = self._get_ssh_prefix(login)

        command = f"test -e '{file_path}'"
        result = self._run_ssh_command(host, command)

        return result.returncode == 0

    def get_disk_usage(
        self, cluster: str, directory: str = "/scratch/local"
    ) -> Dict[str, int]:
        """
        Get disk usage information for a directory on a remote cluster.

        Parameters
        ----------
        cluster : str
            Cluster name ('cpu' or 'gpu')
        directory : str
            Directory to check (default: /scratch/local)

        Returns
        -------
        Dict[str, int]
            Dictionary with 'total', 'used', and 'available' space in bytes
        """
        login = self.cpu_login if cluster == "cpu" else self.gpu_login
        host = self._get_ssh_prefix(login)

        command = f"df -B1 '{directory}' | tail -1"
        result = self._run_ssh_command(host, command)

        if result.returncode != 0:
            raise SSHTransferError(
                f"Failed to get disk usage for {directory}: {result.stderr}"
            )

        try:
            # Parse df output: filesystem total used available use% mountpoint
            parts = result.stdout.strip().split()
            return {
                "total": int(parts[1]),
                "used": int(parts[2]),
                "available": int(parts[3]),
            }
        except (ValueError, IndexError):
            raise SSHTransferError(
                f"Failed to parse disk usage output: {result.stdout}"
            )

    def ensure_directory_exists(self, cluster: str, directory: str) -> None:
        """
        Ensure a directory exists on a remote cluster.

        Parameters
        ----------
        cluster : str
            Cluster name ('cpu' or 'gpu')
        directory : str
            Directory path to create
        """
        login = self.cpu_login if cluster == "cpu" else self.gpu_login
        host = self._get_ssh_prefix(login)

        command = f"mkdir -p '{directory}'"
        result = self._run_ssh_command(host, command)

        if result.returncode != 0:
            raise SSHTransferError(
                f"Failed to create directory {directory}: {result.stderr}"
            )

    def transfer_file(
        self,
        source_cluster: str,
        source_path: str,
        dest_cluster: str,
        dest_path: str,
        compress: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Transfer a file between clusters via local staging.

        Parameters
        ----------
        source_cluster : str
            Source cluster name ('cpu' or 'gpu')
        source_path : str
            Source file path
        dest_cluster : str
            Destination cluster name ('cpu' or 'gpu')
        dest_path : str
            Destination file path
        compress : bool
            Whether to use compression during transfer
        progress_callback : Optional[callable]
            Callback function for progress updates

        Returns
        -------
        Dict[str, Union[float, int]]
            Transfer statistics including duration and throughput
        """
        if source_cluster == dest_cluster:
            raise ValueError("Source and destination clusters cannot be the same")

        start_time = time.time()

        # Get source and destination SSH prefixes
        source_login = self.cpu_login if source_cluster == "cpu" else self.gpu_login
        dest_login = self.cpu_login if dest_cluster == "cpu" else self.gpu_login

        source_host = self._get_ssh_prefix(source_login)
        dest_host = self._get_ssh_prefix(dest_login)

        # Check if source file exists
        if not self.check_file_exists(source_cluster, source_path):
            raise SSHTransferError(f"Source file does not exist: {source_path}")

        # Get file size for progress tracking
        file_size = self.get_file_size(source_cluster, source_path)

        # Create destination directory if needed
        dest_dir = str(Path(dest_path).parent)
        self.ensure_directory_exists(dest_cluster, dest_dir)

        # Use temporary directory for staging
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / Path(source_path).name

            # Download from source cluster
            logger.info(f"Downloading {source_path} from {source_cluster} cluster...")
            scp_cmd = ["scp"]
            if compress:
                scp_cmd.extend(["-C"])  # Enable compression
            scp_cmd.extend([f"{source_host}:{source_path}", str(temp_file)])

            download_start = time.time()
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            download_time = time.time() - download_start

            if result.returncode != 0:
                raise SSHTransferError(f"Failed to download file: {result.stderr}")

            # Upload to destination cluster
            logger.info(f"Uploading {dest_path} to {dest_cluster} cluster...")
            scp_cmd = ["scp"]
            if compress:
                scp_cmd.extend(["-C"])  # Enable compression
            scp_cmd.extend([str(temp_file), f"{dest_host}:{dest_path}"])

            upload_start = time.time()
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            upload_time = time.time() - upload_start

            if result.returncode != 0:
                raise SSHTransferError(f"Failed to upload file: {result.stderr}")

        total_time = time.time() - start_time

        # Calculate transfer statistics
        stats = {
            "file_size_bytes": file_size,
            "total_time_seconds": total_time,
            "download_time_seconds": download_time,
            "upload_time_seconds": upload_time,
            "throughput_mbps": (file_size / (1024 * 1024)) / total_time
            if total_time > 0
            else 0,
            "compression_enabled": compress,
        }

        logger.info(
            f"Transfer completed in {total_time:.2f}s ({stats['throughput_mbps']:.2f} MB/s)"
        )

        return stats

    def transfer_directory(
        self,
        source_cluster: str,
        source_path: str,
        dest_cluster: str,
        dest_path: str,
        compress: bool = True,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Union[float, int]]:
        """
        Transfer a directory between clusters via rsync.

        Parameters
        ----------
        source_cluster : str
            Source cluster name ('cpu' or 'gpu')
        source_path : str
            Source directory path
        dest_cluster : str
            Destination cluster name ('cpu' or 'gpu')
        dest_path : str
            Destination directory path
        compress : bool
            Whether to use compression during transfer
        exclude_patterns : Optional[List[str]]
            List of patterns to exclude from transfer

        Returns
        -------
        Dict[str, Union[float, int]]
            Transfer statistics
        """
        if source_cluster == dest_cluster:
            raise ValueError("Source and destination clusters cannot be the same")

        start_time = time.time()

        # Get source and destination SSH prefixes
        source_login = self.cpu_login if source_cluster == "cpu" else self.gpu_login
        dest_login = self.cpu_login if dest_cluster == "cpu" else self.gpu_login

        source_host = self._get_ssh_prefix(source_login)
        dest_host = self._get_ssh_prefix(dest_login)

        # Check if source directory exists
        if not self.check_file_exists(source_cluster, source_path):
            raise SSHTransferError(f"Source directory does not exist: {source_path}")

        # Create destination directory
        self.ensure_directory_exists(dest_cluster, dest_path)

        # Use rsync over SSH for directory transfer
        rsync_cmd = [
            "rsync",
            "-avz" if compress else "-av",  # archive mode, verbose, compress if enabled
            "--progress",
            "--stats",
        ]

        # Add exclude patterns if provided
        if exclude_patterns:
            for pattern in exclude_patterns:
                rsync_cmd.extend(["--exclude", pattern])

        # Add source and destination
        rsync_cmd.extend(
            [
                f"{source_host}:{source_path}/",  # Trailing slash to copy contents
                f"{dest_host}:{dest_path}/",
            ]
        )

        logger.info(
            f"Transferring directory {source_path} from {source_cluster} to {dest_cluster}..."
        )
        logger.info(f"Rsync command: {' '.join(rsync_cmd)}")

        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        total_time = time.time() - start_time

        if result.returncode != 0:
            raise SSHTransferError(f"Directory transfer failed: {result.stderr}")

        # Parse rsync stats for transfer information
        stats = {
            "total_time_seconds": total_time,
            "compression_enabled": compress,
            "rsync_output": result.stdout,
        }

        # Try to extract transfer stats from rsync output
        if "Number of files transferred:" in result.stdout:
            for line in result.stdout.split("\n"):
                if "Total file size:" in line:
                    try:
                        size_str = line.split()[-2].replace(",", "")
                        stats["total_size_bytes"] = int(size_str)
                    except (ValueError, IndexError):
                        pass
                elif "sent" in line and "received" in line:
                    try:
                        # Parse "sent X bytes received Y bytes Z bytes/sec"
                        parts = line.split()
                        sent_idx = parts.index("sent") + 1
                        received_idx = parts.index("received") + 1
                        stats["bytes_sent"] = int(parts[sent_idx])
                        stats["bytes_received"] = int(parts[received_idx])
                    except (ValueError, IndexError):
                        pass

        if "total_size_bytes" in stats:
            stats["throughput_mbps"] = (
                (stats["total_size_bytes"] / (1024 * 1024)) / total_time
                if total_time > 0
                else 0
            )

        logger.info(f"Directory transfer completed in {total_time:.2f}s")

        return stats

    def transfer_directory_as_zip(
        self,
        source_cluster: str,
        source_path: str,
        dest_cluster: str,
        dest_path: str,
        temp_zip_name: Optional[str] = None,
        cleanup_zip: bool = True,
    ) -> Dict[str, Union[float, int]]:
        """
        Transfer a directory by zipping it first, then transferring the zip file.

        This is more efficient for large directories with many files.

        Parameters
        ----------
        source_cluster : str
            Source cluster name ('cpu' or 'gpu')
        source_path : str
            Source directory path
        dest_cluster : str
            Destination cluster name ('cpu' or 'gpu')
        dest_path : str
            Destination directory path (will be created)
        temp_zip_name : Optional[str]
            Name for temporary zip file (default: auto-generated)
        cleanup_zip : bool
            Whether to clean up temporary zip files after transfer

        Returns
        -------
        Dict[str, Union[float, int]]
            Transfer statistics including zip/unzip times
        """
        if source_cluster == dest_cluster:
            raise ValueError("Source and destination clusters cannot be the same")

        start_time = time.time()

        # Generate zip filename if not provided
        if temp_zip_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_zip_name = f"transfer_{Path(source_path).name}_{timestamp}.zip"

        # Get SSH connection info
        source_login = self.cpu_login if source_cluster == "cpu" else self.gpu_login
        dest_login = self.cpu_login if dest_cluster == "cpu" else self.gpu_login

        source_host = self._get_ssh_prefix(source_login)
        dest_host = self._get_ssh_prefix(dest_login)

        # Check if source directory exists
        if not self.check_file_exists(source_cluster, source_path):
            raise SSHTransferError(f"Source directory does not exist: {source_path}")

        # Temporary zip paths
        source_zip_path = f"/tmp/{temp_zip_name}"
        dest_zip_path = f"/tmp/{temp_zip_name}"

        try:
            # Step 1: Create zip file on source cluster
            logger.info(f"Creating zip file on {source_cluster} cluster...")
            zip_start = time.time()

            # Use zip command to create archive (preserving directory structure)
            # -r: recursive, -q: quiet, -9: maximum compression
            zip_cmd = f"cd '{Path(source_path).parent}' && zip -r -q -9 '{source_zip_path}' '{Path(source_path).name}'"
            result = self._run_ssh_command(
                source_host, zip_cmd, timeout=1800
            )  # 30 min timeout

            if result.returncode != 0:
                raise SSHTransferError(f"Failed to create zip file: {result.stderr}")

            zip_time = time.time() - zip_start

            # Get zip file size
            zip_size = self.get_file_size(source_cluster, source_zip_path)
            logger.info(
                f"Created zip file: {zip_size / (1024 * 1024):.1f} MB in {zip_time:.2f}s"
            )

            # Step 2: Transfer zip file
            logger.info(
                f"Transferring zip file from {source_cluster} to {dest_cluster}..."
            )
            transfer_stats = self.transfer_file(
                source_cluster,
                source_zip_path,
                dest_cluster,
                dest_zip_path,
                compress=False,  # Already compressed as zip
            )

            # Step 3: Create destination directory and unzip
            logger.info(f"Unzipping on {dest_cluster} cluster...")
            unzip_start = time.time()

            # Ensure destination parent directory exists
            self.ensure_directory_exists(dest_cluster, str(Path(dest_path).parent))

            # Unzip to destination (this will create the directory structure)
            # -q: quiet, -o: overwrite without prompting
            unzip_cmd = (
                f"cd '{Path(dest_path).parent}' && unzip -q -o '{dest_zip_path}'"
            )
            result = self._run_ssh_command(dest_host, unzip_cmd, timeout=1800)

            if result.returncode != 0:
                raise SSHTransferError(f"Failed to unzip file: {result.stderr}")

            # If the extracted directory doesn't match dest_path, rename it
            extracted_dir = Path(dest_path).parent / Path(source_path).name
            if str(extracted_dir) != dest_path:
                rename_cmd = f"mv '{extracted_dir}' '{dest_path}'"
                result = self._run_ssh_command(dest_host, rename_cmd)
                if result.returncode != 0:
                    logger.warning(
                        f"Failed to rename extracted directory: {result.stderr}"
                    )

            unzip_time = time.time() - unzip_start

            # Verify destination exists
            if not self.check_file_exists(dest_cluster, dest_path):
                raise SSHTransferError(
                    "Transfer completed but destination directory not found"
                )

            total_time = time.time() - start_time

            # Combine statistics
            stats = {
                "transfer_type": "directory_zip",
                "total_time_seconds": total_time,
                "zip_time_seconds": zip_time,
                "unzip_time_seconds": unzip_time,
                "zip_size_bytes": zip_size,
                "compression_ratio": zip_size / transfer_stats["file_size_bytes"]
                if transfer_stats["file_size_bytes"] > 0
                else 0,
                **transfer_stats,
            }

            # Calculate effective throughput (based on uncompressed size)
            if (
                "file_size_bytes" in transfer_stats
                and transfer_stats["file_size_bytes"] > 0
            ):
                # Estimate original directory size from zip metadata or use transfer size as approximation
                original_size = transfer_stats[
                    "file_size_bytes"
                ]  # This is the zip size
                stats["effective_throughput_mbps"] = (
                    (original_size / (1024 * 1024)) / total_time
                    if total_time > 0
                    else 0
                )

            logger.info(f"Directory zip transfer completed in {total_time:.2f}s")

        finally:
            # Cleanup temporary zip files if requested
            if cleanup_zip:
                logger.info("Cleaning up temporary zip files...")
                try:
                    self.cleanup_files(source_cluster, [source_zip_path])
                except Exception as e:
                    logger.warning(f"Failed to cleanup source zip: {e}")

                try:
                    self.cleanup_files(dest_cluster, [dest_zip_path])
                except Exception as e:
                    logger.warning(f"Failed to cleanup destination zip: {e}")

        return stats

    def get_directory_size(self, cluster: str, directory_path: str) -> int:
        """
        Get the total size of a directory on a remote cluster.

        Parameters
        ----------
        cluster : str
            Cluster name ('cpu' or 'gpu')
        directory_path : str
            Path to the directory

        Returns
        -------
        int
            Directory size in bytes
        """
        login = self.cpu_login if cluster == "cpu" else self.gpu_login
        host = self._get_ssh_prefix(login)

        # Use du command to get directory size
        command = f"du -sb '{directory_path}' | cut -f1"
        result = self._run_ssh_command(host, command)

        if result.returncode != 0:
            raise SSHTransferError(
                f"Failed to get directory size for {directory_path}: {result.stderr}"
            )

        try:
            return int(result.stdout.strip())
        except ValueError:
            raise SSHTransferError(
                f"Invalid directory size output: {result.stdout.strip()}"
            )

    def create_workflow_transfer_script(
        self,
        script_path: str,
        source_cluster: str,
        source_path: str,
        dest_cluster: str,
        dest_path: str,
        transfer_type: str = "zip",
    ) -> None:
        """
        Create a standalone script for workflow transfers.

        Parameters
        ----------
        script_path : str
            Path where to save the transfer script
        source_cluster : str
            Source cluster name
        source_path : str
            Source path
        dest_cluster : str
            Destination cluster name
        dest_path : str
            Destination path
        transfer_type : str
            Type of transfer ('zip' or 'rsync')
        """
        script_content = f"""#!/bin/bash
#SBATCH --job-name=transfer_{source_cluster}_to_{dest_cluster}
#SBATCH --output=transfer_%j.out
#SBATCH --error=transfer_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

# Auto-generated transfer script
# Source: {source_cluster}:{source_path}
# Destination: {dest_cluster}:{dest_path}
# Transfer type: {transfer_type}

echo "Starting transfer at $(date)"
echo "Source: {source_cluster}:{source_path}"
echo "Destination: {dest_cluster}:{dest_path}"

# Load Python environment
source ~/.bashrc
cd /home/menger/git/adata_hf_datasets

# Run transfer using Python utilities
python3 -c "
import sys
sys.path.append('scripts/util')
from adata_hf_datasets.ssh_transfer_utils import SSHTransferUtils

# Initialize transfer utilities
cpu_login = {{'host': '{self.cpu_login["host"]}', 'user': '{self.cpu_login["user"]}'}}
gpu_login = {{'host': '{self.gpu_login["host"]}', 'user': '{self.gpu_login["user"]}'}}
transfer_utils = SSHTransferUtils(cpu_login, gpu_login)

try:
    # Perform transfer
    if '{transfer_type}' == 'zip':
        stats = transfer_utils.transfer_directory_as_zip(
            '{source_cluster}', '{source_path}',
            '{dest_cluster}', '{dest_path}'
        )
    else:
        stats = transfer_utils.transfer_directory(
            '{source_cluster}', '{source_path}',
            '{dest_cluster}', '{dest_path}'
        )

    print(f'Transfer completed successfully!')
    print(f'Time: {{stats[\"total_time_seconds\"]:.2f}} seconds')
    if 'effective_throughput_mbps' in stats:
        print(f'Throughput: {{stats[\"effective_throughput_mbps\"]:.2f}} MB/s')

except Exception as e:
    print(f'Transfer failed: {{e}}')
    sys.exit(1)
"

echo "Transfer completed at $(date)"
"""

        # Write script to file
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable
        import stat

        st = os.stat(script_path)
        os.chmod(script_path, st.st_mode | stat.S_IEXEC)

        logger.info(f"Created transfer script: {script_path}")

    def cleanup_files(self, cluster: str, file_patterns: List[str]) -> None:
        """
        Clean up files on a remote cluster.

        Parameters
        ----------
        cluster : str
            Cluster name ('cpu' or 'gpu')
        file_patterns : List[str]
            List of file patterns to remove
        """
        login = self.cpu_login if cluster == "cpu" else self.gpu_login
        host = self._get_ssh_prefix(login)

        for pattern in file_patterns:
            command = f"rm -rf {pattern}"
            result = self._run_ssh_command(host, command)

            if result.returncode != 0:
                logger.warning(f"Failed to remove {pattern}: {result.stderr}")
            else:
                logger.info(f"Removed {pattern}")
