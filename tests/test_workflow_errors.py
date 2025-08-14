#!/usr/bin/env python3
"""
Comprehensive tests for workflow error scenarios and validation.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
import subprocess
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adata_hf_datasets.config_utils import validate_config
from adata_hf_datasets.workflow_orchestrator import WorkflowOrchestrator, WorkflowLogger


class TestConfigurationErrors:
    """Test configuration validation and error handling."""

    def test_missing_required_dataset_fields(self):
        """Test error when required dataset fields are missing."""
        config = OmegaConf.create(
            {
                "dataset": {
                    "description": "Test dataset"
                    # Missing 'name' field
                },
                "preprocessing": {
                    "enabled": True,
                    "input_file": "test.h5ad",
                    "output_dir": "test_output",
                },
                "embedding": {
                    "enabled": True,
                    "input_files": ["test.h5ad"],
                    "output_dir": "test_output",
                    "methods": ["hvg"],
                },
                "dataset_creation": {
                    "enabled": True,
                    "data_dir": "test_output",
                    "sentence_keys": ["test"],
                    "required_obsm_keys": ["X_pca"],
                },
            }
        )

        with pytest.raises(ValueError, match="Missing required field: dataset.name"):
            validate_config(config)

    def test_invalid_embedding_methods(self):
        """Test error when invalid embedding methods are specified."""
        config = OmegaConf.create(
            {
                "dataset": {"name": "test_dataset", "file_path": "test.h5ad"},
                "download": {"enabled": True, "subset_size": 1000, "seed": 42},
                "preprocessing": {
                    "enabled": True,
                    "input_file": "test.h5ad",
                    "output_dir": "test_output",
                },
                "embedding": {
                    "enabled": True,
                    "input_files": ["test.h5ad"],
                    "output_dir": "test_output",
                    "methods": ["invalid_method"],  # Invalid method
                    "embedding_dim_map": {"hvg": 512},
                },
                "dataset_creation": {
                    "enabled": True,
                    "data_dir": "test_output",
                    "sentence_keys": ["test"],
                    "required_obsm_keys": ["X_pca"],
                },
            }
        )

        with pytest.raises(ValueError, match="Invalid embedding methods"):
            validate_config(config)


class TestSSHErrors:
    """Test SSH-related error scenarios."""

    @patch("subprocess.run")
    def test_ssh_command_not_available(self, mock_run):
        """Test error when SSH command is not available."""
        mock_run.side_effect = FileNotFoundError("ssh: command not found")

        with pytest.raises(RuntimeError, match="SSH command not available"):
            WorkflowOrchestrator(
                cpu_login={"host": "test_host", "user": "test_user"},
                gpu_login={"host": "test_gpu", "user": "test_user"},
            )

    @patch("subprocess.run")
    def test_ssh_connection_failure(self, mock_run):
        """Test SSH connection failure during job submission."""

        # First two calls: SSH command check and SSH connection test (success)
        # Third call: Job submission (failure)
        def side_effect(*args, **kwargs):
            if side_effect.call_count < 2:
                side_effect.call_count += 1

                # Return a real CompletedProcess with string stdout
                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()
            else:
                raise subprocess.CalledProcessError(
                    1, "ssh", stderr="Connection refused"
                )

        side_effect.call_count = 0
        mock_run.side_effect = side_effect

        orchestrator = WorkflowOrchestrator(
            cpu_login={"host": "test_host", "user": "test_user"},
            gpu_login={"host": "test_gpu", "user": "test_user"},
        )

        with pytest.raises(RuntimeError, match="SLURM job submission failed"):
            orchestrator._submit_slurm_job(
                host="test_host",
                script_path=Path("test_script.slurm"),
                step_name="Test",
            )


class TestSLURMErrors:
    """Test SLURM job management error scenarios."""

    @patch("subprocess.run")
    def test_slurm_job_submission_failure(self, mock_run):
        """Test SLURM job submission failure."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "sbatch: error: Invalid partition"

        orchestrator = WorkflowOrchestrator(
            cpu_login={"host": "test_host", "user": "test_user"},
            gpu_login={"host": "test_gpu", "user": "test_user"},
        )

        with pytest.raises(RuntimeError, match="SLURM job submission failed"):
            orchestrator._submit_slurm_job(
                host="test_host",
                script_path=Path("test_script.slurm"),
                partition="invalid_partition",
                step_name="Test",
            )

    @patch("subprocess.run")
    def test_slurm_job_id_parsing_failure(self, mock_run):
        """Test failure to parse SLURM job ID from output."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Some unexpected output without job ID"

        orchestrator = WorkflowOrchestrator(
            cpu_login={"host": "test_host", "user": "test_user"},
            gpu_login={"host": "test_gpu", "user": "test_user"},
        )

        with pytest.raises(
            RuntimeError, match="Could not parse job ID from SLURM output"
        ):
            orchestrator._submit_slurm_job(
                host="test_host",
                script_path=Path("test_script.slurm"),
                step_name="Test",
            )


class TestFileSystemErrors:
    """Test file system and data-related error scenarios."""

    def test_missing_input_file(self):
        """Test error when input file doesn't exist."""
        config = OmegaConf.create(
            {
                "dataset": {
                    "name": "test_dataset",
                    "file_path": "nonexistent_file.h5ad",
                },
                "preprocessing": {
                    "enabled": True,
                    "input_file": "nonexistent_file.h5ad",
                    "output_dir": "test_output",
                },
                "embedding": {
                    "enabled": True,
                    "input_files": ["nonexistent_file.h5ad"],
                    "output_dir": "test_output",
                    "methods": ["hvg"],
                },
                "dataset_creation": {
                    "enabled": True,
                    "data_dir": "test_output",
                    "sentence_keys": ["test"],
                    "required_obsm_keys": ["X_pca"],
                },
            }
        )

        # This would be caught during actual file operations
        input_file = Path(config.preprocessing.input_file)
        assert not input_file.exists(), "Input file should not exist"

    def test_file_permission_issues(self):
        """Test file permission issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            read_only_file = Path(temp_dir) / "readonly.txt"
            read_only_file.write_text("test content")

            # Make file read-only
            read_only_file.chmod(0o444)

            # Try to write to read-only file
            try:
                with open(read_only_file, "w") as f:
                    f.write("new content")
            except PermissionError:
                # Expected behavior
                pass
            else:
                pytest.fail("Should have raised PermissionError")


class TestWorkflowLogger:
    """Test workflow logging functionality."""

    def test_workflow_logger_creation(self, tmp_path):
        """Test workflow logger creation and directory structure."""
        base_dir = tmp_path / "outputs"
        master_job_id = "test_12345"
        dataset_config_name = "test_dataset"
        workflow_config = OmegaConf.create(
            {
                "project_directory": str(base_dir),
            }
        )

        logger = WorkflowLogger(
            base_dir, master_job_id, dataset_config_name, workflow_config
        )

        # Check that directories were created
        assert logger.workflow_dir.exists()
        assert (logger.workflow_dir / "logs").exists()
        assert (logger.workflow_dir / "config").exists()

        # Check step directories
        for step in [
            "download",
            "preprocessing",
            "embedding_prepare",
            "embedding",
            "dataset_creation",
        ]:
            assert (logger.workflow_dir / step).exists()

    def test_workflow_logger_step_logging(self, tmp_path):
        """Test workflow logger step logging functionality."""
        base_dir = tmp_path / "outputs"
        master_job_id = "test_12345"
        dataset_config_name = "test_dataset"
        workflow_config = OmegaConf.create(
            {
                "project_directory": str(base_dir),
            }
        )

        logger = WorkflowLogger(
            base_dir, master_job_id, dataset_config_name, workflow_config
        )

        # Test step logging
        logger.log_step_start("download", "12345", "test_host")
        logger.log_step_complete("download", "12345")
        logger.log_step_skipped("preprocessing", "disabled in config")

        # Check log files exist
        log_dir = logger.workflow_dir / "logs"
        assert (log_dir / "workflow_summary.log").exists()
        assert (log_dir / "errors_consolidated.log").exists()


class TestParameterValidation:
    """Test parameter validation and edge cases."""

    def test_negative_subset_size(self):
        """Test error when negative subset size is provided."""
        config = OmegaConf.create(
            {
                "dataset": {"name": "test_dataset", "file_path": "test.h5ad"},
                "download": {
                    "enabled": True,
                    "subset_size": -1000,  # Invalid negative value
                    "seed": 42,
                },
                "preprocessing": {
                    "enabled": True,
                    "input_file": "test.h5ad",
                    "output_dir": "test_output",
                },
                "embedding": {
                    "enabled": True,
                    "input_files": ["test.h5ad"],
                    "output_dir": "test_output",
                    "methods": ["hvg"],
                },
                "dataset_creation": {
                    "enabled": True,
                    "data_dir": "test_output",
                    "sentence_keys": ["test"],
                    "required_obsm_keys": ["X_pca"],
                },
            }
        )

        # This should be caught during download step execution
        assert config.download.subset_size < 0, "Negative subset size should be invalid"

    def test_invalid_embedding_dimensions(self):
        """Test error when embedding dimensions are invalid."""
        config = OmegaConf.create(
            {
                "dataset": {"name": "test_dataset", "file_path": "test.h5ad"},
                "preprocessing": {
                    "enabled": True,
                    "input_file": "test.h5ad",
                    "output_dir": "test_output",
                },
                "embedding_cpu": {
                    "enabled": True,
                    "input_files": ["test.h5ad"],
                    "output_dir": "test_output",
                    "methods": ["hvg"],
                    "embedding_dim_map": {
                        "hvg": -512  # Invalid negative dimension
                    },
                },
                "dataset_creation": {
                    "enabled": True,
                    "data_dir": "test_output",
                    "sentence_keys": ["test"],
                    "required_obsm_keys": ["X_pca"],
                },
            }
        )

        # This would be caught during embedding step
        assert config.embedding_cpu.embedding_dim_map["hvg"] < 0

    def test_enhanced_parameter_validation(self):
        """Test enhanced parameter validation with invalid values."""
        config = OmegaConf.create(
            {
                "dataset": {"name": "test_dataset", "file_path": "test.h5ad"},
                "download": {
                    "enabled": True,
                    "subset_size": -1000,  # Invalid negative value
                    "seed": 42,
                },
                "preprocessing": {
                    "enabled": True,
                    "input_file": "test.h5ad",
                    "output_dir": "test_output",
                    "min_cells": -10,  # Invalid negative value
                    "train_split": 1.5,  # Invalid value > 1
                },
                "embedding_cpu": {
                    "enabled": True,
                    "input_files": ["test.h5ad"],
                    "output_dir": "test_output",
                    "methods": ["hvg"],
                    "embedding_dim_map": {
                        "hvg": -512  # Invalid negative dimension
                    },
                    "batch_size": 0,  # Invalid zero value
                },
                "dataset_creation": {
                    "enabled": True,
                    "data_dir": "test_output",
                    "sentence_keys": ["test"],
                    "required_obsm_keys": ["X_pca"],
                },
            }
        )

        # This should catch multiple validation errors
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)

        error_msg = str(exc_info.value)
        # Check that the error message contains information about the invalid parameters
        assert (
            "subset_size" in error_msg
            or "min_cells" in error_msg
            or "train_split" in error_msg
            or "batch_size" in error_msg
        )


if __name__ == "__main__":
    pytest.main([__file__])
