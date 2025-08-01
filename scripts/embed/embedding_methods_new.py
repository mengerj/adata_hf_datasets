#!/usr/bin/env python3
"""
New Embedding Methods for Workflow Orchestrator

These methods use the simplified embedding structure:
- Single master script (run_embed_new.slurm)
- Python launcher (embed_launcher.py)
- Simple array worker (embed_array.slurm)

This replaces the complex chain of scripts in the old workflow.
"""

from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


def run_embedding_prepare_step_new(
    self,
    dataset_config_name: str,
    workflow_config: DictConfig,
    dependency_job_id: Optional[int] = None,
) -> Optional[int]:
    """Run the embedding preparation step using the new simplified structure."""
    logger.info("=== Starting Embedding Preparation Step (New) ===")
    script_path = Path("scripts/embed/run_embed_new.slurm")
    dependencies = [dependency_job_id] if dependency_job_id else None

    logger.info(f"Using dataset config: {dataset_config_name}")

    # Pass the dataset config name, workflow directory, and mode settings as environment variables
    env_vars = {
        "DATASET_CONFIG": dataset_config_name,
        "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
        if self.workflow_logger
        else "",
        "MODE": "cpu",  # Preparation typically runs on CPU
        "PREPARE_ONLY": "true",  # This is preparation mode
        "SLURM_PARTITION": workflow_config.cpu_partition,
    }

    job_id = self._submit_slurm_job(
        self.cpu_login["host"],  # Use CPU cluster for preparation
        script_path,
        partition=workflow_config.cpu_partition,  # Use CPU partition
        dependencies=dependencies,
        env_vars=env_vars,
        step_name="Embedding Preparation",
    )
    return job_id


def run_embedding_cpu_step_new(
    self,
    dataset_config_name: str,
    workflow_config: DictConfig,
    dependency_job_id: Optional[int] = None,
) -> Optional[int]:
    """Run the CPU embedding step using the new simplified structure."""
    logger.info("=== Starting CPU Embedding Step (New) ===")
    script_path = Path("scripts/embed/run_embed_new.slurm")
    dependencies = [dependency_job_id] if dependency_job_id else None

    logger.info(f"Using dataset config: {dataset_config_name}")

    # Pass the dataset config name, workflow directory, and mode settings as environment variables
    env_vars = {
        "DATASET_CONFIG": dataset_config_name,
        "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
        if self.workflow_logger
        else "",
        "MODE": "cpu",  # Force CPU mode
        "PREPARE_ONLY": "false",  # This is full embedding mode
        "SLURM_PARTITION": workflow_config.cpu_partition,
    }

    job_id = self._submit_slurm_job(
        self.cpu_login["host"],  # Use CPU cluster for CPU embedding
        script_path,
        partition=workflow_config.cpu_partition,  # Use CPU partition
        dependencies=dependencies,
        env_vars=env_vars,
        step_name="CPU Embedding",
    )
    return job_id


def run_embedding_gpu_step_new(
    self,
    dataset_config_name: str,
    workflow_config: DictConfig,
    dependency_job_id: Optional[int] = None,
) -> Optional[int]:
    """Run the GPU embedding step using the new simplified structure."""
    logger.info("=== Starting GPU Embedding Step (New) ===")
    script_path = Path("scripts/embed/run_embed_new.slurm")
    dependencies = [dependency_job_id] if dependency_job_id else None

    logger.info(f"Using dataset config: {dataset_config_name}")

    # Pass the dataset config name, workflow directory, and mode settings as environment variables
    # IMPORTANT: Master job runs on CPU cluster to avoid consuming GPU resources
    # Only the array jobs will use GPU resources
    env_vars = {
        "DATASET_CONFIG": dataset_config_name,
        "WORKFLOW_DIR": str(self.workflow_logger.workflow_dir)
        if self.workflow_logger
        else "",
        "MODE": "gpu",  # Force GPU mode for array jobs
        "PREPARE_ONLY": "false",  # This is full embedding mode
        "SLURM_PARTITION": workflow_config.gpu_partition,  # Pass GPU partition for array jobs
        "GPU_HOST": f"{self.gpu_login['user']}@{self.gpu_login['host']}"
        if self.gpu_login
        else "",  # GPU cluster info for array job submission
    }

    job_id = self._submit_slurm_job(
        self.cpu_login["host"],  # Use CPU cluster for master job (coordination only)
        script_path,
        partition=workflow_config.cpu_partition,  # Use CPU partition for master job
        dependencies=dependencies,
        env_vars=env_vars,
        step_name="GPU Embedding",
    )
    return job_id


# Example of how to integrate these into the WorkflowOrchestrator class:
"""
To use the new simplified embedding structure, replace the existing methods in
WorkflowOrchestrator with these:

# In workflow_orchestrator.py, replace:
# - run_embedding_prepare_step with run_embedding_prepare_step_new
# - run_embedding_cpu_step with run_embedding_cpu_step_new
# - run_embedding_gpu_step with run_embedding_gpu_step_new

The main benefits:
1. Single master script handles both CPU/GPU and prepare/embed modes
2. Clean Python-only configuration handling
3. Simplified logging and error handling
4. Easier debugging and maintenance

Integration example:
```python
# Add these methods to WorkflowOrchestrator class
WorkflowOrchestrator.run_embedding_prepare_step_new = run_embedding_prepare_step_new
WorkflowOrchestrator.run_embedding_cpu_step_new = run_embedding_cpu_step_new
WorkflowOrchestrator.run_embedding_gpu_step_new = run_embedding_gpu_step_new

# Then replace the method calls in run_workflow() and run_workflow_local()
embedding_prepare_job_id = self.run_embedding_prepare_step_new(...)
embedding_cpu_job_id = self.run_embedding_cpu_step_new(...)
embedding_gpu_job_id = self.run_embedding_gpu_step_new(...)
```
"""
