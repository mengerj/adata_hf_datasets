# Embedding Scripts - Simplified Structure

This directory contains a **simplified embedding workflow** that reduces complexity while maintaining all functionality.

## Overview

The new embedding workflow uses a clean 3-script structure:

1. **`run_embed_new.slurm`** - Single master script for all embedding modes
2. **`embed_launcher.py`** - Python-only configuration and job submission
3. **`embed_array.slurm`** - Simplified SLURM array worker script
4. **`embed_core.py`** - Core embedding logic (unchanged)

## Benefits of New Structure

- **50% fewer scripts** (from 6 to 3 scripts)
- **Cleaner parameter flow** (Python → Python → SLURM)
- **Unified master script** handles CPU/GPU and prepare/embed modes
- **Simplified logging** with consistent directory structure
- **Better error handling** and debugging
- **Easier maintenance** with clear responsibilities
- **Efficient GPU resource usage** (master job runs on CPU, only array jobs use GPU)

## Hardware-Aware Embedding

The workflow supports the same hardware-aware splitting as before:

### CPU Embedding

- Methods: `scvi_fm`, `pca`, `hvg`
- Runs on CPU cluster with CPU partition

### GPU Embedding

- Methods: `geneformer`
- Runs on GPU cluster with GPU partition

### Preparation Mode

- Uses `embedding_preparation` config section if available
- Falls back to CPU config for preparation tasks

## Configuration

The same dataset configuration structure is used:

```yaml
# CPU embedding configuration
embedding_cpu:
  enabled: true
  methods: ["hvg", "pca", "scvi_fm"]
  batch_size: 128
  embedding_dim_map:
    scvi_fm: 50
    pca: 50
    hvg: 512

# GPU embedding configuration
embedding_gpu:
  enabled: true
  methods: ["geneformer"]
  batch_size: 128
  embedding_dim_map:
    geneformer: 512

# Preparation configuration (optional)
embedding_preparation:
  enabled: true
  methods: ["hvg", "pca"] # Methods to prepare
```

## Usage

### Manual Execution

```bash
# CPU embedding
MODE=cpu python scripts/embed/embed_launcher.py --config-name dataset_example

# GPU embedding
MODE=gpu python scripts/embed/embed_launcher.py --config-name dataset_example

# Preparation only
python scripts/embed/embed_launcher.py --config-name dataset_example --prepare-only

# Run master script directly
DATASET_CONFIG=dataset_example MODE=cpu PREPARE_ONLY=false sbatch scripts/embed/run_embed_new.slurm
```

### Workflow Integration

The workflow orchestrator can use the new simplified methods:

```python
# Use new embedding methods in workflow_orchestrator.py
from scripts.embed.embedding_methods_new import (
    run_embedding_prepare_step_new,
    run_embedding_cpu_step_new,
    run_embedding_gpu_step_new
)

# Replace old methods with new ones
WorkflowOrchestrator.run_embedding_prepare_step = run_embedding_prepare_step_new
WorkflowOrchestrator.run_embedding_cpu_step = run_embedding_cpu_step_new
WorkflowOrchestrator.run_embedding_gpu_step = run_embedding_gpu_step_new
```

## Script Descriptions

### `run_embed_new.slurm`

- **Single master script** for all embedding operations
- Handles both CPU/GPU modes via `MODE` environment variable
- Supports prepare-only mode via `PREPARE_ONLY` environment variable
- Sets up logging directories and calls `embed_launcher.py`
- Waits for array jobs to complete with proper error handling

### `embed_launcher.py`

- **Python-only configuration handling**
- Loads dataset config and determines embedding parameters
- Discovers input directories (train/val/test)
- Submits SLURM array jobs for each directory
- Clean error handling and logging

### `embed_array.slurm`

- **Simplified array worker** for processing individual files
- Minimal bash script focused on single-file processing
- Calls `embed_core.py` with proper parameters
- Clean logging with task-specific output directories

## Environment Variables

The scripts use these environment variables:

- `DATASET_CONFIG`: Name of dataset configuration
- `MODE`: Processing mode ("cpu" or "gpu")
- `PREPARE_ONLY`: Set to "true" for preparation mode
- `WORKFLOW_DIR`: Base directory for workflow outputs
- `SLURM_PARTITION`: SLURM partition to use
- `GPU_HOST`: SSH target for GPU cluster (for cross-cluster array job submission)

## Log Structure

Logs are organized consistently:

```
${WORKFLOW_DIR}/
├── embedding_prepare/job_${MASTER_JOB_ID}/
│   ├── master.out/err           # Master job logs
│   └── array_${ARRAY_JOB_ID}/   # Array job logs
│       ├── 0.out/err            # Task-specific logs
│       └── ...
└── embedding/job_${MASTER_JOB_ID}/
    ├── master.out/err
    └── array_${ARRAY_JOB_ID}/
        ├── 0.out/err
        └── ...
```

## Migration from Old Structure

The old scripts are preserved for backward compatibility:

- `embed_configure_submit_parallel_sh.py` → `embed_launcher.py`
- `embed_submit_parallel.sh` → integrated into `embed_launcher.py`
- `embed_chunks_parallel.slurm` → `embed_array.slurm`
- `run_embed.slurm` + `run_embed_prepare.slurm` → `run_embed_new.slurm`

## Error Handling

The new structure provides better error handling:

- **Master script** monitors array job completion and status
- **Python launcher** provides detailed error messages for config issues
- **Array worker** reports individual task failures clearly
- **Consolidated logging** makes debugging easier

## GPU Resource Management

The new structure efficiently handles GPU resources:

- **Master job runs on CPU cluster** (coordination only, doesn't consume GPU resources)
- **Array jobs run on GPU cluster** (actual GPU work with `--gres=gpu:N`)
- **Cross-cluster submission** via SSH when needed
- **Prevents resource deadlocks** in limited GPU environments

## Testing

Test the new structure:

```bash
# Test CPU embedding
DATASET_CONFIG=dataset_test_workflow MODE=cpu PREPARE_ONLY=false \
  sbatch scripts/embed/run_embed_new.slurm

# Test preparation mode
DATASET_CONFIG=dataset_test_workflow MODE=cpu PREPARE_ONLY=true \
  sbatch scripts/embed/run_embed_new.slurm
```
