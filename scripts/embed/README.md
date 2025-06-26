# Embedding Scripts

This directory contains scripts for running embedding steps in the workflow.

## Overview

The embedding step applies various embedding methods to preprocessed AnnData files. Different embedding methods have different hardware requirements:

- **GPU-required methods**: `geneformer`
- **CPU-only methods**: `scvi_fm`, `pca`, `hvg`

## Hardware-Aware Embedding

The workflow now uses a config-based approach to split embedding methods based on hardware requirements:

### CPU Embedding Step

- Runs on CPU cluster (`imbi13`)
- Uses CPU partition (`slurm`)
- Processes methods: `scvi_fm`, `pca`, `hvg`

### GPU Embedding Step

- Runs on GPU cluster (`imbi_gpu_H100`)
- Uses GPU partition (`gpu`)
- Processes methods: `geneformer`

### Configuration Structure

The workflow uses two separate configuration sections in your dataset config:

```yaml
# CPU embedding configuration
embedding_cpu:
  enabled: true
  methods: ["hvg", "pca", "scvi_fm"] # CPU-only methods
  batch_size: 128
  embedding_dim_map:
    scvi_fm: 50
    pca: 50
    hvg: 512

# GPU embedding configuration
embedding_gpu:
  enabled: true
  methods: ["geneformer"] # GPU-required methods
  batch_size: 128
  embedding_dim_map:
    geneformer: 512
```

### Automatic Execution

The workflow automatically:

1. Checks if `embedding_cpu.enabled` is true and runs CPU methods
2. Checks if `embedding_gpu.enabled` is true and runs GPU methods
3. Both steps can run in parallel since they depend on the same preparation step
4. Dataset creation waits for both embedding steps to complete

## Scripts

### `run_embed_parallel.slurm`

Main SLURM script for running embedding jobs. Supports both CPU and GPU modes via the `MODE` environment variable.

### `embed_adata.py`

Main embedding script that processes AnnData files and applies embeddings.

### `run_embed_with_config.py`

Script that extracts embedding parameters from config and runs the embedding pipeline.

## Usage

### Manual Execution

To run embedding manually:

```bash
# CPU embedding
MODE=cpu python scripts/embed/run_embed_with_config.py --config-name dataset_example

# GPU embedding
MODE=gpu python scripts/embed/run_embed_with_config.py --config-name dataset_example
```

### Workflow Integration

The embedding steps are automatically handled by the workflow orchestrator:

```bash
# Run complete workflow (embedding steps will be split automatically)
python scripts/workflow/run_workflow_master.py dataset_example
```

## Configuration

### Environment Variables

- `MODE`: Set to "cpu" or "gpu" to force specific hardware usage
- `DATASET_CONFIG`: Name of the dataset configuration to use
- `WORKFLOW_DIR`: Directory for workflow outputs and logs
- `SLURM_PARTITION`: SLURM partition to use (auto-detected based on MODE)

### Dataset Config Structure

Configure embedding methods in your dataset config with separate CPU and GPU sections:

```yaml
# CPU embedding methods
embedding_cpu:
  enabled: true
  methods: ["hvg", "pca", "scvi_fm"]
  batch_size: 128
  embedding_dim_map:
    scvi_fm: 50
    pca: 50
    hvg: 512

# GPU embedding methods
embedding_gpu:
  enabled: true
  methods: ["geneformer"]
  batch_size: 128
  embedding_dim_map:
    geneformer: 512
```

### Backward Compatibility

The system maintains backward compatibility with the old single `embedding` section. If `embedding_cpu` and `embedding_gpu` are not found, it will fall back to the legacy `embedding` configuration.

## Output

Embedding results are stored in `adata.obsm` with keys:

- `X_hvg`: Highly variable genes embedding
- `X_pca`: PCA embedding
- `X_scvi_fm`: SCVI foundation model embedding
- `X_geneformer`: Geneformer embedding

## Benefits

1. **Clean Separation**: CPU and GPU methods are clearly separated in config
2. **Flexible Control**: Enable/disable CPU or GPU embedding independently
3. **Parallel Execution**: Both steps can run simultaneously
4. **Resource Optimization**: Each step runs on appropriate hardware
5. **Simple Configuration**: No complex filtering logic needed

## Troubleshooting

### No Methods Found

If you see "No CPU/GPU embedding methods found", check your dataset config to ensure the methods are properly configured in the respective sections.

### Hardware Issues

- CPU methods can run on GPU clusters but may be slower
- GPU methods require actual GPU hardware and will fail on CPU-only clusters

### Memory Issues

For large datasets, consider:

- Reducing batch size
- Using smaller embedding dimensions
- Processing data in chunks
