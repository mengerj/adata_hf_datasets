# Embedding Scripts

This directory contains scripts for applying various embedding methods to preprocessed AnnData files.

## Main Script: `embed_adata.py`

The main script `embed_adata.py` now supports two modes of operation:

### 1. Prepare-Only Mode (`++prepare_only=true`)

Runs only the `prepare()` step for each embedding method without saving any embeddings. This is useful for:

- GPU-dependent embedders where the prepare step is more efficient on CPU
- Pre-computing resources that can be reused in subsequent runs
- Testing and debugging the preparation phase separately

**Example:**

```bash
python3 scripts/embed/embed_adata.py ++prepare_only=true ++methods='["scvi_fm", "pca"]'
```

### 2. Full Pipeline Mode (`++prepare_only=false` or default)

Runs the complete pipeline: prepare + embed + save embeddings. This is the default behavior.

**Example:**

```bash
python3 scripts/embed/embed_adata.py ++prepare_only=false ++methods='["scvi_fm", "pca"]'
```

## Input Files Handling

The script intelligently handles input files in two ways:

### Auto-Generated Paths (Default)

When no `input_files` is specified, the script uses auto-generated paths from the dataset configuration:

```bash
# Uses auto-generated paths from dataset config
python3 scripts/embed/embed_adata.py ++prepare_only=true
```

### Command-Line Override

When `input_files` is explicitly specified, it overrides the auto-generated paths:

```bash
# Uses command-line specified files
python3 scripts/embed/embed_adata.py ++prepare_only=true ++input_files='["my_file.h5ad"]'
```

This allows the script to work seamlessly with both:

- **Direct execution**: Run on specific files by passing `++input_files`
- **SLURM array jobs**: Let bash scripts override `input_files` for specific chunks

## Usage Examples

### Basic Usage

```bash
# Prepare-only mode with auto-generated paths
python3 scripts/embed/embed_adata.py ++prepare_only=true

# Full pipeline mode with auto-generated paths
python3 scripts/embed/embed_adata.py
```

### With Specific Files

```bash
# Prepare-only mode on specific file
python3 scripts/embed/embed_adata.py ++prepare_only=true ++input_files='["data/processed/chunk_0.zarr"]'

# Full pipeline on specific file
python3 scripts/embed/embed_adata.py ++input_files='["data/processed/chunk_0.zarr"]'
```

### With Dataset-Centric Config

```bash
# Using dataset config with prepare-only (auto-generated paths)
python3 scripts/embed/embed_adata.py ++prepare_only=true

# Override methods in dataset config
python3 scripts/embed/embed_adata.py ++prepare_only=true ++embedding.methods='["scvi_fm", "pca"]'
```

### SLURM Array Job Compatibility

The script works perfectly with SLURM array jobs where bash scripts set `input_files`:

```bash
# In SLURM script: python3 scripts/embed/embed_adata.py ++input_files='["$this_file"]'
# This will use the specific chunk file, ignoring auto-generated paths
```

## Parameter Overrides

You can override any configuration parameter using Hydra's `++` syntax:

```bash
python3 scripts/embed/embed_adata.py \
  ++prepare_only=true \
  ++methods='["scvi_fm", "pca", "hvg"]' \
  ++batch_key="dataset_title" \
  ++batch_size=32 \
  ++input_files='["custom_file.h5ad"]'
```

## Configuration Files

- `conf/embed_adata.yaml`: Standalone configuration
- `conf/dataset_*.yaml`: Dataset-centric configurations
- `conf/prepare_embed_adata.yaml`: Example configuration for prepare-only mode

## Supported Embedding Methods

- `scvi_fm`: scVI Foundation Model
- `geneformer`: Geneformer transformer model
- `pca`: Principal Component Analysis
- `hvg`: Highly Variable Genes

## Workflow Benefits

1. **Resource Efficiency**: Prepare-only mode can run on CPU nodes, saving GPU resources
2. **Reusability**: Full pipeline can reuse prepared resources from previous runs
3. **Debugging**: Easy to test and debug each step separately
4. **Flexibility**: Works with both standalone and dataset-centric configurations
5. **SLURM Compatibility**: Seamlessly works with array jobs and direct execution

## SLURM Integration

For SLURM-based workflows, see the example script `run_embedding_workflow.sh` which shows how to structure separate jobs for prepare-only and full pipeline modes.

## Migration from Old Scripts

The old `prepare_embed_adata.py` script has been removed. Its functionality is now integrated into `embed_adata.py` with the `++prepare_only=true` parameter.
