# AnnData HuggingFace Datasets Pipeline

A comprehensive pipeline for processing single-cell RNA-seq data and creating HuggingFace datasets for machine learning. The pipeline handles everything from raw data download through preprocessing, embedding generation, to final dataset creation and publication.

## Overview

This pipeline transforms raw single-cell RNA-seq data into ready-to-use HuggingFace datasets through a series of automated steps:

```
1. Download → 2. Preprocessing → 3. Embedding Prep → 4. CPU Embedding → 5. GPU Embedding → 6. Dataset Creation
   (optional)                                                                                    ↓
                                                                                    HuggingFace Hub Publication
```

**Key Features:**

- 🔄 **Automated workflow orchestration** with SLURM or local execution
- 🧬 **Memory-efficient processing** of large-scale single-cell datasets
- 🎯 **Multiple embedding methods** (PCA, scVI, HVG, Geneformer)
- 📊 **Quality control** with automatic plots and metrics
- 🤗 **HuggingFace integration** with rich dataset cards
- 🔧 **Highly configurable** with dataset-centric YAML configs

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
  - [Dataset Configuration](#dataset-configuration)
  - [Workflow Orchestrator Configuration](#workflow-orchestrator-configuration)
- [Quick Start](#quick-start)
  - [Local Execution](#local-execution-macos-linux)
  - [SLURM Cluster Execution](#slurm-cluster-execution)
- [Pipeline Steps](#pipeline-steps)
- [Advanced Usage](#advanced-usage)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Git (with submodule support)
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer
- Python 3.10+ (managed by uv)

### Setup

1. **Clone the repository with submodules:**

```bash
git clone <repository-url>
cd adata_hf_datasets

# Initialize and update submodules (required for Geneformer)
git submodule update --init --recursive
```

2. **Install dependencies using uv:**

```bash
# Install all dependencies including optional extras
uv sync --all-extras
```

This creates a virtual environment and installs all required packages including:

- Core dependencies (anndata, scanpy, datasets)
- Embedding tools (scVI, Geneformer)
- Workflow orchestration tools
- All optional dependencies

3. **Activate the virtual environment:**

```bash
# The virtual environment is created in .venv/
source .venv/bin/activate
```

That's it! The pipeline is ready to use.

---

## Configuration

The pipeline uses two main configuration files:

### Dataset Configuration

Dataset configurations define **what data to process** and **how to process it**. Each dataset has its own YAML file in the `conf/` directory.

**Example:** [`conf/dataset_config_example.yaml`](conf/dataset_config_example.yaml)

**Template with all parameters:** [`conf/dataset_default.yaml`](conf/dataset_default.yaml)

#### Key Sections

1. **Dataset Metadata:**

   ```yaml
   dataset:
     name: "cellxgene_pseudo_bulk_10k"
     description: "CellxGene pseudo bulk dataset"
     download_url: "https://example.com/data.h5ad"
   ```

2. **Common Keys** (used across all steps):

   ```yaml
   batch_key: "dataset_title" # Batch/dataset identifier
   annotation_key: "cell_type" # Cell type annotations
   caption_key: "natural_language_annotation" # Natural language descriptions
   ```

3. **Step Configuration** (enable/disable and configure each step):

   ```yaml
   download:
     enabled: false
     subset_size: 10000

   preprocessing:
     enabled: true
     chunk_size: 10000
     split_dataset: true

   embedding_preparation:
     enabled: true
     methods: ["geneformer"]

   embedding_cpu:
     enabled: true
     methods: ["pca", "scvi_fm", "hvg"]

   embedding_gpu:
     enabled: true
     methods: ["geneformer"]

   dataset_creation:
     enabled: true
     dataset_format: "multiplets"
     negatives_per_sample: 2
   ```

**All available parameters are documented in [`dataset_default.yaml`](conf/dataset_default.yaml).**

### Workflow Orchestrator Configuration

The workflow orchestrator configuration defines **where and how to run** the pipeline.

**Configuration file:** [`conf/workflow_orchestrator.yaml`](conf/workflow_orchestrator.yaml)

#### Execution Mode

Choose between local execution or SLURM cluster:

```yaml
workflow:
  execution_mode: "local" # or "slurm"
```

#### Local Execution Settings

For running on your local machine (macOS/Linux):

```yaml
workflow:
  execution_mode: "local"

  # Output directory for logs and results
  output_directory: "/Users/username/repos/adata_hf_datasets/outputs"

  # Base directory for data files
  local_base_file_path: "/Users/username/repos/adata_hf_datasets/data/RNA"

  # Parallel processing settings
  local_max_workers: 2 # Number of parallel workers
  local_enable_gpu: false # Enable GPU embedding locally (requires CUDA)

  # Project directory
  project_directory: "/Users/username/repos/adata_hf_datasets"

  # Virtual environment path (relative to project_directory)
  venv_path: ".venv"
```

#### SLURM Cluster Settings

For running on SLURM clusters:

```yaml
workflow:
  execution_mode: "slurm"

  # SSH connection settings for CPU cluster
  cpu_login:
    host: "cpu_cluster" # SSH host (must be in ~/.ssh/config)
    user: "username"

  # SSH connection settings for GPU cluster
  gpu_login:
    host: "gpu_cluster" # SSH host (must be in ~/.ssh/config)
    user: "username"

  # SLURM partition names (cluster-specific)
  cpu_partition: "slurm" # CPU partition name
  gpu_partition: "gpu" # GPU partition name

  # Output directory (on cluster, accessible by both CPU and GPU nodes)
  output_directory: "/home/username/outputs"

  # Base data directory (must be accessible by both CPU and GPU nodes)
  slurm_base_file_path: "/scratch/global/username/data/RNA"

  # Project directory on cluster
  project_directory: "/home/username/adata_hf_datasets"

  # Virtual environment path (relative to project_directory)
  venv_path: ".venv"
```

#### Important Notes for SLURM

**SSH Configuration:**

- The SLURM mode requires **passwordless SSH access** to the clusters
- Set up SSH keys so that `ssh cpu_cluster` and `ssh gpu_cluster` work without password prompts
- Configure hosts in `~/.ssh/config` if needed (including ProxyJump if required)

**Shared Storage:**

- `slurm_base_file_path` **must be accessible by both CPU and GPU clusters**
- Typically a global scratch filesystem (e.g., `/scratch/global/username/`)
- Data is written by one cluster and read by another during the workflow

**Cluster-Specific Settings:**

- `cpu_partition` and `gpu_partition` names are cluster-specific
- Check your cluster's SLURM configuration for the correct partition names
- Use `sinfo` on your cluster to list available partitions

---

## Quick Start

### Local Execution (macOS/Linux)

For running the complete pipeline on your local machine:

1. **Configure for local execution:**

Edit `conf/workflow_orchestrator.yaml`:

```yaml
workflow:
  execution_mode: "local"
  output_directory: "/Users/username/repos/adata_hf_datasets/outputs"
  local_base_file_path: "/Users/username/repos/adata_hf_datasets/data/RNA"
  local_max_workers: 2
  local_enable_gpu: false # Set to true if you have CUDA-capable GPU
```

2. **Configure your dataset:**
   Take a close look at the [example dataset config](conf/dataset_config_example.yaml) and the [default config](conf/dataset_default.yaml)

Before attempting to add your own dataset, try running the workflow (Step 3) with the --config-name=dataset_config_example setting.
Edit or create a dataset config in `conf/`, for example `conf/my_dataset.yaml`:

```yaml
defaults:
  - dataset_default.yaml
  - _self_

dataset:
  name: "my_dataset"
  description: "My single-cell dataset"
  download_url: "https://..."
  full_name: "my_dataset_full" #required if subsetting the dataset

# Enable/disable steps as needed
preprocessing:
  enabled: true
embedding_cpu:
  enabled: true
# ... etc
```

3. **Run the workflow:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Run workflow in foreground (recommended for first runs)
python scripts/workflow/submit_workflow_local.py \
    --config-name my_dataset \
    --foreground

# Or run in background (detached)
python scripts/workflow/submit_workflow_local.py \
    --config-name my_dataset
```

**What happens:**

- The workflow runs each step sequentially
- Steps are executed based on the `enabled` flags in your dataset config
- Logs are written to `outputs/{date}/workflow_{timestamp}/`
- Data files are written to the `local_base_file_path` directory

### SLURM Cluster Execution

For running on SLURM clusters with SSH orchestration:

1. **Set up SSH keys:**

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519

# Copy to clusters
ssh-copy-id username@cpu_cluster
ssh-copy-id username@gpu_cluster

# Test passwordless access
ssh cpu_cluster "hostname"
ssh gpu_cluster "hostname"
```

2. **Configure for SLURM:**

Edit `conf/workflow_orchestrator.yaml`:

```yaml
workflow:
  execution_mode: "slurm"

  cpu_login:
    host: "cpu_cluster" # Your CPU cluster SSH alias
    user: "username"

  gpu_login:
    host: "gpu_cluster" # Your GPU cluster SSH alias
    user: "username"

  cpu_partition: "slurm" # Check with `sinfo` on your cluster
  gpu_partition: "gpu" # Check with `sinfo` on your cluster

  output_directory: "/home/username/outputs"
  slurm_base_file_path: "/scratch/global/username/data/RNA" # Must be accessible by both clusters!
  project_directory: "/home/username/adata_hf_datasets"
```

3. **Ensure the repository is synced on the cluster:**

```bash
# On your local machine, push to git
git push

# SSH to the cluster and pull
ssh cpu_cluster
cd /home/username/adata_hf_datasets
git pull
git submodule update --init --recursive
uv sync --all-extras
exit
```

4. **Submit the workflow:**

```bash
# From your local machine
python scripts/workflow/submit_workflow.py \
    --config-name my_dataset
```

**What happens:**

- A master SLURM job is submitted to the CPU cluster
- The master job orchestrates all subsequent steps
- Steps run on appropriate clusters (CPU vs GPU)
- Job dependencies are automatically managed by SLURM
- You can monitor progress with `ssh cpu_cluster "squeue -u username"`

**Output location:**

- Logs: `{output_directory}/{date}/workflow_{job_id}/`
- Data: `{slurm_base_file_path}/` (organized into `raw/`, `processed/`, `processed_with_emb/`)

---

## Pipeline Steps

The pipeline consists of six steps, each with detailed documentation:

### 1. Download (Optional)

Downloads and optionally subsets raw data from a URL.

**Documentation:** [`scripts/download/README.md`](scripts/download/README.md)

**Key Features:**

- Download from URLs or file paths
- Stratified subsetting with preserved proportions
- Validation of downloaded files

**Configuration:**

```yaml
download:
  enabled: true
  subset_size: 10000
  stratify_keys: ["cell_type", "tissue"]
  preserve_proportions: true
```

### 2. Preprocessing

Cleans, filters, and normalizes raw count data.

**Documentation:** [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md)

**Key Features:**

- Quality control with MAD-based outlier detection
- Gene/cell filtering
- Normalization and log-transformation
- Highly variable gene selection
- Optional train/val split
- SRA metadata enrichment

**Configuration:**

```yaml
preprocessing:
  enabled: true
  min_cells: 20
  min_genes: 200
  n_top_genes: 5000
  chunk_size: 200000
  split_dataset: true
  train_split: 0.9
```

### 3. Embedding Preparation (CPU)

Performs CPU-intensive preparation for GPU embedding methods (e.g., Geneformer tokenization).

**Documentation:** [`scripts/embed/README.md`](scripts/embed/README.md)

**Key Features:**

- Separates CPU-intensive prep from GPU computation
- Tokenization for Geneformer
- Cached preparation results

**Configuration:**

```yaml
embedding_preparation:
  enabled: true
  methods: ["geneformer"] # Methods that need preparation
```

### 4. CPU Embedding

Generates embeddings using CPU-based methods.

**Documentation:** [`scripts/embed/README.md`](scripts/embed/README.md)

**Key Features:**

- PCA: Linear dimensionality reduction
- scVI: Deep learning foundation model
- HVG: Highly variable gene selection
- Memory-efficient streaming to disk

**Configuration:**

```yaml
embedding_cpu:
  enabled: true
  methods: ["pca", "scvi_fm", "hvg"]
  embedding_dim_map:
    pca: 50
    scvi_fm: 50
    hvg: 512
```

### 5. GPU Embedding

Generates embeddings using GPU-based methods.

**Documentation:** [`scripts/embed/README.md`](scripts/embed/README.md)

**Key Features:**

- Geneformer: Transformer-based embeddings
- Automatic retry on GPU errors
- Uses preparation results from step 3

**Configuration:**

```yaml
embedding_gpu:
  enabled: true
  methods: ["geneformer"]
  embedding_dim_map:
    geneformer: 768
```

### 6. Dataset Creation

Creates HuggingFace datasets with contrastive learning pairs/multiplets.

**Documentation:** [`scripts/dataset_creation/README.md`](scripts/dataset_creation/README.md)

**Key Features:**

- Multiple dataset formats (multiplets, pairs, single)
- Cell sentence generation
- Intelligent negative sampling
- HuggingFace Hub publication
- Optional Nextcloud integration

**Configuration:**

```yaml
dataset_creation:
  enabled: true
  dataset_format: "multiplets"
  sentence_keys: ["sample_id_og"]
  negatives_per_sample: 2
  required_obsm_keys: ["X_pca", "X_scvi_fm", "X_geneformer"]
  push_to_hub: true
  base_repo_id: "your-username"
```

---

## Advanced Usage

### Running Individual Steps

While the workflow orchestrator runs all enabled steps automatically, you can run individual steps manually:

```bash
# Activate environment
source .venv/bin/activate

# Run preprocessing only
python scripts/preprocessing/preprocess.py --config-name my_dataset

# Run CPU embedding only
python scripts/embed/embed_core.py \
    --config-name my_dataset \
    ++embedding_config_section=embedding_cpu

# Run dataset creation only
python scripts/dataset_creation/create_dataset.py --config-name my_dataset
```

See individual step documentation for more details.

### Configuration Overrides

Override any configuration parameter via command line:

```bash
python scripts/workflow/submit_workflow_local.py \
    --config-name my_dataset \
    ++preprocessing.chunk_size=50000 \
    ++embedding_cpu.methods='["pca"]' \
    ++dataset_creation.push_to_hub=false
```

### Monitoring Progress

**Local execution:**

```bash
# Check logs in real-time
tail -f outputs/{date}/workflow_{timestamp}/logs/workflow_master.out

# View step-specific logs
tail -f outputs/{date}/workflow_{timestamp}/preprocessing/job_*/preprocessing.out
```

**SLURM execution:**

```bash
# Check job queue
ssh cpu_cluster "squeue -u username"

# View master job logs
ssh cpu_cluster "cat /home/username/outputs/{date}/workflow_{job_id}/logs/workflow_master.out"

# View step-specific logs
ssh cpu_cluster "cat /home/username/outputs/{date}/workflow_{job_id}/preprocessing/job_*/preprocessing.out"
```

### Skipping Steps

To skip steps (e.g., if already completed):

```yaml
# In your dataset config
preprocessing:
  enabled: false # Skip preprocessing

embedding_preparation:
  enabled: false # Skip embedding preparation
```

Or via command line:

```bash
python scripts/workflow/submit_workflow_local.py \
    --config-name my_dataset \
    ++preprocessing.enabled=false \
    ++embedding_preparation.enabled=false
```

---

## Documentation

Detailed documentation for each component:

### Pipeline Steps

- **[Download](scripts/download/README.md)** - Data acquisition and subsetting
- **[Preprocessing](scripts/preprocessing/README.md)** - QC, filtering, normalization
- **[Embedding](scripts/embed/README.md)** - PCA, scVI, HVG, Geneformer embeddings
- **[Dataset Creation](scripts/dataset_creation/README.md)** - HuggingFace dataset generation

### Configuration

- **[Dataset Configuration Template](conf/dataset_default.yaml)** - All available parameters
- **[Dataset Configuration Example](conf/dataset_config_example.yaml)** - Working example
- **[Workflow Orchestrator Config](conf/workflow_orchestrator.yaml)** - Workflow settings

### Source Code

- **[`src/adata_hf_datasets/`](src/adata_hf_datasets/)** - Core library code
- **[`scripts/`](scripts/)** - Executable scripts for each step

---

## Troubleshooting

### Installation Issues

**Problem:** `uv sync` fails with missing dependencies

**Solution:**

```bash
# Try with verbose output
uv sync --all-extras -v

# Or install core dependencies first
uv sync
uv sync --extra embed
uv sync --extra workflow
```

**Problem:** Git submodule not initialized

**Solution:**

```bash
git submodule update --init --recursive
```

### Configuration Issues

**Problem:** "Config file not found"

**Solution:**

```bash
# Ensure you're in the project root
cd /path/to/adata_hf_datasets

# Check config file exists
ls conf/my_dataset.yaml

# Use config name without .yaml extension
python scripts/workflow/submit_workflow_local.py --config-name my_dataset
```

**Problem:** "base_file_path is not set"

**Solution:** Ensure `workflow_orchestrator.yaml` has the correct path:

```yaml
workflow:
  local_base_file_path: "/absolute/path/to/data/RNA" # For local
  slurm_base_file_path: "/absolute/path/to/data/RNA" # For SLURM
```

### Execution Issues

**Problem:** SSH timeout when submitting to SLURM

**Solution:**

```bash
# Test SSH connection
ssh cpu_cluster "hostname"

# Check SSH config
cat ~/.ssh/config

# Ensure SSH agent is running
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**Problem:** SLURM job fails immediately

**Solution:**

```bash
# Check SLURM logs on cluster
ssh cpu_cluster "cat /home/username/outputs/{date}/workflow_{job_id}/logs/workflow_master.err"

# Verify partition names
ssh cpu_cluster "sinfo"

# Check virtual environment exists on cluster
ssh cpu_cluster "ls /home/username/adata_hf_datasets/.venv/bin/python"
```

**Problem:** "File not found" errors during workflow

**Solution:**

- Verify `base_file_path` is accessible and has correct permissions
- For SLURM: Ensure `base_file_path` is accessible from both CPU and GPU clusters
- Check that previous steps completed successfully

### Getting Help

For more help:

1. Check the step-specific README files linked above
2. Review log files in `outputs/{date}/workflow_{timestamp}/`
3. Check the `.err` files for error messages
4. Review the dataset config for any misconfigurations

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{adata_hf_datasets,
  title = {AnnData HuggingFace Datasets Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/adata_hf_datasets}
}
```

---

## License

[Specify your license here]

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Acknowledgments

This pipeline builds on:

- [Scanpy](https://scanpy.readthedocs.io/) for single-cell analysis
- [scVI](https://scvi-tools.org/) for probabilistic models
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer) for transformer embeddings
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) for dataset management
