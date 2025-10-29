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
- [Test Run](#test-run)
- [HuggingFace Hub Integration](#huggingface-hub-integration)
- [Configuration](#configuration)
  - [Dataset Configuration](#dataset-configuration)
  - [Workflow Orchestrator Configuration](#workflow-orchestrator-configuration)
- [Quick Start](#quick-start)
  - [Local Execution](#local-execution-macos-linux)
  - [SLURM Cluster Execution](#slurm-cluster-execution)
- [Nextcloud Integration](#nextcloud-integration)
- [Pipeline Steps](#pipeline-steps)
- [Advanced Usage](#advanced-usage)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10-3.13 (see [Python version requirements](#python-version-requirements))
- Git (submodules only needed if installing Geneformer support or running the full pipeline)

### Installation Methods

Choose the installation method based on your use case:

- **Pipeline usage**: Running workflow scripts and orchestrators → Use `uv` (Option 1)
- **Library usage**: Importing as a dependency in other projects → Use `pip` (Option 2)

#### Option 1: Using uv (For Pipeline/Workflow Usage)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer that handles submodule dependencies automatically. **Use this method if you want to run the pipeline scripts and workflows.**

1. **Clone the repository:**

```bash
git clone https://github.com/mengerj/adata_hf_datasets.git
cd adata_hf_datasets
```

2. **Install dependencies:**

```bash
# Install without Geneformer (no submodules needed)
uv sync --all-extras --no-extra geneformer

# OR install with Geneformer (requires submodules)
git submodule update --init --recursive
uv sync --all-extras
```

3. **Activate the virtual environment:**

```bash
source .venv/bin/activate
```

#### Option 2: Using pip (For Library Usage in Other Projects)

**Use this method if you want to import and use this package as a dependency in another project.** This is the recommended approach for library usage - no submodules are required for basic usage (no geneformer)

**From GitHub (recommended for library usage):**

```bash
pip install git+https://github.com/mengerj/adata_hf_datasets.git
```

**Or add to your project's requirements.txt:**

```txt
git+https://github.com/mengerj/adata_hf_datasets.git
```

**From local clone (for development):**

```bash
git clone https://github.com/mengerj/adata_hf_datasets.git
cd adata_hf_datasets
pip install .
# Or for editable install: pip install -e .
```

**Optional: Install Geneformer support** (only if you need Geneformer embeddings):

```bash
# First initialize submodules
git submodule update --init --recursive

# Then install Geneformer from the local submodule
pip install external/Geneformer
```

**Example: Using as a library in your project**

```python
from adata_hf_datasets.embed import GeneSelectEmbedder10k
from adata_hf_datasets.pp import preprocess_adata
import anndata as ad

# Use the embedders and preprocessing functions
embedder = GeneSelectEmbedder10k(n_components=50)
embeddings = embedder.embed(adata=your_adata)

# Or use preprocessing
processed_adata = preprocess_adata(your_raw_adata)
```

**Note:** Geneformer can only be installed on Linux machines with CUDA support and requires submodules to be initialized.

### Python Version Requirements

The package requires Python **3.10, 3.11, 3.12, or 3.13**. The current requirement is `>=3.10,<3.14`. Python 3.9 and earlier, or Python 3.14+ are not supported.

### Submodule Control

Submodules are **optional** and only needed if you want to use Geneformer embeddings:

- **Without submodules:** You can install and use all features except Geneformer embeddings
- **With submodules:** Initialize with `git submodule update --init --recursive` before installing Geneformer

To avoid submodule initialization during clone, use:

```bash
git clone --recurse-submodules=no https://github.com/mengerj/adata_hf_datasets.git
```

### What Gets Installed

The base installation includes:

- Core dependencies (anndata, scanpy, datasets, huggingface-hub)
- Embedding tools (scVI, PCA, HVG)
- Workflow orchestration tools (Hydra)
- All optional dependencies except Geneformer

Geneformer support is only installed if:

1. Submodules are initialized
2. You install from the local path (see above)

---

## Test Run

Before attempting to add your own dataset, try running the workflow with the example data.
This will download a .h5ad, preprocess it, run several embedders and create a HuggingFace dataset.

**Note:** This test run will NOT use HuggingFace Hub or Nextcloud (both are disabled in the example config for simplicity).

### Run the Test Workflow

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run workflow in foreground (recommended for first runs)
python scripts/workflow/submit_workflow_local.py \
    --config-name dataset_config_example \
    --foreground

# Or run in background (detached)
python scripts/workflow/submit_workflow_local.py \
    --config-name dataset_config_example
```

### Monitor Progress

Check the main log to see the progress:

```bash
# View workflow summary (replace date/run_id with your actual run)
tail -f outputs/2025-*/workflow_local_*/logs/workflow_summary.log
```

Specific logs for each step are in their respective subfolders:

- `outputs/{date}/workflow_{run_id}/preprocessing/`
- `outputs/{date}/workflow_{run_id}/embedding/`
- `outputs/{date}/workflow_{run_id}/dataset_creation/`

### Load the Created Dataset

Once the workflow completes successfully, find the dataset location in the logs:

```bash
# Check the dataset creation output (it will show the final location)
cat outputs/*/workflow_local_*/dataset_creation/job_local_*/create_ds_0.out
```

Load and inspect the dataset:

```python
from datasets import load_from_disk

# Replace with your actual path from the logs
dataset_path = "outputs/2025-*/workflow_local_*_*/dataset_creation/job_local_*/job_0/demo_dataset"

# Load the dataset
ds = load_from_disk(dataset_path)

# Inspect it
print(ds)
# DatasetDict({
#     train: Dataset({...})
#     validation: Dataset({...})
# })

# Check a sample
print(ds['train'][0])
```

**Important:** Use `load_from_disk()` to load locally saved datasets. The `load_dataset()` function is for loading from the HuggingFace Hub.

---

## HuggingFace Hub Integration

The pipeline can automatically publish datasets to the HuggingFace Hub for easy sharing and distribution.

### Setup

1. **Create a HuggingFace account:**
   - Visit [https://huggingface.co/join](https://huggingface.co/join)
   - Create an account (free)

2. **Get your access token:**
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with **write** permissions
   - Copy the token

3. **Configure authentication:**

Create a `.env` file in the project root:

```bash
# In the project root directory
cat > .env << 'EOF'
# HuggingFace Hub authentication
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx  # Your token here
EOF
```

Or log in via CLI:

```bash
huggingface-cli login
# Paste your token when prompted
```

4. **Configure dataset publication:**

In your dataset config (or `conf/dataset_default.yaml`):

```yaml
dataset_creation:
  enabled: true
  push_to_hub: true # Enable HuggingFace Hub upload
  base_repo_id: "your-hf-username" # Your HuggingFace username
```

### Usage

When the workflow completes, your dataset will be published to:

```
https://huggingface.co/datasets/{your-username}/{dataset-name}_{format}_{caption-key}
```

Example: `https://huggingface.co/datasets/jo-mengr/cellxgene_pseudo_bulk_10k_multiplets_natural_language_annotation`

The dataset will include:

- ✅ All splits (train/validation or test)
- ✅ Automatically generated README with dataset card
- ✅ Schema and feature descriptions
- ✅ Download statistics and usage examples

### Private vs Public Datasets

By default, datasets are uploaded as **private**. You can control this in your dataset configuration:

```yaml
dataset_creation:
  push_to_hub: true
  base_repo_id: "your-hf-username"
  private: true # true = private (default), false = public
```

To make a dataset public, set `private: false` in your config, or update the repository settings on HuggingFace after upload.

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
     name: "human_pancreas"
     description: "Human pancreas dataset"
     download_url: "https://example.com/data.h5ad"
   ```

2. **Common Keys** (used across all steps):

   ```yaml
   batch_key: "batch_id" # Batch/dataset identifier
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

Before attempting to run on slurm, you have make sure that the repository is installed on the cluster. Follow the same steps as locally to install. UV can be installed without sudo rights.

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
- Always keep the repos synced, for example when changing a configuration file

**Shared Storage:**

- `slurm_base_file_path` **must be accessible by both CPU and GPU clusters**
- Typically a global scratch filesystem (e.g., `/scratch/global/username/`)
- Data is written by one cluster and read by another during the workflow
- Local execution is actually much faster, since I/O speeds on a global filesystem are usually very slow, and the pipeline requires reading data into memory at several steps. If you don't need a gpu, or have a gpu locally, I would recommend to work locally. But if you don't bother waiting a while and want to just submit a bunch of datasets, the cluster is better suited.

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

## Nextcloud Integration

Nextcloud integration allows you to store large AnnData files remotely, making your HuggingFace datasets truly autonomous and shareable without local file dependencies.

### Why Nextcloud?

HuggingFace datasets store only **metadata** (cell sentences, captions, negative indices). The actual AnnData files with expression matrices and embeddings are stored separately. Nextcloud provides:

- ☁️ **Remote storage** for large AnnData files
- 🔗 **Share links** embedded in the dataset for downstream access
- 🌐 **Independence** from local file systems
- 🤝 **Easy sharing** - anyone with the HF dataset can access the data

### Setup

1. **Get Nextcloud access:**
   - Obtain a Nextcloud account (institutional, self-hosted, or cloud provider)
   - You need: URL, username, and password

2. **Configure credentials:**

Create or edit the `.env` file in the project root:

```bash
# In the project root directory
cat >> .env << 'EOF'

# Nextcloud authentication
NEXTCLOUD_URL=https://cloud.example.com  # Your Nextcloud instance URL (what you type in browser)
NEXTCLOUD_USER=your-username              # Your Nextcloud username
NEXTCLOUD_PASSWORD=your-password          # Your Nextcloud password
EOF
```

**Security note:** The `.env` file is in `.gitignore` and will not be committed to git.

3. **Enable Nextcloud in dataset config:**

```yaml
dataset_creation:
  enabled: true
  use_nextcloud: true # Enable Nextcloud upload

  nextcloud_config:
    url: "NEXTCLOUD_URL" # Will be read from .env
    username: "NEXTCLOUD_USER" # Will be read from .env
    password: "NEXTCLOUD_PASSWORD" # Will be read from .env
    remote_path: "" # Automatically set based on dataset
```

The environment variables will be automatically resolved at runtime.

### How It Works

1. **During dataset creation:**
   - Processed AnnData files are uploaded to Nextcloud
   - Share links are generated for each file
   - Links are embedded in the HuggingFace dataset

2. **When using the dataset:**
   - The `adata_link` column contains Nextcloud share URLs
   - Downstream models can download files on-demand
   - No local file dependencies needed

### Nextcloud Directory Structure

Files are organized in Nextcloud as:

```
{remote_path}/
└── {dataset_name}/
    ├── train/
    │   ├── chunk_0.zarr.zip
    │   └── chunk_1.zarr.zip
    └── validation/
        └── chunk_0.zarr.zip
```

### Other Storage Backends

**Currently supported:** Nextcloud only

**Want other backends?** If you need support for other cloud storage providers (AWS S3, Google Drive, Figshare, Zenodo, etc.), please [open an issue](https://github.com/mengerj/adata_hf_datasets/issues) describing your use case. We're interested in adding compatibility for additional storage backends!

**For developers:** The storage interface is in `src/adata_hf_datasets/file_utils.py`. Contributions for new backends are welcome!

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
- Memory-efficient streaming to disk

**Configuration:**

```yaml
embedding_cpu:
  enabled: true
  methods: ["pca", "scvi_fm", "gs10k"]
  embedding_dim_map:
    pca: 50
    scvi_fm: 50
    gs10k: 10000
```

### 5. GPU Embedding

Generates embeddings using GPU-based methods.

**Documentation:** [`scripts/embed/README.md`](scripts/embed/README.md)

**Key Features:**

- Geneformer: Transformer-based embeddings (! Needs Cuda device !)
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
uv sync --all-extras --no-extra geneformer -v
```

**Problem:** Python version error: "requires a different Python: X.X.X not in '<3.14,>=3.10'"

**Solution:**
The package requires Python 3.10, 3.11, 3.12, or 3.13. If you see this error, you're likely using Python 3.14+ or an older version (3.9 or earlier). Check your Python version:

```bash
python --version
```

If you need to use a different Python version, consider using `pyenv` or a virtual environment with the correct version.

**Problem:** Pip installation is very slow or fails

**Solution:**
Pip installation can be slow due to dependency resolution. Consider:

1. Using `uv` instead (much faster): `uv pip install .`
2. Using pre-built wheels: `pip install --only-binary :all: .` (may not work for local packages)
3. Installing in a clean virtual environment

**Problem:** Git submodule issues/maintenance

**Solutions:**

```bash
# Initialize submodules (only needed for Geneformer)
git submodule update --init --recursive

# Skip submodules entirely when cloning
git clone --recurse-submodules=no https://github.com/mengerj/adata_hf_datasets.git

# Remove submodule initialization if already cloned
git submodule deinit --all -f
```

**Note:** Submodules are optional and only needed for Geneformer support. You can install and use the package without them.

**Problem:** `pip install .[geneformer]` doesn't install Geneformer

**Solution:**
The `geneformer` extra is a marker only. To install Geneformer:

1. Initialize submodules: `git submodule update --init --recursive`
2. Install from local path: `pip install external/Geneformer`
3. Or use `uv sync --all-extras` which handles this automatically

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

### Dataset Loading Issues

**Problem:** `ValueError: Couldn't infer the same data file format for all splits`

**Solution:** Use `load_from_disk()` instead of `load_dataset()` for locally saved datasets:

```python
# ✅ Correct - for locally saved datasets
from datasets import load_from_disk
ds = load_from_disk("/path/to/dataset")

# ❌ Wrong - this is for HuggingFace Hub
from datasets import load_dataset
ds = load_dataset("/path/to/dataset")  # Will fail!
```

`load_dataset()` is only for loading datasets from the HuggingFace Hub or inferring formats from raw files. For datasets saved with `save_to_disk()`, always use `load_from_disk()`.

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
  author = {Jonatan Menger},
  year = {2025},
  url = {https://github.com/mengerj/adata_hf_datasets}
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

- [AnnData](https://anndata.readthedocs.io/en/stable/) for handling data matrices
- [Scanpy](https://scanpy.readthedocs.io/) for single-cell analysis
- [scVI](https://scvi-tools.org/) for probabilistic models
- [Geneformer](https://huggingface.co/ctheodoris/Geneformer) for transformer embeddings
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/) for dataset management
