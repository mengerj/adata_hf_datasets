# Preprocessing Pipeline

This document provides a comprehensive guide to the preprocessing pipeline for single-cell RNA-seq data in the `adata_hf_datasets` project. The preprocessing step transforms raw AnnData files into cleaned, normalized, and quality-controlled datasets ready for embedding generation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Workflow Integration](#workflow-integration)
- [Dataset Configuration](#dataset-configuration)
  - [Configuration Structure](#configuration-structure)
  - [Common Keys](#common-keys)
  - [Preprocessing Parameters](#preprocessing-parameters)
- [Processing Pipeline](#processing-pipeline)
  - [Input Handling](#input-handling)
  - [Train/Val Split](#trainval-split)
  - [Chunk-Based Processing](#chunk-based-processing)
  - [Core Processing Steps](#core-processing-steps)
- [Execution Options](#execution-options)
  - [Running on SLURM](#running-on-slurm)
  - [Running Locally](#running-locally)
  - [Environment Variables](#environment-variables)
- [Advanced Options](#advanced-options)
  - [Category Consolidation](#category-consolidation)
  - [SRA Metadata Fetching](#sra-metadata-fetching)
  - [Bimodal Splitting](#bimodal-splitting)
  - [Layer Management](#layer-management)
- [Output Structure](#output-structure)
- [Quality Control](#quality-control)
- [Troubleshooting](#troubleshooting)

---

## Overview

The preprocessing pipeline performs the following high-level operations:

1. **Data Loading**: Loads large AnnData files in "backed" mode to minimize memory usage
2. **Train/Validation Split**: Optionally splits data into training and validation sets
3. **Quality Control**: Filters low-quality cells based on QC metrics (mitochondrial content, gene counts, etc.)
4. **Gene/Cell Filtering**: Removes genes expressed in too few cells and cells expressing too few genes
5. **Category Consolidation**: Merges or removes low-frequency categories in metadata
6. **Normalization**: Normalizes and log-transforms gene expression data
7. **Highly Variable Gene Selection**: Identifies highly variable genes for downstream analysis
8. **Geneformer Preparation**: Adds Ensembl IDs and prepares data for Geneformer embeddings
9. **SRA Metadata Enrichment**: Optionally fetches additional metadata from SRA database
10. **Chunk-Based Output**: Writes processed data in chunks to handle large datasets

The pipeline is designed to handle datasets of any size through memory-efficient chunk-based processing.

---

## Quick Start

### Basic Usage (SLURM)

```bash
# Set the dataset configuration
export DATASET_CONFIG=dataset_example

# Submit the preprocessing job
sbatch scripts/preprocessing/run_preprocess.slurm
```

### Basic Usage (Local)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run preprocessing with a specific config
python scripts/preprocessing/preprocess.py --config-name=dataset_example
```

---

## Workflow Integration

The preprocessing step is typically the **second step** in the complete data processing workflow:

```
1. Download       → 2. Preprocessing → 3. Embedding Prep → 4. CPU Embedding → 5. GPU Embedding → 6. Dataset Creation
   (optional)        (this step)
```

### Automatic Workflow Execution

For comprehensive information on running the complete workflow (including preprocessing as part of the automated pipeline), please refer to:

**📖 [Workflow Orchestration Guide](../workflow/README.md)**

The workflow orchestrator handles:

- Automatic execution of preprocessing after download
- Input/output path management between stages
- SLURM job submission and dependency tracking
- Unified logging and error consolidation
- Configuration propagation across pipeline stages

### Manual Execution

For debugging or running preprocessing as a standalone step, you can execute it independently:

```bash
# Run with custom input file
python scripts/preprocessing/preprocess.py \
    --config-name=dataset_example \
    ++base_file_path=/path/to/raw_data.h5ad
```

See the [Execution Options](#execution-options) section below for detailed standalone execution instructions.

---

## Dataset Configuration

### Configuration Structure

Preprocessing configuration uses a **dataset-centric** design where all pipeline parameters are defined in a single YAML file. This configuration inherits from `dataset_default.yaml` and can override any defaults.

**Example Configuration** (`conf/dataset_example.yaml`):

```yaml
defaults:
  - dataset_default.yaml
  - _self_

# Dataset metadata
dataset:
  name: "example_dataset"
  description: "Example dataset for demonstration"
  download_url: "https://example.com/data.h5ad"
  full_name: "example_dataset_full"

# Common keys used across all pipeline stages
batch_key: "dataset_title"
annotation_key: "cell_type"
caption_key: "natural_language_annotation"
instrument_key: "assay"
other_bio_labels: ["tissue", "disease"]

# Preprocessing configuration
preprocessing:
  # Dataset-specific filtering parameters
  min_cells: 15 # Minimum cells per gene
  min_genes: 150 # Minimum genes per cell
  n_top_genes: 512 # Number of highly variable genes
  count_layer_key: "counts" # Layer with raw counts

  # Category consolidation
  category_threshold: 5 # Minimum category frequency
  remove_low_frequency: true # Remove or rename low-frequency categories

  # Train/validation split
  split_dataset: true # Enable splitting
  train_split: 0.8 # Training set fraction
  random_seed: 42 # Random seed for reproducibility

  # Execution parameters
  chunk_size: 100000 # Number of cells per processing chunk
  output_format: "zarr" # Output format: "zarr" or "h5ad"
  geneformer_pp: true # Enable Geneformer preprocessing

  # Quality control
  metrics_of_interest:
    - "n_genes_by_counts"
    - "total_counts"
    - "pct_counts_mt"

  # SRA metadata fetching
  skip_sra_fetch: false
  sra_max_retries: 3
  sra_continue_on_fail: true
  sra_chunk_size: 10000
  sra_extra_cols: ["library_layout", "library_source", "instrument"]

  # Layer management
  layers_to_delete: null # List of layers to remove
```

### Common Keys

Common keys are defined once at the top level and automatically propagated to all workflow stages:

- **`batch_key`**: Column in `.obs` containing batch/dataset identifiers (e.g., `"dataset_title"`, `"donor_id"`)
- **`annotation_key`**: Column in `.obs` containing cell type annotations (e.g., `"cell_type"`)
- **`caption_key`**: Column in `.obs` containing natural language descriptions (can be `null`)
- **`instrument_key`**: Column in `.obs` containing sequencing instrument information (can be `null`)
- **`other_bio_labels`**: List of additional biological metadata columns for consolidation (e.g., `["tissue", "disease", "age"]`)

### Preprocessing Parameters

The `preprocessing` section contains both dataset-specific and execution parameters:

#### Filtering Parameters

| Parameter         | Type | Default | Description                                                |
| ----------------- | ---- | ------- | ---------------------------------------------------------- |
| `min_cells`       | int  | 20      | Genes expressed in fewer cells are removed                 |
| `min_genes`       | int  | 200     | Cells expressing fewer genes are removed                   |
| `n_top_genes`     | int  | 5000    | Number of highly variable genes to select                  |
| `count_layer_key` | str  | `null`  | Key for raw counts in `.layers` (if `null`, uses `.raw.X`) |

#### Consolidation Parameters

| Parameter                  | Type | Default | Description                                                                             |
| -------------------------- | ---- | ------- | --------------------------------------------------------------------------------------- |
| `category_threshold`       | int  | 5       | Categories with fewer samples are consolidated                                          |
| `remove_low_frequency`     | bool | false   | If `true`, removes cells; if `false`, renames to "remaining {category}"                 |
| `consolidation_categories` | list | auto    | Categories to consolidate (auto-generated from `annotation_key` and `other_bio_labels`) |

#### Split Parameters

| Parameter       | Type  | Default | Description                                              |
| --------------- | ----- | ------- | -------------------------------------------------------- |
| `split_dataset` | bool  | false   | Whether to create train/val split                        |
| `train_split`   | float | 0.9     | Fraction of data for training                            |
| `random_seed`   | int   | 42      | Random seed for split                                    |
| `split_fn`      | str   | `null`  | Custom split function (e.g., `"my_module:custom_split"`) |

#### Execution Parameters

| Parameter       | Type | Default  | Description                              |
| --------------- | ---- | -------- | ---------------------------------------- |
| `chunk_size`    | int  | 200000   | Number of cells processed per chunk      |
| `output_format` | str  | `"zarr"` | Output format: `"zarr"` or `"h5ad"`      |
| `geneformer_pp` | bool | true     | Enable Geneformer-specific preprocessing |

#### SRA Metadata Parameters

| Parameter              | Type | Default  | Description                     |
| ---------------------- | ---- | -------- | ------------------------------- |
| `skip_sra_fetch`       | bool | false    | Skip SRA metadata fetching      |
| `sra_chunk_size`       | int  | `null`   | Chunk size for SRA queries      |
| `sra_extra_cols`       | list | `[null]` | Additional SRA columns to fetch |
| `sra_max_retries`      | int  | 3        | Maximum retry attempts for SRA  |
| `sra_continue_on_fail` | bool | true     | Continue if SRA fetching fails  |

#### Advanced Parameters

| Parameter          | Type | Default | Description                           |
| ------------------ | ---- | ------- | ------------------------------------- |
| `split_bimodal`    | bool | false   | Enable bimodal distribution splitting |
| `bimodal_col`      | str  | `null`  | Column for bimodal splitting          |
| `layers_to_delete` | list | `null`  | Layer names to remove from `.layers`  |

---

## Processing Pipeline

### Input Handling

The preprocessing script uses **backed mode** for initial data loading to minimize memory usage:

```python
# Opens file without loading data into memory
ad_bk = safe_read_h5ad_backed(infile)

# Generate QC plots on backed file
subset_sra_and_plot(adata_bk=ad_bk, cfg=preprocess_cfg, run_dir=run_dir + "/before")
```

**Key Features:**

- `safe_read_h5ad_backed()` handles remote files (e.g., Nextcloud) by creating temporary local copies
- Automatic cleanup of temporary files after processing
- Supports both `.h5ad` and `.zarr` input formats

### Train/Val Split

If `split_dataset: true`, the data is split into training and validation sets:

```python
def default_split_fn(adata, train_frac=0.8, random_state=0):
    """Random train/val split returning lists of obs indices."""
    n = adata.n_obs
    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    cut = int(train_frac * n)
    return idx[:cut].tolist(), idx[cut:].tolist()
```

**Custom Split Functions:**

You can provide a custom split function via `split_fn`:

```yaml
preprocessing:
  split_dataset: true
  split_fn: "my_pipeline.splits:stratified_split"
```

Your custom function must have the signature:

```python
def custom_split(adata: AnnData, train_frac: float, random_state: int) -> tuple[list[int], list[int]]:
    # Return (train_indices, val_indices)
    ...
```

### Chunk-Based Processing

To handle large datasets, preprocessing operates on chunks:

```python
# Initialize chunk loader
loader = BatchChunkLoader(infile, chunk_size, batch_key=batch_key, file_format="zarr")

# Process each chunk
for i, adata_chunk in enumerate(loader):
    # Apply preprocessing steps
    processed_chunk = preprocess_chunk(adata_chunk)

    # Write chunk to disk
    chunk_path = chunk_dir / f"chunk_{i}.zarr"
    processed_chunk.write_zarr(chunk_path)
```

**Benefits:**

- Processes arbitrarily large datasets
- Memory usage bounded by `chunk_size`
- Parallelizable (future enhancement)

### Core Processing Steps

Each chunk undergoes the following transformations (in order):

#### 1. Raw Counts Layer Creation

```python
ensure_raw_counts_layer(adata, raw_layer_key=count_layer_key)
```

- Ensures raw counts are stored in `adata.layers["counts"]`
- If `count_layer_key` is specified, uses that layer
- Otherwise, checks `adata.raw.X` or `adata.X`
- Critical for downstream normalization and embedding generation

#### 2. Layer Deletion (Optional)

```python
adata = delete_layers(adata, layers_to_delete)
```

- Removes specified layers from `adata.layers` to reduce file size
- Useful when raw data contains unnecessary technical replicates or temporary layers

#### 3. Bimodal Splitting (Optional)

```python
if split_bimodal and bimodal_col in adata.obs:
    log_col = f"{bimodal_col}_log"
    adata.obs[log_col] = np.log1p(adata.obs[bimodal_col].values)
    adata_splits = split_if_bimodal(adata, column_name=log_col)
```

- Splits data based on bimodal distribution in a metadata column
- Useful for datasets with mixed cell types or conditions
- Each split is processed independently and then recombined

#### 4. SRA Metadata Fetching (Optional)

```python
adata = maybe_add_sra_metadata(
    adata,
    chunk_size=sra_chunk_size,
    new_cols=sra_extra_cols,
    skip_sra_fetch=skip_sra_fetch,
    max_retries=sra_max_retries,
    continue_on_fail=sra_continue_on_fail,
)
```

- Detects SRA-based identifiers (e.g., SRX, SRR IDs)
- Fetches additional metadata from SRA database using `pysradb`
- Common columns: `library_layout`, `library_source`, `instrument`
- Adds fallback values if fetching fails and `continue_on_fail=True`

#### 5. Quality Control

```python
adata = pp_quality_control(adata)
```

**QC Metrics Calculated:**

- Total counts per cell (`total_counts`)
- Number of genes per cell (`n_genes_by_counts`)
- Percentage of mitochondrial genes (`pct_counts_mt`)
- Percentage of ribosomal genes (`pct_counts_ribo`)
- Percentage of hemoglobin genes (`pct_counts_hb`)
- Percentage of counts in top N genes (`pct_counts_in_top_20_genes`)

**Outlier Detection:**

- Uses Median Absolute Deviation (MAD) for outlier detection
- Default: 5 MADs for main metrics, 3 MADs for mitochondrial content
- Absolute threshold: 8% mitochondrial content
- Flags and removes cells exceeding thresholds

**Gene Labeling:**

```python
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
adata.var["hb"] = adata.var_names.str.upper().str.contains(r"^HB[^P]")
```

#### 6. General Preprocessing

```python
adata = pp_adata_general(
    adata,
    min_cells=min_cells,
    min_genes=min_genes,
    batch_key=batch_key,
    n_top_genes=n_top_genes,
    categories=consolidation_categories,
    category_threshold=category_threshold,
    remove=remove_low_frequency,
)
```

**Steps:**

1. **Make observation names unique**
2. **Filter genes and cells:**
   - Remove genes expressed in < `min_cells` cells
   - Remove cells expressing < `min_genes` genes
3. **Consolidate low-frequency categories:**
   - Identifies categories with < `category_threshold` samples
   - Either removes cells (if `remove=True`) or renames to "remaining {category}"
4. **Normalize and log-transform:**
   - Normalizes counts to 10,000 reads per cell
   - Applies log1p transformation
   - Stores raw counts in `adata.layers["counts"]`
5. **Highly variable gene selection:**
   - Selects `n_top_genes` highly variable genes
   - Uses batch-aware HVG selection if `batch_key` is provided
   - Marks genes in `adata.var["highly_variable"]` (does not filter)

#### 7. Geneformer Preprocessing (Optional)

```python
if geneformer_pp:
    adata = pp_adata_geneformer(adata)
```

**Steps:**

1. **Add Ensembl IDs:**
   - Fetches Ensembl IDs from gene names using `pybiomart`
   - Stores in `adata.var["ensembl_id"]`
   - Required for Geneformer token vocabulary

2. **Add QC metrics:**
   - Calculates `n_counts` if not present
   - Adds `percent_top` metrics for top 50/100/200/500 genes

3. **Add sample index:**
   - Creates stable numeric index in `adata.obs["sample_index"]`
   - Required for Geneformer tokenization

#### 8. Instrument Description Prepending (Optional)

```python
if instrument_key and description_key:
    prepend_instrument_to_description(
        adata,
        instrument_key=instrument_key,
        description_key=description_key,
    )
```

- Prepends instrument/assay information to natural language descriptions
- Example: `"10x 3' v3: Memory B cells from lung tissue"`

---

## Execution Options

### Running on SLURM

The SLURM script (`run_preprocess.slurm`) handles job submission with proper environment setup:

```bash
#!/bin/bash
#SBATCH --job-name=pp
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --partition=slurm

# Set dataset configuration
export DATASET_CONFIG=dataset_example

# Optional: Override input file
export BASE_FILE_PATH=/path/to/custom_input.h5ad

# Submit job
sbatch scripts/preprocessing/run_preprocess.slurm
```

**Environment Variables:**

- **`DATASET_CONFIG`**: Name of the dataset configuration file (without `.yaml`)
- **`BASE_FILE_PATH`**: (Optional) Override the input file path
- **`WORKFLOW_DIR`**: (Set by orchestrator) Workflow directory for unified logging
- **`PROJECT_DIR`**: (Optional) Project root directory (default: `/home/menger/git/adata_hf_datasets`)
- **`VENV_PATH`**: (Optional) Virtual environment path (default: `.venv`)

**SLURM Resource Recommendations:**

| Dataset Size    | Memory | Time | Partition     |
| --------------- | ------ | ---- | ------------- |
| < 100k cells    | 60G    | 4h   | slurm         |
| 100k-500k cells | 120G   | 12h  | slurm         |
| 500k-1M cells   | 250G   | 24h  | slurm         |
| > 1M cells      | 500G   | 48h  | slurm-highmem |

### Running Locally

For smaller datasets or debugging:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default config
python scripts/preprocessing/preprocess.py --config-name=dataset_example

# Override specific parameters
python scripts/preprocessing/preprocess.py \
    --config-name=dataset_example \
    ++base_file_path=/path/to/input.h5ad \
    ++preprocessing.chunk_size=50000 \
    ++preprocessing.min_genes=150 \
    ++preprocessing.split_dataset=false

# Specify custom output directory
python scripts/preprocessing/preprocess.py \
    --config-name=dataset_example \
    ++hydra.run.dir=outputs/my_custom_run
```

### Environment Variables

Key environment variables used by the preprocessing pipeline:

| Variable                | Purpose                    | Set By                |
| ----------------------- | -------------------------- | --------------------- |
| `WORKFLOW_DIR`          | Unified workflow directory | Workflow orchestrator |
| `DATASET_CONFIG`        | Dataset configuration name | User/SLURM script     |
| `BASE_FILE_PATH`        | Override input file path   | User/orchestrator     |
| `PROJECT_DIR`           | Project root directory     | SLURM script          |
| `VENV_PATH`             | Virtual environment path   | SLURM script          |
| `HDF5_USE_FILE_LOCKING` | Disable HDF5 locking       | Preprocessing script  |

---

## Advanced Options

### Category Consolidation

Category consolidation removes or renames rare categories in metadata columns to improve downstream analysis:

**Configuration:**

```yaml
preprocessing:
  category_threshold: 5 # Minimum samples per category
  remove_low_frequency: false # Rename instead of remove

# Consolidation applies to these columns:
annotation_key: "cell_type" # e.g., cell type
other_bio_labels: ["tissue", "disease", "age"]
```

**Behavior:**

- **`remove_low_frequency: true`**: Removes cells with rare categories
  - Example: Cell type appearing in only 3 cells → cell is removed

- **`remove_low_frequency: false`**: Renames rare categories
  - Example: `cell_type="rare_T_cell"` (3 cells) → `cell_type="remaining cell_type"` (preserves cells)

**Auto-generation:**

If not specified, `consolidation_categories` is automatically generated from `annotation_key` and `other_bio_labels`:

```python
consolidation_categories = [annotation_key] + other_bio_labels
# Example: ["cell_type", "tissue", "disease", "age"]
```

### SRA Metadata Fetching

For datasets with SRA identifiers (SRX, SRR, etc.), additional metadata can be fetched:

**Configuration:**

```yaml
preprocessing:
  skip_sra_fetch: false
  sra_chunk_size: 10000 # Query 10k samples at a time
  sra_max_retries: 3 # Retry on network errors
  sra_continue_on_fail: true # Don't fail if SRA is down
  sra_extra_cols:
    - "library_layout" # SINGLE or PAIRED
    - "library_source" # TRANSCRIPTOMIC, GENOMIC, etc.
    - "instrument" # Sequencing instrument
```

**Detection Logic:**

```python
# Checks if data is from SRA by looking for SRX pattern
if adata.obs.index[0].startswith("SRX"):
    fetch_sra_metadata(adata, ...)
```

**Fetched Columns:**

Common SRA metadata columns:

- `library_layout`: Single-end vs paired-end
- `library_source`: TRANSCRIPTOMIC, GENOMIC, etc.
- `instrument`: Illumina HiSeq, NextSeq, NovaSeq, etc.
- `library_strategy`: RNA-Seq, ChIP-Seq, etc.
- `library_selection`: cDNA, PCR, etc.

**Error Handling:**

- If SRA database is unreachable and `continue_on_fail=true`, adds placeholder: `"unknown_sra_unavailable"`
- Implements exponential backoff with `max_retries`
- Processes in chunks to avoid overwhelming SRA API

### Bimodal Splitting

Some datasets have bimodal distributions in certain metadata (e.g., high/low expression of a marker gene):

**Configuration:**

```yaml
preprocessing:
  split_bimodal: true
  bimodal_col: "CD8_expression" # Column with bimodal distribution
```

**Processing:**

1. Log-transforms the bimodal column: `log_col = log1p(bimodal_col)`
2. Detects modes using Gaussian Mixture Model or similar
3. Splits data into two groups
4. Processes each group independently (different filtering/normalization)
5. Recombines splits with a `bimodal_split` label in `.obs`

**Use Cases:**

- Separating naive vs activated T cells based on activation markers
- Splitting by high vs low mitochondrial content
- Separating cycling vs non-cycling cells

### Layer Management

AnnData objects can contain multiple layers that may not be needed:

**Configuration:**

```yaml
preprocessing:
  layers_to_delete: ["replicate_1", "replicate_2", "technical_replicate"]
```

**Behavior:**

```python
# Removes specified layers to reduce file size
for layer in layers_to_delete:
    if layer in adata.layers:
        del adata.layers[layer]
```

**Common Layers to Remove:**

- Technical replicates (`replicate_1`, `replicate_2`)
- Intermediate processing layers (`spliced`, `unspliced` if not doing RNA velocity)
- Temporary layers created during previous processing

---

## Output Structure

### Path Generation

The preprocessing pipeline uses a **path generation system** that automatically creates directory structures based on the dataset configuration. The `generate_paths_from_config()` utility function (in `src/adata_hf_datasets/workflow/config_utils.py`) generates all paths from the `base_file_path` parameter.

**Key Parameter:**

- **`base_file_path`**: Base directory for all data files (defined in dataset config or via environment variable `BASE_FILE_PATH`)

**Path Structure:**

The system creates three main subdirectories within `base_file_path`:

1. **`raw/`**: Downloaded raw data
2. **`processed/`**: Preprocessed data (output of this step)
3. **`processed_with_emb/`**: Data with embeddings (output of embedding steps)

Each subdirectory is further organized by:

- **Training/Test designation**: `train/` or `test/` (based on `preprocessing.split_dataset`)
- **Dataset name**: Subdirectory named after `dataset.name`

### Data Output Directory Layout

**For training datasets** (`preprocessing.split_dataset: true`):

```
$base_file_path/
├── raw/
│   └── train/
│       ├── {dataset_name}_full.h5ad    # Full downloaded file (if download enabled)
│       └── {dataset_name}.h5ad         # Subset (input to preprocessing)
│
├── processed/
│   └── train/
│       └── {dataset_name}/             # Preprocessed data (THIS STEP)
│           ├── train/                  # Training split
│           │   ├── chunk_0.zarr/       # First chunk
│           │   ├── chunk_1.zarr/       # Additional chunks (if dataset is large)
│           │   └── ...
│           └── val/                    # Validation split
│               └── chunk_0.zarr/
│
└── processed_with_emb/
    └── train/
        └── {dataset_name}/             # With embeddings (future step)
            ├── train/
            │   └── chunk_0.zarr
            └── val/
                └── chunk_0.zarr
```

**For test/inference datasets** (`preprocessing.split_dataset: false`):

```
$base_file_path/
├── raw/
│   └── test/
│       └── {dataset_name}.h5ad
│
├── processed/
│   └── test/
│       └── {dataset_name}/
│           └── all/                    # Single split (no train/val)
│               └── chunk_0.zarr/
│
└── processed_with_emb/
    └── test/
        └── {dataset_name}/
            └── all/
                └── chunk_0.zarr
```

**Generated Paths in Config:**

When `apply_all_transformations(cfg)` is called, these paths are automatically added to the config:

```yaml
# Auto-generated paths added to config
preprocessing:
  input_file: "{base_file_path}/raw/train/{dataset_name}.h5ad"
  output_dir: "{base_file_path}/processed/train/{dataset_name}"

embedding_cpu:
  input_files:
    - "{base_file_path}/processed/train/{dataset_name}/train/chunk_0.zarr"
    - "{base_file_path}/processed/train/{dataset_name}/val/chunk_0.zarr"
  output_dir: "{base_file_path}/processed_with_emb/train/{dataset_name}"
```

### Logs and Metadata Directory Layout

Separate from the data files, logs and QC outputs are stored based on execution context:

**When run as part of a workflow:**

```
$WORKFLOW_DIR/
└── preprocessing/
    └── job_{SLURM_JOB_ID}/
        ├── preprocessing.out           # STDOUT
        ├── preprocessing.err           # STDERR
        ├── before/                     # QC plots before processing
        │   ├── sra_metrics.png
        │   └── qc_violin.png
        ├── after/                      # QC plots after processing
        │   ├── qc_metrics.png
        │   ├── category_dist.png
        │   └── hvg_selection.png
        ├── system_monitor.json         # System resource usage
        ├── system_monitor_plot.png     # Resource usage plot
        └── .hydra/                     # Hydra configuration
            ├── config.yaml             # Resolved configuration
            ├── hydra.yaml
            └── overrides.yaml
```

**When run standalone (not in workflow):**

```
outputs/
└── {YYYY-MM-DD}/
    └── preprocessing/
        └── {RUN_ID}/
            ├── preprocessing.out
            ├── preprocessing.err
            ├── before/                 # QC plots
            ├── after/                  # QC plots
            ├── system_monitor.json
            ├── system_monitor_plot.png
            └── .hydra/
```

**Note:** The data files themselves are always stored in `$base_file_path/processed/...`, regardless of whether running in workflow or standalone mode. Only logs, plots, and metadata are stored in the workflow/output directories.

### Output Files

Each split (train/val or all) contains:

**Zarr Format** (default):

```
chunk_0.zarr/
├── .zattrs                    # Metadata
├── .zgroup                    # Group info
├── X/                         # Expression matrix (log-normalized)
│   ├── .zarray
│   └── 0.0, 0.1, ...         # Data chunks
├── obs/                       # Cell metadata
│   ├── cell_type/...
│   ├── batch/...
│   └── sample_index/...
├── var/                       # Gene metadata
│   ├── highly_variable/...
│   ├── ensembl_id/...
│   └── n_cells/...
├── layers/                    # Data layers
│   └── counts/               # Raw counts
│       ├── .zarray
│       └── 0.0, 0.1, ...
└── obsm/                      # Embeddings (empty after preprocessing)
```

**H5AD Format** (if `output_format: "h5ad"`):

```
chunk_0.h5ad                   # Single HDF5 file
├── /X                         # Expression matrix
├── /obs                       # Cell metadata
├── /var                       # Gene metadata
├── /layers/counts             # Raw counts
└── /obsm                      # Embeddings (empty)
```

### Key Data Contents

After preprocessing, the AnnData object contains:

**`.X`**: Log-normalized expression matrix

- Normalized to 10,000 reads per cell
- Log1p transformed
- Sparse CSR matrix

**`.layers["counts"]`**: Raw count matrix

- Integer counts
- Preserved for downstream analysis

**`.obs`**: Cell metadata (expanded)

- Original metadata columns
- QC metrics: `n_genes_by_counts`, `total_counts`, `pct_counts_mt`, etc.
- Outlier flags: `outlier`, `mt_outlier`
- Sample index: `sample_index` (for Geneformer)
- SRA metadata (if fetched): `library_layout`, `instrument`, etc.

**`.var`**: Gene metadata

- `highly_variable`: Boolean, selected HVGs
- `ensembl_id`: Ensembl gene IDs (if Geneformer preprocessing)
- `n_cells`: Number of cells expressing each gene
- Gene type labels: `mt`, `ribo`, `hb`

**`.uns`**: Unstructured metadata

- `log1p`: Dictionary with normalization parameters
- `hvg`: HVG selection parameters

---

## Quality Control

### QC Plots

The pipeline generates QC plots before and after processing:

**Before Processing** (`before/`):

- Distribution of raw counts per cell
- Number of genes per cell
- Percentage of mitochondrial genes
- Batch-specific metrics

**After Processing** (`after/`):

- Distribution of normalized expression
- HVG selection plots
- Category frequencies
- Batch effects
- Metrics of interest (from config)

**Metrics of Interest:**
These metrics are used in the qc plots
Configurable via `preprocessing.metrics_of_interest`:

```yaml
preprocessing:
  metrics_of_interest:
    - "n_genes_by_counts" # Number of genes detected
    - "total_counts" # Total UMI counts
    - "pct_counts_mt" # Mitochondrial content
    - "pct_counts_ribo" # Ribosomal content
```

### System Monitoring

The preprocessing script includes a `SystemMonitor` that tracks:

- Memory usage (RAM, swap)
- CPU usage
- Disk I/O
- Processing time per chunk

Monitoring data is saved to:

```
<run_dir>/system_monitor.json
<run_dir>/system_monitor_plot.png
```

### Log Files

**Standard Output** (`preprocessing.out`):

- Progress messages
- Chunk processing status
- Cell/gene counts at each step
- QC summary statistics

**Standard Error** (`preprocessing.err`):

- Warnings (e.g., low gene counts in certain batches)
- Error messages
- Stack traces (if failures occur)

**Consolidated Error Log** (workflow mode):

```
$WORKFLOW_DIR/logs/errors_consolidated.log
```

All ERROR-level messages from all steps are aggregated here.

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:**

- Job killed with exit code 137
- Error message: "MemoryError" or "unable to allocate array"

**Solutions:**

- Reduce `chunk_size` in config:
  ```yaml
  preprocessing:
    chunk_size: 50000 # Reduce from default 200000
  ```
- Increase SLURM memory allocation:
  ```bash
  #SBATCH --mem=500G
  ```
- Use Zarr output format (more memory-efficient than H5AD)

#### 2. HDF5 File Locking Errors

**Symptoms:**

- Error: "Unable to open file (file is already open)"
- BlockingIOError on shared filesystems

**Solutions:**

- The script automatically sets `HDF5_USE_FILE_LOCKING=FALSE`
- Ensure no other processes are accessing the file
- Use Zarr format to avoid HDF5 locking issues entirely

#### 3. SRA Metadata Fetching Fails

**Symptoms:**

- Error: "Failed to connect to SRA database"
- Warning: "SRA metadata fetching failed"

**Solutions:**

- Enable `continue_on_fail`:
  ```yaml
  preprocessing:
    sra_continue_on_fail: true
  ```
- Skip SRA fetching entirely:
  ```yaml
  preprocessing:
    skip_sra_fetch: true
  ```
- Increase retry attempts:
  ```yaml
  preprocessing:
    sra_max_retries: 10
  ```

#### 4. No Cells Left After Filtering

**Symptoms:**

- Error: "No cells left after filtering. Exiting."
- ValueError: "No cells left after filtering"

**Solutions:**

- Relax filtering thresholds:
  ```yaml
  preprocessing:
    min_genes: 100 # Reduce from 200
    min_cells: 5 # Reduce from 20
    category_threshold: 2 # Reduce from 5
  ```
- Check QC thresholds in source code (hardcoded):
  - `nmads_main=5` → reduce to 7-10 for more lenient filtering
  - `pct_counts_mt_threshold=8.0` → increase to 10-15 for datasets with naturally higher MT content

#### 5. Ensembl ID Fetching Fails

**Symptoms:**

- Warning: "Could not fetch Ensembl IDs for some genes"
- Many genes missing `ensembl_id`

**Solutions:**

- Ensure gene names are standard (HGNC symbols for human, MGI for mouse)
- Check species parameter in `pp_adata_geneformer.py` (default: `"hsapiens"`)
- For non-human datasets, modify source to use correct species:
  ```python
  add_ensembl_ids(adata, ensembl_col="ensembl_id", species="mmusculus")
  ```

#### 6. Backed File Access Errors

**Symptoms:**

- Error: "Unable to create temporary local copy"
- Permission denied when accessing remote files

**Solutions:**

- Ensure sufficient disk space in `/tmp` or `$TMPDIR`
- Check network connectivity to remote storage (e.g., Nextcloud)
- Provide local copy of input file:
  ```bash
  cp remote/path/data.h5ad local/path/data.h5ad
  python scripts/preprocessing/preprocess.py \
      --config-name=dataset_example \
      ++base_file_path=local/path/data.h5ad
  ```

### Debugging Tips

#### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Check Intermediate Files

After each chunk, inspect the output:

```python
import anndata

# Read first chunk
adata = anndata.read_zarr("outputs/.../train/chunk_0.zarr")

# Check dimensions
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

# Check QC metrics
print(adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe())

# Check HVG selection
print(f"HVGs: {adata.var['highly_variable'].sum()}")

# Check for Ensembl IDs
if "ensembl_id" in adata.var.columns:
    print(f"Genes with Ensembl ID: {adata.var['ensembl_id'].notna().sum()}")
```

#### Run on Small Subset

Test preprocessing on a small subset first:

```python
# In download config
download:
  enabled: true
  subset_size: 5000  # Only 5k cells for testing
```

#### Check Configuration Resolution

See the exact configuration used by Hydra:

```bash
python scripts/preprocessing/preprocess.py \
    --config-name=dataset_example \
    --cfg job  # Print full config and exit
```

### Getting Help

If issues persist:

1. Check logs in `<run_dir>/preprocessing.err`
2. Review QC plots in `<run_dir>/before/` and `<run_dir>/after/`
3. Check system monitor metrics: `<run_dir>/system_monitor.json`
4. Look for consolidated errors: `$WORKFLOW_DIR/logs/errors_consolidated.log`
5. Open an issue with:
   - Full error message
   - Dataset config file
   - System specifications (memory, CPU)
   - Input file size and format

---

## Summary

The preprocessing pipeline transforms raw single-cell RNA-seq data into analysis-ready AnnData objects through:

✅ **Quality Control**: Removes low-quality cells and genes
✅ **Normalization**: Log-normalizes expression data
✅ **HVG Selection**: Identifies highly variable genes
✅ **Metadata Enrichment**: Fetches SRA metadata, adds Ensembl IDs
✅ **Memory Efficiency**: Chunk-based processing for large datasets
✅ **Reproducibility**: Seed-controlled splitting and comprehensive logging
✅ **Flexibility**: Extensive configuration options for diverse datasets

The output is ready for the next pipeline stage: **embedding generation** (CPU/GPU).
