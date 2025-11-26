# Embedding Pipeline

This document explains the embedding generation pipeline for single-cell RNA-seq data in the `adata_hf_datasets` project. The embedding step transforms preprocessed AnnData files into files enriched with multiple embedding representations (PCA, scVI, HVG, Geneformer, etc.).

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
  - [Two-Phase Processing](#two-phase-processing)
  - [Memory-Efficient Design](#memory-efficient-design)
  - [Hardware-Aware Execution](#hardware-aware-execution)
- [Embedding Methods](#embedding-methods)
- [Configuration](#configuration)
- [Processing Pipeline](#processing-pipeline)
  - [Phase 1: Preparation (CPU)](#phase-1-preparation-cpu)
  - [Phase 2: Embedding (CPU/GPU)](#phase-2-embedding-cpugpu)
  - [Array Job Parallelization](#array-job-parallelization)
- [Output Structure](#output-structure)
- [Workflow Integration](#workflow-integration)
- [Advanced Topics](#advanced-topics)
  - [Manual Execution](#manual-execution)
  - [Debugging](#debugging)

---

## Overview

The embedding pipeline is the **third, fourth, and fifth step** in the complete data processing workflow:

```
1. Download → 2. Preprocessing → 3. Embedding Prep → 4. CPU Embedding → 5. GPU Embedding → 6. Dataset Creation
                                    (this step)      (this step)       (this step)
```

The embedding pipeline performs the following operations:

1. **Preparation Phase**: Runs CPU-intensive preprocessing for each embedding method (e.g., tokenization for Geneformer)
2. **CPU Embedding Phase**: Generates embeddings using CPU-based methods (PCA, HVG, scVI foundation model)
3. **GPU Embedding Phase**: Generates embeddings using GPU-based methods (Geneformer)
4. **Efficient Storage**: Streams embeddings into Zarr files without loading entire datasets into memory
5. **Parallel Processing**: Launches array jobs for each chunk created during preprocessing

**Key Features:**

- ✅ Memory-efficient: Streams embeddings without loading full datasets
- ✅ Hardware-optimized: Runs preparation on CPU, embedding on appropriate hardware
- ✅ Parallelizable: Uses SLURM array jobs for processing multiple chunks
- ✅ Incremental: Checks for existing embeddings to avoid redundant computation
- ✅ Robust: Includes retry logic for GPU-related errors

---

## Core Concepts

### Two-Phase Processing

The embedding pipeline is split into **two distinct phases** to optimize resource utilization:

#### Phase 1: Preparation (CPU-Intensive)

**Why separate preparation?**

Some embedding methods (particularly Geneformer) require CPU-intensive preprocessing before GPU computation:

- **Geneformer**: Tokenization of gene expression data
  - Reads raw counts from `adata.layers["counts"]`
  - Ranks genes by expression within each cell
  - Maps gene symbols to token IDs using vocabulary
  - Creates integer token sequences
  - **CPU-bound**: No GPU benefit, wastes GPU resources if run on GPU cluster

- **Other methods**: Similar CPU-intensive setup steps
  - Data loading and validation
  - Batch key extraction
  - Gene filtering
  - Format conversions

**Benefits of separate preparation:**

- 🚀 **Resource efficiency**: Doesn't waste precious GPU time on CPU work
- 💰 **Cost reduction**: GPU clusters are typically more expensive than CPU clusters
- ⚡ **Parallelization**: Preparation can run on many CPU nodes simultaneously
- 🔄 **Flexibility**: Can prepare on one cluster, embed on another

#### Phase 2: Embedding (Hardware-Specific)

After preparation is complete, the embedding phase runs on hardware appropriate for each method:

- **CPU methods** (PCA, HVG, scVI): Run on CPU cluster
- **GPU methods** (Geneformer): Run on GPU cluster with CUDA support

The embedding phase:

- Loads prepared resources (e.g., tokenized data)
- Computes the actual embedding
- Streams results directly to output file

### Memory-Efficient Design

The core embedding script (`embed_core.py`) implements several strategies to handle large datasets efficiently:

#### 1. Incremental Embedding Addition

Instead of loading an entire AnnData object, modifying it, and writing it back, the pipeline uses `append_embedding()`:

```python
def append_embedding(
    adata_path: Path,
    embedding: np.ndarray,
    outfile: Path,
    obsm_key: str,
    chunk_rows: int = 16_384,
) -> Path:
    """
    Append embedding to AnnData file without loading entire dataset.

    For Zarr files:
    - Opens file in read-write mode
    - Creates/overwrites specific obsm dataset
    - Streams embedding data in chunks

    Memory usage: O(chunk_rows) instead of O(n_cells)
    """
```

**Process:**

1. If output file doesn't exist, copy/convert input file once
2. Open Zarr file in read-write mode (`mode="r+"`)
3. Create or overwrite the specific `obsm/{key}` dataset
4. Stream embedding data in chunks of `chunk_rows` (default: 16,384 cells)

**Memory benefit:**

- Traditional approach: Load full dataset → modify → write (memory = full dataset)
- Streaming approach: Only chunk in memory at once (memory = 16K cells × embedding_dim)

#### 2. Lazy Loading in Embedders

The `InitialEmbedder` class uses lazy loading strategies:

```python
# Only reads necessary data from file
embedder.prepare(adata_path=str(infile), batch_key=batch_key)

# Reads data in batches during embedding
emb_matrix = embedder.embed(
    adata_path=str(infile),
    obsm_key=obsm_key,
    batch_key=batch_key,
    batch_size=128,  # Process 128 cells at a time
)
```

Different methods implement different loading strategies:

- **Geneformer**: Reads counts layer, processes in batches through model
- **PCA**: Can use incremental PCA for very large datasets
- **scVI**: Loads model weights, processes data in batches

#### 3. Existing Embedding Detection

Before processing, the pipeline checks which embeddings already exist:

```python
def check_existing_embeddings(file_path: Path, input_format: str) -> set[str]:
    """
    Check which embeddings exist WITHOUT loading the entire dataset.

    For Zarr: Directly checks obsm group keys (instant)
    For H5AD: Opens file in backed mode and checks obsm
    """
```

This allows:

- ✅ Skipping completed embeddings (saves time)
- ✅ Resuming failed jobs without reprocessing
- ✅ Adding new embeddings to existing files

### Hardware-Aware Execution

The pipeline is designed to run different methods on appropriate hardware:

#### CPU Embedding Configuration

```yaml
embedding_cpu:
  enabled: true
  methods: ["hvg", "pca", "scvi_fm"] # CPU-compatible methods
  batch_size: 128
  embedding_dim_map:
    scvi_fm: 50
    pca: 50
    hvg: 512
```

**CPU methods characteristics:**

- No GPU requirement
- Can run on standard compute nodes
- Often faster on CPU than GPU (for small models)

#### GPU Embedding Configuration

```yaml
embedding_gpu:
  enabled: true
  methods: ["geneformer"] # GPU-required methods
  batch_size: 128
  embedding_dim_map:
    geneformer: 768
```

**GPU methods characteristics:**

- Require CUDA-capable GPU
- Run on specialized GPU nodes
- Preparation phase should run on CPU first

#### Preparation Configuration

```yaml
embedding_preparation:
  enabled: true
  methods: ["geneformer"] # Methods to prepare (typically GPU methods)
```

**Preparation phase:**

- Runs on CPU cluster
- Performs tokenization, data format conversion, etc.
- Creates cached resources for embedding phase
- No embeddings are saved to output files

---

## Embedding Methods

The pipeline supports multiple embedding methods, each with different characteristics:

| Method         | Type              | Hardware | Dimension | Description                                         |
| -------------- | ----------------- | -------- | --------- | --------------------------------------------------- |
| **pca**        | Linear projection | CPU      | 50        | Principal Component Analysis on log-normalized data |
| **scvi_fm**    | Deep learning     | CPU      | 50        | scVI foundation model pretrained embeddings         |
| **geneformer** | Transformer       | GPU      | 768       | Geneformer transformer model embeddings             |
| **gs**         | Gene selection    | CPU      | 3936      | Geneformer-selected genes (3,936 genes)             |
| **gs10k**      | Gene selection    | CPU      | 10000     | Extended gene selection (10,000 genes)              |

### Method Details

#### PCA (Principal Component Analysis)

- **Purpose**: Linear dimensionality reduction
- **Input**: Log-normalized expression on HVGs
- **Output**: 50-dimensional dense embedding
- **Configuration**: `embedding_dim: 50` (standard)
- **Use case**: Fast, interpretable, captures linear variance

#### scVI Foundation Model

- **Purpose**: Deep learning-based embeddings from pretrained model
- **Input**: Raw counts
- **Output**: 50-dimensional dense embedding (fixed by pretrained model)
- **Configuration**: Model weights loaded automatically
- **Use case**: Captures non-linear biological structure, batch-corrected

#### Geneformer

- **Purpose**: Transformer-based embeddings of cellular state
- **Input**: Ranked gene expression (via tokenization)
- **Output**: 768-dimensional dense embedding (model dependent)
- **Configuration**: Requires Ensembl IDs, special tokenization
- **Preparation**: Tokenization (CPU-intensive, runs separately)
- **Embedding**: Transformer forward pass (GPU-intensive)
- **Use case**: State-of-the-art representation, captures complex gene interactions

---

## Configuration

### Dataset Configuration Structure

Embedding configuration is part of the dataset-centric config structure:

```yaml
# conf/dataset_example.yaml

# Common keys (used across all steps)
batch_key: "dataset_title"
annotation_key: "cell_type"

# Embedding preparation configuration
embedding_preparation:
  enabled: true
  methods: ["geneformer"] # Methods that need CPU-intensive preparation

  # Resource allocation
  memory_gb: 60

  # Execution parameters
  input_format: "auto" # "auto", "h5ad", or "zarr"
  output_format: "zarr" # Output format
  overwrite: false # Skip existing embeddings
  chunk_rows: 16384 # Rows per chunk when streaming
  batch_size: 128 # Batch size for embedders

  # Embedding dimensions
  embedding_dim_map:
    geneformer: 768
    scvi_fm: 50
    pca: 50
    hvg: 512

# CPU embedding configuration
embedding_cpu:
  enabled: true
  methods: ["hvg", "pca", "scvi_fm"]

  memory_gb: 60
  input_format: "auto"
  output_format: "zarr"
  overwrite: false
  chunk_rows: 16384
  batch_size: 128

  embedding_dim_map:
    scvi_fm: 50
    pca: 50
    hvg: 512

# GPU embedding configuration
embedding_gpu:
  enabled: true
  methods: ["geneformer"]

  memory_gb: 60
  input_format: "auto"
  output_format: "zarr"
  overwrite: false
  chunk_rows: 16384
  batch_size: 128

  embedding_dim_map:
    geneformer: 768
```

### Auto-Generated Paths

When the config is loaded, the `apply_all_transformations()` function automatically adds input/output paths:

```yaml
# Auto-generated by path transformation
embedding_preparation:
  input_files:
    - "{base_file_path}/processed/train/{dataset_name}/train/chunk_0.zarr"
    - "{base_file_path}/processed/train/{dataset_name}/val/chunk_0.zarr"
  output_dir: "{base_file_path}/processed_with_emb/train/{dataset_name}"

embedding_cpu:
  input_files: [...] # Same as above
  output_dir: [...] # Same as above

embedding_gpu:
  input_files: [...] # Same as above
  output_dir: [...] # Same as above
```

**Note:** The `input_files` list is automatically populated based on:

- Training vs test dataset (`preprocessing.split_dataset`)
- Number of chunks created during preprocessing
- Output format from preprocessing

---

## Processing Pipeline

### Phase 1: Preparation (CPU)

The preparation phase runs on CPU nodes and performs CPU-intensive setup:

**Script:** `embed_core.py` with `++prepare_only=true`

**What it does:**

```python
# For each input file (chunk):
for input_file in embedding_cfg.input_files:
    # For each method requiring preparation:
    for method in embedding_cfg.methods:
        embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)

        # Run preparation (CPU-intensive)
        embedder.prepare(
            adata_path=str(infile),
            batch_key=embedding_cfg.batch_key,
        )
        # Results are cached internally by embedder

# NO OUTPUT FILES ARE WRITTEN
```

**Example: Geneformer Preparation**

1. Loads raw counts from `adata.layers["counts"]`
2. Loads Ensembl IDs from `adata.var["ensembl_id"]`
3. For each cell:
   - Ranks genes by expression level
   - Filters to genes in vocabulary
   - Creates integer token sequence
4. Caches tokenized data for embedding phase

**Why this matters:**

- Tokenization for 100K cells can take 1-2 hours on CPU
- Would waste GPU resources if done on GPU nodes
- Can be parallelized across chunks with array jobs

### Phase 2: Embedding (CPU/GPU)

The embedding phase runs on appropriate hardware and generates actual embeddings:

**Script:** `embed_core.py` with `++prepare_only=false` (default)

**What it does:**

```python
# For each input file (chunk):
for input_file in embedding_cfg.input_files:
    infile = Path(input_file)
    # e.g., /data/processed/train/dataset/train/chunk_0.zarr

    # Determine output path
    split_name = infile.parent.name  # "train" or "val" or "all"
    output_dir = Path(output_dir_base) / split_name
    outfile = output_dir / f"{infile.stem}.zarr"
    # e.g., /data/processed_with_emb/train/dataset/train/chunk_0.zarr

    # Check which embeddings already exist
    existing_obsm_keys = check_existing_embeddings(outfile if outfile.exists() else infile)

    # Determine which methods need to run
    methods_to_run = []
    for method in embedding_cfg.methods:
        obsm_key = f"X_{method}"
        if obsm_key not in existing_obsm_keys or embedding_cfg.overwrite:
            methods_to_run.append(method)

    # Compute missing embeddings
    for method in methods_to_run:
        # Initialize embedder
        embedder = InitialEmbedder(method=method, embedding_dim=emb_dim)

        # Run preparation (uses cache if available from prep phase)
        embedder.prepare(
            adata_path=str(input_for_processing),
            batch_key=embedding_cfg.batch_key,
        )

        # Run embedding computation
        emb_matrix = embedder.embed(
            adata_path=str(input_for_processing),
            obsm_key=obsm_key,
            batch_key=embedding_cfg.batch_key,
            batch_size=embedding_cfg.batch_size,
        )
        # Returns: np.ndarray of shape (n_cells, embedding_dim)

        # Stream embedding to output file
        append_embedding(
            adata_path=str(input_for_processing),
            embedding=emb_matrix,
            outfile=str(outfile),
            obsm_key=obsm_key,
            chunk_rows=16384,
        )
```

**Key behaviors:**

1. **First-time processing:**
   - Copies input file to output location
   - Adds first embedding to `obsm`

2. **Subsequent embeddings:**
   - Opens existing output file
   - Adds new embedding to existing `obsm`

3. **Resuming failed jobs:**
   - Checks which embeddings exist
   - Only computes missing embeddings
   - Preserves existing work

4. **Error handling:**
   - Retries GPU-related errors (CUDA initialization, OOM)
   - Exponential backoff between retries
   - Clears CUDA cache between attempts

### Array Job Parallelization

The preprocessing step creates multiple chunks per split:

```
processed/train/dataset_name/
├── train/
│   ├── chunk_0.zarr  ← Input file 1
│   ├── chunk_1.zarr  ← Input file 2
│   └── chunk_2.zarr  ← Input file 3
└── val/
    └── chunk_0.zarr  ← Input file 4
```

The embedding pipeline launches **SLURM array jobs** where each task processes one chunk:

```bash
# Array job with 4 tasks (one per input file)
sbatch --array=0-3 embed_array.slurm
```

**Task assignment:**

- Task 0 → `train/chunk_0.zarr`
- Task 1 → `train/chunk_1.zarr`
- Task 2 → `train/chunk_2.zarr`
- Task 3 → `val/chunk_0.zarr`

**Benefits:**

- ⚡ **Parallel processing**: All chunks processed simultaneously
- 🔄 **Fault tolerance**: Failed tasks can be rerun independently
- 📊 **Resource efficiency**: Each task requests appropriate resources
- 🎯 **Scalability**: Handles datasets of any size

**Resource allocation per task:**

- Preparation: CPU node, 60GB memory
- CPU embedding: CPU node, 60GB memory
- GPU embedding: GPU node, 60GB memory, 1 GPU

---

## Output Structure

### Directory Layout

The embedding steps write to the `processed_with_emb/` directory:

```
$base_file_path/
├── processed/                         # Input (from preprocessing)
│   └── train/
│       └── {dataset_name}/
│           ├── train/
│           │   ├── chunk_0.zarr       # Input
│           │   └── chunk_1.zarr
│           └── val/
│               └── chunk_0.zarr
│
└── processed_with_emb/                # Output (this step)
    └── train/
        └── {dataset_name}/
            ├── train/
            │   ├── chunk_0.zarr       # Output with embeddings
            │   └── chunk_1.zarr
            └── val/
                └── chunk_0.zarr
```

### Output File Structure

Each output file is a Zarr directory containing the original data plus embeddings:

```
chunk_0.zarr/
├── X/                         # Log-normalized expression (from preprocessing)
├── layers/
│   └── counts/               # Raw counts (from preprocessing)
├── obs/                      # Cell metadata (from preprocessing)
├── var/                      # Gene metadata (from preprocessing)
├── uns/                      # Unstructured metadata
└── obsm/                     # EMBEDDINGS (added by this step)
    ├── X_pca/                # PCA embedding (50-dim)
    ├── X_scvi_fm/            # scVI embedding (50-dim)
    ├── X_hvg/                # HVG embedding (512-dim)
    └── X_geneformer/         # Geneformer embedding (768-dim)
```

### Progressive Enrichment

The pipeline progressively adds embeddings:

**After preparation:** No files written (preparation state cached internally)

**After CPU embedding:**

```
obsm/
├── X_pca/
├── X_scvi_fm/
└── X_hvg/
```

**After GPU embedding:**

```
obsm/
├── X_pca/
├── X_scvi_fm/
├── X_hvg/
└── X_geneformer/  ← Added by GPU embedding
```

### Logs and Metadata

Separate from data files, logs are stored in workflow directories:

```
$WORKFLOW_DIR/
├── embedding_prepare/
│   └── job_{SLURM_JOB_ID}/
│       ├── master.out               # Master job stdout
│       ├── master.err               # Master job stderr
│       ├── array_{ARRAY_JOB_ID}/
│       │   ├── task_0.out          # Task 0 stdout
│       │   ├── task_0.err          # Task 0 stderr
│       │   └── ...
│       └── .hydra/
│
├── embedding_cpu/
│   └── job_{SLURM_JOB_ID}/
│       ├── master.out
│       ├── master.err
│       └── array_{ARRAY_JOB_ID}/
│           └── task_*.out/err
│
└── embedding_gpu/
    └── job_{SLURM_JOB_ID}/
        ├── master.out
        ├── master.err
        └── array_{ARRAY_JOB_ID}/
            └── task_*.out/err
```

---

## Workflow Integration

The embedding steps are designed to run as part of the complete workflow orchestration.

**For comprehensive information on running the complete workflow (including embeddings as part of the automated pipeline), please refer to:**

**📖 [Workflow Orchestration Guide](../workflow/README.md)**

### Automatic Execution

When using the workflow orchestrator:

1. **Preprocessing** creates chunked output in `processed/`
2. **Embedding preparation** runs on CPU cluster for GPU methods
3. **CPU embedding** runs for CPU-compatible methods
4. **GPU embedding** runs for GPU-dependent methods
5. **Dataset creation** uses the enriched files from `processed_with_emb/`

The orchestrator handles:

- ✅ Automatic path management between steps
- ✅ SLURM job submission with proper dependencies
- ✅ Resource allocation (CPU vs GPU clusters)
- ✅ Cross-cluster job submission (if CPU and GPU on different clusters)
- ✅ Error tracking and consolidated logging
- ✅ Array job submission for parallel chunk processing

### Step Dependencies

```
Download (optional)
    ↓
Preprocessing
    ↓
    ├─→ Embedding Preparation (CPU) ─→ GPU Embedding
    │                                       ↓
    └─→ CPU Embedding ─────────────────────┤
                                            ↓
                                    Dataset Creation
```

**Dependency logic:**

- GPU embedding depends on embedding preparation completion
- CPU embedding runs independently (no preparation needed)
- Dataset creation waits for both CPU and GPU embedding completion

---

## Advanced Topics

### Manual Execution

**⚠️ Important:** The embedding pipeline is complex and tightly integrated with the workflow orchestrator. **We strongly recommend using the workflow orchestrator** rather than running embedding scripts manually.

If you need to run embedding steps manually (for debugging or custom workflows), here's the general approach:

#### Prerequisites

1. **Completed preprocessing** with output in `$base_file_path/processed/`
2. **Dataset configuration** with embedding sections defined
3. **Virtual environment** activated

#### Manual Embedding Execution

```bash
# Activate virtual environment
source .venv/bin/activate

# Set base file path (if not in config)
export BASE_FILE_PATH=/path/to/data

# Run embedding preparation (optional, for GPU methods)
python scripts/embed/embed_core.py \
    --config-path=../../conf \
    --config-name=dataset_example \
    ++embedding_config_section=embedding_preparation \
    ++prepare_only=true

# Run CPU embedding
python scripts/embed/embed_core.py \
    --config-path=../../conf \
    --config-name=dataset_example \
    ++embedding_config_section=embedding_cpu \
    ++prepare_only=false

# Run GPU embedding (requires CUDA)
python scripts/embed/embed_core.py \
    --config-path=../../conf \
    --config-name=dataset_example \
    ++embedding_config_section=embedding_gpu \
    ++prepare_only=false
```

**Key parameters:**

- `--config-path`: Path to config directory (relative to script location)
- `--config-name`: Dataset configuration name (without `.yaml`)
- `++embedding_config_section`: Which embedding config to use (`embedding_preparation`, `embedding_cpu`, or `embedding_gpu`)
- `++prepare_only`: Whether to run only preparation (`true`) or full pipeline (`false`)

#### Limitations of Manual Execution

- ❌ No automatic array job submission for multiple chunks
- ❌ No automatic resource allocation
- ❌ No cross-cluster job submission
- ❌ No dependency management between steps
- ❌ Manual path management required
- ❌ No consolidated error logging

**For production use, please use the workflow orchestrator.**

### Debugging

#### Check Existing Embeddings

```python
import zarr
import anndata as ad

# Check what embeddings exist
adata = ad.read_zarr("processed_with_emb/train/dataset/train/chunk_0.zarr")
print("Existing embeddings:", list(adata.obsm.keys()))

# Check embedding shapes
for key in adata.obsm.keys():
    print(f"{key}: {adata.obsm[key].shape}")
```

#### Test Single Method

```bash
# Test preparation only
python scripts/embed/embed_core.py \
    --config-name=dataset_example \
    ++embedding_config_section=embedding_preparation \
    ++prepare_only=true \
    ++embedding_preparation.methods='["geneformer"]'

# Test single method embedding
python scripts/embed/embed_core.py \
    --config-name=dataset_example \
    ++embedding_config_section=embedding_cpu \
    ++embedding_cpu.methods='["pca"]'
```

#### Verify Input Files

```python
from pathlib import Path
import anndata as ad

# Check if input files exist and are valid
input_files = [
    "processed/train/dataset/train/chunk_0.zarr",
    "processed/train/dataset/val/chunk_0.zarr",
]

for f in input_files:
    path = Path(f)
    if not path.exists():
        print(f"❌ Missing: {f}")
    else:
        try:
            adata = ad.read_zarr(path)
            print(f"✓ Valid: {f} ({adata.n_obs} cells, {adata.n_vars} genes)")

            # Check for required data
            if "counts" not in adata.layers:
                print(f"  ⚠️ Missing 'counts' layer")
            if "ensembl_id" not in adata.var.columns:
                print(f"  ⚠️ Missing 'ensembl_id' (needed for Geneformer)")
        except Exception as e:
            print(f"❌ Invalid: {f} - {e}")
```

#### Monitor GPU Usage

```bash
# On GPU node
watch -n 1 nvidia-smi

# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Common Issues

**Issue: "No embedding configuration found"**

- **Cause**: Missing `embedding_config_section` parameter
- **Solution**: Add `++embedding_config_section=embedding_cpu` to command

**Issue: "Input file not found"**

- **Cause**: Preprocessing didn't complete or paths incorrect
- **Solution**: Check `$base_file_path/processed/` for input files

**Issue: "Missing ensembl_id for Geneformer"**

- **Cause**: Preprocessing didn't run with `geneformer_pp: true`
- **Solution**: Rerun preprocessing with Geneformer preparation enabled

**Issue: "CUDA out of memory"**

- **Cause**: Batch size too large for GPU
- **Solution**: Reduce `batch_size` in embedding config (e.g., from 128 to 64 or 32)

**Issue: "All embeddings already exist, skipping"**

- **Cause**: Embeddings already computed
- **Solution**: Set `++embedding_cpu.overwrite=true` to recompute

---

## Summary

The embedding pipeline efficiently generates multiple embedding representations through:

✅ **Two-phase processing**: CPU-intensive preparation separated from embedding
✅ **Memory efficiency**: Streams embeddings without loading full datasets
✅ **Hardware optimization**: Runs methods on appropriate hardware (CPU/GPU)
✅ **Parallel processing**: Uses array jobs for multiple chunks
✅ **Incremental computation**: Skips existing embeddings
✅ **Robust error handling**: Retries GPU errors, validates inputs

**The embedding output is ready for the final pipeline stage: dataset creation**.

For running the complete pipeline including embeddings, see the **[Workflow Orchestration Guide](../workflow/README.md)**.
