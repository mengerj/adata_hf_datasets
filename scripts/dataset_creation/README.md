# Dataset Creation Pipeline

This document explains the dataset creation pipeline for single-cell RNA-seq data in the `adata_hf_datasets` project. The dataset creation step transforms processed AnnData files with embeddings into Hugging Face datasets ready for training machine learning models.

## Table of Contents

- [Overview](#overview)
- [Dataset Formats](#dataset-formats)
  - [Multiplets Format](#multiplets-format)
  - [Pairs Format](#pairs-format)
  - [Single Format](#single-format)
- [Cell Sentences](#cell-sentences)
  - [What are Cell Sentences?](#what-are-cell-sentences)
  - [Sentence Keys Configuration](#sentence-keys-configuration)
  - [Semantic Sentences](#semantic-sentences)
- [Negative Sampling](#negative-sampling)
  - [Resolved vs Index-Based Negatives](#resolved-vs-index-based-negatives)
  - [Negative Types](#negative-types)
  - [Batch-Aware Sampling](#batch-aware-sampling)
- [Configuration](#configuration)
- [Hugging Face Hub Integration](#hugging-face-hub-integration)
  - [Authentication](#authentication)
  - [Public vs Private Datasets](#public-vs-private-datasets)
  - [Repository Naming](#repository-naming)
  - [Automatic Versioning](#automatic-versioning)
- [Template Customization](#template-customization)
- [Nextcloud Integration](#nextcloud-integration)
- [Output Structure](#output-structure)
- [Workflow Integration](#workflow-integration)
- [Advanced Topics](#advanced-topics)

---

## Overview

The dataset creation pipeline is the **final step** in the complete data processing workflow:

### Quick Start: Recommended Configuration

For most use cases, we recommend:

- ✅ **Format**: `multiplets` (most tested and flexible)
- ✅ **Sentence keys**: Single key with `sample_id_og` (for numeric/embedding approaches)
- ✅ **Negatives**: Index-based (default, not resolved)
- ✅ **Negative count**: 2-4 negatives per sample
- ❌ **Avoid**: Semantic sentences (experimental, not well-tested)
- ❌ **Avoid**: Pairs format with multiple sentence keys (not tested)

**Example minimal config:**

```yaml
dataset_creation:
  dataset_format: "multiplets"
  sentence_keys: ["sample_id_og"]
  negatives_per_sample: 2
  resolve_negatives: false
```

This configuration has been most extensively used and tested in practice.

### Pipeline Position

```
1. Download → 2. Preprocessing → 3. Embedding Prep → 4. CPU Embedding → 5. GPU Embedding → 6. Dataset Creation
                                                                                            (this step)
```

### What It Does

The dataset creation pipeline performs the following operations:

1. **Load AnnData files**: Reads processed files with embeddings from `processed_with_emb/`
2. **Create cell sentences**: Generates text representations of cells (gene lists, IDs, semantic descriptions)
3. **Generate negative samples**: Creates contrastive learning pairs/multiplets with proper negatives
4. **Upload to Nextcloud** (optional): Makes AnnData files accessible via download links
5. **Build HuggingFace Dataset**: Creates `datasets.Dataset` objects for each split
6. **Push to Hub** (optional): Uploads the dataset to Hugging Face Hub with rich metadata

**Key Features:**

- ✅ Multiple dataset formats for different training paradigms
- ✅ Flexible cell sentence generation (text-based and ID-based)
- ✅ Intelligent negative sampling within batches
- ✅ Memory-efficient construction using generators
- ✅ Automatic HuggingFace Hub integration with versioning
- ✅ Rich metadata and templates for dataset cards

---

## Dataset Formats

The `AnnDataSetConstructor` class supports three dataset formats, each designed for different use cases:

### Multiplets Format

**Purpose:** Contrastive learning with multiple negatives per anchor

**Use case:** Training models with hard negative mining, where each sample has one positive caption and multiple negative samples.

**⭐ Recommended Format:** This is the recommended format for most training scenarios. It provides the most flexibility and has been extensively tested.

**Structure:**

```python
{
    "sample_idx": "AAACCTGAGAAACCAT-1",           # Original cell barcode
    "cell_sentence_1": "AAACCTGAGAAACCAT-1",     # Cell ID (for numeric methods)
    "cell_sentence_2": "CD3D CD8A ...",          # Gene list (for text methods)
    "cell_sentence_3": "T cell from lung",       # Semantic true label
    "cell_sentence_4": "B cell from spleen",     # Semantic similar label
    "positive": "This is a T cell from lung tissue. The donor was a healthy 45-year-old male.",
    "negative_1_idx": "AAACCTGAGCATCATC-1",      # Caption negative (different caption)
    "negative_2_idx": "AAACCTGCAGGCGATA-1",      # Sentence negative (different sample)
    "negative_3_idx": "AAACCTGTCAGCAACT-1",      # Caption negative
    "negative_4_idx": "AAACGGGAGCCACAAG-1",      # Sentence negative
    "adata_link": "https://nextcloud.example.com/share/abc123"
}
```

**Key characteristics:**

- **Alternating negatives**: Even indices (1, 3, 5...) are caption negatives (different cell type), odd indices (2, 4, 6...) are sentence negatives (different sample, any type)
- **Index-based by default**: Negatives stored as sample indices, allowing downstream models to choose which cell sentence representation to use
- **Flexible representation**: Same dataset can serve both text-based and embedding-based models

**Configuration:**

```yaml
dataset_creation:
  dataset_format: "multiplets"
  negatives_per_sample: 4 # Total negatives per sample
  sentence_keys:
    - "sample_id_og" # Cell ID
    - "cell_sentence" # Gene list
    - "semantic_true" # True cell type
    - "semantic_similar" # Similar cell type
```

### Pairs Format

**Purpose:** Binary classification with positive/negative pairs

**Use case:** Training models with explicit positive/negative labels, where each anchor generates individual records.

**⚠️ Limited Testing:** This format has been less extensively tested, especially with multiple cell sentences. **We recommend using only one `sentence_key` when using pairs format.**

**Structure:**

```python
# Positive pair
{
    "sample_idx": "AAACCTGAGAAACCAT-1",
    "cell_sentence_1": "AAACCTGAGAAACCAT-1",
    "caption": "This is a T cell from lung tissue. The donor was a healthy 45-year-old male.",
    "label": 1.0,
    "adata_link": "https://nextcloud.example.com/share/abc123"
}

# Negative pair (same sample, different caption)
{
    "sample_idx": "AAACCTGAGAAACCAT-1",
    "cell_sentence_1": "AAACCTGAGAAACCAT-1",
    "caption": "This is a B cell from spleen tissue. The donor was a sick 60-year-old female.",
    "label": 0.0,
    "adata_link": "https://nextcloud.example.com/share/abc123"
}
```

**Key characteristics:**

- **Explicit labels**: Binary `label` column (1.0 = positive, 0.0 = negative)
  - The label indicates the match between `cell_sentence_1` and `caption`
  - 1.0 = `cell_sentence_1` and `caption` describe the same cell
  - 0.0 = `cell_sentence_1` and `caption` describe different cells
- **Duplicated samples**: Each sample appears twice (once with positive, once with negative caption)
- **Simple training**: Easy to use with standard binary classification losses

**Configuration:**

```yaml
dataset_creation:
  dataset_format: "pairs"
  negatives_per_sample: 1 # One negative per positive (creates 2 records per sample)
  sentence_keys:
    - "sample_id_og" # Recommended: use single sentence_key for pairs format
```

### Single Format

**Purpose:** Inference/test datasets without contrastive learning

**Use case:** Test datasets, inference, or when you only need the omics representation without captions.

**Structure:**

```python
{
    "sample_idx": "AAACCTGAGAAACCAT-1",
    "cell_sentence_1": "AAACCTGAGAAACCAT-1",
    "cell_sentence_2": "CD3D CD8A CCR7 IL7R ...",
    "adata_link": "https://nextcloud.example.com/share/abc123"
}
```

**Key characteristics:**

- **No captions**: Omits the `positive`/`caption` column
- **No negatives**: No negative sampling
- **Minimal structure**: Only cell sentences and data link

**Configuration:**

```yaml
dataset_creation:
  dataset_format: "single"
  sentence_keys:
    - "sample_id_og"
    - "cell_sentence"
  caption_key: null # No caption needed
```

---

## Cell Sentences

### What are Cell Sentences?

A **cell sentence** is a string representation of a single cell that captures its molecular or semantic characteristics. Cell sentences serve as input features for machine learning models.

The pipeline supports multiple types of cell sentences simultaneously:

1. **Cell ID sentences**: The cell barcode/identifier
   - Example: `"AAACCTGAGAAACCAT-1"`
   - Use: Allows models to retrieve numeric embeddings from the AnnData file
   - **⭐ Recommended for numeric/embedding-based approaches**: Use the sample ID which can be tokenized later, with embeddings extracted via the `adata_link`

2. **Gene list sentences**: Space-separated list of expressed genes
   - Example: `"CD3D CD8A CCR7 IL7R SELL LEF1 TCF7 ..."`
   - Use: Text-based models can process gene names directly

3. **Semantic sentences**: Natural language descriptions combining gene list and cell type
   - Example: `"CD3D CD8A CCR7 IL7R ... T cell from lung tissue"`
   - Use: Experimental feature combining gene information with cell type labels
   - **⚠️ Limited use in practice**: This format has not been extensively used or tested

### Design Philosophy: Multiple Cell Sentences

The pipeline was originally designed to support **multiple cell sentence types in the same dataset** to enable comparing different cell representations without creating separate datasets for each approach. For example, you could include both a cell ID (for embedding-based models) and a gene list (for text-based models) in the same dataset.

**However, in practice**, the easiest and most common approach is to:

- **Use a single `sentence_key`** containing the sample ID (`sample_id_og`)
- Let downstream models tokenize this ID and extract embeddings via `adata_link`
- This approach works well for numeric/embedding-based methods

If you do use multiple cell sentences, downstream models must explicitly select which `cell_sentence_N` to use during training.

### Sentence Keys Configuration

The `sentence_keys` parameter defines which columns from `adata.obs` become cell sentences.

**Recommended Simple Configuration:**

```yaml
dataset_creation:
  dataset_format: "multiplets" # Recommended format
  sentence_keys:
    - "sample_id_og" # Single cell ID for numeric approaches
  negatives_per_sample: 2
```

**Advanced Multi-Sentence Configuration** (for comparison experiments):

```yaml
dataset_creation:
  dataset_format: "multiplets" # Only tested with multiplets
  sentence_keys:
    - "sample_id_og" # Cell ID → cell_sentence_1
    - "cell_sentence" # Gene list → cell_sentence_2
    - "semantic_true" # True cell type → cell_sentence_3 (experimental)
    - "semantic_similar" # Similar cell type → cell_sentence_4 (experimental)
  negatives_per_sample: 2
```

**Output mapping:**

```python
# In the resulting HuggingFace dataset:
{
    "cell_sentence_1": adata.obs["sample_id_og"][i],
    "cell_sentence_2": adata.obs["cell_sentence"][i],        # Optional
    "cell_sentence_3": adata.obs["semantic_true"][i],        # Optional
    "cell_sentence_4": adata.obs["semantic_similar"][i],     # Optional
}
```

**Important Notes:**

- **For pairs format**: Only use a single `sentence_key` (e.g., `["sample_id_og"]`)
- **For multiplets format**: You can use multiple sentence keys, but training code must explicitly select which one to use
- **Semantic sentences** are experimental and haven't been extensively tested in practice

**Creating cell sentences:**

The `create_cell_sentences()` function automatically generates cell sentences during dataset creation:

```python
from adata_hf_datasets.dataset import create_cell_sentences

adata = create_cell_sentences(
    adata=adata,
    gene_name_column="gene_name",      # Column in adata.var with gene names
    annotation_column="cell_type",     # Column in adata.obs with cell types
    cs_length=4096,                    # Maximum number of genes in gene list
)
```

**What it creates:**

1. **`sample_id_og`**: Copies `adata.obs.index` to a column

   ```python
   "AAACCTGAGAAACCAT-1"
   ```

2. **`cell_sentence`**: Creates gene list from top expressed genes:

   ```python
   # For each cell, ranks genes by expression and takes top cs_length
   "CD3D CD8A CCR7 IL7R SELL LEF1 TCF7 MAL LDHB NOSIP IL32 ..."
   ```

3. **`semantic_true`** (if `annotation_column` provided): Gene list WITH cell type label

   ```python
   # Note: This is NOT just the cell type, but gene list + cell type
   "CD3D CD8A CCR7 IL7R ... CD8+ T cell"
   ```

4. **`semantic_similar`** (if `annotation_column` provided): Gene list WITH similar cell type
   ```python
   # Gene list + similar cell type label
   "CD3D CD8A CCR7 IL7R ... CD4+ T cell"
   ```

**⚠️ Important:** Semantic sentences (`semantic_true` and `semantic_similar`) are experimental features that combine the gene list with cell type labels in a single string. They have not been extensively used or validated in practice. For most use cases, using just `sample_id_og` is recommended.

### Semantic Sentences

Semantic sentences are experimental features that combine gene lists with cell type labels in a single string (e.g., `"CD3D CD8A ... CD8+ T cell"`).

**⚠️ Limited Practical Use:** These features were designed for potential cross-modal learning experiments but have not been extensively tested or used in practice.

**We recommend:**

- **For most use cases**: Use only `sample_id_og` as your single sentence key
- **For text-based experiments**: Use `cell_sentence` (gene list only)
- **Skip semantic sentences** unless you have a specific experimental need

**Configuration:**

```yaml
# Enable semantic sentences (experimental, not recommended for production)
annotation_key: "cell_type" # Column in adata.obs

dataset_creation:
  sentence_keys:
    - "sample_id_og"
    - "cell_sentence"
    - "semantic_true" # Requires annotation_key (experimental)
    - "semantic_similar" # Requires annotation_key (experimental)
```

**If `annotation_key` is `null`:**

- Semantic sentence keys are automatically removed from `sentence_keys`
- Only cell ID and gene list sentences are created
- This is the recommended configuration for most datasets

---

## Negative Sampling

Negative sampling is crucial for contrastive learning. The pipeline implements intelligent negative sampling strategies.

### Resolved vs Index-Based Negatives

The `AnnDataSetConstructor` supports two modes for storing negatives:

#### Index-Based Negatives (Default, Recommended)

**What it stores:**

```python
{
    "negative_1_idx": "AAACCTGAGCATCATC-1",  # Sample index
    "negative_2_idx": "AAACCTGCAGGCGATA-1",
    "negative_3_idx": "AAACCTGTCAGCAACT-1",
    "negative_4_idx": "AAACGGGAGCCACAAG-1",
}
```

**Why use indices?**

The original design goal was to support **comparing different cell representations** without creating separate datasets. By storing indices instead of resolved content, the same dataset can serve multiple modeling approaches.

**Benefits:**

- **Flexibility**: Downstream models choose which cell sentence representation to use
- **Single dataset**: Same dataset serves text-based, embedding-based, and hybrid models
- **Memory efficient**: Stores small string indices instead of full content
- **Dynamic loading**: Models retrieve negative content on-demand

**In practice:** Most models use `cell_sentence_1` (sample ID) for all samples and negatives, then load embeddings via `adata_link` during training.

**Example usage at training time:**

```python
# Text-based model uses cell_sentence_2 (gene lists)
def get_negative_text(row, neg_idx):
    neg_sample_id = row[f"negative_{neg_idx}_idx"]
    return dataset_lookup[neg_sample_id]["cell_sentence_2"]

# Embedding-based model uses cell_sentence_1 (IDs) to load embeddings
def get_negative_embedding(row, neg_idx):
    neg_sample_id = row[f"negative_{neg_idx}_idx"]
    adata = load_adata(dataset_lookup[neg_sample_id]["adata_link"])
    return adata[neg_sample_id].obsm["X_geneformer"]
```

#### Resolved Negatives (Not Recommended)

**Configuration:**

```yaml
dataset_creation:
  resolve_negatives: true # Only works with single sentence_key
  sentence_keys:
    - "sample_id_og" # Must be exactly one key
```

**What it stores:**

```python
{
    "negative_1_idx": "AAACCTGAGCATCATC-1",
    "negative_1": "This is a B cell from spleen tissue...",  # Resolved caption
    "negative_2_idx": "AAACCTGCAGGCGATA-1",
    "negative_2": "CD19 CD79A MS4A1 CD79B TCL1A ...",       # Resolved gene list
    "negative_3_idx": "AAACCTGTCAGCAACT-1",
    "negative_3": "This is a Monocyte from blood...",       # Resolved caption
    "negative_4_idx": "AAACGGGAGCCACAAG-1",
    "negative_4": "S100A9 LYZ S100A8 FCN1 ...",            # Resolved gene list
}
```

**When to use resolved negatives:**

- ✅ Debugging and inspection (to see negative content directly)
- ✅ Simple training setup with one cell sentence type
- ❌ **Not recommended for production**: Index-based negatives offer more flexibility
- ❌ Multi-representation datasets (incompatible with multiple sentence_keys)
- ❌ Dynamic negative selection at training time

**Recommendation:** Use the default index-based negatives unless you have a specific reason to resolve them.

### Negative Types

The pipeline generates two types of negatives in alternating order:

#### Caption Negatives (Odd indices: 1, 3, 5...)

**Definition:** Samples with **different captions** (different cell types/conditions)

**Purpose:** Hard negatives that differ in biological interpretation

**Selection strategy:**

1. Prefer same batch, different caption
2. Fallback to cross-batch, different caption

**Example:**

```python
# Anchor: T cell from lung
{
    "positive": "This is a CD8+ T cell from lung tissue.",
    "negative_1_idx": "...",  # → "This is a B cell from spleen tissue."
    "negative_3_idx": "...",  # → "This is a Monocyte from blood."
}
```

#### Sentence Negatives (Even indices: 2, 4, 6...)

**Definition:** Different samples, regardless of caption

**Purpose:** Ensures diversity in negative sampling

**Selection strategy:**

1. Prefer same batch, different sample
2. Fallback to cross-batch, different sample
3. Exclude already-selected negatives

**Example:**

```python
# Anchor: T cell from lung
{
    "positive": "This is a CD8+ T cell from lung tissue.",
    "negative_2_idx": "...",  # → Any other cell (could even be another T cell)
    "negative_4_idx": "...",  # → Any other cell
}
```

### Batch-Aware Sampling

Negatives are preferentially sampled **within the same batch** to ensure meaningful contrastive pairs.

**Why batch-aware?**

- Technical variation (different sequencing runs) can overshadow biological differences
- Within-batch negatives ensure the model learns biological distinctions, not technical artifacts

**Batch definition:**

```yaml
batch_key: "dataset_title" # Or "donor_id", "sequencing_run", etc.
```

**Sampling hierarchy:**

1. **Same batch, different caption** (most preferred)
2. **Same batch, different sample**
3. **Cross-batch, different caption** (fallback)
4. **Cross-batch, different sample** (last resort)

---

## Configuration

### Dataset Configuration Structure

Dataset creation configuration is part of the dataset-centric config.

**Recommended Simple Configuration:**

```yaml
# conf/dataset_example.yaml

# Common keys (used across all steps)
batch_key: "dataset_title"
annotation_key: null # Set to null if no cell type annotations
caption_key: "natural_language_annotation"

# Dataset creation configuration
dataset_creation:
  enabled: true

  # Recommended settings
  dataset_format: "multiplets" # Recommended format
  sentence_keys:
    - "sample_id_og" # Single sentence key (recommended)

  # Cell sentence generation
  gene_name_column: "gene_name" # Column in adata.var
  cs_length: 4096 # Max genes in gene list

  # Negative sampling
  negatives_per_sample: 2 # Number of negatives per anchor
  resolve_negatives: false # Keep as indices (recommended)

  # Required embeddings (validation)
  required_obsm_keys: ["X_pca", "X_scvi_fm", "X_geneformer"]

  # Output configuration
  output_dir: "data/hf_datasets"
  push_to_hub: true
  base_repo_id: "jo-mengr"
  use_nextcloud: false
```

**Advanced Multi-Sentence Configuration** (for experimental comparisons):

```yaml
# Only use if comparing different cell representations
dataset_creation:
  dataset_format: "multiplets" # Only tested with multiplets
  sentence_keys:
    - "sample_id_og" # Cell ID
    - "cell_sentence" # Gene list
    # Note: semantic sentences not recommended
  negatives_per_sample: 2
  resolve_negatives: false # Must be false for multiple sentence keys

  # Required embeddings (validation)
  required_obsm_keys: ["X_pca", "X_scvi_fm", "X_geneformer"]

  # Output configuration
  output_dir: "data/hf_datasets"
  push_to_hub: true
  base_repo_id: "jo-mengr"

  # Nextcloud upload (optional)
  use_nextcloud: false
  force_reupload: true
  nextcloud_config:
    url: "NEXTCLOUD_URL"
    username: "NEXTCLOUD_USER"
    password: "NEXTCLOUD_PASSWORD"
    remote_path: ""
```

### Key Parameters

| Parameter              | Type      | Default        | Description                                                  |
| ---------------------- | --------- | -------------- | ------------------------------------------------------------ |
| `dataset_format`       | str       | `"multiplets"` | Dataset format: `"multiplets"`, `"pairs"`, or `"single"`     |
| `sentence_keys`        | list[str] | -              | Columns in `adata.obs` to use as cell sentences              |
| `gene_name_column`     | str       | `"gene_name"`  | Column in `adata.var` with gene names                        |
| `cs_length`            | int       | 4096           | Maximum number of genes in cell sentence                     |
| `negatives_per_sample` | int       | 2              | Number of negatives per anchor                               |
| `resolve_negatives`    | bool      | false          | Resolve negatives to content (only with single sentence_key) |
| `required_obsm_keys`   | list[str] | `[]`           | Required embeddings (validates before processing)            |
| `push_to_hub`          | bool      | true           | Push to HuggingFace Hub                                      |
| `base_repo_id`         | str       | -              | HuggingFace username or organization                         |
| `use_nextcloud`        | bool      | false          | Upload AnnData files to Nextcloud                            |

### Auto-Generated Paths

The `apply_all_transformations()` function automatically sets:

```yaml
# Auto-generated paths
dataset_creation:
  data_dir: "{base_file_path}/processed_with_emb/train/{dataset_name}"

  # If required_obsm_keys is empty (no embeddings), falls back to:
  # data_dir: "{base_file_path}/processed/train/{dataset_name}"
```

---

## Hugging Face Hub Integration

### Authentication

Before pushing datasets to the Hub, you **must authenticate**:

#### Method 1: Using the CLI (Recommended)

```bash
# Login to HuggingFace Hub
huggingface-cli login

# Follow the prompts to enter your token
# Token can be found at: https://huggingface.co/settings/tokens
```

This stores your token in `~/.huggingface/token` for all future operations.

#### Method 2: Using Environment Variables

```bash
# Set in your environment or .env file
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxx"

# For upload operations, you can also use:
export HF_TOKEN_UPLOAD="hf_xxxxxxxxxxxxxxxxxxxxx"
```

**Note:** The `HF_TOKEN_UPLOAD` environment variable takes precedence over `HF_TOKEN` for upload operations.

#### Method 3: Programmatic Login

```python
from huggingface_hub import login

login(token="hf_xxxxxxxxxxxxxxxxxxxxx")
```

### Public vs Private Datasets

**Currently, datasets are pushed as PRIVATE by default** for security reasons.

**Configuration location:**

In `create_dataset.py`, line 256:

```python
annotate_and_push_dataset(
    dataset=hf_dataset,
    embedding_generation=embedding_generation,
    dataset_type_explanation=dataset_type_explanation,
    repo_id=final_repo_id,
    readme_template_name="cellwhisperer_train",
    metadata=metadata,
    private=True,  # ← CHANGE THIS to False for public datasets
)
```

**To make datasets public:**

You have two options:

#### Option 1: Modify the Source Code

Edit `scripts/dataset_creation/create_dataset.py`:

```python
# Line 256
private=False,  # Make datasets public
```

#### Option 2: Change Visibility After Upload

```bash
# Make a dataset public after upload
huggingface-cli repo update jo-mengr/dataset_name --visibility public

# Or via Python
from huggingface_hub import HfApi
api = HfApi()
api.update_repo_visibility(repo_id="jo-mengr/dataset_name", private=False)
```

**⚠️ Important Considerations:**

- **Private datasets**: Only visible to you and collaborators you explicitly add
- **Public datasets**: Anyone can view, download, and use
- **Data sensitivity**: Ensure your data doesn't contain patient-identifiable information before making public
- **License**: Consider adding an appropriate license to your dataset

### Repository Naming

The final repository ID is automatically generated from configuration:

```python
def build_repo_id(
    base_repo_id: str,           # Your HF username/org
    dataset_names: str,          # Dataset name
    dataset_format: str,         # "multiplets", "pairs", or "single"
    caption_key: str,            # Caption key or "no_caption"
) -> str:
    return f"{base_repo_id}/{dataset_names}_{dataset_format}_{caption_key}"
```

**Example:**

```yaml
base_repo_id: "jo-mengr"
dataset.name: "cellxgene_pseudo_bulk_35k"
dataset_format: "multiplets"
caption_key: "natural_language_annotation"
```

**Result:** `jo-mengr/cellxgene_pseudo_bulk_35k_multiplets_natural_language_annotation`

### Automatic Versioning

If a dataset with the same name already exists, the pipeline **automatically adds a version suffix**:

```python
# First upload
"jo-mengr/dataset_example_multiplets_cell_type"

# Second upload (repo exists)
"jo-mengr/dataset_example_multiplets_cell_type_v2"

# Third upload
"jo-mengr/dataset_example_multiplets_cell_type_v3"
```

**Implementation:**

```python
def check_and_version_repo_id(base_repo_id: str) -> str:
    """Check if repository exists and add version suffix if needed."""
    try:
        api.repo_info(repo_id=base_repo_id, token=token)
        # Repo exists, increment version
        version = 2
        while True:
            versioned_repo_id = f"{base_repo_id}_v{version}"
            try:
                api.repo_info(repo_id=versioned_repo_id, token=token)
                version += 1  # This version exists too
            except RepositoryNotFoundError:
                return versioned_repo_id  # Found available version
    except RepositoryNotFoundError:
        return base_repo_id  # Original name available
```

**Benefits:**

- ✅ Never overwrites existing datasets
- ✅ Preserves previous versions
- ✅ No manual intervention needed

---

## Template Customization

The dataset creation pipeline uses **Jinja2 templates** to generate rich README files (dataset cards) for the Hugging Face Hub.

### Template Location

Templates are stored in:

```
src/adata_hf_datasets/templates/
├── cellwhisperer_train.md        # Main training dataset template
├── cellwhisperer_test.md         # Test dataset template
└── custom_template.md            # Your custom templates
```

### Default Template: `cellwhisperer_train.md`

The default template includes:

1. **Dataset Overview**: Brief description
2. **Dataset Structure**: Splits and sizes
3. **Data Fields**: Detailed field descriptions
4. **Source Data**: Information about source AnnData files
5. **Embeddings**: List of included embeddings
6. **Usage Examples**: Code snippets for loading and using the dataset
7. **Metadata**: Custom metadata (cs_length, share links, etc.)

### Customizing Templates

#### Step 1: Create a Custom Template

Create a new template file in `src/adata_hf_datasets/templates/`:

````markdown
## <!-- src/adata_hf_datasets/templates/my_custom_template.md -->

license: cc-by-4.0
task_categories:

- text-classification
- feature-extraction
  tags:
- single-cell
- biology
- omics
  size_categories:
- {{ size_category }}

---

# {{ dataset_name }}

## Overview

{{ dataset_description }}

## Dataset Statistics

- **Total samples**: {{ total_samples }}
- **Splits**: {{ splits }}
- **Embeddings**: {{ embeddings }}

## Cell Sentences

This dataset includes {{ num_sentence_keys }} cell sentence types:
{% for key in sentence_keys %}

- `{{ key }}`: {{ sentence_descriptions[key] }}
  {% endfor %}

## Custom Section

Add your custom content here. Available variables:

- embedding_generation: {{ embedding_generation }}
- dataset_type_explanation: {{ dataset_type_explanation }}
- example_share_link: {{ example_share_link }}
- cs_length: {{ cs_length }}

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{{ repo_id }}")
train_data = dataset["train"]

# Your usage example here
```
````

## Citation

```bibtex
@dataset{your_citation_here,
  author = {Your Name},
  title = { {{dataset_name}} },
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/{{ repo_id }}}
}
```

````

#### Step 2: Update the Script to Use Your Template

Edit `scripts/dataset_creation/create_dataset.py`, line 254:

```python
annotate_and_push_dataset(
    dataset=hf_dataset,
    embedding_generation=embedding_generation,
    dataset_type_explanation=dataset_type_explanation,
    repo_id=final_repo_id,
    readme_template_name="my_custom_template",  # ← Change this
    metadata=metadata,
    private=True,
)
````

#### Step 3: Add Custom Metadata

Pass custom metadata to the template:

```python
# In create_dataset.py, around line 244
metadata = {
    "cs_length": cs_length,
    "example_share_link": example_share_link,
    # Add your custom fields
    "dataset_version": "v1.0",
    "preprocessing_date": "2025-01-15",
    "organism": "Homo sapiens",
    "tissue_types": ["lung", "spleen", "blood"],
}

annotate_and_push_dataset(
    dataset=hf_dataset,
    repo_id=final_repo_id,
    readme_template_name="my_custom_template",
    metadata=metadata,  # All metadata fields available in template
    private=True,
)
```

### Available Template Variables

The following variables are automatically available in templates:

| Variable                    | Type    | Description                        |
| --------------------------- | ------- | ---------------------------------- |
| `embedding_generation`      | str     | Description of embeddings included |
| `dataset_type_explanation`  | str     | Explanation of dataset format      |
| `repo_id`                   | str     | Full repository ID                 |
| `dataset_name`              | str     | Dataset name (from config)         |
| `total_samples`             | int     | Total number of samples            |
| `splits`                    | list    | List of split names                |
| Any keys in `metadata` dict | various | Custom metadata passed to function |

### Template Best Practices

1. **Include YAML frontmatter**: HuggingFace uses this for metadata

   ```yaml
   ---
   license: cc-by-4.0
   task_categories:
     - feature-extraction
   ---
   ```

2. **Provide usage examples**: Help users get started quickly

3. **Document data fields**: Explain each column in the dataset

4. **Add citations**: Give credit to data sources and methods

5. **Include limitations**: Be transparent about dataset scope

---

## Nextcloud Integration

Nextcloud integration allows you to upload AnnData files to a cloud storage and reference them via share links in the HuggingFace dataset.

### Why Use Nextcloud?

- **Large files**: AnnData files can be gigabytes; storing download links in HF dataset is more efficient than embedding data
- **Versioning**: Keep multiple versions of raw data without re-uploading to HF Hub
- **Accessibility**: Share data with collaborators who may not need the HF dataset

### Configuration

```yaml
dataset_creation:
  use_nextcloud: true # Enable Nextcloud upload
  force_reupload: true # Re-upload even if files exist

  nextcloud_config:
    url: "https://nextcloud.example.com"
    username: "your_username"
    password: "your_password" # Or use app password
    remote_path: "" # Auto-set to split_dir
```

**⚠️ Security Note:** Store credentials in environment variables or `.env` file:

```bash
# .env file
NEXTCLOUD_URL="https://nextcloud.example.com"
NEXTCLOUD_USER="your_username"
NEXTCLOUD_PASSWORD="your_app_password"
```

Update config to use environment variables:

```yaml
nextcloud_config:
  url: "${NEXTCLOUD_URL}"
  username: "${NEXTCLOUD_USER}"
  password: "${NEXTCLOUD_PASSWORD}"
```

### How It Works

1. **Upload**: Each AnnData file is zipped and uploaded to Nextcloud

   ```
   train/chunk_0.zarr → chunk_0.zarr.zip → Nextcloud
   ```

2. **Share link generation**: Nextcloud creates a public share link for each file

   ```
   "https://nextcloud.example.com/s/abc123xyz"
   ```

3. **Store in dataset**: The share link is stored in the `adata_link` column

   ```python
   {
       "sample_idx": "AAACCTGAGAAACCAT-1",
       "cell_sentence_1": "...",
       "adata_link": "https://nextcloud.example.com/s/abc123xyz"
   }
   ```

4. **Downstream usage**: Models download the AnnData file when needed

   ```python
   import anndata as ad
   import requests

   # Download from Nextcloud share link
   response = requests.get(row["adata_link"])
   with open("temp.zarr.zip", "wb") as f:
       f.write(response.content)

   # Extract and load
   import zipfile
   with zipfile.ZipFile("temp.zarr.zip", "r") as zip_ref:
       zip_ref.extractall("temp.zarr")
   adata = ad.read_zarr("temp.zarr")
   ```

### Disabling Nextcloud

If you don't need cloud storage (e.g., for local use):

```yaml
dataset_creation:
  use_nextcloud: false
```

**What happens:**

- Files are **not uploaded** to Nextcloud
- `adata_link` contains **local absolute paths** instead of share links
- Dataset is only usable on the same machine or with accessible file paths

---

## Output Structure

### Directory Layout

```
$base_file_path/
└── processed_with_emb/
    └── train/
        └── {dataset_name}/
            ├── train/
            │   ├── chunk_0.zarr
            │   └── chunk_1.zarr
            └── val/
                └── chunk_0.zarr
```

### Logs and Metadata

```
$WORKFLOW_DIR/
└── dataset_creation/
    └── job_{SLURM_JOB_ID}/
        ├── create_ds.out              # STDOUT
        ├── create_ds.err              # STDERR
        ├── {dataset_name}/            # Local copy of dataset
        │   ├── train/
        │   │   ├── data-00000-of-00001.arrow
        │   │   ├── dataset_info.json
        │   │   └── state.json
        │   └── val/
        │       └── ...
        └── .hydra/
            ├── config.yaml
            └── overrides.yaml
```

### HuggingFace Dataset Structure

The created dataset has the following structure:

```python
DatasetDict({
    train: Dataset({
        features: {
            'sample_idx': Value(dtype='string'),
            'cell_sentence_1': Value(dtype='string'),
            'cell_sentence_2': Value(dtype='string'),
            'cell_sentence_3': Value(dtype='string'),
            'cell_sentence_4': Value(dtype='string'),
            'positive': Value(dtype='string'),
            'negative_1_idx': Value(dtype='string'),
            'negative_2_idx': Value(dtype='string'),
            'negative_3_idx': Value(dtype='string'),
            'negative_4_idx': Value(dtype='string'),
            'adata_link': Value(dtype='string'),
        },
        num_rows: 95000
    }),
    val: Dataset({
        features: { ... },
        num_rows: 5000
    })
})
```

---

## Workflow Integration

The dataset creation step is designed to run as part of the complete workflow orchestration.

**For comprehensive information on running the complete workflow (including dataset creation), please refer to:**

**📖 [Workflow Orchestration Guide](../workflow/README.md)**

### Automatic Execution

When using the workflow orchestrator, dataset creation automatically:

1. Waits for embedding steps to complete
2. Loads data from `processed_with_emb/` directory
3. Validates required embeddings exist
4. Creates cell sentences
5. Builds HuggingFace dataset
6. Optionally uploads to Nextcloud
7. Pushes to HuggingFace Hub

### Step Dependencies

```
Download → Preprocessing → Embed Prep → CPU Embed → GPU Embed → Dataset Creation
                                                                   (this step)
```

---

## Advanced Topics

### Manual Execution

**⚠️ Important:** Dataset creation is integrated with the workflow orchestrator. **We recommend using the workflow** for production use.

For debugging or custom workflows:

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export BASE_FILE_PATH=/path/to/data
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Run dataset creation
python scripts/dataset_creation/create_dataset.py \
    --config-name=dataset_example

# With overrides
python scripts/dataset_creation/create_dataset.py \
    --config-name=dataset_example \
    ++dataset_creation.cs_length=2048 \
    ++dataset_creation.push_to_hub=false \
    ++caption_key="description"
```

### SLURM Execution

```bash
# Set dataset configuration
export DATASET_CONFIG=dataset_example

# Optional overrides
export CS_LENGTH_OVERRIDE=2048
export CAPTION_KEY_OVERRIDE="description"
export BASE_FILE_PATH=/custom/path

# Submit job
sbatch scripts/dataset_creation/run_create_ds.slurm
```

### Debugging Tips

#### Check Input Files

```python
from pathlib import Path
import anndata as ad

data_dir = Path("data/processed_with_emb/train/dataset_name")

for split in ["train", "val"]:
    split_dir = data_dir / split
    print(f"\n{split} split:")

    for f in split_dir.glob("*.zarr"):
        adata = ad.read_zarr(f)
        print(f"  {f.name}: {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"    Embeddings: {list(adata.obsm.keys())}")
        print(f"    Required cols: {['cell_type' in adata.obs.columns]}")
```

#### Test Cell Sentence Generation

```python
from adata_hf_datasets.dataset import create_cell_sentences
import anndata as ad

adata = ad.read_zarr("data/processed_with_emb/train/dataset/train/chunk_0.zarr")

# Create cell sentences
adata = create_cell_sentences(
    adata=adata,
    gene_name_column="gene_name",
    annotation_column="cell_type",
    cs_length=4096,
)

# Check results
print("Cell sentence keys:", [k for k in adata.obs.columns if "sentence" in k or "semantic" in k])
print("\nExample cell sentence:")
print(adata.obs.iloc[0]["cell_sentence"])
```

#### Test Dataset Construction

```python
from adata_hf_datasets.dataset import AnnDataSetConstructor
import anndata as ad

constructor = AnnDataSetConstructor(
    dataset_format="multiplets",
    negatives_per_sample=2,
    resolve_negatives=False,
)

adata = ad.read_zarr("data/processed_with_emb/train/dataset/train/chunk_0.zarr")

constructor.add_anndata(
    adata=adata,
    sentence_keys=["sample_id_og", "cell_sentence"],
    caption_key="natural_language_annotation",
    batch_key="dataset_title",
    adata_link="test_link",
)

dataset = constructor.get_dataset()
print(f"Created dataset with {len(dataset)} samples")
print(f"Features: {list(dataset.features.keys())}")
print(f"\nExample record:")
print(dataset[0])
```

### Common Issues

**Issue: "Missing required obsm keys"**

- **Cause**: Embedding step didn't complete or embeddings weren't generated
- **Solution**: Check `required_obsm_keys` in config or run embedding steps

**Issue: "annotation_key is None"**

- **Cause**: No cell type annotations available
- **Solution**: Set `annotation_key: null` and remove semantic sentence keys

**Issue: "Repository already exists"**

- **Cause**: Dataset with same name on HF Hub
- **Solution**: Automatic versioning will add `_v2`, `_v3`, etc.

**Issue: "Authentication error" when pushing to Hub**

- **Cause**: Not logged in to HuggingFace
- **Solution**: Run `huggingface-cli login` or set `HF_TOKEN` environment variable

**Issue: "Nextcloud upload failed"**

- **Cause**: Invalid credentials or network issues
- **Solution**: Verify credentials, check network, or set `use_nextcloud: false`

---

## Summary

The dataset creation pipeline transforms processed AnnData files into HuggingFace datasets ready for training through:

✅ **Flexible formats**: Multiplets, pairs, or single for different training paradigms
✅ **Rich cell sentences**: Multiple representations (IDs, gene lists, semantics)
✅ **Intelligent negative sampling**: Batch-aware, alternating caption/sentence negatives
✅ **Memory efficiency**: Generator-based construction, index-based negatives
✅ **HuggingFace integration**: Automatic upload with versioning and rich templates
✅ **Nextcloud support**: Cloud storage for large AnnData files
✅ **Customizable templates**: Full control over dataset card appearance

**The output is a publication-ready HuggingFace dataset for training contrastive learning models on single-cell data.**

For running the complete pipeline including dataset creation, see the **[Workflow Orchestration Guide](../workflow/README.md)**.
