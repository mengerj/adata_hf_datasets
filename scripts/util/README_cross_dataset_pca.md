# Cross-Dataset PCA Fitting

This utility fits a single PCA model across multiple datasets to ensure consistent dimensionality reduction that can be reused across different studies and analyses.

## Problem Statement

When applying PCA independently to different datasets, you get inconsistent feature spaces:

- **Different principal components**: PC1 in dataset A ≠ PC1 in dataset B
- **No transferability**: Can't apply trained models across datasets
- **Incomparable results**: Can't meaningfully compare or integrate analyses

## Solution: Cross-Dataset PCA

This script solves the problem by:

1. **Loading multiple datasets** from zarr chunk directories
2. **Aligning genes** across all datasets (intersection or predefined list)
3. **Fitting a single PCA model** on combined data from all datasets
4. **Saving the model** for consistent application to new datasets

## Critical Requirements for Reusable PCA

⚠️ **IMPORTANT**: For PCA to be reusable across datasets, all datasets must have:

- ✅ **Exact same genes** (same gene identifiers)
- ✅ **Same gene order** (consistent ordering)
- ✅ **Same number of features** (same gene count)
- ✅ **Consistent preprocessing** (same normalization, scaling)

## Features

- **Multi-dataset support**: Handles multiple zarr chunk directories
- **Flexible gene selection**: Use intersection or predefined gene lists
- **Consistent ordering**: Maintains strict gene order across datasets
- **Automatic scaling**: Detects and applies scaling if needed
- **Memory efficient**: Processes data in chunks when needed
- **Reusable models**: Saves PCA model + scaler for downstream use
- **Usage examples**: Generates example code for applying saved models

## Installation

No additional dependencies beyond the existing `adata_hf_datasets` environment:

```bash
# Core dependencies should already be installed
pip install scanpy anndata pandas numpy pyyaml scikit-learn
```

## Quick Start

### 1. Prepare your data

Ensure your datasets are preprocessed and stored as zarr chunks:

```
dataset1_chunks/
├── chunk_0.zarr
├── chunk_1.zarr
└── chunk_2.zarr

dataset2_chunks/
├── chunk_0.zarr
└── chunk_1.zarr
```

### 2. Create configuration file

```yaml
# pca_config.yaml
dataset_directories:
  - "path/to/dataset1_chunks/"
  - "path/to/dataset2_chunks/"

n_components: 50
gene_list_file: "stable_genes.txt" # optional
output_dir: "pca_results"
```

### 3. Run PCA fitting

```bash
python scripts/util/cross_dataset_pca.py --config pca_config.yaml
```

### 4. Apply to new datasets

```python
import pickle
import anndata as ad

# Load saved model
with open("pca_results/my_cross_pca_model.pkl", "rb") as f:
    saved_data = pickle.load(f)

pca_model = saved_data['pca_model']
scaler = saved_data['scaler']
gene_order = saved_data['gene_order']

# Apply to new dataset
new_adata = ad.read_h5ad("new_dataset.h5ad")
new_adata = new_adata[:, gene_order]  # Subset to same genes
X_scaled = scaler.transform(new_adata.X.toarray())
X_pca = pca_model.transform(X_scaled)
new_adata.obsm["X_pca"] = X_pca
```

## Configuration Parameters

### Required Parameters

```yaml
dataset_directories: # List of paths to zarr chunk directories
  - "path/to/dataset1/"
  - "path/to/dataset2/"

n_components: 50 # Number of PCA components
output_dir: "results" # Output directory
```

### Optional Parameters

```yaml
# Gene specification
gene_list_file:
  "genes.txt" # File with gene list (one per line)
  # If not provided, uses intersection

# Output naming
output_prefix: "my_pca" # Prefix for output files

# Reproducibility
random_state: 42 # Seed for reproducible results

# Future extensions (not implemented yet)
max_cells_per_dataset: 50000 # Subsample large datasets
subsample_strategy: "random" # Subsampling method
chunk_size: 10000 # Memory optimization
```

## Gene Selection Strategies

### Strategy 1: Intersection (Default)

Uses genes present in **ALL** datasets:

```yaml
# No gene_list_file specified
dataset_directories: [...]
n_components: 50
```

- **Pros**: Ensures all genes are present in all datasets
- **Cons**: May lose many genes if datasets differ significantly

### Strategy 2: Predefined Gene List (Recommended)

Uses a fixed gene list (e.g., from stable HVG selection):

```yaml
dataset_directories: [...]
gene_list_file: "stable_hvg_gene_list.txt"
n_components: 50
```

- **Pros**: Consistent, reproducible, can optimize gene selection
- **Cons**: Requires prior gene selection step

## Output Files

The script generates several files:

### Core Outputs

- **`{prefix}_model.pkl`**: Complete model package with PCA, scaler, genes
- **`{prefix}_genes.txt`**: Gene list in exact order (one per line)
- **`{prefix}_metadata.json`**: Summary statistics and model info

### Documentation

- **`{prefix}_usage_example.py`**: Example code for applying the model

### Model Package Contents

The pickle file contains:

```python
{
    'pca_model': fitted_sklearn_PCA_object,
    'scaler': fitted_StandardScaler_or_None,
    'gene_order': list_of_genes_in_exact_order,
    'metadata': summary_statistics,
    'config': original_configuration
}
```

## Usage Patterns

### Pattern 1: Training Set PCA

Fit PCA on training datasets, apply to validation/test:

```yaml
# Fit on training data
dataset_directories:
  - "data/train/dataset1/"
  - "data/train/dataset2/"
```

Then apply the saved model to validation and test sets.

### Pattern 2: Cross-Study PCA

Fit PCA across multiple studies for meta-analysis:

```yaml
dataset_directories:
  - "data/study1_processed/"
  - "data/study2_processed/"
  - "data/study3_processed/"
```

### Pattern 3: Stable Gene Set PCA

Combine with stable HVG selection:

```bash
# 1. First run stable HVG selection
python scripts/util/stable_hvg_selection.py --config hvg_config.yaml

# 2. Use stable genes for PCA
python scripts/util/cross_dataset_pca.py --config pca_config.yaml
# where pca_config.yaml specifies:
# gene_list_file: "stable_hvg_results/stable_hvg_gene_list.txt"
```

## Integration with Existing Workflows

### With Initial Embedder

Replace the per-dataset PCA in `initial_embedder.py`:

```python
# Instead of fitting new PCA
# self._pca_model = PCA(n_components=self.embedding_dim)
# self._pca_model.fit(X)

# Load pre-fitted model
with open("cross_pca_model.pkl", "rb") as f:
    saved_data = pickle.load(f)
    self._pca_model = saved_data['pca_model']
    self._scaler = saved_data['scaler']
```

### With Preprocessing Pipeline

Integrate into the preprocessing workflow:

```python
# After preprocessing
adata = pp_adata_general(adata, ...)

# Apply cross-dataset PCA instead of per-dataset PCA
adata = apply_saved_pca(adata, "cross_pca_model.pkl", obsm_key="X_pca")
```

## Best Practices

### Data Preparation

1. **Consistent preprocessing**: Use the same preprocessing pipeline for all datasets
2. **Quality control**: Remove low-quality datasets before PCA fitting
3. **Gene filtering**: Consider using stable HVG selection first
4. **Normalization**: Ensure consistent normalization across datasets

### Model Fitting

1. **Representative sampling**: Include diverse datasets in training
2. **Sufficient data**: Use enough cells/datasets for robust PCA
3. **Component selection**: Choose appropriate number of components
4. **Validation**: Test on held-out datasets

### Model Application

1. **Gene matching**: Always check gene availability before applying
2. **Order preservation**: Maintain exact gene order
3. **Scaling consistency**: Apply same scaling as during fitting
4. **Quality checks**: Validate transformed data makes sense

## Troubleshooting

### Common Issues

**"No genes common to all datasets"**

- **Cause**: Datasets have very different gene sets
- **Solution**: Use a predefined gene list or filter datasets

**"Dataset missing X required genes"**

- **Cause**: New dataset missing genes from trained model
- **Solution**: Check gene naming consistency, update gene list

**"Memory error during fitting"**

- **Cause**: Too much data loaded simultaneously
- **Solution**: Implement chunked processing (future feature)

**"PCA results look wrong"**

- **Cause**: Inconsistent preprocessing or scaling
- **Solution**: Verify preprocessing pipeline consistency

### Performance Tips

**For Large Datasets:**

- Consider subsampling cells for PCA fitting
- Use stable HVG selection to reduce gene count
- Process datasets sequentially if memory constrained

**For Memory Efficiency:**

- Use zarr format for large datasets
- Consider chunked processing
- Monitor memory usage during fitting

## Algorithm Details

### Gene Alignment Process

1. **Load datasets**: Read all zarr chunks for each dataset
2. **Combine chunks**: Concatenate chunks within each dataset
3. **Find common genes**: Intersection or from gene list
4. **Subset datasets**: Extract same genes in same order
5. **Verify alignment**: Ensure identical gene sets

### PCA Fitting Process

1. **Combine datasets**: Concatenate all cell data
2. **Check scaling**: Use `is_data_scaled()` to detect scaling
3. **Apply scaling**: Fit `StandardScaler` if needed
4. **Fit PCA**: Train sklearn PCA on combined scaled data
5. **Store components**: Save model, scaler, and metadata

### Model Application Process

1. **Load model**: Read PCA model, scaler, and gene order
2. **Subset genes**: Extract required genes in correct order
3. **Apply scaling**: Transform using saved scaler
4. **Transform data**: Apply PCA transformation
5. **Store results**: Add to `adata.obsm`

## Comparison with Alternatives

| Approach              | Consistency | Transferability | Memory    | Setup      |
| --------------------- | ----------- | --------------- | --------- | ---------- |
| **Per-dataset PCA**   | ❌ Poor     | ❌ None         | ✅ Low    | ✅ Simple  |
| **Cross-dataset PCA** | ✅ High     | ✅ Full         | ⚠️ Medium | ⚠️ Medium  |
| **Harmony/Scanorama** | ✅ High     | ⚠️ Limited      | ❌ High   | ❌ Complex |

## Future Enhancements

Potential improvements:

- **Chunked processing**: Handle very large datasets
- **Incremental PCA**: Online learning for streaming data
- **Batch correction**: Integrate with batch correction methods
- **GPU acceleration**: Speed up for large datasets
- **Model validation**: Cross-validation and metrics

## Examples

### Example 1: Basic Cross-Study Analysis

```yaml
# config.yaml
dataset_directories:
  - "study1_processed_chunks/"
  - "study2_processed_chunks/"
n_components: 50
output_dir: "cross_study_pca"
```

### Example 2: With Stable Gene Set

```yaml
# config.yaml
dataset_directories:
  - "atlas1_chunks/"
  - "atlas2_chunks/"
  - "atlas3_chunks/"
gene_list_file: "stable_hvg_1500_genes.txt"
n_components: 100
output_dir: "multi_atlas_pca"
output_prefix: "atlas_pca"
```

### Example 3: Minimal Setup

```yaml
# minimal_config.yaml
dataset_directories: ["data1/", "data2/"]
n_components: 30
output_dir: "results"
```

This comprehensive utility ensures that your PCA-based dimensionality reduction is consistent, transferable, and reproducible across all your datasets and analyses.
