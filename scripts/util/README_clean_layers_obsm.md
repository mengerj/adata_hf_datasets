# Clean Layers and Obsm Utility with HVG Selection

This utility helps you clean up layers and obsm entries from h5ad and zarr files, with optional highly variable gene (HVG) selection. It can work with either a single h5ad file or a directory containing zarr files.

## Files

- `clean_layers_obsm.py` - Main Python script
- `run_clean_layers_obsm.slurm` - Slurm job script

## Usage

### Direct Python Script

```bash
# Clean a single h5ad file
python scripts/util/clean_layers_obsm.py --input path/to/file.h5ad --layers layer1,layer2 --obsm X_pca,X_umap

# Clean all zarr files in a directory
python scripts/util/clean_layers_obsm.py --input path/to/data/ --layers counts,logcounts --obsm X_pca,X_umap

# Perform HVG selection only
python scripts/util/clean_layers_obsm.py --input path/to/data/ --hvg --n-top-genes 3000

# Batch-aware HVG selection
python scripts/util/clean_layers_obsm.py --input file.h5ad --hvg --batch-key batch --n-top-genes 2000

# Clean and perform HVG selection with custom output
python scripts/util/clean_layers_obsm.py --input data/ --output processed/ --layers counts --hvg --n-top-genes 2000

# Dry run to see what would be done (recommended first)
python scripts/util/clean_layers_obsm.py --input path/to/data/ --layers counts --obsm X_pca --hvg --dry-run
```

### Using Slurm

```bash
# Submit a slurm job - clean layers and obsm
sbatch scripts/util/run_clean_layers_obsm.slurm data/ "" "layer1,layer2" "X_pca,X_umap" false 2000 "" false

# HVG selection only with custom output
sbatch scripts/util/run_clean_layers_obsm.slurm data/ processed/ "" "" true 3000 "" false

# Batch-aware HVG selection
sbatch scripts/util/run_clean_layers_obsm.slurm file.h5ad "" "" "" true 2000 batch false

# Clean and perform HVG selection
sbatch scripts/util/run_clean_layers_obsm.slurm data/ processed/ "counts" "X_pca" true 2000 "" false

# Dry run with slurm
sbatch scripts/util/run_clean_layers_obsm.slurm data/ "" "counts" "X_pca" true 2000 "" true
```

## Arguments

### Python Script Arguments

- `--input` / `-i`: Path to h5ad file or directory containing zarr files (required)
- `--output` / `-out`: Output path for processed files (optional, defaults to input location)
- `--layers` / `-l`: Comma-separated list of layer names to remove (optional)
- `--obsm` / `-o`: Comma-separated list of obsm keys to remove (optional)
- `--hvg`: Enable highly variable gene selection (optional)
- `--n-top-genes`: Number of top highly variable genes to keep (default: 2000)
- `--batch-key`: Key in adata.obs for batch-aware HVG selection (optional)
- `--min-genes-per-cell`: Minimum number of genes per cell for preprocessing (default: 200)
- `--min-cells-per-gene`: Minimum number of cells per gene for preprocessing (default: 3)
- `--dry-run` / `-d`: Show what would be done without actually doing it (optional)

### Slurm Script Arguments

1. **Input path**: Path to h5ad file or directory containing zarr files
2. **Output path**: Output path for processed files (use empty string "" for input location)
3. **Layers**: Comma-separated list of layer names to remove (use empty string "" if none)
4. **Obsm keys**: Comma-separated list of obsm keys to remove (use empty string "" if none)
5. **HVG**: "true" to enable HVG selection, "false" to disable
6. **N top genes**: Number of top highly variable genes to keep (default: 2000)
7. **Batch key**: Key in adata.obs for batch-aware HVG selection (use empty string "" if none)
8. **Min genes per cell**: Minimum number of genes per cell for preprocessing (default: 200)
9. **Min cells per gene**: Minimum number of cells per gene for preprocessing (default: 3)
10. **Dry run**: "true" for dry run, "false" for actual processing

## Key Features

### Data Preprocessing

When HVG selection is enabled, the script automatically performs robust preprocessing using the existing pipeline functions:

- **Quality Control**: Outlier detection using median absolute deviation, mitochondrial gene filtering, ribosomal gene labeling
- **Gene/Cell Filtering**: Filters genes expressed in fewer than `min_cells_per_gene` cells and cells with fewer than `min_genes_per_cell` genes
- **Normalization**: Automatic detection and handling of raw count data with proper normalization and log transformation
- **Infinite Value Handling**: Removes genes with infinite or NaN values to prevent downstream errors
- **HVG Selection**: Batch-aware or standard highly variable gene selection with proper error handling
- This preprocessing uses the existing `pp_quality_control` and `pp_adata_general` functions to ensure robust data processing

### Batch-aware HVG Selection

- Supports batch correction using any categorical variable in `adata.obs`
- Automatically consolidates low-frequency batch categories (< 5 cells) to avoid errors
- Falls back to standard HVG selection if batch key is not found

## Examples

### Common Use Cases

1. **Remove PCA and UMAP embeddings from all zarr files:**

   ```bash
   python scripts/util/clean_layers_obsm.py --input data/ --obsm X_pca,X_umap
   ```

2. **Remove count layers from a single h5ad file:**

   ```bash
   python scripts/util/clean_layers_obsm.py --input file.h5ad --layers counts,logcounts
   ```

3. **Perform HVG selection to keep top 3000 genes:**

   ```bash
   python scripts/util/clean_layers_obsm.py --input data/ --hvg --n-top-genes 3000
   ```

4. **Batch-aware HVG selection:**

   ```bash
   python scripts/util/clean_layers_obsm.py --input file.h5ad --hvg --batch-key batch_id --n-top-genes 2000
   ```

5. **Clean layers and perform HVG selection with custom output:**

   ```bash
   python scripts/util/clean_layers_obsm.py --input data/ --output processed_data/ --layers counts --hvg --n-top-genes 2000
   ```

6. **Check what would be done (dry run):**
   ```bash
   python scripts/util/clean_layers_obsm.py --input data/ --layers counts --obsm X_pca --hvg --dry-run
   ```

### Slurm Examples

1. **Clean zarr files in a directory:**

   ```bash
   sbatch scripts/util/run_clean_layers_obsm.slurm data/ "" "counts,logcounts" "X_pca,X_umap" false 2000 "" false
   ```

2. **HVG selection only with custom output:**

   ```bash
   sbatch scripts/util/run_clean_layers_obsm.slurm data/ processed/ "" "" true 3000 "" false
   ```

3. **Batch-aware HVG selection:**

   ```bash
   sbatch scripts/util/run_clean_layers_obsm.slurm file.h5ad "" "" "" true 2000 batch_id false
   ```

4. **Clean and perform HVG selection:**

   ```bash
   sbatch scripts/util/run_clean_layers_obsm.slurm data/ processed/ "counts" "X_pca" true 2000 "" false
   ```

5. **Dry run first to check:**
   ```bash
   sbatch scripts/util/run_clean_layers_obsm.slurm data/ "" "counts" "X_pca" true 2000 "" true
   ```

## Notes

- The script will show you what layers and obsm keys are available in each file
- Always run with `--dry-run` first to see what would be done
- Files can be saved in place or to a custom output location
- The script handles both h5ad and zarr formats
- For directories, only zarr files are processed
- HVG selection can be performed with or without batch correction
- The slurm job is configured with 32GB RAM and 4 CPUs - adjust as needed for your files
- Output files will maintain the same format as input files (h5ad/zarr)

## Error Handling

- The script will warn if a specified layer or obsm key is not found
- If batch key is specified but not found, HVG selection will fail
- If any file fails to process, the script will continue with others
- Exit codes indicate success (0) or failure (1)
- All processing is logged for debugging

## Memory and Performance

- For large files, consider increasing the memory allocation in the slurm script
- The script loads entire files into memory, so ensure sufficient RAM
- HVG selection adds computational overhead, especially with batch correction
- Processing time depends on file size, number of files, and whether HVG selection is enabled
- Batch-aware HVG selection may require additional memory for large datasets
