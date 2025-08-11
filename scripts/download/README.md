# Dataset Download Scripts

This directory contains scripts for downloading datasets (primarily h5ad files for single-cell genomics) with optional subsetting capabilities, designed to run on Slurm clusters.

## Files

- `download_dataset.py` - Main Python script for downloading and optionally subsetting datasets
- `submit_download.sh` - Slurm batch script for job submission
- `submit_download_job.sh` - Helper script for easy job submission with presets
- `README.md` - This documentation

## Quick Start

### 1. Basic Download (using presets)

```bash
# Download CellxGene dataset
./scripts/submit_download_job.sh cellxgene

# Download immune health atlas with 10k subset
./scripts/submit_download_job.sh immune-subset

# Download full immune health atlas
./scripts/submit_download_job.sh immune-atlas
```

### 2. Custom Download

```bash
./scripts/submit_download_job.sh custom \
    --url "https://example.com/dataset.h5ad" \
    --output "data/my_dataset.h5ad" \
    --subset-size 5000
```

## Detailed Usage

### Main Python Script (`download_dataset.py`)

The main script can be run directly for testing or used within Slurm jobs:

```bash
python scripts/download_dataset.py --help
```

#### Basic Examples

```bash
# Simple download
python scripts/download_dataset.py \
    --url "https://datasets.cellxgene.cziscience.com/f886c7d9-1392-4f09-9e10-31b953afa2da.h5ad" \
    --output "data/cellxgene.h5ad"

# Download with random subset
python scripts/download_dataset.py \
    --url "https://example.com/large_dataset.h5ad" \
    --output "data/subset.h5ad" \
    --subset-size 10000 \
    --seed 42

# Download with stratified subset (preserving cell type proportions)
python scripts/download_dataset.py \
    --url "https://example.com/dataset.h5ad" \
    --output "data/stratified_subset.h5ad" \
    --subset-size 5000 \
    --stratify-by "cell_type" \
    --preserve-proportions
```

#### Parameters

**Required:**

- `--url`: URL to download from
- `--output`: Local path to save the file

**Optional:**

- `--subset-size`: Number of observations (cells) to include in subset
- `--seed`: Random seed for reproducibility (default: 42)
- `--stratify-by`: Column name in obs to stratify subset by
- `--preserve-proportions`: Preserve proportions when stratifying
- `--temp-dir`: Directory for temporary files
- `--validate`: Validate downloaded file format
- `--keep-temp`: Keep temporary files when subsetting

### Slurm Job Submission (`submit_download_job.sh`)

The helper script provides easy presets and handles Slurm job submission:

```bash
./scripts/submit_download_job.sh --help
```

#### Available Presets

1. **cellxgene** - CellxGene example dataset
2. **immune-atlas** - Full immune health atlas (~16GB)
3. **immune-subset** - Immune health atlas with 10k cell subset
4. **custom** - Custom URL (requires `--url`)

#### Examples

```bash
# Quick preset usage
./scripts/submit_download_job.sh cellxgene

# Preset with modifications
./scripts/submit_download_job.sh immune-subset \
    --subset-size 5000 \
    --output "data/immune_5k.h5ad"

# Custom download with specific resources
./scripts/submit_download_job.sh custom \
    --url "https://example.com/big_dataset.h5ad" \
    --output "data/big_data.h5ad" \
    --subset-size 20000 \
    --mem 128G \
    --time 24:00:00

# Stratified subset preserving cell type proportions
./scripts/submit_download_job.sh immune-subset \
    --subset-size 8000 \
    --stratify-by "cell_type" \
    --preserve-proportions

# Dry run to see what would be executed
./scripts/submit_download_job.sh immune-atlas --dry-run
```

#### Slurm Configuration Options

- `--time`: Job time limit (default: 6:00:00)
- `--mem`: Memory allocation (default: 32G)
- `--cpus`: Number of CPUs (default: 4)
- `--partition`: Slurm partition (default: gpu)
- `--account`: Slurm account (default: menger)

## Advanced Features

### Stratified Subsetting

When working with single-cell data, you often want to preserve the proportions of different cell types or conditions in your subset:

```bash
# Preserve cell type proportions in a 5000-cell subset
python scripts/download_dataset.py \
    --url "https://example.com/dataset.h5ad" \
    --output "data/balanced_subset.h5ad" \
    --subset-size 5000 \
    --stratify-by "cell_type" \
    --preserve-proportions
```

This ensures that if your original dataset has 30% T cells, 20% B cells, etc., your subset will maintain similar proportions.

### Large File Handling

For very large files, the script automatically uses temporary storage during subsetting:

```bash
# Large file with custom temp directory
./scripts/submit_download_job.sh custom \
    --url "https://example.com/huge_dataset.h5ad" \
    --output "data/huge_subset.h5ad" \
    --subset-size 50000 \
    --temp-dir "/scratch/$USER" \
    --mem 256G \
    --time 48:00:00
```

### Monitoring Jobs

After submitting a job, you'll get commands to monitor it:

```bash
# Monitor job status
squeue -j <JOB_ID>

# View real-time output
tail -f logs/download_<JOB_ID>.out

# View errors
tail -f logs/download_<JOB_ID>.err

# Cancel job if needed
scancel <JOB_ID>
```

## Configuration

### Cluster-Specific Setup

You may need to modify the Slurm script (`submit_download.sh`) for your cluster:

```bash
# Update these lines in submit_download.sh:
#SBATCH --account=YOUR_ACCOUNT     # Your Slurm account
#SBATCH --partition=YOUR_PARTITION # Your preferred partition

# Uncomment and modify module loading:
# module load python/3.11
# module load gcc/9.3.0

# Uncomment and set your virtual environment:
# source /path/to/your/venv/bin/activate
```

### Default URLs in Presets

The preset URLs are defined in `submit_download_job.sh` and can be modified:

```bash
# Edit these URLs in submit_download_job.sh as needed
"cellxgene")
    URL="https://datasets.cellxgene.cziscience.com/..."
"immune-atlas")
    URL="https://allenimmunology.org/public/publication/..."
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase `--mem` parameter for large files
2. **Time Limit**: Increase `--time` for slow downloads or large files
3. **Permission Errors**: Check output directory permissions
4. **Network Issues**: Downloads will retry automatically

### Log Files

All output is logged to files in the `logs/` directory:

- `logs/download_<JOB_ID>.out` - Standard output
- `logs/download_<JOB_ID>.err` - Error output
- `download_dataset.log` - Detailed download log

### File Validation

The script can validate downloaded files:

```bash
# Enable validation (default)
python scripts/download_dataset.py --url "..." --output "..." --validate

# Skip validation for faster processing
python scripts/download_dataset.py --url "..." --output "..." --no-validate
```

## File Formats Supported

- `.h5ad` - AnnData HDF5 format (primary use case)
- `.zarr` - Zarr format for AnnData
- Other formats are downloaded but not validated

## Best Practices

1. **Use appropriate memory allocation**: Large h5ad files need sufficient RAM
2. **Set reasonable subset sizes**: Start small for testing
3. **Use stratified sampling** for balanced subsets when working with biological data
4. **Monitor disk space**: Ensure sufficient space for both temp and final files
5. **Use scratch storage** for temporary files when available
6. **Test with dry runs** before submitting large jobs

## Examples for Different Use Cases

### Quick Testing

```bash
# Small subset for testing pipelines
./scripts/submit_download_job.sh cellxgene \
    --subset-size 1000 \
    --output "data/test_data.h5ad"
```

### Production Dataset

```bash
# Full dataset with validation
./scripts/submit_download_job.sh immune-atlas \
    --validate \
    --mem 64G \
    --time 12:00:00
```

### Balanced Research Dataset

```bash
# Stratified subset for research
./scripts/submit_download_job.sh immune-subset \
    --subset-size 15000 \
    --stratify-by "donor_id" \
    --preserve-proportions \
    --output "data/research_cohort.h5ad"
```
