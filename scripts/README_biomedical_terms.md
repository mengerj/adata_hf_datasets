# Biomedical Term List Generation

This directory contains scripts to fetch comprehensive lists of biomedical terms and generate descriptions for them using reliable scientific databases.

## Overview

The system consists of two main scripts:

1. **`fetch_biomedical_term_lists.py`** - Fetches term lists from authoritative sources
2. **`use_saved_term_lists.py`** - Uses saved lists with the term description system

## Quick Start

### 1. Install Dependencies

Make sure you have the required packages:

```bash
pip install requests pandas transformers
# Optional: For CAP client
pip install cap-python-client
```

### 2. Fetch Term Lists

```bash
python scripts/fetch_biomedical_term_lists.py
```

This will create these files in the `resources/` folder:

- `cell_types.txt` - Cell types from CAP and Cell Ontology
- `genes.txt` - Human genes from HGNC
- `diseases.txt` - Diseases from MONDO ontology
- `tissues.txt` - Tissues from Uberon ontology

### 3. Generate Descriptions

```bash
# Quick test with 10 cell types
python scripts/use_saved_term_lists.py test

# Full run with 100 terms per category
python scripts/use_saved_term_lists.py full
```

## Data Sources

### Cell Types

- **Primary**: Cell Annotation Platform (CAP) via `cap-python-client`
- **Fallback**: Cell Ontology (CL) via EBI OLS API
- **Quality**: Human-curated cell type annotations with ontology IDs

### Genes

- **Primary**: HGNC (HUGO Gene Nomenclature Committee) REST API
- **Fallback**: Common gene list with Gene Ontology terms
- **Quality**: Official approved human gene symbols

### Diseases

- **Source**: MONDO Disease Ontology via EBI OLS API
- **Quality**: Standardized disease terms with cross-references
- **Filtering**: Excludes overly technical terms and those with many numbers

### Tissues

- **Source**: Uberon anatomy ontology via EBI OLS API
- **Quality**: Cross-species anatomical terms
- **Filtering**: Excludes developmental and very technical terms

## Configuration

### Term Limits

Edit `fetch_biomedical_term_lists.py` to adjust the number of terms fetched:

```python
limits = {
    'cell_types': 3000,
    'genes': 15000,
    'diseases': 4000,
    'tissues': 1500
}
```

### Description Generation

Edit `use_saved_term_lists.py` to adjust description parameters:

```python
# Number of terms to process (for testing)
subset_size = 50

# Advanced settings
batch_size = 25  # Terms per API batch
include_synonyms = True
calculate_token_length = True
tokenizer_model = 'neuml/pubmedbert-base-embeddings'
```

## Output Structure

### Term Lists (`resources/`)

```
resources/
├── cell_types.txt    # One cell type per line
├── genes.txt         # One gene symbol per line
├── diseases.txt      # One disease name per line
└── tissues.txt       # One tissue name per line
```

### Descriptions (`out/data/{data_dir}/descriptions/`)

```
out/data/comprehensive_descriptions/descriptions/
├── cell_types_descriptions.csv
├── genes_descriptions.csv
├── diseases_descriptions.csv
└── tissues_descriptions.csv
```

Each CSV contains:

- `term` - The biomedical term
- `term_type` - Category (cell_types, genes, etc.)
- `description` - Authoritative definition
- `mesh_id` - Ontology ID
- `synonyms` - Alternative names
- `mesh_tree` - Ontology classification
- `source_database` - Source database
- `definition_source` - Specific source
- `token_length` - Description length in tokens

### HuggingFace Datasets

If `save_to_hf=True`, creates anchor-positive pair datasets:

- `{data_dir}_term_descriptions_cell_types_anchor_positive`
- `{data_dir}_term_descriptions_genes_anchor_positive`
- `{data_dir}_term_descriptions_diseases_anchor_positive`
- `{data_dir}_term_descriptions_tissues_anchor_positive`

Each contains:

- `anchor` - Term (including synonyms)
- `positive` - Clean definition
- `source_database` - Database with proper attribution
- `source_link` - Direct ontology link
- `is_synonym` - Boolean flag for synonym pairs

## Usage Examples

### Load Terms Programmatically

```python
from scripts.fetch_biomedical_term_lists import BiomedicalTermFetcher

fetcher = BiomedicalTermFetcher()

# Load specific term lists
cell_types = fetcher.load_terms_from_file('cell_types.txt')
genes = fetcher.load_terms_from_file('genes.txt')

print(f"Loaded {len(cell_types)} cell types")
print(f"Sample: {cell_types[:5]}")
```

### Custom Description Generation

```python
from adata_hf_datasets.config import Config, TermDescriptionConfig
from adata_hf_datasets.term_descriptions import gen_term_descriptions

# Custom configuration
config = Config(
    data_dir="my_custom_descriptions",
    save_to_hf=False  # Don't upload to HuggingFace
)

description_config = TermDescriptionConfig(
    config=config,
    email="your.email@institution.edu",
    cell_types=cell_types[:20],  # Just 20 cell types
    pull_descriptions=True
)

gen_term_descriptions(description_config)
```

## Troubleshooting

### CAP Client Issues

If CAP client fails:

```
Warning: CAP direct method failed: [error]
Info: Fetching cell types from Cell Ontology...
```

This is expected - the script will automatically fall back to Cell Ontology.

### API Rate Limits

The scripts include delays between requests:

- 0.1-0.3 seconds between NCBI requests
- 0.1 seconds between EBI OLS requests

If you encounter rate limits, the scripts will retry with exponential backoff.

### Large Term Lists

For very large lists:

1. Increase `subset_size` gradually
2. Use `batch_size=10` for slower but more reliable processing
3. Set `search_books=False` to avoid slow searches

### Email Configuration

Update the email in both scripts:

```python
email = "your.email@institution.edu"  # REQUIRED for NCBI API
```

## Citations

When using the generated data, please cite the original ontologies:

- **Cell Ontology**: Diehl, A.D., et al. (2016). Journal of Biomedical Semantics, 7(1), 44.
- **HGNC**: HUGO Gene Nomenclature Committee
- **MONDO**: Vasilevsky, N.A., et al. (2022). medRxiv.
- **Uberon**: Mungall, C.J., et al. (2012). Genome Biology, 13(1), R5.

## Support

For issues with:

- **Term fetching**: Check network connectivity and API status
- **Description generation**: Verify email configuration and NCBI access
- **HuggingFace upload**: Check HF token configuration in `hf_config.py`
