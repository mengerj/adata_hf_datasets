---
language:
  - code
pretty_name: "Transcriptome with text annotations - paired dataset"
tags:
  - multimodal
  - omics
  - sentence-transformers
  - anndata
license: "mit"
task_categories:
  - zero-shot-classification
---

## Description

This dataset contains a representation of **RNA sequencing data** and text descriptions.
$dataset_type_explanation

**Cell Sentence Length**: The cell sentences in this dataset have a length of $cs_length genes.

The **RNA sequencing data** used for training was originally gathered and annotated in the **CellWhisperer** project. It is derived from
**CellxGene** and **GEO**. Detailed information on the gathering and annotation of the data can be read in the CellWhisperer Manuscript.

## Example Data Row

The dataset contains the following column structure (example from the first row):

```
$example_data_formatted
```

The processed .h5ad files used to create this dataset are stored remotely. An example file can be accessed here: $example_share_link

The AnnData Objects were processed and converted into a Hugging Face dataset using the [adata_hf_datasets](https://github.com/mengerj/adata_hf_datasets) Python package.
The dataset can be used to train a multimodal model, aligning transcriptome and text modalities with the **sentence-transformers** framework.
See [mmcontext](https://github.com/mengerj/mmcontext) for examples on how to train such a model.

The anndata objects are stored on nextcloud and a sharelink is provided as part of the dataset to download them. These anndata objects contain
intial embeddings generated like this: $embedding_generation
These initial embeddings are used as inputs for downstream model training / inference.

## Source

- **Original Data:**
  CZ CELLxGENE Discover: **A single-cell data platform for scalable exploration, analysis and modeling of aggregated data CZI Single-Cell Biology, et al. bioRxiv 2023.10.30**
  [Publication](https://doi.org/10.1101/2023.10.30.563174)

  GEO Database: Edgar R, Domrachev M, Lash AE.
  Gene Expression Omnibus: NCBI gene expression and hybridization array data repository
  Nucleic Acids Res. 2002 Jan 1;30(1):207-10

- **Annotated Data:**
  Cell Whisperer: _Multimodal learning of transcriptomes and text enables interactive single-cell RNA-seq data exploration with natural-language chats_
  _Moritz Schaefer, Peter Peneder, Daniel Malzl, Mihaela Peycheva, Jake Burton, Anna Hakobyan, Varun Sharma, Thomas Krausgruber, Jörg Menche, Eleni M. Tomazou, Christoph Bock_
  [Publication](https://doi.org/10.1101/2024.10.15.618501)
  Annotated Data: [CellWhisperer website](https://cellwhisperer.bocklab.org/)
- **Embedding Methods:**
  scVI: _Lopez, R., Regier, J., Cole, M.B. et al. Deep generative modeling for single-cell transcriptomics. Nat Methods 15, 1053–1058 (2018). https://doi.org/10.1038/s41592-018-0229-2_
  geneformer: _Theodoris, C.V., Xiao, L., Chopra, A. et al. Transfer learning enables predictions in network biology. Nature 618, 616–624 (2023)._ [Publication](https://doi.org/10.1038/s41586-023-06139-9)
- **Further important packages**
  anndata: _Isaac Virshup, Sergei Rybakov, Fabian J. Theis, Philipp Angerer, F. Alexander Wolf. anndata: Annotated data. bioRxiv 2021.12.16.473007_
  [Publication](https://doi.org/10.1101/2021.12.16.473007)
  scnapy: _Wolf, F., Angerer, P. & Theis, F. SCANPY: large-scale single-cell gene expression data analysis. Genome Biol 19, 15 (2018)._
  [Publication](https://doi.org/10.1186/s13059-017-1382-0)

## Usage

To use this dataset in Python:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("$repo_id")
```

### Understanding the Data Structure

- **sample_idx**: This column maps to the `adata.obs.index` of the original AnnData objects
- **Chunking**: Larger datasets were chunked, so each AnnData object contains only a subset of the indices from the complete dataset
- **Share Links**: Each row contains a `share_link` that can be used with requests to download the corresponding AnnData object

### Loading AnnData Objects

The share links in the dataset can be used to download the corresponding AnnData objects:

```python
import requests
import anndata as ad

# Get the share link from a dataset row
row = dataset["train"][0]  # First row as example
share_link = row["share_link"]
sample_idx = row["sample_idx"]

# Download and load the AnnData object
response = requests.get(share_link)
if response.status_code == 200:
    with open("adata.h5ad", "wb") as f:
        f.write(response.content)
    adata = ad.read_h5ad("adata.h5ad")

    # The sample_idx corresponds to adata.obs.index
    sample_data = adata[adata.obs.index == sample_idx]
    print(f"Found sample: {sample_data.shape}")
else:
    print("Failed to download AnnData object")
```
