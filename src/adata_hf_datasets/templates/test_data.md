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

This dataset contains **RNA sequencing data** (or actually references to anndata files stored on a cloud). It is a dataset supposed to be used for testing and held-out from training.

The Test data is partly taken from the cellwhisperer project (bowel disease dataset) and from Luecken et. al.
It was processed and converted into a Hugging Face dataset using the [adata_hf_datasets](https://github.com/mengerj/adata_hf_datasets) Python package.
The dataset can be used to train a multimodal model, aligning transcriptome and text modalities with the **sentence-transformers** framework.
See [mmcontext](https://github.com/mengerj/mmcontext) for examples on how to train such a model.

The anndata objects are stored on nextcloud and a sharelink is provided as part of the dataset to download them. These anndata objects contain
intial embeddings generated like this: $embedding_generation
These initial embeddings are used as inputs for downstream model inference.
$caption_info

## Source

- **Original Data:**
  CZ CELLxGENE Discover: **A single-cell data platform for scalable exploration, analysis and modeling of aggregated data CZI Single-Cell Biology, et al. bioRxiv 2023.10.30**
  [Publication](https://doi.org/10.1101/2023.10.30.563174)

  Bowel Disease: _Parikh, Kaushal, Agne Antanaviciute, David Fawkner-Corbett, Marta Jagielowicz, Anna Aulicino, Christoffer Lagerholm, Simon Davis, et al. 2019. “Colonic Epithelial Cell Diversity in Health and Inflammatory Bowel Disease.” Nature 567 (7746): 49–55_
  [GEO accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116222)

  Other Test Data: Luecken, Malte D., M. Büttner, K. Chaichoompu, A. Danese, M. Interlandi, M. F. Mueller, D. C. Strobl, et al. “Benchmarking Atlas-Level Data Integration in Single-Cell Genomics.” Nature Methods 19, no. 1 (January 2022): 41–50.
  [Publication](https://doi.org/10.1038/s41592-021-01336-8).

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

dataset = load_dataset("$repo_id")
```

The anndata reference is a json string which contains a share_link to the remotly stored anndata object. It can be obtained like this:

```python
import json
import anndata
import requests

# hf makes a
adata_ref = json.loads(dataset["train]["anndata_ref"][0])
share_link = adata_ref["file_path"]
sample_id = adata_ref["sample_id"]
save_path = "data"
response = requests.get(share_link)
if response.status_code == 200:
  # Write the content of the response to a local file
  with open(save_path, "wb") as file:
    file.write(response.content)
else:
  print("Failed to read data from share link.")

adata = anndata.read_h5ad(save_path)
# The dataset contains several pre-computed embeddings. Lets for example get the embeddings computed with "scvi":
sample_idx = adata.obs.index == sample_id
sample_embedding = adata.obsm["X_scvi"][sample_idx]
# This sample embedding is described the the caption (loaded above)
# Note that you can cache your anndata files so you don't need to reload the anndata object if the filepath is still the same.
```
