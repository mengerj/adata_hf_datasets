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

This dataset contains **pseudo-bulk and scRNA sequencing data** originally gathered and annotated in the **CellWhisperer** project.
It is not a paired transcriptome - text dataset, as would be used for training, but a test dataset meant for evaluation.
It was processed and converted into a Hugging Face dataset using the [adata_hf_datasets](https://github.com/mengerj/adata_hf_datasets) Python package.
The dataset can be used for inference with a multimodal model, trained to align transcriptome and text modalities with the **sentence-transformers** framework.
See [mmcontext](https://github.com/mengerj/mmcontext) for examples on how to train such a model.

The anndata objects are stored on nextcloud and a sharelink is provided as part of the dataset to download them. These anndata objects contain
intial embeddings generated like this: $embedding_generation
These initial embeddings are the input for the alignment model. Each sample is represented by one embedding vector.

$caption_info

## Source

- **Original Data:**
  Bowel Disease: _Parikh, Kaushal, Agne Antanaviciute, David Fawkner-Corbett, Marta Jagielowicz, Anna Aulicino, Christoffer Lagerholm, Simon Davis, et al. 2019. “Colonic Epithelial Cell Diversity in Health and Inflammatory Bowel Disease.” Nature 567 (7746): 49–55_
  Raw Data: [GEO accession](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116222)
- **Annotated Data:**
  Bowel Disease: _Multimodal learning of transcriptomes and text enables interactive single-cell RNA-seq data exploration with natural-language chats_
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
