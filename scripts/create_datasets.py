from adata_hf_datasets.utils import setup_logging
import anndata
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.utils import split_anndata
import os
from pathlib import Path
from dotenv import load_dotenv
from adata_hf_datasets.adata_ref_ds import AnnDataSetConstructor
from adata_hf_datasets.adata_ref_ds import SimpleCaptionConstructor
from adata_hf_datasets.utils import annotate_and_push_dataset
from datasets import DatasetDict

method = "scvi"
data_name = "cellxgene_pseudo_bulk"
caption_key = "natural_language_annotation"
#caption_key = "cluster_label"

nextcloud_config = {
    "url": "https://nxc-fredato.imbi.uni-freiburg.de",
    "username": "NEXTCLOUD_USER",  # env will we obtained within code
    "password": "NEXTCLOUD_PASSWORD",
    "remote_path": "",
}

project_dir = Path(__file__).resolve().parents[1]
def main():
    setup_logging()
    load_dotenv(override=True)
    adata = anndata.read_h5ad(f"{project_dir}/data/RNA/raw/{data_name}.h5ad")
    # Delete objects that are not needed and are taking up space
    del adata.obsm["natural_language_annotation_replicates"]
    del adata.layers

    dataset_name = f"{data_name}_{method}"

    embedder = InitialEmbedder(method=method)
    embedder.fit(adata)
    embedder.embed(adata)

    train_path = f"{project_dir}/data/RNA/processed/{method}/{data_name}/train.h5ad"
    val_path = f"{project_dir}/data/RNA/processed/{method}/{data_name}/val.h5ad"
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    train_data, val_adata = split_anndata(adata, train_size=0.9)
    train_data.write_h5ad(train_path)
    val_adata.write_h5ad(val_path)

    train_remote_path = f"datasets/{method}/train/bowel_disease.h5ad"
    val_remote_path = f"datasets/{method}/val/bowel_disease.h5ad"

    hf_dataset = DatasetDict()
    # Create caption constructor with desired obs keys
    for split, path in zip(["train", "val"], [train_path, val_path]):
        caption_constructor = SimpleCaptionConstructor(obs_keys=caption_key)
        nextcloud_config["remote_path"] = eval(f"{split}_remote_path")
        constructor = AnnDataSetConstructor(
            caption_constructor=caption_constructor,
            store_nextcloud=True,
            nextcloud_config=nextcloud_config,
        )
        constructor.add_anndata(file_path=path)
        # Get dataset
        dataset = constructor.get_dataset()
        hf_dataset[split] = dataset
   

    caption_generation = f"""Captions were generated with the SimpleCaptionConstructor class. That means the previosly added annotation from the
                    following obs_keys were concatenated: {caption_constructor.obs_keys}."""

    embedding_generation = f"""Embeddings were generated with the InitialEmbedder class from the adata_hf_datasets package, with method = {method}, they have 
            {embedder.embedding_dim} dimensions, and are stored in adata.obsm['X_{method}']"""

    annotate_and_push_dataset(
        dataset=hf_dataset,
        caption_generation=caption_generation,
        embedding_generation=embedding_generation,
        repo_id=f"jo-mengr/{dataset_name}",
        readme_template_name="cellxgene_pseudo_bulk",
    )
    
if __name__ == "__main__":
    main()
