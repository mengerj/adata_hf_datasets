import logging

import anndata
import scipy.sparse as sp
from pathlib import Path
import os
import psutil

logger = logging.getLogger(__name__)


class BaseAnnDataEmbedder:
    """Abstract base class for an embedding method that works on AnnData objects."""

    def __init__(self, **kwargs):
        """
        Base constructor for embedder.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for subclass-specific initialization.
        """
        self.kwargs = kwargs  # Store additional arguments for debugging/logging

    def fit(self, adata: anndata.AnnData, **kwargs) -> None:
        """
        Train or fit the embedding model on `adata`. Some methods may not require training.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to be used for training/fitting the embedding.
        kwargs : dict
            Additional keyword arguments for fitting.
        """
        raise NotImplementedError

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pp", **kwargs) -> None:
        """
        Transform the data into the learned embedding space and store in `adata.obsm[obsm_key]`.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to transform into the embedding space.
        obsm_key : str, optional
            The key under which the embedding will be stored in `adata.obsm`.
        kwargs : dict
            Additional keyword arguments for embedding.
        """
        raise NotImplementedError


class PCAEmbedder(BaseAnnDataEmbedder):
    """PCA-based embedding for single-cell data stored in AnnData."""

    def __init__(self, embedding_dim: int = 64, **kwargs):
        """
        Initialize the PCA embedder.

        Parameters
        ----------
        embedding_dim : int
            Number of principal components to retain.
        kwargs : dict
            Additional keyword arguments. Not used.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self._pca_model = None

    def fit(self, adata: anndata.AnnData, **kwargs) -> None:
        """Fit a PCA model to the AnnData object's .X matrix."""
        logger.info("Fitting PCA with %d components.", self.embedding_dim)
        from sklearn.decomposition import PCA

        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        self._pca_model = PCA(
            n_components=self.embedding_dim, **kwargs
        )  # Pass kwargs to PCA
        self._pca_model.fit(X)

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pca", **kwargs) -> None:
        """Transform the data via PCA and store in `adata.obsm[obsm_key]`."""
        if self._pca_model is None:
            raise RuntimeError("PCA model is not fit yet. Call `fit(adata)` first.")

        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.obsm[obsm_key] = self._pca_model.transform(X)


class SCVIEmbedder(BaseAnnDataEmbedder):
    """SCVI Encoder."""

    def __init__(self, embedding_dim: int = 64, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = None

    def fit(
        self, adata: anndata.AnnData, batch_key, layer_key="counts", **kwargs
    ) -> None:
        """Set up scVI model and train on the data."""
        try:
            import scvi
        except ImportError:
            raise ImportError("scvi-tools is not installed.")

        logger.info("Setting up scVI model with embedding_dim=%d", self.embedding_dim)
        if not adata.layers[layer_key]:
            adata.layers[layer_key] = adata.X.copy()
        scvi.model.SCVI.setup_anndata(adata, layer=layer_key, batch_key=batch_key)
        self.model = scvi.model.SCVI(adata, n_latent=self.embedding_dim, **kwargs)

        logger.info("Training scVI model.")
        self.model.train()

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_scvi", **kwargs) -> None:
        """Use the trained scVI model to compute latent embeddings for each cell."""
        if self.model is None:
            raise RuntimeError("scVI model not trained. Call `fit(adata)` first.")
        adata.obsm[obsm_key] = self.model.get_latent_representation(adata)


class GeneformerEmbedder(BaseAnnDataEmbedder):
    """Geneformer Encoder

    To use this method, geneformer needs to be installed according to the instructions at https://geneformer.readthedocs.io/en/latest/
    Not all embedding dimension are supported and different models can be chosen.
    There are models with 2048 dimensions, trained on 30M cells and 4096 dimensions trained on 95M cells.
    """

    def __init__(
        self,
        model_input_size: int = 4096,
        num_layers: int = 12,
        special_model: bool | str = False,
        EmbExtractor_init: dict = {},
        **kwargs,
    ):
        """
        Parameters
        ----------
        model_input_size : int, optional
            The input size of the geneformer model. Older models were trained with 2048 genes per cell, newer models with 4096 genes per cell.
        num_layers : int, optional
            Number of layers in the geneformer model. Only 6, 12, 20 are supported.
        special_model : bool | str, optional
            You can provide a string. Currently the only supported special model is "CLcancer" which was finetuned for cancer data.
        EmbExtractor_init : dict, optional
            Dictionary with additional parameters for the EmbExtractor. See geneformer documentation for more information.
        kwargs : dict
            Additional keyword arguments. Not used.
        """
        super().__init__()
        self.model = None
        self.embedding_dim = 512  # geneformer embeddings are always 512. Not enforced but used for documentation.
        if model_input_size not in [2048, 4096]:
            raise ValueError(
                "Only embedding dimensions of 2048 and 4096 are supported for geneformer initial embeddings."
            )
        else:
            self.model_input_size = model_input_size
            logger.info(
                "Setting up an Geneformer model with model_input_size=%d",
                self.model_input_size,
            )
        # get the path of this repository
        project_dir = Path(__file__).resolve().parents[2]
        if model_input_size == 2048:
            if num_layers not in [6, 12]:
                raise ValueError(
                    "Only 6 and 12 layers are supported for a model dimension of 2048 for geneformer initial embeddings."
                )
            dictionary_dir = (
                f"{project_dir}/external/Geneformer/geneformer/gene_dictionaries_30M"
            )
            self.model_dir = (
                f"{project_dir}/external/Geneformer/gf-{num_layers}L-30M-i2048"
            )
            self.ensembl_mapping_dict = (
                f"{dictionary_dir}/ensembl_mapping_dict_gc30M.pkl"
            )
            self.token_dictionary_file = f"{dictionary_dir}/token_dictionary_gc30M.pkl"
            self.gene_median_file = f"{dictionary_dir}/gene_median_dictionary_gc30M.pkl"
        if model_input_size == 4096:
            if num_layers not in [12, 20]:
                raise ValueError(
                    "Only 12 and 20 layers are supported for a model dimension of 4096 for geneformer initial embeddings."
                )
            dictionary_dir = f"{project_dir}/external/Geneformer/geneformer"
            self.model_dir = (
                f"{project_dir}/external/Geneformer/gf-{num_layers}L-95M-i4096"
            )
            if special_model:
                self.model_dir = f"{project_dir}/external/Geneformer/gf-{num_layers}L-95M-i4096_{special_model}"
            self.ensembl_mapping_dict = (
                f"{dictionary_dir}/ensembl_mapping_dict_gc95M.pkl"
            )
            self.token_dictionary_file = f"{dictionary_dir}/token_dictionary_gc95M.pkl"
            self.gene_median_file = f"{dictionary_dir}/gene_median_dictionary_gc95M.pkl"
            self.gene_name_id_dict = f"{dictionary_dir}/gene_name_id_dict_gc95M.pkl"

        self.input_dict_defaults = {
            "model_type": "Pretrained",
            "num_classes": 0,
            "emb_mode": "cls",
            "cell_emb_style": "mean_pool",
            "gene_emb_style": "mean_pool",
            "filter_data": None,
            "max_ncells": None,
            "emb_layer": -1,
            "emb_label": [
                "sample_index"
            ],  # This should not be changed if you want sample based embeddings
            "labels_to_plot": None,
            "forward_batch_size": 32,
            "nproc": 1,
            "summary_stat": None,
            "token_dictionary_file": self.token_dictionary_file,
        }
        self.input_dict_defaults.update(EmbExtractor_init)

        # geneformer by defaults reads from an anndata file and writes the embeddings to a csv. We want to work with the embeddings directly.
        # We define a tmp dir for all those files
        self.tmp_dir = Path(project_dir) / "tmp_geneformer"
        self.tmp_adata_dir = self.tmp_dir / "adata"
        self.tmp_dir.mkdir(exist_ok=True)
        self.tmp_adata_dir.mkdir(exist_ok=True)

    def fit(self, **kwargs) -> None:
        """We dont really fit the geneformer model, we load a pretrained model. This method is here to keep the interface consistent."""
        logging.info("Geneformer model is pretrained and does not require fitting.")
        pass

    def embed(self, adata, obsm_key: str = "X_geneformer") -> None:
        """Calling the embedding function of geneformer, based on the previously tokenized dataset and the chosen model configurations.
        If you encoder memory issues, try lowering the batch_size. Default is 10.

        Check out the geneformer documentation for more information on the input parameters.
        If you want to overwrite any of the defaults, provide them in the EmbExtractor_init dictionary."""
        try:
            from geneformer import TranscriptomeTokenizer, EmbExtractor
        except ImportError:
            raise ImportError(
                "geneformer is not installed. Please install geneformer to use GeneformerEmbedder. Follow the instructions from the geneformer documentation."
            )
        from adata_hf_datasets.utils import add_ensembl_ids
        import scanpy as sc

        # Load the data for preprocessing
        if "ensemble_ids" not in adata.var.columns:
            add_ensembl_ids(adata)
            logger.info("Added Ensembl IDs to adata.var['ensembl_id'].")
        if "n_counts" not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            adata.obs["n_counts"] = adata.obs.total_counts
            logger.info("Added 'n_counts' to adata.obs.")

        # To keep track of the order of the sample we include an index
        # make a string "s0", "s1" .. for the index
        adata.obs["sample_index"] = range(adata.shape[0])
        # Should improve this as it is inefficient to save the data again for large files
        # Don't like to overwrite the original dataset though
        adata.write_h5ad(self.tmp_adata_dir / "adata.h5ad")
        del adata

        self.dataset_name = "geneformer"
        if not os.path.exists(f"{self.tmp_dir}/{self.dataset_name}.dataset"):
            # initialise the transcriptome tokenizer
            tk = TranscriptomeTokenizer(
                custom_attr_name_dict={"sample_index": "sample_index"},
                nproc=1,
                gene_median_file=self.gene_median_file,
                token_dictionary_file=self.token_dictionary_file,
                gene_mapping_file=self.ensembl_mapping_dict,
            )
            tk.tokenize_data(
                self.tmp_adata_dir,  # This has to point to the directory not a file
                self.tmp_dir,
                self.dataset_name,
                output_torch_embs=False,
                file_format="h5ad",
            )
            logger.info(
                "Tokenized geneformer dataset created and stored in %s",
                self.out_dataset_dir,
            )
        else:
            logger.info(
                "Tokenized geneformer dataset already exists. Skipping tokenization."
            )
        self.out_dataset_dir = self.tmp_dir
        extractor = EmbExtractor(**self.input_dict_defaults)

        embs = extractor.extract_embs(
            self.model_dir,
            f"{self.out_dataset_dir}/{self.dataset_name}.dataset",
            self.tmp_dir,
            output_prefix=f"{self.dataset_name}_embeddings",
            cell_state=None,
        )
        # reorder the embeddings after the index in embs["sample_index"]
        embs_sorted = embs.loc[embs["sample_index"].sort_values().index]
        embs_matrix = embs_sorted.drop(columns=["sample_index"]).values
        adata = anndata.read_h5ad(self.tmp_adata_dir / "adata.h5ad")
        adata.obsm[obsm_key] = embs_matrix
        return adata
        # Clean up the tmp files and processes
        self._kill_process()
        # Clean up the tmp files

    def _kill_process(self):
        """Had issues with stalling processes after first exection. This function kills all child processes."""
        # List all child processes
        parent_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "ppid", "name"]):
            if proc.info["ppid"] == parent_pid:
                print(f"Killing process {proc.info['name']} (PID {proc.info['pid']})")
                proc.kill()  # Force kill the process


class InitialEmbedder:
    """Main interface for creating embeddings of single-cell data."""

    def __init__(
        self,
        method: str = "pca",
        embedding_dim: int = 64,
        **init_kwargs,
    ):
        """
        Initialize the manager and select the embedding method.

        Parameters
        ----------
        method : str
            The embedding method to use.
        embedding_dim : int
            Dimensionality of the output embedding space.
        init_kwargs : dict, optional
            Additional keyword arguments to pass to the embedding method initializer. Checkout the specific embedder for available options.
            Especially useful for geneformer.
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.init_kwargs = init_kwargs or {}

        # Dispatch to the correct embedder class
        embedder_classes = {
            "pca": PCAEmbedder,
            "scvi": SCVIEmbedder,
            "geneformer": GeneformerEmbedder,
        }

        if method not in embedder_classes:
            raise ValueError(f"Unknown embedding method: {method}")

        self.embedder = embedder_classes[method](
            embedding_dim=embedding_dim, **self.init_kwargs
        )

    def fit(self, adata: anndata.AnnData, **fit_kwargs) -> None:
        """
        Fit/train the embedding model on the provided AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to be used for fitting.
        fit_kwargs : dict
            Additional keyword arguments for fitting. Check the specific embedder for available options.
        """
        logger.info(
            "Fitting method '%s' with embedding_dim=%d", self.method, self.embedding_dim
        )
        self.embedder.fit(adata=adata, **fit_kwargs)

    def embed(self, adata: anndata.AnnData, **embed_kwargs) -> None:
        """
        Transform the data into the learned embedding space.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to be transformed.
        obsm_key : str
            The key under which the embedding will be stored in `adata.obsm`.
        embed_kwargs : dict
            Additional keyword arguments for embedding. Check the specific embedder for available options.
        """
        logger.info(
            f"Embedding data using method {self.method}. Storing embeddings in X_{self.method}."
        )
        self.embedder.embed(adata, obsm_key=f"X_{self.method}", **embed_kwargs)
