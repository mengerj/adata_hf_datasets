import anndata
import logging
from adata_hf_datasets.utils import (
    fix_non_numeric_nans,
    ensure_log_norm,
    is_data_scaled,
)
from scvi.hub import HubModel
from scvi.model import SCVI
from pathlib import Path
import os
import tempfile
import psutil
import scanpy as sc
from anndata.experimental import AnnLoader
import scipy.sparse as sp
import numpy as np
from appdirs import user_cache_dir
import pandas as pd

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """
    Base class for all embedders. Defines a common interface.
    """

    def __init__(self, embedding_dim, **init_kwargs):
        """
        Initialize the embedder.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the output embedding space.
        init_kwargs : dict
            Additional keyword arguments for the embedder.
        """
        self.embedding_dim = embedding_dim
        self.init_kwargs = init_kwargs

    def prepare(self, adata, **kwargs):
        """
        Prepare the embedder for embedding. Subclasses decide whether
        to train from scratch, load from hub, or load from S3, etc.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset.
        **kwargs : dict
            Additional keyword arguments used for preparing.
        """
        raise NotImplementedError("Subclasses must implement 'prepare'")

    def embed(self, adata, obsm_key: str, **kwargs):
        """
        Transform the data into the learned embedding space and store in `adata.obsm`.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to be transformed.
        obsm_key : str
            The key in `adata.obsm` under which to store the embedding.
        **kwargs : dict
            Additional keyword arguments for embedding.
        """
        raise NotImplementedError("Subclasses must implement 'embed'")


class HighlyVariableGenesEmbedder(BaseEmbedder):
    """
    Selects the top `n_top` highly variable genes from an AnnData object and uses them as an embedding.
    """

    def __init__(self, embedding_dim: int = 2000, **kwargs):
        """
        Parameters
        ----------
        n_top : int, optional
            The number of highly variable genes to select.
        kwargs : dict
            Additional keyword arguments. Not used.
        """
        super().__init__(embedding_dim=embedding_dim)

    def prepare(self, adata_path: anndata.AnnData, **kwargs) -> None:
        """
        Identifies the top `embedding_dim` highly variable genes in `adata`.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data to analyze.
        kwargs : dict
            Additional keyword arguments for `scanpy.pp.highly_variable_genes`.
        """
        logger.info("Normalizing and log-transforming data before HVG selection.")
        adata = sc.read(adata_path, backed="r")
        # check if the data is already normalized
        ensure_log_norm(adata)
        # First save the raw counts as a layer

    def embed(
        self,
        adata: anndata.AnnData,
        obsm_key: str = "X_hvg",
        batch_key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Stores the expression of the selected highly variable genes as an embedding in `adata.obsm`.

        Parameters
        ----------
        adata : anndata.AnnData
            The single-cell data containing highly variable genes.
        obsm_key : str, optional
            The key under which the embedding will be stored in `adata.obsm`.
        kwargs : dict
            Additional keyword arguments. Not used.
        """
        logger.info("Selecting top %d highly variable genes.", self.embedding_dim)

        # check if any genes contain inf values
        # remove cells with infinity values
        sc.pp.highly_variable_genes(
            adata, n_top_genes=self.embedding_dim, layer="log-norm", batch_key=batch_key
        )

        if "highly_variable" not in adata.var:
            raise ValueError("Failed to compute highly variable genes.")
        logger.info("Successfully identified highly variable genes.")
        # check if there a different number of highly variable genes was selected
        if self.embedding_dim != np.sum(adata.var["highly_variable"]):
            logger.warning(
                "Selected %d highly variable genes, but %d were requested.",
                np.sum(adata.var["highly_variable"]),
                self.embedding_dim,
            )
            # if more genes were selected, take the top ones
            if np.sum(adata.var["highly_variable"]) > self.embedding_dim:
                logger.warning(
                    "Manually enforcing exact number of highly variable genes."
                )
                self._enforce_exact_hvg_from_dispersion(adata, self.embedding_dim)
            else:
                raise ValueError(
                    f"Selected less highly variable genes than requested. Requested {self.embedding_dim}, selected {np.sum(adata.var['highly_variable'])}. Full data contains {adata.shape[1]} genes."
                )

        hvg_mask = adata.var["highly_variable"].values
        X = (
            adata.layers["log-norm"].toarray()
            if sp.issparse(adata.layers["log-norm"])
            else adata.layers["log-norm"]
        )
        adata.obsm[obsm_key] = X[:, hvg_mask]
        # return to sparse matrix
        adata.obsm[obsm_key] = sp.csr_matrix(adata.obsm[obsm_key])
        logger.info(
            f"Stored highly variable gene expression in adata.obsm[{obsm_key}], with shape {adata.obsm[obsm_key].shape}"
        )
        logger.info(
            "Stored highly variable gene expression in adata.obsm[%s]", obsm_key
        )
        return adata

    def _enforce_exact_hvg_from_dispersion(self, adata: sc.AnnData, n_top: int) -> None:
        """Enforce exact number of HVGs by selecting top genes by normalized dispersion.

        Parameters
        ----------
        adata : AnnData
            AnnData object after running `highly_variable_genes`.
        n_top : int
            Number of top genes to retain.
        """
        if "dispersions_norm" not in adata.var.columns:
            raise ValueError(
                "Missing 'dispersions_norm'. Run `sc.pp.highly_variable_genes` with `flavor='seurat'` or similar."
            )

        top_genes = adata.var["dispersions_norm"].nlargest(n_top).index
        adata.var["highly_variable"] = False
        adata.var.loc[top_genes, "highly_variable"] = True


class PCAEmbedder(BaseEmbedder):
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
        super().__init__(embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim
        self._pca_model = None

    def prepare(self, adata_path: anndata.AnnData, n_cells=10000, **kwargs) -> None:
        """Fit a PCA model to the AnnData object's .X matrix."""
        logger.info(
            "Fitting PCA with %d components on %d.", self.embedding_dim, n_cells
        )
        from sklearn.decomposition import PCA

        adata_sub = sc.read(adata_path, backed="r")
        # get a random subset of cells with random
        logger.info("Subsampling %d cells for PCA.", n_cells)
        if n_cells < adata_sub.shape[0]:
            adata_sub = adata_sub[
                np.random.choice(adata_sub.shape[0], n_cells, replace=False), :
            ]
        adata_sub = adata_sub.to_memory()

        if not is_data_scaled(adata_sub.X):
            logger.info("Data is not scaled. Scaling data before PCA.")
            sc.pp.scale(adata_sub)
        X = adata_sub.X.toarray() if sp.issparse(adata_sub.X) else adata_sub.X
        self._pca_model = PCA(n_components=self.embedding_dim)  # Pass kwargs to PCA
        self._pca_model.fit(X)

    def embed(self, adata: anndata.AnnData, obsm_key: str = "X_pca", **kwargs) -> None:
        """Transform the data via PCA and store in `adata.obsm[obsm_key]`."""
        if self._pca_model is None:
            raise RuntimeError("PCA model is not fit yet. Call `prepare(adata)` first.")
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        adata.obsm[obsm_key] = self._pca_model.transform(X)
        return adata


class GeneformerEmbedder(BaseEmbedder):
    """
    Geneformer Encoder for single-cell data embeddings.

    This class uses the Geneformer package to generate a 512-dimensional embedding
    for single-cell data. It supports pre-trained models with either 2048 or 4096
    input genes, as well as different numbers of layers.

    References
    ----------
    The Geneformer package and models must be installed and available locally.
    For installation instructions, see https://geneformer.readthedocs.io/en/latest/.
    """

    def __init__(
        self,
        model_input_size: int = 4096,
        num_layers: int = 12,
        special_model: bool | str = False,
        emb_extractor_init: dict = None,
        **kwargs,
    ):
        """
        Initialize the Geneformer embedder configuration.

        Parameters
        ----------
        model_input_size : int, optional
            The input size of the Geneformer model. Valid options are 2048 or 4096.
            Older models were trained with 2048 genes per cell, newer models with 4096.
        num_layers : int, optional
            Number of layers in the Geneformer model. Valid options depend on `model_input_size`:
            - For 2048 genes: 6 or 12 layers
            - For 4096 genes: 12 or 20 layers
        special_model : bool or str, optional
            If False, the default model is used. If a string (e.g., "CLcancer"),
            uses a specialized, fine-tuned model variant with that suffix.
        emb_extractor_init : dict, optional
            Dictionary with additional parameters for the EmbExtractor from Geneformer.
            See geneformer documentation for more information.
        kwargs : dict
            Additional keyword arguments for the embedder, not used here but included
            for interface consistency.

        Raises
        ------
        ValueError
            If `model_input_size` or `num_layers` are invalid.
        """
        super().__init__(embedding_dim=512)
        self.model = None
        # self.embedding_dim = 512 # Geneformer outputs 512-dimensional embeddings. Is just used for downstream methods to know.
        self.model_input_size = model_input_size
        if model_input_size not in [2048, 4096]:
            raise ValueError(
                "Geneformer only supports model_input_size in [2048, 4096]."
            )

        # Check num_layers against model_input_size.
        valid_combinations = {
            2048: [6, 12],
            4096: [12, 20],
        }
        if num_layers not in valid_combinations[model_input_size]:
            raise ValueError(
                f"For model_input_size={model_input_size}, valid layers are {valid_combinations[model_input_size]}."
            )

        # Construct directory paths
        self.num_layers = num_layers
        self.special_model = special_model
        self.project_dir = Path(__file__).resolve().parents[2]

        # Set up dictionary paths based on model size
        if model_input_size == 2048:
            dictionary_dir = (
                self.project_dir
                / "external"
                / "Geneformer"
                / "geneformer"
                / "gene_dictionaries_30M"
            )
            self.model_dir = (
                self.project_dir
                / "external"
                / "Geneformer"
                / f"gf-{num_layers}L-30M-i2048"
            )
            self.ensembl_mapping_dict = str(
                dictionary_dir / "ensembl_mapping_dict_gc30M.pkl"
            )
            self.token_dictionary_file = str(
                dictionary_dir / "token_dictionary_gc30M.pkl"
            )
            self.gene_median_file = str(
                dictionary_dir / "gene_median_dictionary_gc30M.pkl"
            )
        else:  # model_input_size == 4096
            dictionary_dir = self.project_dir / "external" / "Geneformer" / "geneformer"
            base_dir_4096 = (
                self.project_dir
                / "external"
                / "Geneformer"
                / f"gf-{num_layers}L-95M-i4096"
            )
            if special_model:
                base_dir_4096 = Path(str(base_dir_4096) + f"_{special_model}")
            self.model_dir = base_dir_4096
            self.ensembl_mapping_dict = str(
                dictionary_dir / "ensembl_mapping_dict_gc95M.pkl"
            )
            self.token_dictionary_file = str(
                dictionary_dir / "token_dictionary_gc95M.pkl"
            )
            self.gene_median_file = str(
                dictionary_dir / "gene_median_dictionary_gc95M.pkl"
            )
            self.gene_name_id_dict = str(dictionary_dir / "gene_name_id_dict_gc95M.pkl")

        # Default parameters for EmbExtractor
        self.emb_extractor_init = {
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
            ],  # do not change if you want sample-based embeddings
            "labels_to_plot": None,
            "forward_batch_size": 16,
            "nproc": 8,
            "summary_stat": None,
            "token_dictionary_file": self.token_dictionary_file,
        }
        if emb_extractor_init is not None:
            self.emb_extractor_init.update(emb_extractor_init)

        # Create a unique temp directory for saving intermediate data
        # time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.tmp_dir = self.project_dir / f"tmp_geneformer_{time_tag}"
        # self.tmp_adata_dir = self.tmp_dir / "adata"
        # self.tmp_adata_dir.mkdir(parents=True, exist_ok=True)

        # Name of the tokenized dataset
        self.dataset_name = "geneformer"
        # Where the output tokenized dataset is stored

        logger.info(
            "Initialized GeneformerEmbedder with model_input_size=%d, num_layers=%d, special_model=%s",
            self.model_input_size,
            self.num_layers,
            self.special_model,
        )

    def prepare(self, adata_path: str, do_tokenization: bool = True, **kwargs) -> None:
        """
        Prepare (preprocess + tokenize) the data for Geneformer embeddings.

        This includes:
         - Adding Ensembl IDs to `adata.var["ensembl_id"]` if not present.
         - Calculating and storing `adata.obs["n_counts"]` if not present.
         - Creating a `sample_index` column in `adata.obs`.
         - Writing the AnnData to a temporary H5AD file.
         - Optionally tokenizing the data using `TranscriptomeTokenizer` if it
           has not been tokenized previously (i.e., if the .dataset file doesn't exist).

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to prepare.
        do_tokenization : bool
            Whether to run the tokenization step if no tokenized dataset is found.
        **kwargs : dict
            Extra parameters for future extension, not used here.

        References
        ----------
        - The data is user-provided.
        - Geneformer tokenization is performed by `TranscriptomeTokenizer`.
        """
        try:
            from geneformer import TranscriptomeTokenizer
        except ImportError:
            raise ImportError(
                "To use the Geneformer embedder, ensure `git lfs` is installed and "
                "run 'git submodule update --init --recursive'. Then install geneformer: "
                "`pip install external/Geneformer`."
            )
        self.adata_path = Path(adata_path)
        # save the tokenized dataset in the same directory as the adata
        self.adata_dir = self.adata_path.parent
        self.out_dataset_dir = self.adata_path.parent / "geneformer"
        adata = sc.read(self.adata_path, backed="r")
        # 1. Make sure the data has the required fields
        if "ensembl_id" not in adata.var.columns:
            logger.error(
                "ensembl_id not found in adata.var. Run preprocessing script or pp_geneformer first."
            )
            raise ValueError(
                "ensembl_id not found in adata.var. Run preprocessing script or pp_geneformer first."
            )
        if "n_counts" not in adata.obs.columns:
            logger.error(
                "n_counts not found in adata.obs. Run preprocessing script or pp_geneformer first."
            )
            raise ValueError(
                "n_counts not found in adata.obs. Run preprocessing script or pp_geneformer first."
            )
        if "sample_index" not in adata.obs.columns:
            logger.error(
                "sample_index not found in adata.obs. Run preprocessing script or pp_geneformer first."
            )
            raise ValueError(
                "sample_index not found in adata.obs. Run preprocessing script or pp_geneformer first."
            )

        # 3. Write to a temporary h5ad
        # h5ad_path = self.tmp_adata_dir / "adata.h5ad"
        # adata.write_h5ad(h5ad_path)
        # logger.info("Wrote AnnData to temporary file: %s", h5ad_path)

        # 4. Tokenize data if needed
        dataset_path = self.out_dataset_dir / f"{self.dataset_name}.dataset"
        if dataset_path.exists():
            logger.info(
                "Tokenized geneformer dataset already exists at %s. Skipping tokenization.",
                dataset_path,
            )
            return

        if do_tokenization:
            logger.info("Tokenizing data with TranscriptomeTokenizer...")
            tk = TranscriptomeTokenizer(
                custom_attr_name_dict={"sample_index": "sample_index"},
                nproc=6,
                gene_median_file=self.gene_median_file,
                token_dictionary_file=self.token_dictionary_file,
                gene_mapping_file=self.ensembl_mapping_dict,
            )
            # The tokenizer expects a directory containing .h5ad => pass self.tmp_adata_dir
            tk.tokenize_data(
                str(self.adata_dir),
                str(self.out_dataset_dir),
                self.dataset_name,
                file_format="h5ad",
            )
            logger.info("Created tokenized dataset: %s", dataset_path)
        else:
            logger.warning(
                "No tokenized dataset found and do_tokenization=False. "
                "Embedding will fail unless the tokenized dataset is created elsewhere."
            )

    def embed(
        self,
        adata_path: str,
        obsm_key: str = "X_geneformer",
        batch_size: int = 16,
        cleanup: bool = True,
        **kwargs,
    ) -> anndata.AnnData:
        """
        Run Geneformer embedding on the data.

        This method:
         - Reads the tokenized dataset from `self.tmp_dir`.
         - Invokes Geneformer's `EmbExtractor` to generate embeddings.
         - Re-reads the (processed) AnnData from the temporary directory.
         - Aligns the embeddings with the sample order, storing them in `adata.obsm[obsm_key]`.
         - (Optionally) cleans up temporary files.

        Parameters
        ----------
        adata_path : str
            Path to the AnnData file to embed. Has to already be prepared and tokenized.
        obsm_key : str, optional
            Key in `adata.obsm` to store the final embeddings. Defaults to "X_geneformer".
        batch_size : int, optional
            Forward batch size used by the Geneformer model for embedding extraction.
        cleanup : bool, optional
            Whether to remove the temporary directory after generating embeddings.
        **kwargs : dict
            Additional arguments (unused here, but kept for interface consistency).

        Returns
        -------
        anndata.AnnData
            The same AnnData object with `adata.obsm[obsm_key]` filled with
            the Geneformer embeddings.

        References
        ----------
        - The tokenized dataset must exist (either from `prepare(do_tokenization=True)`
          or manually created).
        - The final embeddings have dimensionality 512.

        Raises
        ------
        ImportError
            If geneformer package is not installed.
        ValueError
            If the tokenized dataset is missing and cannot be embedded.
        """
        try:
            from geneformer import EmbExtractor
        except ImportError:
            raise ImportError(
                "To use the Geneformer embedder, ensure it is installed via "
                "`git submodule update --init --recursive` and `pip install external/Geneformer`."
            )

        dataset_path = self.out_dataset_dir / f"{self.dataset_name}.dataset"
        if not dataset_path.exists():
            raise ValueError(
                f"No tokenized dataset found at {dataset_path}. "
                "Did you run `prepare(..., do_tokenization=True)` first?"
            )
        if not self.adata_path:
            raise ValueError("Run prepare first to set the adata_path.")

        # Check if csv with embeddings already exists (is simultaniously created for both splits of the dataset and therefore doesnt need to be recreated)
        embs_csv_path = self.out_dataset_dir / "geneformer_embeddings.csv"
        if not embs_csv_path.exists():
            # Create the extractor with updated batch size
            extractor_params = dict(self.emb_extractor_init)
            extractor_params["forward_batch_size"] = batch_size
            extractor = EmbExtractor(**extractor_params)

            logger.info(
                "Extracting geneformer embeddings from model at %s...", self.model_dir
            )
            embs_df = extractor.extract_embs(
                str(self.model_dir),
                str(dataset_path),
                str(self.out_dataset_dir),
                output_prefix=f"{self.dataset_name}_embeddings",
                cell_state=None,
            )
        else:
            logger.info(
                "Geneformer embeddings already exist at %s. Skipping extraction.",
                embs_csv_path,
            )
            embs_df = pd.read_csv(embs_csv_path)

        # Load the adata to attach the embeddings
        processed_adata_path = self.adata_path
        if not processed_adata_path.exists():
            raise ValueError(f"No processed AnnData found at {processed_adata_path}.")
        adata = sc.read(processed_adata_path, backed="r")
        og_ids = adata.obs["sample_index"].values
        # Filter and sort embs_df to align with og_ids
        # drop the "Unamed: 0" column
        embs_df = embs_df.drop(columns=["Unnamed: 0"], errors="ignore")
        embs_sorted = embs_df.set_index("sample_index").loc[og_ids].reset_index()
        embs_matrix = embs_sorted.drop(columns=["sample_index"]).values
        adata.obsm[obsm_key] = embs_matrix
        logger.info(
            "Stored Geneformer embeddings of shape %s in adata.obsm[%r].",
            embs_matrix.shape,
            obsm_key,
        )
        return adata

    def _kill_process(self):
        """
        Kill any child processes spawned by Geneformer.

        Sometimes the Geneformer library spawns additional processes that may not
        terminate gracefully. This method forcibly kills them.
        """
        parent_pid = os.getpid()
        for proc in psutil.process_iter(["pid", "ppid", "name"]):
            if proc.info["ppid"] == parent_pid:
                logger.warning(
                    "Killing process %s (PID %s)", proc.info["name"], proc.info["pid"]
                )
                proc.kill()


class SCVIEmbedder(BaseEmbedder):
    """
    Class to load a pretrained SCVI model and apply embeddings.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        **kwargs,
    ):
        """
        Initialize the SCVI embedder.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the output embeddings.
        model_cache_dir : str | None
            Directory to cache downloaded models. If None, uses '../models/scvi_cellxgene'.
        **kwargs
            Additional arguments passed to SCVI setup.
        """
        super().__init__(embedding_dim=embedding_dim)
        self.model = None
        self.init_kwargs = kwargs

    def prepare(self, adata_path: str, **kwargs):
        """
        Prepare the SCVI model for embedding.

        For example:
        - `hub_repo_id` to load from HF
        - `reference_s3_bucket` = "cellxgene-contrib-public" and `reference_s3_path` = "models/scvi/2024-02-12/homo_sapiens/modelhub" to load from S3
        - `reference_adata_url` = "https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/homo_sapiens/adata-spinal-cord-minified.h5ad" to load reference adata (jointly with s3)

        The user can pass these as part of `init_kwargs` or `kwargs`.

        Parameters
        ----------
        adata : anndata.AnnData
            The query AnnData if it is already available at prepare-time.
        **kwargs : dict
            Additional configuration for loading from S3 or HF.
        """
        logger.info("Preparing SCVI model, loading from S3 or HF if needed.")

        # Decide which approach to use based on the presence of certain kwargs.
        hub_repo_id = kwargs.get("hub_repo_id", self.init_kwargs.get("hub_repo_id"))
        reference_s3_bucket = kwargs.get(
            "reference_s3_bucket", self.init_kwargs.get("reference_s3_bucket")
        )
        reference_s3_path = kwargs.get(
            "reference_s3_path", self.init_kwargs.get("reference_s3_path")
        )
        reference_adata_url = kwargs.get(
            "reference_adata_url",
            self.init_kwargs.get("reference_adata_url"),
        )
        cache_dir = kwargs.get("cache_dir", self.init_kwargs.get("cache_dir"))
        file_cache_dir = kwargs.get(
            "file_cache_dir", self.init_kwargs.get("file_cache_dir")
        )

        # Load the model from either HF hub or S3
        if hub_repo_id is not None:
            logger.info("Loading SCVI model from HF hub: %s", hub_repo_id)
            model = HubModel.pull_from_huggingface_hub(hub_repo_id, revision="main")
        elif reference_s3_bucket is not None and reference_s3_path is not None:
            logger.info(
                "Loading SCVI model from S3 bucket %s, path %s",
                reference_s3_bucket,
                reference_s3_path,
            )
            import botocore

            model = HubModel.pull_from_s3(
                s3_bucket=reference_s3_bucket,
                s3_path=reference_s3_path,
                pull_anndata=False,
                config=botocore.config.Config(signature_version=botocore.UNSIGNED),
                cache_dir=Path(cache_dir),
            )
        else:
            raise ValueError("No valid SCVI loading parameters provided.")

        # Load reference AnnData if needed
        if reference_adata_url is not None:
            if file_cache_dir is None:
                save_dir = tempfile.TemporaryDirectory()
            else:
                if not file_cache_dir.endswith("/"):
                    file_cache_dir += "/"
                save_dir = Path(file_cache_dir)
            adata_path = os.path.join(save_dir, "cellxgene_reference_adata.h5ad")
            logger.info("Reading reference adata from URL %s", reference_adata_url)
            reference_adata = sc.read(adata_path, backup_url=reference_adata_url)
        else:
            # Try to get reference adata from model
            try:
                reference_adata = model.adata
            except AttributeError as e:
                raise ValueError(
                    "No reference AnnData available in model or via URL."
                ) from e

        # Set up the scVI model with reference data
        self.scvi_model = self._setup_model_with_ref(model, reference_adata)

    def _prepare_query_adata(self, query_adata: str | Path):
        """
        Private helper to prepare query data for scVI inference.

        Parameters
        ----------
        query_adata_path : str | Path
            Single-cell dataset to be used as 'query'.
        """
        logger.info("Preparing query AnnData and loading into SCVI model.")

        # Check if counts layer exists
        if "counts" not in query_adata.layers:
            raise ValueError(
                "No 'counts' layer found in adata. Run preprocessing first."
            )

        # X will be modified to match the genes in the reference data and the training size of the scvi Model. But we
        # need to keep the original object for later.
        self.adata_backup = query_adata.copy()
        query_adata.X = query_adata.layers["counts"].copy()

        # Set batch key as expected from training data
        query_adata.obs["batch"] = query_adata.obs[self.batch_key].astype("category")

        # Prepare scvi fields
        SCVI.prepare_query_anndata(query_adata, self.scvi_model)

        # fix non-numerics
        fix_non_numeric_nans(query_adata)

        # Load query data
        query_model = SCVI.load_query_data(query_adata, self.scvi_model)
        query_model.is_trained = True
        self.scvi_model = query_model

        logger.info("Successfully prepared query data for SCVI")

    @staticmethod
    def _setup_model_with_ref(model, reference_adata):
        """
        Set up the SCVI model with the reference AnnData.

        Parameters
        ----------
        model : HubModel
            The loaded HubModel instance.
        reference_adata : anndata.AnnData
            The reference dataset.

        Returns
        -------
        SCVI
            The SCVI model, loaded with reference AnnData.
        """
        logger.info("Preparing SCVI model with reference data...")
        # didn't quite understand why, but this property has to be deleted
        try:
            del reference_adata.uns["_scvi_adata_minify_type"]
        except KeyError:
            pass
        if "is_primary_data" in reference_adata.obs:
            model.load_model(
                adata=reference_adata[reference_adata.obs["is_primary_data"]].copy()
            )
        else:
            model.load_model(adata=reference_adata.copy())
        scvi_model = model.model
        return scvi_model

    def embed(
        self,
        adata_path: str | None = None,
        adata: anndata.AnnData | None = None,
        obsm_key: str = "X_scvi",
        batch_key="batch",
        **kwargs,
    ):
        """
        Transform the data into the SCVI latent space.

        If the SCVI model has not yet been set up with the query data, it does so here.

        Parameters
        ----------
        adata : anndata.AnnData
            The query dataset to be embedded.
        obsm_key : str
            The key in `adata.obsm` under which to store the SCVI embedding.
        batch_key : str
            The batch key in `adata.obs` to use for batch correction. Most likely the exact categories don't align completly witht the reference data which limits batch correction.
            But this is how scVI can be used zero shot.
        **kwargs : dict
            Additional keyword arguments (unused).

        Returns
        -------
        anndata.AnnData
            The same AnnData with latent representation stored in `adata.obsm[obsm_key]`.

        References
        ----------
        The reference data is loaded from S3 or the Hub (depending on configuration).
        Query data is the user-provided `adata`.
        """
        self.batch_key = batch_key
        if self.scvi_model is None:
            raise ValueError("SCVI model is not prepared. Call `prepare(...)` first.")
        # If the query hasn't been loaded yet, load it now:
        if adata is None and adata_path is not None:
            # if only a path is given, maybe load the entire data
            adata = sc.read(adata_path)
        if not hasattr(self.scvi_model, "adata") or self.scvi_model.adata is not adata:
            self._prepare_query_adata(adata)

        logger.info(
            "Computing SCVI latent representation, storing in `%s`...", obsm_key
        )
        latent_repr = self.scvi_model.get_latent_representation()
        # restore the original adata
        adata = self.adata_backup
        adata.obsm[obsm_key] = latent_repr
        return adata


class SCVIEmbedderFM(SCVIEmbedder):
    """
    SCVI embedder preconfigured as a foundation model (FM) loading weights from an scvi model trained on the cellxgene corpus.
    """

    def __init__(self, embedding_dim: int = 50, **init_kwargs):
        """
        Initialize the SCVI FM embedder with defaults for bucket/paths.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the embedding.
        init_kwargs : dict
            Additional keyword arguments including:
            - cache_dir: Override default cache directory for model files
            - file_cache_dir: Override default cache directory for reference data files

        References
        ----------
        - The large pretrained SCVI model is assumed to be hosted in the
          'cellxgene-contrib-public' S3 bucket at the specified path.
        - The reference AnnData is provided via the reference_adata_url parameter.
        """
        # Get the default cache directories using appdirs
        app_name = "adata_hf_datasets"
        default_cache_root = user_cache_dir(app_name)

        # Create default paths for different types of cache
        default_model_cache = os.path.join(
            default_cache_root, "models", "scvi_cellxgene"
        )
        default_data_cache = os.path.join(default_cache_root, "reference_data", "scvi")

        # Ensure the cache directories exist
        os.makedirs(default_model_cache, exist_ok=True)
        os.makedirs(default_data_cache, exist_ok=True)

        # Use provided cache dirs or defaults
        cache_dir = init_kwargs.pop("cache_dir", default_model_cache)
        file_cache_dir = init_kwargs.pop("file_cache_dir", default_data_cache)

        default_kwargs = {
            "reference_s3_bucket": "cellxgene-contrib-public",
            "reference_s3_path": "models/scvi/2024-02-12/homo_sapiens/modelhub",
            "reference_adata_url": "https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/homo_sapiens/adata-spinal-cord-minified.h5ad",
            "cache_dir": cache_dir,
            "file_cache_dir": file_cache_dir,
        }

        # Log cache locations
        logger.info(f"Using model cache directory: {cache_dir}")
        logger.info(f"Using reference data cache directory: {file_cache_dir}")

        # Merge user overrides with defaults
        default_kwargs.update(init_kwargs)

        # Call parent constructor
        super().__init__(embedding_dim=embedding_dim, **default_kwargs)


class InitialEmbedder:
    """
    A manager class for creating embeddings of single-cell data from file paths.

    Parameters
    ----------
    method : str
        The embedding method to use. For example: ["scvi_fm", "geneformer", ...].
    embedding_dim : int
        Dimensionality of the output embedding space.
    init_kwargs : dict, optional
        Additional keyword arguments to pass to the chosen embedder.
    """

    def __init__(
        self,
        method: str,
        embedding_dim: int = 64,
        **init_kwargs,
    ):
        """
        Initialize the manager and select the embedding method.

        Parameters
        ----------
        method : str
            The embedding method to use. For example: ["scvi_fm", "geneformer", ...].
        embedding_dim : int
            Dimensionality of the output embedding space.
        init_kwargs : dict, optional
            Additional keyword arguments to pass to the chosen embedder.
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.init_kwargs = init_kwargs or {}

        # You already have these classes defined (SCVIEmbedder, SCVIEmbedderFM, GeneformerEmbedder, etc.)
        embedder_classes = {
            "scvi_fm": SCVIEmbedderFM,  # for scVI
            "geneformer": GeneformerEmbedder,
            "pca": PCAEmbedder,
            "hvg": HighlyVariableGenesEmbedder,
        }

        if method not in embedder_classes:
            raise ValueError(f"Unknown embedding method: {method}")

        self.embedder = embedder_classes[method](
            embedding_dim=embedding_dim, **self.init_kwargs
        )

        # For convenience, you might store a flag if the method requires AnnData in memory:
        # (Here we check if it's an SCVIEmbedder or you can define a property in each embedder.)
        self.requires_mem_adata = isinstance(
            self.embedder, (SCVIEmbedder, HighlyVariableGenesEmbedder, PCAEmbedder)
        )
        logger.info(
            "Initialized InitialEmbedder with method=%s, embedding_dim=%d. requires_mem_adata=%s",
            self.method,
            self.embedding_dim,
            self.requires_mem_adata,
        )

    def prepare(self, adata_path: str, **prepare_kwargs):
        """
        Prepare the embedder. For methods like SCVI, this might load from S3/Hub;
        for Geneformer, it might tokenize, etc.

        This now accepts a file path instead of an in-memory AnnData.

        Parameters
        ----------
        adata_path : str
            Path to the single-cell data file (e.g., .h5ad) to prepare.
        **prepare_kwargs : dict
            Additional keyword arguments passed to the embedder's `prepare()` method.

        Notes
        -----
        The data source is a user-provided .h5ad file.
        """
        logger.info(
            "Preparing method '%s' with embedding_dim=%d",
            self.method,
            self.embedding_dim,
        )
        # Most embedders do not strictly require the entire data in memory just for "prepare".
        # So we can simply pass the file path.
        self.embedder.prepare(adata_path=adata_path, **prepare_kwargs)

    def embed(
        self,
        adata_path: str,
        obsm_key: str | None = None,
        output_path: str | None = None,
        chunk_size: int = 10000,
        **embed_kwargs,
    ):
        """
        Transform the data into the learned embedding space. If the embedder
        requires a full AnnData in memory (e.g., SCVI), we perform chunk-based
        processing and write out a concatenated result on disk.

        Parameters
        ----------
        adata_path : str
            Path to the single-cell data file (e.g., .h5ad) to embed.
        obsm_key : str, optional
            Key in `adata.obsm` to store the embeddings (default: "X_{self.method}").
        output_path : str, optional
            Where to write the embedded AnnData. If None, overwrites `adata_path`.
        chunk_size : int, optional
            Number of observations to load in each chunk for memory-limited processing.
        **embed_kwargs : dict
            Additional kwargs for the underlying embedder's `embed` method.

        Returns
        -------
        anndata.AnnData
            AnnData object on which embedding was performed (with an updated `obsm`).
            - If chunk-based mode, returns the final *in-memory* concatenation.
            - The same result is also written to `output_path` on disk.

        Notes
        -----
        - Data is user-provided as `.h5ad` and loaded in chunks if needed.
        - If the embedder does not require full AnnData, we simply call
          `self.embedder.embed(adata_path, ...)` directly.
        - If the embedder does need a full AnnData in memory (like scVI),
          we load in chunks from disk using `AnnLoader`, call the embedder
          on each chunk in memory, and concatenate the chunks on disk.
        """
        if output_path is None:
            # make a new outputpath based on input path +_"method name"_emb
            output_path = adata_path.replace(".h5ad", f"_{self.method}_emb.h5ad")
        if obsm_key is None:
            obsm_key = f"X_{self.method}"

        # if the output file already exists, remove it and issue a warning
        if os.path.exists(output_path):
            logger.warning(
                "Output file %s already exists and will be overwritten.", output_path
            )
            os.remove(output_path)

        logger.info(
            "Embedding data using method '%s'. Output to '%s'. obsm_key=%s",
            self.method,
            output_path,
            obsm_key,
        )

        # If the method doesn't require in memory AnnData, we can pass it directly.
        if not self.requires_mem_adata:
            # The embedder itself is prepared to accept file paths directly:
            adata_emb = self.embedder.embed(
                adata_path=adata_path, obsm_key=obsm_key, **embed_kwargs
            )
            adata_emb.write_h5ad(output_path)
            return adata_emb

        # Otherwise, we do chunk-based processing to avoid loading the whole adata object into memory:
        logger.info("Using chunk-based approach for method '%s'.", self.method)
        adata = sc.read(adata_path, backed="r")
        loader = AnnLoader(adatas=adata, batch_size=chunk_size)
        chunk_list = []

        # Use a context manager to ensure the temporary directory is properly cleaned up
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, chunk in enumerate(loader):
                logger.info(
                    "Processing chunk %d with shape obs x var = %s x %s",
                    i,
                    chunk.n_obs,
                    chunk.n_vars,
                )

                # Move chunk fully into memory
                chunk_in_memory = chunk.to_adata()

                # Let the embedder produce embeddings
                chunk_adata = self.embedder.embed(
                    adata_path=None,  # Not used because we're passing chunk_in_memory directly
                    adata=chunk_in_memory,
                    obsm_key=obsm_key,
                    **embed_kwargs,
                )

                # Create a path for the chunk file within the temporary directory
                chunk_path = Path(tmp_dir) / f"chunk_{i}.h5ad"
                chunk_adata.write_h5ad(chunk_path)
                chunk_list.append(chunk_path)
                del chunk_adata

            # Concatenate all embedded chunks in memory
            anndata.experimental.concat_on_disk(
                in_files=chunk_list, out_file=output_path
            )
        # Write the final result to disk
        logger.info("Wrote final embedded AnnData to %s", output_path)
        # return the backed adata object
        return sc.read(output_path, backed="r")


'''
class InitialEmbedder:
    """
    Main interface for creating embeddings of single-cell data.
    """

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
            The embedding method to use. One of ["hvg", "pca", "scvi", "geneformer", ...].
        embedding_dim : int
            Dimensionality of the output embedding space.
        init_kwargs : dict, optional
            Additional keyword arguments to pass to the chosen embedder.
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.init_kwargs = init_kwargs or {}

        # Dispatch to the correct embedder class
        embedder_classes = {
            # "hvg": HighlyVariableGenesEmbedder,
            # "pca": PCAEmbedder,
            "scvi_fm": SCVIEmbedderFM,
            "geneformer": GeneformerEmbedder,
        }

        if method not in embedder_classes:
            raise ValueError(f"Unknown embedding method: {method}")

        self.embedder = embedder_classes[method](
            embedding_dim=embedding_dim, **self.init_kwargs
        )

    def prepare(self, adata_path=None, **prepare_kwargs):
        """
        Prepare the embedder. For methods like PCA, this trains the model;
        for SCVI, this loads from S3 or the Hub, etc.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The dataset to use for either training (PCA) or preparing query data (SCVI).
        **prepare_kwargs : dict
            Additional keyword arguments passed to the embedder's `prepare()` method.
        """
        logger.info(
            "Preparing method '%s' with embedding_dim=%d",
            self.method,
            self.embedding_dim,
        )
        self.embedder.prepare(adata=adata_path, **prepare_kwargs)

    def embed(self, adata_path, obsm_key=None, **embed_kwargs):
        """
        Transform the data into the learned embedding space.

        Parameters
        ----------
        adata_path : str
            Path to the AnnData file to embed.
        obsm_key : str, optional
            Key in `adata.obsm` to store the embeddings (default: "X_{method}").
        **embed_kwargs : dict
            Additional kwargs for the underlying embedder's `embed` method.

        Returns
        -------
        anndata.AnnData
            The same AnnData with the new embeddings in `adata.obsm[obsm_key]`.
        """
        if obsm_key is None:
            obsm_key = f"X_{self.method}"

        logger.info(
            "Embedding data using method '%s'. Storing in '%s'.", self.method, obsm_key
        )
        return self.embedder.embed(adata_path = adata_path, obsm_key=obsm_key, **embed_kwargs)
'''
