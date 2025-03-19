import anndata
import logging
from adata_hf_datasets.utils import fix_non_numeric_nans
from scvi.hub import HubModel
from scvi.model import SCVI
from pathlib import Path
import os
from datetime import datetime
import shutil
import psutil
import scanpy as sc

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
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_dir = self.project_dir / f"tmp_geneformer_{time_tag}"
        self.tmp_adata_dir = self.tmp_dir / "adata"
        self.tmp_adata_dir.mkdir(parents=True, exist_ok=True)

        # Name of the tokenized dataset
        self.dataset_name = "geneformer"
        # Where the output tokenized dataset is stored
        self.out_dataset_dir = self.tmp_dir

        logger.info(
            "Initialized GeneformerEmbedder with model_input_size=%d, num_layers=%d, special_model=%s",
            self.model_input_size,
            self.num_layers,
            self.special_model,
        )

    def prepare(
        self, adata: anndata.AnnData, do_tokenization: bool = True, **kwargs
    ) -> None:
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
            from adata_hf_datasets.pp import add_ensembl_ids
            from geneformer import TranscriptomeTokenizer
        except ImportError:
            raise ImportError(
                "To use the Geneformer embedder, ensure `git lfs` is installed and "
                "run 'git submodule update --init --recursive'. Then install geneformer: "
                "`pip install external/Geneformer`."
            )

        # 1. Make sure the data has the required fields
        if "ensembl_id" not in adata.var.columns:
            add_ensembl_ids(adata)
            logger.info("Added Ensembl IDs to adata.var['ensembl_id'].")
        if "n_counts" not in adata.obs.columns:
            sc.pp.calculate_qc_metrics(adata, inplace=True)
            adata.obs["n_counts"] = adata.obs.total_counts
            logger.info("Added 'n_counts' to adata.obs.")

        # 2. Attach a stable sample index
        adata.obs["sample_index"] = range(adata.shape[0])

        # 3. Write to a temporary h5ad
        h5ad_path = self.tmp_adata_dir / "adata.h5ad"
        adata.write_h5ad(h5ad_path)
        logger.info("Wrote AnnData to temporary file: %s", h5ad_path)

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
                str(self.tmp_adata_dir),
                str(self.tmp_dir),
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
        adata: anndata.AnnData,
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
        adata : anndata.AnnData
            Single-cell dataset. The same object used during `prepare(...)`.
            We store the final embeddings in this object's `.obsm`.
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
            str(self.tmp_dir),
            output_prefix=f"{self.dataset_name}_embeddings",
            cell_state=None,
        )

        # Sort by sample_index to align with the original order
        embs_sorted = embs_df.loc[embs_df["sample_index"].sort_values().index]
        embs_matrix = embs_sorted.drop(columns=["sample_index"]).values

        # Reload the same data from the temporary .h5ad so we can retrieve obs, var, etc.
        processed_adata_path = self.tmp_adata_dir / "adata.h5ad"
        if not processed_adata_path.exists():
            raise ValueError(f"No processed AnnData found at {processed_adata_path}.")

        adata = anndata.read_h5ad(processed_adata_path)
        # logger.info("Loaded processed AnnData from %s.", processed_adata_path)

        # Attach embeddings (in the correct order) back to the *original* adata
        # here we rely on the fact that `processed_adata` and `adata` have the same shape
        # and that sample_index was introduced in `.prepare()`.
        # If you want to strictly match them by 'sample_index', you can reindex `adata`
        # or reorder the matrix accordingly. For simplicity, we assume consistent ordering.
        adata.obsm[obsm_key] = embs_matrix
        logger.info(
            "Stored Geneformer embeddings of shape %s in adata.obsm[%r].",
            embs_matrix.shape,
            obsm_key,
        )

        # Optional cleanup
        if cleanup:
            logger.info("Cleaning up temporary directory: %s", self.tmp_dir)
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
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

    def __init__(self, embedding_dim: int = 64, **init_kwargs):
        """
        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the embedding (not strictly used by scVI, but kept for consistency).
        init_kwargs : dict
            Additional parameters that may include S3 bucket, HF repo, etc.
        """
        super().__init__(embedding_dim=embedding_dim, **init_kwargs)
        self.scvi_model = None

    def prepare(self, **kwargs):
        """
        Load (or set up) the SCVI model from the provided parameters.

        For example:
        - `hub_repo_id` to load from HF
        - `reference_s3_bucket` = "cellxgene-contrib-public" and `reference_s3_path` = "models/scvi/2024-02-12/homo_sapiens/modelhub to load from S3
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
            "reference_adata_url", self.init_kwargs.get("reference_adata_url")
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
                cache_dir=cache_dir,
            )
        else:
            raise ValueError("No valid SCVI loading parameters provided.")

        # Load reference AnnData if needed
        if reference_adata_url is not None:
            import tempfile
            import os
            import scanpy as sc

            if file_cache_dir is None:
                save_dir = tempfile.TemporaryDirectory()
            else:
                if not file_cache_dir.endswith("/"):
                    file_cache_dir += "/"
                save_dir = Path(file_cache_dir + "adata.h5ad")
            adata_path = os.path.join(save_dir.name, "cellxgene_reference_adata.h5ad")
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

    def _prepare_query_adata(self, query_adata: anndata.AnnData):
        """
        Private helper to prepare query data for scVI inference.

        Parameters
        ----------
        query_adata : anndata.AnnData
            Single-cell dataset to be used as 'query'.
        """
        logger.info("Preparing query AnnData and loading into SCVI model.")

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
        adata: anndata.AnnData,
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
        if not hasattr(self.scvi_model, "adata") or self.scvi_model.adata is not adata:
            self._prepare_query_adata(adata)

        logger.info(
            "Computing SCVI latent representation, storing in `%s`...", obsm_key
        )
        latent_repr = self.scvi_model.get_latent_representation()
        adata.obsm[obsm_key] = latent_repr
        return adata


class SCVIEmbedderFM(SCVIEmbedder):
    """
    SCVI embedder preconfigured as a foundation model (FM) loading weights from an scvi model trained on the cellxgene corpus.
    """

    def __init__(self, embedding_dim: int = 64, **init_kwargs):
        """
        Initialize the SCVI FM embedder with defaults for bucket/paths.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the embedding.
        init_kwargs : dict
            Additional keyword arguments.

        References
        ----------
        - The large pretrained SCVI model is assumed to be hosted in the
          'cellxgene-contrib-public' S3 bucket at the specified path.
        - The reference AnnData is provided via the reference_adata_url parameter.
        """
        default_kwargs = {
            "reference_s3_bucket": "cellxgene-contrib-public",
            "reference_s3_path": "models/scvi/2024-02-12/homo_sapiens/modelhub",
            "reference_adata_url": "https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/homo_sapiens/adata-spinal-cord-minified.h5ad",
            "cache_dir": "../models/scvi_cellxgene",
            "file_cache_dir": "data/RNA/scvi_reference_data/cellxgene_2024-02-12_homo_sapiens/",
        }
        # Merge user overrides
        default_kwargs.update(init_kwargs)

        # Call parent constructor
        super().__init__(embedding_dim=embedding_dim, **default_kwargs)


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

    def prepare(self, adata=None, **prepare_kwargs):
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
        self.embedder.prepare(adata=adata, **prepare_kwargs)

    def embed(self, adata, obsm_key=None, **embed_kwargs):
        """
        Transform the data into the learned embedding space.

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to be transformed.
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
        return self.embedder.embed(adata, obsm_key=obsm_key, **embed_kwargs)
