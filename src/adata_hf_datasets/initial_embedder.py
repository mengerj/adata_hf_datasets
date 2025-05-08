import anndata
import logging
from adata_hf_datasets.utils import (
    fix_non_numeric_nans,
)
from adata_hf_datasets.pp.utils import (
    ensure_log_norm,
    is_data_scaled,
    check_enough_genes_per_batch,
    consolidate_low_frequency_categories,
)
from adata_hf_datasets.pp.pybiomart_utils import add_ensembl_ids, ensure_ensembl_index
from scvi.hub import HubModel
import shutil
import tempfile
import uuid
import psutil
from scvi.model import SCVI
from pathlib import Path
import os
import scanpy as sc
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

    def prepare(self, adata: anndata.AnnData, **kwargs) -> None:
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
        # check if the data is already normalized
        if "highly_variable" not in adata.var:
            ensure_log_norm(adata, var_threshold=1)
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
        redo_hvg = True
        # Check if the highly variable genes have already been computed and if there are enough
        if "highly_variable" not in adata.var:
            n_hvg = np.sum(adata.var["highly_variable"])
            if n_hvg >= self.embedding_dim:
                logger.info(
                    "Found %d highly variable genes. No need to recompute.",
                    n_hvg,
                )
                redo_hvg = False
        # only compute if not already included (from pp)
        if redo_hvg:
            # Convert to dense for checking
            if sp.issparse(adata.X):
                X_arr = adata.X.toarray()
            else:
                X_arr = adata.X
            # Find genes (columns) with any infinite values
            finite_mask = np.isfinite(X_arr).all(axis=0)
            n_bad = np.count_nonzero(~finite_mask)
            if n_bad > 0:
                logger.warning(
                    "Found %d genes with infinite values. Removing those genes.", n_bad
                )
                # Remove genes with infinite values
                adata = adata[:, finite_mask]
            # Check if enough valid genes are present in each batch
            if batch_key is not None:
                consolidate_low_frequency_categories(
                    adata, columns=[batch_key], threshold=3, remove=False
                )
                check_enough_genes_per_batch(adata, batch_key, self.embedding_dim)
            # remove cells with infinity values
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.embedding_dim, batch_key=batch_key
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
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
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

    def prepare(self, adata: anndata.AnnData, n_cells=10000, **kwargs) -> None:
        """Fit a PCA model to the AnnData object's .X matrix."""
        logger.info(
            "Fitting PCA with %d components on %d.", self.embedding_dim, n_cells
        )
        from sklearn.decomposition import PCA

        adata_sub = adata.copy()
        # get a random subset of cells with random
        logger.info("Subsampling %d cells for PCA.", n_cells)
        if n_cells < adata_sub.shape[0]:
            adata_sub = adata_sub[
                np.random.choice(adata_sub.shape[0], n_cells, replace=False), :
            ]

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

    def prepare(
        self,
        adata: anndata.AnnData,
        adata_path=str,
        do_tokenization: bool = True,
        **kwargs,
    ) -> None:
        """
        Prepare (preprocess + tokenize) the data for Geneformer embeddings.

        This includes:
         - Adding Ensembl IDs to `adata.var["ensembl_id"]` if not present.
         - Calculating and storing `adata.obs["n_counts"]` if not present.
         - Writing the AnnData to a temporary H5AD file.
         - Optionally tokenizing the data using `TranscriptomeTokenizer` if it
           has not been tokenized previously (i.e., if the .dataset file doesn't exist).

        Parameters
        ----------
        adata : anndata.AnnData
            Single-cell dataset to prepare.
        adata_path : str
            Path to the AnnData file, to save the tokenized dataset. If not provided, save in "geneformer" folder in current directory.
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
        if adata_path is None:
            adata_path = str(
                Path(__file__).resolve().parents[2] / "geneformer" / "adata.h5ad"
            )
        self.in_adata_path = Path(adata_path)
        # adata_name = self.in_adata_path.stem
        # save the tokenized dataset in the same directory as the adata
        self.adata_dir = self.in_adata_path.parent
        # self.adata_path = self.adata_dir / adata_name
        # adata.write_h5ad(self.adata_path)
        self.out_dataset_dir = self.adata_dir / "geneformer"
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
        adata: anndata.AnnData,
        obsm_key: str = "X_geneformer",
        batch_size: int = 16,
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
        adata : anndata.Anndata
            The AnnData object to embed.
        obsm_key : str, optional
            Key in `adata.obsm` to store the final embeddings. Defaults to "X_geneformer".
        batch_size : int, optional
            Forward batch size used by the Geneformer model for embedding extraction.
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

        # Attach the embeddings to the adata object
        og_ids = adata.obs["sample_index"].values
        # Filter and sort embs_df to align with og_ids
        # drop the "Unamed: 0" column
        embs_sorted = self._deduplicate_and_reindex_embeddings(embs_df, og_ids)
        embs_matrix = embs_sorted.values
        adata.obsm[obsm_key] = embs_matrix
        logger.info(
            "Stored Geneformer embeddings of shape %s in adata.obsm[%r].",
            embs_matrix.shape,
            obsm_key,
        )
        return adata

    def _deduplicate_and_reindex_embeddings(self, embs_df, og_ids):
        """
        Ensure one row per 'sample_index' in embs_df, in the exact order of og_ids.

        If 'sample_index' occurs multiple times in embs_df, we replace
        that embedding with a row of zeros. If 'sample_index' is missing,
        we also fill with zeros.

        Parameters
        ----------
        embs_df : pd.DataFrame
            DataFrame containing columns ['sample_index', ...embedding cols...].
            May have duplicate sample_index values.
        og_ids : array-like
            The array of sample_index values from AnnData (og_ids).

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by 'sample_index', with each index unique and in the
            exact order of og_ids.
        """

        # Drop the "Unnamed: 0" column if it exists
        embs_df = embs_df.drop(columns=["Unnamed: 0"], errors="ignore")

        # Group by sample_index. For any group of size > 1, we'll produce a single row of zeros.
        def handle_duplicates(group):
            if group.shape[0] == 1:
                # Exactly one row for this sample_index, keep it as is
                return group
            else:
                # Duplicate index => create a single row of zeros (except for the sample_index).
                row = group.iloc[[0]].copy()
                for c in row.columns:
                    if c != "sample_index":
                        row[c] = 0
                return row

        deduped_df = embs_df.groupby(
            "sample_index", group_keys=False, as_index=False
        ).apply(handle_duplicates)
        # Now deduped_df has at most one row per sample_index.

        # Set index to sample_index (verify_integrity=False is safe now since we just deduplicated).
        deduped_df = deduped_df.set_index("sample_index", verify_integrity=False)

        # Reindex to the exact order of og_ids.
        # Missing IDs become NaN, we fill with zeros.
        deduped_df = deduped_df.reindex(og_ids).fillna(0)

        return deduped_df

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

    def prepare(self, **kwargs):
        """
        Prepare the SCVI model for embedding.

        For example:
        - `hub_repo_id` to load from HF
        - `reference_s3_bucket` = "cellxgene-contrib-public" and `reference_s3_path` = "models/scvi/2024-02-12/homo_sapiens/modelhub" to load from S3
        - `reference_adata_url` = "https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/homo_sapiens/adata-spinal-cord-minified.h5ad" to load reference adata (jointly with s3)

        The user can pass these as part of `init_kwargs` or `kwargs`.

        Parameters
        ----------
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
            ref_adata_path = os.path.join(save_dir, "cellxgene_reference_adata.h5ad")
            logger.info("Reading reference adata from URL %s", reference_adata_url)
            reference_adata = sc.read(ref_adata_path, backup_url=reference_adata_url)
        else:
            # Try to get reference adata from model
            try:
                reference_adata = model.adata
            except AttributeError as e:
                raise ValueError(
                    "No reference AnnData available in model or via URL."
                ) from e

        # Set up the scVI model with reference data
        # loaded minified reference adata with is expected to give a warning about empty cells
        # but this is expected and can be ignored
        logging.info("Reference AnnData is expected to be minified with empty cells.")
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

        ensure_ensembl_index(
            query_adata,
            ensembl_col=self.init_kwargs.get("ensembl_col", "ensembl_id"),
            add_fn=add_ensembl_ids,
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
    def _is_shared_path(path: Path) -> bool:
        """
        Heuristic – consider paths on NFS/Lustre or in $CACHE_DIR 'shared'.
        Adapt if you have a better way to decide.
        """
        return not path.is_symlink() and "/tmp/" not in path.as_posix()

    @staticmethod
    def _localize_hubmodel(model: HubModel) -> HubModel:
        """
        Copy the weight file (model.pt) into a unique temp dir and return a *new*
        HubModel instance that points there.  This eliminates cross-process races.
        """
        orig_dir = Path(model.local_dir)
        tmp_dir = Path(tempfile.gettempdir()) / f"scvi_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # copy the weight file; other files are usually tiny, copy if you need them
        shutil.copy2(orig_dir / "model.pt", tmp_dir / "model.pt")
        if (orig_dir / "adata.h5ad").is_file():
            shutil.copy2(orig_dir / "adata.h5ad", tmp_dir / "adata.h5ad")
        # reuse the existing metadata / model-card objects
        return HubModel(
            local_dir=str(tmp_dir),
            metadata=model.metadata,
            model_card=model.model_card,
        )

    def _setup_model_with_ref(self, model, reference_adata):
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
        if self._is_shared_path(Path(model.local_dir)):
            logger.info(
                "Copying SCVI weight file to a worker-unique temp dir to avoid NFS races."
            )
            model = self._localize_hubmodel(model)

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
    Manager for creating embeddings of single-cell data.

    Some embedding methods require loading the entire AnnData into memory,
    while others can operate directly on file paths.

    Parameters
    ----------
    method : str
        Embedding method. One of:
        - "scvi_fm"
        - "geneformer"
        - "pca"
        - "hvg"
    embedding_dim : int, default=64
        Dimensionality of the output embedding.
    **init_kwargs
        Additional keyword arguments passed to the chosen embedder.
    """

    def __init__(
        self,
        method: str,
        embedding_dim: int = 64,
        **init_kwargs,
    ):
        # Map supported methods to their embedder classes
        embedder_classes = {
            "scvi_fm": SCVIEmbedderFM,
            "geneformer": GeneformerEmbedder,
            "pca": PCAEmbedder,
            "hvg": HighlyVariableGenesEmbedder,
        }
        if method not in embedder_classes:
            raise ValueError(f"Unknown embedding method: {method}")

        self.method = method
        self.embedding_dim = embedding_dim
        self.init_kwargs = init_kwargs or {}
        self.embedder = embedder_classes[method](
            embedding_dim=embedding_dim, **self.init_kwargs
        )

        # Determine if full in-memory AnnData is required
        # Embedders should expose a `requires_mem_adata` attribute; default True
        self.requires_mem_adata = getattr(self.embedder, "requires_mem_adata", True)

        logger.info(
            "Initialized InitialEmbedder(method=%s, embedding_dim=%d, requires_mem_adata=%s)",
            self.method,
            self.embedding_dim,
            self.requires_mem_adata,
        )

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **prepare_kwargs,
    ) -> None:
        """
        Prepare the embedder from a file path.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The AnnData object to prepare the embedder. Has to be preprocessed already, if embedding method requires it.
        adata_path : str
            Path to the AnnData file (.h5ad). Only needed for geneformer.
        **prepare_kwargs
            Keyword arguments passed to the embedder's prepare().
        """
        logger.info("Preparing embedder '%s' with file %s", self.method, adata_path)
        self.embedder.prepare(adata=adata, adata_path=adata_path, **prepare_kwargs)

    def embed(
        self,
        adata: anndata.AnnData,
        obsm_key: str | None = None,
        batch_key: str = "batch",
        **embed_kwargs,
    ):
        """
        Embed data and write the result to disk.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object to embed.
        obsm_key : str, optional
            Key under which embeddings are stored in .obsm.
            Defaults to "X_{method}".
        batch_key : str, default="batch"
            Observation column for batch labels (used by some embedders).
        **embed_kwargs
            Additional keyword arguments for the embedders embed().

        Returns
        -------
        AnnData
            The AnnData object with embeddings in .obsm[obsm_key].
        """
        # Derive defaults
        if obsm_key is None:
            obsm_key = f"X_{self.method}"

        logger.info(f"Embedding method: {self.method}")

        adata = self.embedder.embed(
            adata=adata,
            obsm_key=obsm_key,
            batch_key=batch_key,
            **embed_kwargs,
        )
        logger.info("Embedding complete. Returning the embedding matrix")
        return adata
