import anndata
import logging
import pandas as pd
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
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import numpy as np
from appdirs import user_cache_dir
import errno
import time
import random

logger = logging.getLogger(__name__)


def _redirect_tmp_cache_dir(
    cache_dir: str, cluster_tmp_dir: str = "/scratch/global/menger/tmp"
) -> str:
    """
    Redirect cache directories from /tmp/ to a cluster-appropriate location.

    Parameters
    ----------
    cache_dir : str
        The original cache directory path.
    cluster_tmp_dir : str, optional
        The cluster tmp directory to use instead of /tmp/.
        Defaults to "/scratch/global/menger/tmp".

    Returns
    -------
    str
        The potentially redirected cache directory path.
    """
    cache_path = Path(cache_dir)

    # Check if the path starts with /tmp/
    if str(cache_path).startswith("/tmp/"):
        # Get the relative part after /tmp/
        relative_path = cache_path.relative_to("/tmp")
        # Create new path using cluster tmp directory
        new_cache_dir = os.path.join(cluster_tmp_dir, str(relative_path))
        logger.info(f"Redirecting cache from {cache_dir} to {new_cache_dir}")

        # Ensure the new directory exists
        os.makedirs(new_cache_dir, exist_ok=True)

        return new_cache_dir

    return cache_dir


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

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **kwargs,
    ):
        """
        Prepare the embedder for embedding. Subclasses decide whether
        to train from scratch, load from hub, or load from S3, etc.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Single-cell dataset in memory.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        **kwargs : dict
            Additional keyword arguments used for preparing.

        Raises
        ------
        ValueError
            If neither adata nor adata_path is provided.
        """

        raise NotImplementedError("Subclasses must implement 'prepare'")

    def embed(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str = "X_embedding",
        **kwargs,
    ) -> np.ndarray:
        """
        Transform the data into the learned embedding space.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Single-cell dataset in memory.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        obsm_key : str
            The key in `adata.obsm` under which to store the embedding.
        **kwargs : dict
            Additional keyword arguments for embedding.

        Returns
        -------
        np.ndarray
            The embedding matrix of shape (n_cells, embedding_dim).

        Raises
        ------
        ValueError
            If neither adata nor adata_path is provided.
        """
        raise NotImplementedError("Subclasses must implement 'embed'")


def _check_load_adata(
    adata: anndata.AnnData | None = None, adata_path: str | None = None
):
    if adata is None and adata_path is None:
        raise ValueError("Either adata or adata_path must be provided")
    if adata is not None and adata_path is not None:
        raise ValueError("Only one of adata or adata_path must be provided")
    if adata_path is not None:
        path = Path(adata_path)
        if path.suffix == ".zarr":
            adata = ad.read_zarr(path)
        else:
            adata = ad.read_h5ad(path)
    return adata


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

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Identifies the top `embedding_dim` highly variable genes in `adata`.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The single-cell data to analyze.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        kwargs : dict
            Additional keyword arguments for `scanpy.pp.highly_variable_genes`.
        """

        logger.info("No preperation done for HVG embedder. Done in embed method.")

    def embed(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str = "X_hvg",
        batch_key: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Stores the expression of the selected highly variable genes as an embedding.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The single-cell data containing highly variable genes.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        obsm_key : str, optional
            The key under which the embedding will be stored in `adata.obsm`.
        batch_key : str, optional
            The batch key in `adata.obs` to use for batch correction.
        **kwargs : dict
            Additional keyword arguments. Not used.

        Returns
        -------
        np.ndarray
            The embedding matrix of shape (n_cells, n_hvg_genes).
        """
        adata = _check_load_adata(adata, adata_path)
        logger.info("Selecting top %d highly variable genes.", self.embedding_dim)
        redo_hvg = True
        # Check if the highly variable genes have already been computed and if there are enough
        if "highly_variable" in adata.var:
            n_hvg = np.sum(adata.var["highly_variable"])
            if n_hvg >= self.embedding_dim:
                logger.info(
                    "Found %d highly variable genes. No need to recompute.",
                    n_hvg,
                )
                redo_hvg = False
        # only compute if not already included (from pp)
        if redo_hvg:
            logger.info("Normalizing and log-transforming data before HVG selection.")
            ensure_log_norm(adata, var_threshold=1)
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
        embedding_matrix = X[:, hvg_mask]
        # Store in adata for compatibility
        adata.obsm[obsm_key] = sp.csr_matrix(embedding_matrix)
        logger.info(
            f"Stored highly variable gene expression in adata.obsm[{obsm_key}], with shape {embedding_matrix.shape}"
        )
        return embedding_matrix

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


class PCAEmbedder_old(BaseEmbedder):
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

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        n_cells=10000,
        **kwargs,
    ) -> None:
        """Fit a PCA model to the AnnData object's .X matrix."""
        adata = _check_load_adata(adata, adata_path)
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

    def embed(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str = "X_pca",
        **kwargs,
    ) -> np.ndarray:
        """Transform the data via PCA and return the embedding matrix."""
        adata = _check_load_adata(adata, adata_path)
        if self._pca_model is None:
            raise RuntimeError("PCA model is not fit yet. Call `prepare(adata)` first.")
        X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
        embedding_matrix = self._pca_model.transform(X)
        # Store in adata for compatibility
        adata.obsm[obsm_key] = embedding_matrix
        return embedding_matrix


class PCAEmbedder(BaseEmbedder):
    """
    Pre-trained PCA embedder using saved cross-dataset PCA model.

    This embedder loads a pre-trained PCA model and gene list from the resources
    directory and applies it to new datasets, ensuring consistent dimensionality
    reduction across all analyses.
    """

    def __init__(self, embedding_dim: int = 50, **kwargs):
        """
        Initialize the pre-trained PCA embedder.

        Parameters
        ----------
        embedding_dim : int, optional
            Number of principal components to retain. Defaults to 50.
            This should match the number of components in the saved model.
        kwargs : dict
            Additional keyword arguments including:
            - model_path: Path to saved PCA model (default: resources/cellxgene_geo_pca_3936_to_50.pkl)
            - gene_list_path: Path to gene list file (default: resources/gene_selection_ENSG_8k.txt)
        """
        super().__init__(embedding_dim=embedding_dim)
        self.embedding_dim = embedding_dim

        # Set default paths
        self.model_path = kwargs.get(
            "model_path", "resources/cellxgene_geo_pca_3936_to_50.pkl"
        )
        self.gene_list_path = kwargs.get(
            "gene_list_path", "resources/gene_selection_ENSG_8k.txt"
        )

        # Initialize model components
        self.pca_model = None
        self.scaler = None
        self.gene_order = None
        self.metadata = None

        logger.info(f"Initialized PCAEmbedder with model_path: {self.model_path}")
        logger.info(
            f"Initialized PCAEmbedder with gene_list_path: {self.gene_list_path}"
        )

    def _load_model(self) -> None:
        """Load the saved PCA model and gene list."""
        import pickle

        # Load PCA model
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"PCA model file not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        # Extract components
        self.pca_model = model_data["pca_model"]
        self.scaler = model_data["scaler"]
        self.gene_order = model_data["gene_order"]
        self.metadata = model_data["metadata"]

        logger.info(
            f"Loaded PCA model: {self.metadata['n_components']} components, {len(self.gene_order)} genes"
        )

        # Validate embedding dimension matches model
        if self.embedding_dim != self.pca_model.n_components_:
            logger.warning(
                f"Embedding dimension ({self.embedding_dim}) doesn't match model components ({self.pca_model.n_components_}). Using model components."
            )
            self.embedding_dim = self.pca_model.n_components_

    def _load_gene_list(self) -> None:
        """Load gene list from file if not already loaded from model."""
        if self.gene_order is None:
            gene_list_path = Path(self.gene_list_path)
            if not gene_list_path.exists():
                raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

            with open(gene_list_path, "r") as f:
                self.gene_order = [line.strip() for line in f if line.strip()]

            logger.info(f"Loaded {len(self.gene_order)} genes from {gene_list_path}")

    def _subset_and_order_genes(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Subset dataset to required genes in the correct order.
        Missing genes are filled with zeros to ensure consistent dimensionality.

        Parameters
        ----------
        adata : ad.AnnData
            Input dataset

        Returns
        -------
        ad.AnnData
            Subsetted dataset with genes in correct order
        """
        # Ensure ensembl IDs are available
        ensure_ensembl_index(adata, ensembl_col="ensembl_id")

        # Create a new AnnData object with the exact gene list
        n_cells = adata.n_obs

        # Initialize data matrix with zeros for all genes
        if sp.issparse(adata.X):
            X_new = np.zeros((n_cells, len(self.gene_order)), dtype=adata.X.dtype)
        else:
            X_new = np.zeros((n_cells, len(self.gene_order)), dtype=adata.X.dtype)

        # Create new var DataFrame with the exact gene list
        var_new = pd.DataFrame(index=self.gene_order)

        # Copy available genes from original dataset
        available_genes = [g for g in self.gene_order if g in adata.var_names]
        missing_genes = [g for g in self.gene_order if g not in adata.var_names]

        if len(available_genes) == 0:
            raise ValueError("Dataset has no genes from the required gene set")

        # Copy data for available genes
        for j, gene in enumerate(self.gene_order):
            if gene in adata.var_names:
                # Find the column index in the original dataset
                orig_idx = list(adata.var_names).index(gene)
                X_new[:, j] = (
                    adata.X[:, orig_idx].toarray().flatten()
                    if sp.issparse(adata.X)
                    else adata.X[:, orig_idx]
                )

                # Copy var metadata for this gene
                if gene in adata.var.columns:
                    for col in adata.var.columns:
                        if col not in var_new.columns:
                            var_new[col] = None
                        var_new.loc[gene, col] = adata.var.loc[gene, col]

        # Create new AnnData object
        adata_subset = ad.AnnData(
            X=X_new,
            obs=adata.obs.copy(),
            var=var_new,
            uns=adata.uns.copy() if adata.uns else {},
        )

        # Log information about missing genes
        if missing_genes:
            logger.warning(
                f"Dataset missing {len(missing_genes)} genes, filled with zeros"
            )
            logger.debug(f"First few missing genes: {missing_genes[:5]}")

        logger.info(
            f"Dataset subsetted to {len(self.gene_order)} genes (exact order from model)"
        )
        return adata_subset

    def _prepare_data(self, adata: ad.AnnData) -> np.ndarray:
        """
        Prepare data for PCA transformation.

        Parameters
        ----------
        adata : ad.AnnData
            Subsetted dataset

        Returns
        -------
        np.ndarray
            Prepared data matrix
        """
        # Convert to dense if sparse
        if sp.issparse(adata.X):
            X = adata.X.toarray()
        else:
            X = adata.X.copy()

        # Apply scaling if scaler was used during training
        if self.scaler is not None:
            logger.debug("Applying saved scaling transformation")
            X = self.scaler.transform(X)
        else:
            logger.debug("No scaling applied (none was used during training)")

        return X

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Load the pre-trained PCA model and gene list.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Not used for preparation, but kept for interface consistency.
        adata_path : str, optional
            Not used for preparation, but kept for interface consistency.
        **kwargs : dict
            Additional keyword arguments (unused).
        """
        logger.info("Loading pre-trained PCA model and gene list...")
        self._load_model()
        self._load_gene_list()
        logger.info("PCA model preparation complete")

    def embed(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str = "X_pca",
        **kwargs,
    ) -> np.ndarray:
        """
        Apply pre-trained PCA transformation to a dataset.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Input dataset
        adata_path : str, optional
            Path to the AnnData file (.h5ad or .zarr)
        obsm_key : str, default "X_pca"
            Key for storing PCA results in adata.obsm

        Returns
        -------
        np.ndarray
            PCA embedding matrix
        """
        adata = _check_load_adata(adata, adata_path)

        if self.pca_model is None:
            raise RuntimeError("PCA model is not loaded. Call `prepare()` first.")

        logger.info(f"Applying pre-trained PCA to dataset: {adata.shape}")

        # 1. Subset to required genes in correct order
        adata_subset = self._subset_and_order_genes(adata)

        # 2. Prepare data (scaling, etc.)
        X_prepared = self._prepare_data(adata_subset)

        # 3. Apply PCA transformation
        X_pca = self.pca_model.transform(X_prepared)

        # 4. Store results in original adata
        adata.obsm[obsm_key] = X_pca

        logger.info(f"PCA applied: {adata.shape} -> {X_pca.shape}")
        logger.info(f"Results stored in adata.obsm['{obsm_key}']")

        return X_pca


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

        # Name of the tokenized dataset
        self.dataset_name = "geneformer"

        logger.info(
            "Initialized GeneformerEmbedder with model_input_size=%d, num_layers=%d, special_model=%s",
            self.model_input_size,
            self.num_layers,
            self.special_model,
        )

    def _read_sample_indices(self, file_path: str | Path) -> np.ndarray:
        """
        Efficiently read sample indices from an AnnData file (h5ad or zarr) without loading the entire object.

        Parameters
        ----------
        file_path : str or Path
            Path to the AnnData file (.h5ad or .zarr).

        Returns
        -------
        np.ndarray
            Array of sample indices.

        Raises
        ------
        ValueError
            If the file format is not supported or if sample_index is not found.
        """
        file_path = Path(file_path)
        if file_path.suffix == ".h5ad":
            import h5py

            with h5py.File(file_path, "r") as f:
                if "obs" not in f or "sample_index" not in f["obs"]:
                    raise ValueError("sample_index not found in obs")
                return f["obs/sample_index"][:]
        elif file_path.suffix == ".zarr":
            import zarr

            store = zarr.DirectoryStore(file_path)
            root = zarr.group(store=store)
            if "obs" not in root or "sample_index" not in root["obs"]:
                raise ValueError("sample_index not found in obs")
            return root["obs/sample_index"][:]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
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
        adata : anndata.AnnData, optional
            Single-cell dataset to prepare.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
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
            raise ValueError(
                "adata_path must be provided to save the tokenized dataset."
            )
        # quick fix: Always use "processed" dir and not "processed_with_emb" to avoid retokenization
        if "processed_with_emb" in adata_path:
            adata_path = adata_path.replace("processed_with_emb", "processed")
        adata = _check_load_adata(adata, adata_path)
        self.in_adata_path = Path(adata_path)
        adata_name = self.in_adata_path.stem
        # save the tokenized dataset in the same directory as the adata
        self.og_adata_dir = self.in_adata_path.parent
        self.adata_dir = self.og_adata_dir / "geneformer" / adata_name / "adata"
        self.adata_dir.mkdir(parents=True, exist_ok=True)
        # write the adata to the directory, if the file doesnt exist yet
        gf_adata_file = self.adata_dir / f"{adata_name}.h5ad"
        if not gf_adata_file.exists():
            logger.info("Writing AnnData to %s", gf_adata_file)
            adata.write_h5ad(gf_adata_file)
        else:
            logger.info(
                "AnnData already exists at %s. Skipping writing.",
                gf_adata_file,
            )
        self.out_dataset_dir = (
            self.og_adata_dir / "geneformer" / adata_name / "tokenized_ds"
        )
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
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str = "X_geneformer",
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        """
        Run Geneformer embedding on the data.

        This method:
         - Reads the tokenized dataset from `self.tmp_dir`.
         - Invokes Geneformer's `EmbExtractor` to generate embeddings.
         - Re-reads the (processed) AnnData from the temporary directory.
         - Aligns the embeddings with the sample order, storing them in `adata.obsm[obsm_key]`.

        Parameters
        ----------
        adata : anndata.Anndata, optional
            The AnnData object to embed.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        obsm_key : str, optional
            Key in `adata.obsm` to store the final embeddings. Defaults to "X_geneformer".
        batch_size : int, optional
            Forward batch size used by the Geneformer model for embedding extraction.
        **kwargs : dict
            Additional arguments (unused here, but kept for interface consistency).

        Returns
        -------
        np.ndarray
            The embedding matrix of shape (n_cells, 512).

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

        # Get sample indices efficiently without loading the entire AnnData
        if adata_path is not None:
            og_ids = self._read_sample_indices(adata_path)
        else:
            og_ids = adata.obs["sample_index"].values

        # Filter and sort embs_df to align with og_ids
        # drop the "Unamed: 0" column
        embs_sorted = self._deduplicate_and_reindex_embeddings(embs_df, og_ids)
        embedding_matrix = embs_sorted.values
        # Store in adata for compatibility if provided
        if adata is not None:
            adata.obsm[obsm_key] = embedding_matrix
            logger.info(
                "Stored Geneformer embeddings of shape %s in adata.obsm[%r].",
                embedding_matrix.shape,
                obsm_key,
            )
        return embedding_matrix

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
        file_copy_max_retries : int, optional
            Maximum number of retry attempts for NFS file copying (default: 5)
        file_copy_base_delay : float, optional
            Base delay in seconds for exponential backoff during file copying (default: 1.0)
        **kwargs
            Additional arguments passed to SCVI setup.
        """
        super().__init__(embedding_dim=embedding_dim)
        self.model = None
        self.init_kwargs = kwargs

        # File copying configuration for NFS robustness
        self.file_copy_max_retries = kwargs.get("file_copy_max_retries", 5)
        self.file_copy_base_delay = kwargs.get("file_copy_base_delay", 1.0)

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **kwargs,
    ):
        """
        Prepare the SCVI model for embedding.

        For example:
        - `hub_repo_id` to load from HF
        - `reference_s3_bucket` = "cellxgene-contrib-public" and `reference_s3_path` = "models/scvi/2024-02-12/homo_sapiens/modelhub" to load from S3
        - `reference_adata_url` = "https://cellxgene-contrib-public.s3.amazonaws.com/models/scvi/2024-02-12/homo_sapiens/adata-spinal-cord-minified.h5ad" to load reference adata (jointly with s3)

        The user can pass these as part of `init_kwargs` or `kwargs`.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The AnnData object to prepare the embedder.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        **kwargs : dict
            Additional configuration for loading from S3 or HF.
        """
        adata = _check_load_adata(adata, adata_path)
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

        # Redirect cache directories if they're in /tmp/
        if cache_dir is not None:
            cache_dir = _redirect_tmp_cache_dir(cache_dir)
        if file_cache_dir is not None:
            file_cache_dir = _redirect_tmp_cache_dir(file_cache_dir)

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
                temp_base_dir = _redirect_tmp_cache_dir(tempfile.gettempdir())
                save_dir = tempfile.TemporaryDirectory(dir=temp_base_dir)
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
        # Convert to string first to ensure consistent data types, then to categorical
        query_adata.obs["batch"] = (
            query_adata.obs[self.batch_key].astype(str).astype("category")
        )

        # Clear varm to prevent dimension mismatch errors during SCVI preparation
        # The varm field may contain PCA components or other variable-level metadata
        # that was computed on a different set of genes than what SCVI expects
        if len(query_adata.varm) > 0:
            logger.info(
                "Clearing varm field to prevent dimension mismatch during SCVI preparation"
            )
            query_adata.varm.clear()

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
    def _robust_copy_file(
        src: Path, dst: Path, max_retries: int = 5, base_delay: float = 1.0
    ) -> None:
        """
        Robustly copy a file with retry logic for NFS/shared filesystem issues.

        Parameters
        ----------
        src : Path
            Source file path
        dst : Path
            Destination file path
        max_retries : int
            Maximum number of retry attempts
        base_delay : float
            Base delay in seconds for exponential backoff

        Raises
        ------
        OSError
            If all retry attempts fail
        """
        # NFS-related error codes that should trigger retries
        nfs_errors = {
            errno.ESTALE,  # Stale file handle (70 on macOS, varies on Linux)
            errno.EIO,  # 5: I/O error
            errno.EBUSY,  # 16: Device or resource busy
            errno.EAGAIN,  # 11: Try again (35 on macOS)
            errno.EINTR,  # 4: Interrupted system call
            70,  # ESTALE on some systems
            116,  # Stale file handle on Linux systems (from original error)
        }

        for attempt in range(max_retries):
            try:
                # Try different copy strategies on each attempt
                if attempt == 0:
                    # First attempt: use shutil.copy2 (preserves metadata)
                    shutil.copy2(src, dst)
                elif attempt == 1:
                    # Second attempt: use shutil.copyfile (no metadata)
                    shutil.copyfile(src, dst)
                else:
                    # Subsequent attempts: manual chunked copy
                    SCVIEmbedder._chunked_copy(src, dst)

                # If we get here, the copy succeeded
                logger.debug(
                    f"Successfully copied {src} to {dst} on attempt {attempt + 1}"
                )
                return

            except OSError as e:
                # Check for NFS-related errors using multiple approaches
                error_code = getattr(e, "errno", None)
                # Also check the args in case errno isn't set properly
                if error_code is None and hasattr(e, "args") and len(e.args) > 0:
                    error_code = e.args[0] if isinstance(e.args[0], int) else None

                if error_code in nfs_errors and attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)

                    logger.warning(
                        f"NFS error (errno {error_code}) copying {src} to {dst}. "
                        f"Retry {attempt + 1}/{max_retries} in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Non-NFS error or max retries exceeded
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to copy {src} to {dst} after {max_retries} attempts. "
                            f"Last error: {e}"
                        )
                    raise

            except Exception as e:
                # Non-OSError exceptions should not be retried
                logger.error(f"Unexpected error copying {src} to {dst}: {e}")
                raise

    @staticmethod
    def _chunked_copy(src: Path, dst: Path, chunk_size: int = 64 * 1024) -> None:
        """
        Copy a file in chunks to avoid NFS issues with large files.

        Parameters
        ----------
        src : Path
            Source file path
        dst : Path
            Destination file path
        chunk_size : int
            Size of each chunk in bytes
        """
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                chunk = fsrc.read(chunk_size)
                if not chunk:
                    break
                fdst.write(chunk)

        # Copy file permissions
        try:
            shutil.copystat(src, dst)
        except OSError:
            # If we can't copy stats, that's usually not critical
            pass

    @staticmethod
    def _localize_hubmodel(
        model: HubModel, max_retries: int = 5, base_delay: float = 1.0
    ) -> HubModel:
        """
        Copy the weight file (model.pt) into a unique temp dir and return a *new*
        HubModel instance that points there.  This eliminates cross-process races.

        Uses robust file copying with retry logic to handle NFS/shared filesystem issues.

        Parameters
        ----------
        model : HubModel
            The HubModel to localize
        max_retries : int
            Maximum number of retry attempts for file copying
        base_delay : float
            Base delay in seconds for exponential backoff
        """
        orig_dir = Path(model.local_dir)
        temp_base_dir = _redirect_tmp_cache_dir(tempfile.gettempdir())
        tmp_dir = Path(temp_base_dir) / f"scvi_{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy the weight file with robust retry logic
            model_pt_src = orig_dir / "model.pt"
            model_pt_dst = tmp_dir / "model.pt"

            if model_pt_src.exists():
                SCVIEmbedder._robust_copy_file(
                    model_pt_src, model_pt_dst, max_retries, base_delay
                )
            else:
                raise FileNotFoundError(f"Model file not found: {model_pt_src}")

            # Copy adata file if it exists (with retry logic)
            adata_src = orig_dir / "adata.h5ad"
            if adata_src.is_file():
                adata_dst = tmp_dir / "adata.h5ad"
                SCVIEmbedder._robust_copy_file(
                    adata_src, adata_dst, max_retries, base_delay
                )

            # reuse the existing metadata / model-card objects
            return HubModel(
                local_dir=str(tmp_dir),
                metadata=model.metadata,
                model_card=model.model_card,
            )

        except Exception as e:
            # Clean up temp directory on failure
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors

            logger.error(f"Failed to localize HubModel from {orig_dir}: {e}")
            raise

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
            model = self._localize_hubmodel(
                model, self.file_copy_max_retries, self.file_copy_base_delay
            )

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
        adata_path: str | None = None,
        obsm_key: str = "X_scvi",
        batch_key: str = "batch",
        **kwargs,
    ) -> np.ndarray:
        """
        Transform the data into the SCVI latent space.

        If the SCVI model has not yet been set up with the query data, it does so here.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The query dataset to be embedded.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        obsm_key : str
            The key in `adata.obsm` under which to store the SCVI embedding.
        batch_key : str
            The batch key in `adata.obs` to use for batch correction.
        **kwargs : dict
            Additional keyword arguments (unused).

        Returns
        -------
        np.ndarray
            The embedding matrix of shape (n_cells, embedding_dim).

        References
        ----------
        The reference data is loaded from S3 or the Hub (depending on configuration).
        Query data is the user-provided `adata`.
        """
        adata = _check_load_adata(adata, adata_path)
        self.batch_key = batch_key
        if self.scvi_model is None:
            raise ValueError("SCVI model is not prepared. Call `prepare(...)` first.")
        # If the query hasn't been loaded yet, load it now:
        if not hasattr(self.scvi_model, "adata") or self.scvi_model.adata is not adata:
            self._prepare_query_adata(adata)

        logger.info(
            "Computing SCVI latent representation, storing in `%s`...", obsm_key
        )
        embedding_matrix = self.scvi_model.get_latent_representation()
        # Store in adata for compatibility
        adata.obsm[obsm_key] = embedding_matrix
        return embedding_matrix


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

        # Redirect cache directories if they're in /tmp/
        default_model_cache = _redirect_tmp_cache_dir(default_model_cache)
        default_data_cache = _redirect_tmp_cache_dir(default_data_cache)

        # Ensure the cache directories exist
        os.makedirs(default_model_cache, exist_ok=True)
        os.makedirs(default_data_cache, exist_ok=True)

        # Use provided cache dirs or defaults (and redirect if needed)
        cache_dir = _redirect_tmp_cache_dir(
            init_kwargs.pop("cache_dir", default_model_cache)
        )
        file_cache_dir = _redirect_tmp_cache_dir(
            init_kwargs.pop("file_cache_dir", default_data_cache)
        )

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


class GeneSelectEmbedder(BaseEmbedder):
    """
    Gene selection embedder that uses the same gene set as the scVI foundation model
    but returns the raw expression values instead of scVI embeddings.

    This embedder uses the gene list from resources/gene_selection_ENSG_8k.txt, which
    contains exactly the same genes that the scVI foundation model selects. This approach
    is much faster than loading the full scVI model since it only requires the gene list.

    The gene list contains 8000 genes that were selected by the scVI foundation model
    for consistent gene selection across datasets.
    """

    def __init__(self, embedding_dim: int = None, **init_kwargs):
        """
        Initialize the GeneSelect embedder with gene list from resources.

        Parameters
        ----------
        embedding_dim : int, optional
            Will be dynamically determined by the number of genes in the gene list.
            If provided, it will be ignored with a warning.
        init_kwargs : dict
            Additional keyword arguments including:
            - gene_list_path: Path to gene list file (default: resources/gene_selection_ENSG_8k.txt)
        """
        if embedding_dim is not None:
            logger.warning(
                "GeneSelectEmbedder embedding_dim is determined by the gene list. "
                f"Ignoring provided value: {embedding_dim}"
            )

        # Initialize with a placeholder - will be updated after loading the gene list
        super().__init__(embedding_dim=0)

        # Set default gene list path
        self.gene_list_path = init_kwargs.get(
            "gene_list_path", "resources/gene_selection_common_genes.txt"
        )

        # Initialize gene list
        self.gene_order = None

        logger.info(
            f"Initialized GeneSelectEmbedder with gene_list_path: {self.gene_list_path}"
        )

    def _load_gene_list(self) -> None:
        """Load gene list from file."""
        gene_list_path = Path(self.gene_list_path)
        if not gene_list_path.exists():
            raise FileNotFoundError(f"Gene list file not found: {gene_list_path}")

        with open(gene_list_path, "r") as f:
            self.gene_order = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(self.gene_order)} genes from {gene_list_path}")

    def _subset_and_order_genes(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Subset dataset to required genes in the correct order.
        Missing genes are filled with zeros to ensure consistent dimensionality.

        Parameters
        ----------
        adata : ad.AnnData
            Input dataset

        Returns
        -------
        ad.AnnData
            Subsetted dataset with genes in correct order
        """
        # Ensure ensembl IDs are available
        ensure_ensembl_index(adata, ensembl_col="ensembl_id")

        # Create a new AnnData object with the exact gene list
        n_cells = adata.n_obs

        # Initialize data matrix with zeros for all genes
        if sp.issparse(adata.X):
            X_new = np.zeros((n_cells, len(self.gene_order)), dtype=adata.X.dtype)
        else:
            X_new = np.zeros((n_cells, len(self.gene_order)), dtype=adata.X.dtype)

        # Create new var DataFrame with the exact gene list
        var_new = pd.DataFrame(index=self.gene_order)

        # Copy available genes from original dataset
        available_genes = [g for g in self.gene_order if g in adata.var_names]
        missing_genes = [g for g in self.gene_order if g not in adata.var_names]

        if len(available_genes) == 0:
            raise ValueError("Dataset has no genes from the required gene set")

        # Copy data for available genes
        for j, gene in enumerate(self.gene_order):
            if gene in adata.var_names:
                # Find the column index in the original dataset
                orig_idx = list(adata.var_names).index(gene)
                X_new[:, j] = (
                    adata.X[:, orig_idx].toarray().flatten()
                    if sp.issparse(adata.X)
                    else adata.X[:, orig_idx]
                )

                # Copy var metadata for this gene
                if gene in adata.var.columns:
                    for col in adata.var.columns:
                        if col not in var_new.columns:
                            var_new[col] = None
                        var_new.loc[gene, col] = adata.var.loc[gene, col]

        # Create new AnnData object
        adata_subset = ad.AnnData(
            X=X_new,
            obs=adata.obs.copy(),
            var=var_new,
            uns=adata.uns.copy() if adata.uns else {},
        )

        # Log information about missing genes
        if missing_genes:
            logger.warning(
                f"Dataset missing {len(missing_genes)} genes, filled with zeros"
            )
            logger.debug(f"First few missing genes: {missing_genes[:5]}")

        logger.info(
            f"Dataset subsetted to {len(self.gene_order)} genes (exact order from gene list)"
        )
        return adata_subset

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **kwargs,
    ):
        """
        Load the gene list from resources.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            Not used for preparation, but kept for interface consistency.
        adata_path : str, optional
            Not used for preparation, but kept for interface consistency.
        **kwargs : dict
            Additional keyword arguments (unused).
        """
        logger.info("Loading gene list from resources...")
        self._load_gene_list()

        # Update embedding dimension based on the number of genes in the list
        self.embedding_dim = len(self.gene_order)
        logger.info(
            f"GeneSelect embedder prepared. Gene set contains {self.embedding_dim} genes."
        )

    def embed(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str = "X_gs",
        batch_key: str = "batch",
        **kwargs,
    ) -> np.ndarray:
        """
        Apply gene selection and return the processed gene expression matrix.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The query dataset to be processed.
        adata_path : str, optional
            Path to the AnnData file (.h5ad).
        obsm_key : str
            The key in `adata.obsm` under which to store the gene-selected matrix.
        batch_key : str
            The batch key in `adata.obs` to use for batch information (unused but kept for interface consistency).
        **kwargs : dict
            Additional keyword arguments (unused).

        Returns
        -------
        np.ndarray
            The gene-selected expression matrix of shape (n_cells, n_selected_genes).
        """
        adata = _check_load_adata(adata, adata_path)

        if self.gene_order is None:
            raise ValueError("Gene list is not loaded. Call `prepare(...)` first.")

        logger.info("Applying gene selection using scVI foundation model gene set...")

        # Keep a backup of the original data
        adata_backup = adata.copy()

        # Subset to required genes in correct order
        adata_subset = self._subset_and_order_genes(adata)

        # Get the processed expression matrix
        X = adata_subset.X.toarray() if sp.issparse(adata_subset.X) else adata_subset.X
        embedding_matrix = X.copy().astype(np.float32)

        # Store in adata for compatibility
        adata.obsm[obsm_key] = embedding_matrix

        logger.info(
            f"Gene selection complete. Selected {embedding_matrix.shape[1]} genes "
            f"from {adata_backup.shape[1]} original genes. "
            f"Stored in adata.obsm[{obsm_key}]."
        )

        return embedding_matrix


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
        - "gs" (gene select)
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
            "gs": GeneSelectEmbedder,
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

    def _validate_file_path(self, file_path: str | Path) -> None:
        """
        Validate that the file path exists and has a supported format.

        Parameters
        ----------
        file_path : str or Path
            Path to the file to validate.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix not in [".h5ad", ".zarr"]:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Only .h5ad and .zarr formats are supported."
            )

    def prepare(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        **prepare_kwargs,
    ) -> None:
        """
        Prepare the embedder from a file path or AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The AnnData object to prepare the embedder. Has to be preprocessed already, if embedding method requires it.
        adata_path : str, optional
            Path to the AnnData file (.h5ad or .zarr). Only needed for geneformer.
        **prepare_kwargs
            Keyword arguments passed to the embedder's prepare().

        Raises
        ------
        ValueError
            If neither adata nor adata_path is provided.
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported.
        """
        if adata is None and adata_path is None:
            raise ValueError("Either adata or adata_path must be provided")

        if adata_path is not None:
            self._validate_file_path(adata_path)

        logger.info("Preparing embedder '%s'", self.method)
        if adata_path is not None:
            logger.info("Using file path: %s", adata_path)
        self.embedder.prepare(adata=adata, adata_path=adata_path, **prepare_kwargs)

    def embed(
        self,
        adata: anndata.AnnData | None = None,
        adata_path: str | None = None,
        obsm_key: str | None = None,
        batch_key: str = "batch",
        **embed_kwargs,
    ) -> np.ndarray:
        """
        Embed data and return the embedding matrix.

        Parameters
        ----------
        adata : anndata.AnnData, optional
            The AnnData object to embed.
        adata_path : str, optional
            Path to the AnnData file (.h5ad or .zarr).
        obsm_key : str, optional
            Key under which embeddings are stored in .obsm.
            Defaults to "X_{method}".
        batch_key : str, default="batch"
            Observation column for batch labels (used by some embedders).
        **embed_kwargs
            Additional keyword arguments for the embedders embed().

        Returns
        -------
        np.ndarray
            The embedding matrix of shape (n_cells, embedding_dim).

        Raises
        ------
        ValueError
            If neither adata nor adata_path is provided.
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is not supported.
        """
        if adata is None and adata_path is None:
            raise ValueError("Either adata or adata_path must be provided")

        if adata_path is not None:
            self._validate_file_path(adata_path)

        # Derive defaults
        if obsm_key is None:
            obsm_key = f"X_{self.method}"

        logger.info(f"Embedding method: {self.method}")
        if adata_path is not None:
            logger.info("Using file path: %s", adata_path)

        embedding_matrix = self.embedder.embed(
            adata=adata,
            adata_path=adata_path,
            obsm_key=obsm_key,
            batch_key=batch_key,
            **embed_kwargs,
        )

        logger.info("Embedding complete. Returning the embedding matrix")
        return embedding_matrix
