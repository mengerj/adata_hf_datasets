import logging
from pathlib import Path
from typing import Iterator, List, Literal, Optional
import numpy as np

from anndata import AnnData
import anndata as ad

logger = logging.getLogger(__name__)


class BatchChunkLoader:
    """
    Yield AnnData chunks either by batches or randomly up to a target size.

    Parameters
    ----------
    path : Path
        Path to the AnnData file (.h5ad or .zarr).
    chunk_size : int
        Minimum number of cells per chunk.
    batch_key : str, optional
        Column in `.obs` that holds batch labels. Required for batch-based chunking.
    file_format : Literal["h5ad", "zarr"], default="h5ad"
        Format of the input file. Must be either "h5ad" or "zarr".
    random_chunking : bool, optional
        If True, use random chunking (even if batch_key is provided).
        If False and batch_key is None, random chunking is still used (batch-based not possible).
        If None, auto-detect: random if no batch_key, batch-based if batch_key provided.
    random_seed : int, optional
        Random seed for reproducible random chunking. Only used if random chunking is used.
    """

    def __init__(
        self,
        path: Path,
        chunk_size: int,
        batch_key: Optional[str] = None,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
        random_chunking: Optional[bool] = None,
        random_seed: Optional[int] = None,
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.batch_key = batch_key
        self.file_format = file_format
        self.random_seed = random_seed

        # Determine chunking mode
        if random_chunking is True:
            # Explicitly requested random chunking
            self.random_chunking = True
            if batch_key is not None:
                logger.info(
                    f"random_chunking=True: Using random chunking despite batch_key='{batch_key}'"
                )
        elif random_chunking is False:
            # Explicitly requested batch-based chunking
            if batch_key is None:
                logger.warning(
                    "random_chunking=False but no batch_key provided. "
                    "Falling back to random chunking."
                )
                self.random_chunking = True
            else:
                self.random_chunking = False
        else:
            # Auto-detect: random if no batch_key, batch-based if batch_key provided
            self.random_chunking = batch_key is None
            if self.random_chunking:
                logger.info("No batch_key provided, using random chunking")
            else:
                logger.info(
                    f"batch_key='{batch_key}' provided, using batch-based chunking"
                )

        if file_format not in ["h5ad", "zarr"]:
            raise ValueError("file_format must be either 'h5ad' or 'zarr'")

        if path.suffix != f".{file_format}":
            raise ValueError(
                f"File extension {path.suffix} does not match specified format {file_format}"
            )

    def __iter__(self) -> Iterator[AnnData]:
        logger.info("Opening %s file in read mode", self.file_format)
        if self.file_format == "zarr":
            adata_backed = ad.read_zarr(self.path)
        else:
            adata_backed = ad.read_h5ad(self.path, backed="r")

        if self.random_chunking:
            yield from self._iter_random_chunks(adata_backed)
        else:
            yield from self._iter_batch_chunks(adata_backed)

        adata_backed.file.close()

    def _iter_batch_chunks(self, adata_backed: AnnData) -> Iterator[AnnData]:
        """Iterate over chunks grouped by batches."""
        obs = adata_backed.obs
        if self.batch_key not in obs.columns:
            raise KeyError(f"batch_key '{self.batch_key}' not in adata.obs")

        logger.info("Using batch-based chunking with batch_key='%s'", self.batch_key)

        # preserve batch order, group by label
        seen, batches = set(), []
        for b in obs[self.batch_key].tolist():
            if b not in seen:
                seen.add(b)
                batches.append(b)

        current, count = [], 0
        for b in batches:
            n = int((obs[self.batch_key] == b).sum())
            if current and count + n > self.chunk_size:
                yield self._load_batch_chunk(adata_backed, current)
                current, count = [], 0
            current.append(b)
            count += n

        if current:
            yield self._load_batch_chunk(adata_backed, current)

    def _iter_random_chunks(self, adata_backed: AnnData) -> Iterator[AnnData]:
        """Iterate over randomly shuffled chunks."""
        logger.info("Using random chunking")

        n_cells = adata_backed.n_obs
        if self.random_seed is not None:
            rng = np.random.default_rng(self.random_seed)
            logger.info(f"Using random seed: {self.random_seed}")
        else:
            rng = np.random.default_rng()

        # Create shuffled indices
        indices = np.arange(n_cells)
        rng.shuffle(indices)

        # Yield chunks
        for start_idx in range(0, n_cells, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_cells)
            chunk_indices = indices[start_idx:end_idx]
            yield self._load_random_chunk(adata_backed, chunk_indices)

    def _load_batch_chunk(self, adata_backed: AnnData, batches: List) -> AnnData:
        """Load a chunk based on batch labels."""
        mask = adata_backed.obs[self.batch_key].isin(batches)
        logger.info("Loading chunk of %d batches: %s", len(batches), batches)
        return adata_backed[mask].to_memory()

    def _load_random_chunk(self, adata_backed: AnnData, indices: np.ndarray) -> AnnData:
        """Load a chunk based on random indices."""
        logger.info("Loading random chunk of %d cells", len(indices))
        return adata_backed[indices].to_memory()
