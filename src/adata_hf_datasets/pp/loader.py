import logging
from pathlib import Path
from typing import Iterator, List, Literal

from anndata import AnnData
import anndata as ad

logger = logging.getLogger(__name__)


class BatchChunkLoader:
    """
    Yield AnnData chunks consisting of whole batches up to a target size.

    Parameters
    ----------
    path : Path
        Path to the AnnData file (.h5ad or .zarr).
    chunk_size : int
        Minimum number of cells per chunk.
    batch_key : str
        Column in `.obs` that holds batch labels.
    file_format : Literal["h5ad", "zarr"], default="h5ad"
        Format of the input file. Must be either "h5ad" or "zarr".
    """

    def __init__(
        self,
        path: Path,
        chunk_size: int,
        batch_key: str,
        file_format: Literal["h5ad", "zarr"] = "h5ad",
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.batch_key = batch_key
        self.file_format = file_format

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

        obs = adata_backed.obs
        if self.batch_key not in obs.columns:
            raise KeyError(f"batch_key '{self.batch_key}' not in adata.obs")

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
                yield self._load_chunk(adata_backed, current)
                current, count = [], 0
            current.append(b)
            count += n

        if current:
            yield self._load_chunk(adata_backed, current)

        adata_backed.file.close()

    def _load_chunk(self, adata_backed: AnnData, batches: List) -> AnnData:
        mask = adata_backed.obs[self.batch_key].isin(batches)
        logger.info("Loading chunk of %d batches: %s", len(batches), batches)
        return adata_backed[mask].to_memory()
