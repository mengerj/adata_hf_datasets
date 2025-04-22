# src/my_pipeline/preprocessing.py
import logging
from pathlib import Path
from adata_hf_datasets.pp import (
    pp_quality_control,
    pp_adata_general,
    pp_adata_geneformer,
    maybe_add_sra_metadata,
    prepend_instrument_to_description,
)
from adata_hf_datasets.utils import H5ADChunkLoader

logger = logging.getLogger(__name__)


def preprocess_h5ad(
    infile: Path,
    outfile: Path,
    *,
    chunk_size: int = 1_000,
    min_cells: int = 10,
    min_genes: int = 200,
    batch_key: str = "batch",
    n_top_genes: int = 1_000,
    call_geneformer: bool = True,
    sra_chunk_size: int | None = None,
    extra_sra_cols: list[str] | None = None,
    instrument_key: str | None = None,
    description_key: str | None = None,
) -> None:
    """
    Preprocess a large .h5ad file in chunks and write a concatenated output.

    This function:
      1. Iterates over smaller AnnData chunks.
      2. Applies QC, general filtering, optional geneformer step.
      3. Optionally adds SRA metadata and instrument descriptions.
      4. Writes each chunk to disk, then concatenates them into `outfile`.

    Parameters
    ----------
    infile : Path
        Path to the raw .h5ad file.
    outfile : Path
        Path where the final preprocessed file is written.
    chunk_size : int, optional
        Number of cells per chunk (default 1000).
    min_cells : int, optional
        Minimum cells per gene for filtering (default 10).
    min_genes : int, optional
        Minimum genes per cell for filtering (default 200).
    batch_key : str, optional
        Column in `.obs` for batch labels (default "batch").
    n_top_genes : int, optional
        Number of HVGs to select (default 1000).
    call_geneformer : bool, optional
        Whether to run the Geneformer step (default True).
    sra_chunk_size : int, optional
        If provided, chunk size for SRA metadata queries.
    extra_sra_cols : list[str], optional
        Additional columns to fetch from SRA.
    instrument_key : str, optional
        Key in `.obs` holding instrument names.
    description_key : str, optional
        Key in `.obs` holding sample descriptions.

    References
    ----------
    - Data source: single-cell RNA‑seq count matrix stored in H5AD format.
    """
    chunk_dir = outfile.with_suffix("").parent / f"{outfile.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    loader = H5ADChunkLoader(infile, chunk_size)

    try:
        for i, adata in enumerate(loader):
            logger.info("Preprocessing chunk %d", i)
            adata = pp_quality_control(adata)
            adata = pp_adata_general(
                adata,
                min_cells=min_cells,
                min_genes=min_genes,
                batch_key=batch_key,
                n_top_genes=n_top_genes,
            )
            if call_geneformer:
                adata = pp_adata_geneformer(adata)
            if sra_chunk_size and extra_sra_cols:
                maybe_add_sra_metadata(
                    adata, chunk_size=sra_chunk_size, new_cols=extra_sra_cols
                )
            if instrument_key and description_key:
                prepend_instrument_to_description(
                    adata,
                    instrument_key=instrument_key,
                    description_key=description_key,
                )
            chunk_path = chunk_dir / f"chunk_{i}.h5ad"
            adata.write_h5ad(chunk_path)
            chunks.append(chunk_path)

        # Concatenate on disk
        from anndata.experimental import concat_on_disk

        concat_on_disk(in_files=chunks, out_file=str(outfile))
        logger.info("Wrote final file to %s", outfile)

    finally:
        # clean up
        for f in chunks:
            try:
                f.unlink()
            except OSError:
                logger.warning("Could not delete chunk %s", f)
        try:
            chunk_dir.rmdir()
        except OSError:
            pass
