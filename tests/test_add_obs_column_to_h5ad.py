# tests/test_add_obs_column_to_h5ad.py
import h5py
import numpy as np
import anndata as ad
from pathlib import Path
from adata_hf_datasets.file_utils import add_obs_column_to_h5ad


def _make_h5ad(tmp_path: Path) -> Path:
    fn = tmp_path / "dummy.h5ad"
    X = np.random.poisson(1.0, size=(5, 3))
    a = ad.AnnData(X)
    a.raw = a  # give it a raw section
    a.write_h5ad(fn)
    return fn


def _delete_raw_X(fn: Path):
    with h5py.File(fn, "r+") as f:
        del f["raw"]["X"]  # simulate corrupted / missing raw matrix


def test_add_column_handles_missing_raw(tmp_path):
    src = _make_h5ad(tmp_path)
    _delete_raw_X(src)

    dst = tmp_path / "out.h5ad"
    # the function should succeed after the fix
    out = add_obs_column_to_h5ad(
        infile=src,
        temp_out=dst,
        column_name="sample_index",
        column_data=np.arange(5, dtype=np.int64),
    )

    # verify new column exists and file opens fine
    a = ad.read_h5ad(out)
    assert "sample_index" in a.obs.columns
    a.file.close()
