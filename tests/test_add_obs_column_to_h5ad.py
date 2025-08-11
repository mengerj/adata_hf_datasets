# tests/test_add_obs_column_to_h5ad.py
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from anndata import AnnData
from adata_hf_datasets.file_utils import add_obs_column_to_h5ad  # adjust import


def _make_h5ad_with_csr_raw(fn: Path):
    ad = AnnData(X=sp.random(30, 40, 0.1, format="csr"))
    ad.raw = ad
    ad.write_h5ad(fn)


def test_add_obs_fails_on_csr_raw(tmp_path):
    src = tmp_path / "csr_raw.h5ad"
    _make_h5ad_with_csr_raw(src)
    dst = tmp_path / "out.h5ad"

    # with pytest.raises(AttributeError, match="raw X"):
    add_obs_column_to_h5ad(
        infile=src,
        temp_out=dst,
        column_name="sample_index",
        column_data=np.arange(30),
    )
