import numpy as np
import scanpy as sc
from anndata import AnnData
from adata_hf_datasets.pp import preprocess_h5ad


class MockSRA:
    @staticmethod
    def maybe_add_sra_metadata(adata, **kwargs):
        print("PATCHED FUNCTION CALLED!")  # Debug print
        adata.obs["foo"] = ["X"] * adata.n_obs
        return adata


def test_preprocess_h5ad_end_to_end(tmp_path, monkeypatch):
    n_cells, n_genes = 50, 100
    X = np.random.poisson(lam=1.0, size=(n_cells, n_genes)).astype(int)
    # Create batches unevenly split into three groups
    batches = np.random.choice(
        ["batch1", "batch2", "batch3"], size=n_cells, p=[0.5, 0.3, 0.2]
    )
    # Instruments and descriptions as simple strings
    insts = [f"I{idx % 4}" for idx in range(n_cells)]
    descrs = [f"desc{idx}" for idx in range(n_cells)]
    obs = {
        "batch": batches,
        "inst": insts,
        "descr": descrs,
    }
    ad = AnnData(X=X, obs=obs)
    ad.var_names = [f"gene{i}" for i in range(n_genes)]
    infile = tmp_path / "in.h5ad"
    outdir = tmp_path / "out"
    ad.write_h5ad(infile)

    # Stub out each processing step
    monkeypatch.setattr(
        "adata_hf_datasets.pp.ensure_raw_counts_layer",
        lambda adata, raw_layer_key=None: None,
    )
    monkeypatch.setattr("adata_hf_datasets.pp.pp_quality_control", lambda adata: adata)
    monkeypatch.setattr(
        "adata_hf_datasets.pp.pp_adata_general", lambda adata, **kwargs: adata
    )
    monkeypatch.setattr("adata_hf_datasets.pp.pp_adata_geneformer", lambda adata: adata)

    # fake maybe_add_sra_metadata: inject 'foo' in place
    def fake_sra(adata, **kwargs):
        print("PATCHED FUNCTION CALLED!")  # Debug print
        adata.obs["foo"] = ["X"] * adata.n_obs
        return adata

    monkeypatch.setattr(
        "adata_hf_datasets.pp.orchestrator.maybe_add_sra_metadata", fake_sra
    )

    # Dummy concat to copy first chunk file to outfile
    def fake_concat(in_files, out_file):
        import shutil

        shutil.copy(in_files[0], out_file)

    monkeypatch.setattr("anndata.experimental.concat_on_disk", fake_concat)

    # Run with single chunk (chunk_size larger than n_obs)
    preprocess_h5ad(
        infile,
        outdir,
        chunk_size=10,
        min_cells=0,
        min_genes=0,
        batch_key="batch",
        count_layer_key="counts",
        n_top_genes=1,
        geneformer_pp=False,
        sra_chunk_size=1,
        sra_extra_cols=["foo"],
        instrument_key="inst",
        description_key="descr",
        output_format="h5ad",
    )

    # Check that outfile exists and content is as expected
    result = sc.read_h5ad(outdir / "chunk_0.h5ad")
    # Original obs keys plus 'foo' and modified 'descr'
    assert "foo" in result.obs.columns
    assert "This measurement was conducted with" in result.obs["descr"].values[0]
    # Ensure counts layer was created
    assert "counts" in result.layers
