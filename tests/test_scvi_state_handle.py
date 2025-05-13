# tests/test_scvi_stale_handle.py
import multiprocessing as mp
import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from pathlib import Path
import pytest
import scvi
from scvi.model import SCVI
from scvi.hub import HubModel
from scvi.hub._metadata import HubMetadata
from huggingface_hub import ModelCard
import sys
import traceback
import pathlib
from adata_hf_datasets.initial_embedder import SCVIEmbedder


def _train_tiny_scvi(model_dir: Path) -> None:
    """Train a 1-epoch tiny SCVI, save weights + adata."""
    adata = AnnData(X=np.random.poisson(1.0, size=(20, 10)))
    adata.obs["batch"] = "b0"
    SCVI.setup_anndata(adata, batch_key="batch")
    m = SCVI(adata, n_latent=2)
    m.train(max_epochs=1)
    m.save(model_dir, overwrite=True, save_anndata=True)


def _build_local_hub(model_dir: Path) -> HubModel:
    meta = HubMetadata(
        scvi_version=scvi.__version__,
        anndata_version=sc.__version__,
        model_cls_name="SCVI",
    )
    card = ModelCard("# dummy")
    return HubModel(model_dir, metadata=meta, model_card=card)


# -------- child worker -------------------------------------------------
def _worker(model_dir, shared_pt, q):
    import scanpy as sc

    # patch torch.load so accessing shared_pt raises
    orig_load = torch.load

    def flaky(path, *a, **kw):
        if str(path) == shared_pt:
            raise OSError(116, "Stale file handle")
        return orig_load(path, *a, **kw)

    torch.load = flaky

    try:
        hub = _build_local_hub(pathlib.Path(model_dir))
        emb = SCVIEmbedder()
        adata = sc.read_h5ad(pathlib.Path(model_dir) / "adata.h5ad")
        emb._setup_model_with_ref(hub, adata)  # <- triggers error
        q.put(None)  # success sentinel
    except Exception:
        q.put(traceback.format_exc())
        sys.exit(1)


def test_stale_handle_queue(tmp_path):
    model_dir = tmp_path / "tiny_scvi"
    _train_tiny_scvi(model_dir)
    shared_pt = str(model_dir / "model.pt")

    ctx = mp.get_context("spawn")  # macOS safe
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(str(model_dir), shared_pt, q))
    p.start()
    p.join()

    try:
        tb = q.get(timeout=1)  # wait up to 1 s for message
    except mp.queues.Empty:
        pytest.fail("Child process produced no queue message")

    if tb is not None:
        print("\n--- traceback from child ---\n", tb)
    assert tb is None  # no error in child process
