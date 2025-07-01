# tests/test_scvi_stale_handle.py
import multiprocessing as mp
import numpy as np
import scanpy as sc
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
    import shutil

    # patch shutil.copy2 so accessing shared_pt raises stale file handle error
    orig_copy2 = shutil.copy2
    orig_copyfile = shutil.copyfile

    call_count = {"copy2": 0, "copyfile": 0}

    def flaky_copy2(src, dst, *a, **kw):
        call_count["copy2"] += 1
        if str(src) == shared_pt and call_count["copy2"] <= 2:  # Fail first 2 attempts
            raise OSError(116, "Stale file handle")
        return orig_copy2(src, dst, *a, **kw)

    def flaky_copyfile(src, dst, *a, **kw):
        call_count["copyfile"] += 1
        if str(src) == shared_pt and call_count["copyfile"] <= 1:  # Fail first attempt
            raise OSError(116, "Stale file handle")
        return orig_copyfile(src, dst, *a, **kw)

    shutil.copy2 = flaky_copy2
    shutil.copyfile = flaky_copyfile

    try:
        hub = _build_local_hub(pathlib.Path(model_dir))
        emb = SCVIEmbedder()
        adata = sc.read_h5ad(pathlib.Path(model_dir) / "adata.h5ad")
        emb._setup_model_with_ref(hub, adata)  # <- should succeed with retry logic
        q.put(None)  # success sentinel
    except Exception:
        q.put(traceback.format_exc())
        sys.exit(1)
    finally:
        # Restore original functions
        shutil.copy2 = orig_copy2
        shutil.copyfile = orig_copyfile


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


def test_robust_copy_file_with_retries(tmp_path):
    """Test the robust file copying mechanism with simulated NFS errors."""
    import shutil
    from adata_hf_datasets.initial_embedder import SCVIEmbedder

    # Create a test file
    src_file = tmp_path / "test_source.txt"
    dst_file = tmp_path / "test_dest.txt"
    test_content = "Test file content for robust copying"
    src_file.write_text(test_content)

    # Test 1: Normal copy should work
    SCVIEmbedder._robust_copy_file(src_file, dst_file)
    assert dst_file.exists()
    assert dst_file.read_text() == test_content
    dst_file.unlink()  # Clean up

    # Test 2: Simulate stale file handle errors with retry success
    orig_copy2 = shutil.copy2
    orig_copyfile = shutil.copyfile

    call_count = {"copy2": 0, "copyfile": 0}

    def flaky_copy2(src, dst, *a, **kw):
        call_count["copy2"] += 1
        if call_count["copy2"] <= 2:  # Fail first 2 attempts
            raise OSError(116, "Stale file handle")
        return orig_copy2(src, dst, *a, **kw)

    def flaky_copyfile(src, dst, *a, **kw):
        call_count["copyfile"] += 1
        if call_count["copyfile"] <= 1:  # Fail first attempt
            raise OSError(116, "Stale file handle")
        return orig_copyfile(src, dst, *a, **kw)

    try:
        shutil.copy2 = flaky_copy2
        shutil.copyfile = flaky_copyfile

        # This should succeed after retries
        SCVIEmbedder._robust_copy_file(
            src_file, dst_file, max_retries=5, base_delay=0.1
        )
        assert dst_file.exists()
        assert dst_file.read_text() == test_content

        # Verify that retries were attempted
        # copy2 should have been called once (attempt 0) and failed
        assert call_count["copy2"] == 1
        # copyfile should have been called once (attempt 1) and failed
        assert call_count["copyfile"] == 1
        # The third attempt should have used chunked copy and succeeded

    finally:
        shutil.copy2 = orig_copy2
        shutil.copyfile = orig_copyfile


def test_robust_copy_file_max_retries_exceeded(tmp_path):
    """Test that robust copying fails after max retries are exceeded."""
    import shutil
    from adata_hf_datasets.initial_embedder import SCVIEmbedder

    # Create a test file
    src_file = tmp_path / "test_source.txt"
    dst_file = tmp_path / "test_dest.txt"
    src_file.write_text("Test content")

    # Mock all copy methods to always fail with stale file handle
    orig_copy2 = shutil.copy2
    orig_copyfile = shutil.copyfile

    def always_fail_copy2(src, dst, *a, **kw):
        raise OSError(116, "Persistent stale file handle")

    def always_fail_copyfile(src, dst, *a, **kw):
        raise OSError(116, "Persistent stale file handle")

    try:
        shutil.copy2 = always_fail_copy2
        shutil.copyfile = always_fail_copyfile

        # Mock the chunked copy to also fail
        orig_chunked_copy = SCVIEmbedder._chunked_copy

        def always_fail_chunked(src, dst, chunk_size=64 * 1024):
            raise OSError(116, "Persistent stale file handle")

        SCVIEmbedder._chunked_copy = always_fail_chunked

        # This should fail after max retries
        with pytest.raises(OSError) as exc_info:
            SCVIEmbedder._robust_copy_file(
                src_file, dst_file, max_retries=3, base_delay=0.1
            )

        assert exc_info.value.errno == 116
        assert "Persistent stale file handle" in str(exc_info.value)

        # Restore chunked copy
        SCVIEmbedder._chunked_copy = orig_chunked_copy

    finally:
        shutil.copy2 = orig_copy2
        shutil.copyfile = orig_copyfile


def test_chunked_copy(tmp_path):
    """Test the chunked copy mechanism."""
    from adata_hf_datasets.initial_embedder import SCVIEmbedder

    # Create a test file with some content
    src_file = tmp_path / "test_source.txt"
    dst_file = tmp_path / "test_dest.txt"
    test_content = "This is a test file for chunked copying mechanism." * 1000
    src_file.write_text(test_content)

    # Test chunked copy
    SCVIEmbedder._chunked_copy(src_file, dst_file, chunk_size=64)

    assert dst_file.exists()
    assert dst_file.read_text() == test_content
