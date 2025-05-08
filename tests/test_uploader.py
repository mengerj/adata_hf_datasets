# tests/test_uploader.py
import json
from pathlib import Path

import pytest

import adata_hf_datasets.file_utils as uploader


@pytest.fixture
def tmp_data(tmp_path):
    """Create a dummy data folder with a single *.h5ad* file."""
    (tmp_path / "a.h5ad").write_bytes(b"dummy-bytes-not-real-hdf5")
    return tmp_path


def test_first_run_uploads_everything(monkeypatch, tmp_data):
    uploaded = {}

    def fake_save_and_upload(local, cfg, *, create_share_link=True):
        uploaded[Path(local).name] = f"https://nc/{cfg['remote_path']}"
        return uploaded[Path(local).name]

    # No verify needed on first run
    monkeypatch.setattr(uploader, "save_and_upload_adata", fake_save_and_upload)
    monkeypatch.setattr(uploader, "verify_share_link", lambda url, *_: False)

    mapping = uploader.upload_folder_to_nextcloud(tmp_data, {"remote_path": ""})
    saved = json.loads(mapping.read_text("utf-8"))
    assert "a.h5ad" in saved
    assert uploaded["a.h5ad"] == saved["a.h5ad"]


def test_stored_link_is_respected(monkeypatch, tmp_data):
    # Pretend we already have a mapping
    share_map = {"a.h5ad": "https://nc/data/a.h5ad"}
    (tmp_data / "share_map.json").write_text(json.dumps(share_map))

    # save_and_upload_adata should NOT be called …
    monkeypatch.setattr(uploader, "save_and_upload_adata", lambda *a, **k: None)
    # … because the stored link verifies OK.
    monkeypatch.setattr(uploader, "verify_share_link", lambda *_: True)

    uploader.upload_folder_to_nextcloud(tmp_data, {"remote_path": ""})
    # file unchanged
    after = json.loads((tmp_data / "share_map.json").read_text())
    assert after == share_map


def test_broken_link_triggers_reupload(monkeypatch, tmp_data):
    share_map = {"a.h5ad": "https://nc/broken"}
    (tmp_data / "share_map.json").write_text(json.dumps(share_map))

    monkeypatch.setattr(uploader, "verify_share_link", lambda *_: False)

    new_link = "https://nc/new/a.h5ad"
    monkeypatch.setattr(
        uploader,
        "save_and_upload_adata",
        lambda *a, **k: new_link,
    )

    uploader.upload_folder_to_nextcloud(tmp_data, {"remote_path": ""})
    after = json.loads((tmp_data / "share_map.json").read_text())
    assert after["a.h5ad"] == new_link
