# tests/test_uploader.py
import json
import os

import pytest

import adata_hf_datasets.file_utils as uploader


@pytest.fixture
def tmp_data(tmp_path):
    """Create a dummy data folder with a single *.h5ad* file."""
    (tmp_path / "a.h5ad").write_bytes(b"dummy-bytes-not-real-hdf5")
    return tmp_path


@pytest.fixture
def mock_nextcloud_config():
    """Create a mock nextcloud config for testing."""
    return {
        "url": "MOCK_NEXTCLOUD_URL",
        "username": "MOCK_NEXTCLOUD_USERNAME",
        "password": "MOCK_NEXTCLOUD_PASSWORD",
    }


def test_first_run_uploads_everything(monkeypatch, tmp_data, mock_nextcloud_config):
    uploaded_files = []

    def mock_getenv(key):
        if key == "MOCK_NEXTCLOUD_URL":
            return "https://mock-nextcloud.example.com"
        elif key == "MOCK_NEXTCLOUD_USERNAME":
            return "testuser"
        elif key == "MOCK_NEXTCLOUD_PASSWORD":
            return "testpass"
        return None

    def mock_session_put(self, url, **kwargs):
        # Mock successful upload
        uploaded_files.append(url)
        response = type("MockResponse", (), {})()
        response.ok = True
        response.status_code = 200
        return response

    def mock_session_request(self, method, url, **kwargs):
        # Mock directory creation
        response = type("MockResponse", (), {})()
        response.status_code = 201  # Created
        return response

    def mock_get_share_link(nc_url, username, password, remote_path):
        return f"https://mock-nextcloud.example.com/s/mock-token-{remote_path.replace('/', '-')}"

    # Mock environment variables
    monkeypatch.setattr(os, "getenv", mock_getenv)

    # Mock requests Session methods
    monkeypatch.setattr("requests.Session.put", mock_session_put)
    monkeypatch.setattr("requests.Session.request", mock_session_request)

    # Mock share link creation
    monkeypatch.setattr(uploader, "get_share_link", mock_get_share_link)

    # Mock verify_share_link to return False (so upload is triggered)
    monkeypatch.setattr(uploader, "verify_share_link", lambda url, *_: False)

    # Run the upload
    share_map = uploader.upload_folder_to_nextcloud(tmp_data, mock_nextcloud_config)

    # Check that the share map was created correctly
    assert "a.h5ad.zip" in share_map
    assert share_map["a.h5ad.zip"].startswith("https://mock-nextcloud.example.com/s/")

    # Check that upload was attempted
    assert len(uploaded_files) > 0


def test_stored_link_is_respected(monkeypatch, tmp_data, mock_nextcloud_config):
    # Pretend we already have a mapping
    share_map = {"a.h5ad.zip": "https://nc/data/a.h5ad.zip"}
    (tmp_data / "share_map.json").write_text(json.dumps(share_map))

    uploaded_files = []

    def mock_session_put(self, url, **kwargs):
        # This should NOT be called since link is valid
        uploaded_files.append(url)
        response = type("MockResponse", (), {})()
        response.ok = True
        response.status_code = 200
        return response

    def mock_getenv(key):
        if key == "MOCK_NEXTCLOUD_URL":
            return "https://mock-nextcloud.example.com"
        elif key == "MOCK_NEXTCLOUD_USERNAME":
            return "testuser"
        elif key == "MOCK_NEXTCLOUD_PASSWORD":
            return "testpass"
        return None

    # Mock environment variables
    monkeypatch.setattr(os, "getenv", mock_getenv)

    # Mock requests Session methods
    monkeypatch.setattr("requests.Session.put", mock_session_put)

    # Mock verify_share_link to return True (link is valid)
    monkeypatch.setattr(uploader, "verify_share_link", lambda *_: True)

    # Run the upload
    result_map = uploader.upload_folder_to_nextcloud(tmp_data, mock_nextcloud_config)

    # File should be unchanged since link is valid
    after = json.loads((tmp_data / "share_map.json").read_text())
    assert after == share_map

    # No uploads should have been attempted
    assert len(uploaded_files) == 0


def test_broken_link_triggers_reupload(monkeypatch, tmp_data, mock_nextcloud_config):
    share_map = {"a.h5ad.zip": "https://nc/broken"}
    (tmp_data / "share_map.json").write_text(json.dumps(share_map))

    uploaded_files = []

    def mock_getenv(key):
        if key == "MOCK_NEXTCLOUD_URL":
            return "https://mock-nextcloud.example.com"
        elif key == "MOCK_NEXTCLOUD_USERNAME":
            return "testuser"
        elif key == "MOCK_NEXTCLOUD_PASSWORD":
            return "testpass"
        return None

    def mock_session_put(self, url, **kwargs):
        # Mock successful upload
        uploaded_files.append(url)
        response = type("MockResponse", (), {})()
        response.ok = True
        response.status_code = 200
        return response

    def mock_session_request(self, method, url, **kwargs):
        # Mock directory creation
        response = type("MockResponse", (), {})()
        response.status_code = 201  # Created
        return response

    def mock_get_share_link(nc_url, username, password, remote_path):
        return f"https://mock-nextcloud.example.com/s/new-token-{remote_path.replace('/', '-')}"

    # Mock environment variables
    monkeypatch.setattr(os, "getenv", mock_getenv)

    # Mock requests Session methods
    monkeypatch.setattr("requests.Session.put", mock_session_put)
    monkeypatch.setattr("requests.Session.request", mock_session_request)

    # Mock share link creation
    monkeypatch.setattr(uploader, "get_share_link", mock_get_share_link)

    # Mock verify_share_link to return False (broken link)
    monkeypatch.setattr(uploader, "verify_share_link", lambda *_: False)

    # Run the upload
    uploader.upload_folder_to_nextcloud(tmp_data, mock_nextcloud_config)

    # Check that new link was created
    after = json.loads((tmp_data / "share_map.json").read_text())
    assert after["a.h5ad.zip"] != share_map["a.h5ad.zip"]  # Should be different
    assert after["a.h5ad.zip"].startswith("https://mock-nextcloud.example.com/s/")

    # Upload should have been attempted
    assert len(uploaded_files) > 0
