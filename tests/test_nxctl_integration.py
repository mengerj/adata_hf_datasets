import os
from pathlib import Path
import uuid
import pytest
import requests
from requests.auth import HTTPBasicAuth

from adata_hf_datasets.file_utils import upload_folder_to_nextcloud

# Environment variable names for Nextcloud credentials
NC_URL_ENV = "NEXTCLOUD_URL"
NC_USER_ENV = "NEXTCLOUD_USER"
NC_PWD_ENV = "NEXTCLOUD_PASSWORD"


def _has_nextcloud_creds():
    """Check if all required Nextcloud credentials are available as environment variables."""
    return all(
        [
            os.getenv(NC_URL_ENV),
            os.getenv(NC_USER_ENV),
            os.getenv(NC_PWD_ENV),
        ]
    )


needs_creds = pytest.mark.skipif(
    not _has_nextcloud_creds(),
    reason="Nextcloud credentials not available (missing environment variables)",
)


@needs_creds
def test_real_upload(tmp_path: Path):
    # create one tiny file
    p = tmp_path / "hello.txt"
    p.write_text("hi")

    remote_root = f"tmp/pytest-{uuid.uuid4()}"
    share_map = upload_folder_to_nextcloud(
        tmp_path,
        nextcloud_config={
            "url": NC_URL_ENV,  # Function expects env var name, resolves internally
            "username": NC_USER_ENV,
            "password": NC_PWD_ENV,
            "remote_path": "",  # overwritten inside helper
            "progress": True,
        },
    )

    # check share link works
    link = next(iter(share_map.values()))
    r = requests.get(link + "/download", allow_redirects=True)
    assert r.ok and r.text.strip() == "hi"

    # cleanup remote artefacts
    auth = HTTPBasicAuth(os.getenv(NC_USER_ENV), os.getenv(NC_PWD_ENV))
    requests.request(
        "DELETE",
        f"{os.getenv(NC_URL_ENV).rstrip('/')}/remote.php/dav/files/{os.getenv(NC_USER_ENV)}/{remote_root}",
        auth=auth,
    )
