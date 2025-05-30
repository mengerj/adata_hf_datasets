import os
from pathlib import Path
import uuid
import pytest
import requests
from requests.auth import HTTPBasicAuth

from adata_hf_datasets.file_utils import upload_folder_to_nextcloud

# gets the env variables internally
NC_URL = "NEXTCLOUD_URL"
NC_USER = "NEXTCLOUD_USER"
NC_PWD = "NEXTCLOUD_PASSWORD"

needs_creds = pytest.mark.skipif(
    not all([NC_URL, NC_USER, NC_PWD]), reason="Nextcloud creds missing"
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
            "url": NC_URL,
            "username": NC_USER,
            "password": NC_PWD,
            "remote_path": "",  # overwritten inside helper
            "progress": True,
        },
    )

    # check share link works
    link = next(iter(share_map.values()))
    r = requests.get(link + "/download", allow_redirects=True)
    assert r.ok and r.text.strip() == "hi"

    # cleanup remote artefacts
    auth = HTTPBasicAuth(os.getenv(NC_USER), os.getenv(NC_PWD))
    requests.request(
        "DELETE",
        f"{os.getenv(NC_URL).rstrip('/')}/remote.php/dav/files/{os.getenv(NC_USER)}/{remote_root}",
        auth=auth,
    )
