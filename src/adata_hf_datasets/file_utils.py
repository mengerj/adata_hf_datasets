import h5py
import os
import logging
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET


def save_and_upload_adata(
    adata, local_path, nextcloud_config=None, create_share_link=True
):
    """
    Saves an AnnData object to a file and optionally uploads it to a Nextcloud server based on provided configuration.

    Parameters:
        adata (AnnData): The AnnData object to save.
        local_path (str): Local path to save the .h5ad file.
        nextcloud_config (dict, optional): Configuration dictionary for Nextcloud which contains:
                                           'url' (str): URL to the Nextcloud server.
                                           'username' (str): Username for Nextcloud.
                                           'password' (str): Password for Nextcloud.
                                           'remote_path' (str): Remote path in Nextcloud where the file will be uploaded.

    Example:
        adata = AnnData(np.random.rand(10, 10))  # Example AnnData object
        save_and_upload_adata(adata, 'local_file.h5ad', nextcloud_config={
                              'url': 'https://nxc-fredato.imbi.uni-freiburg.de',
                              'username': 'your_username',
                              'password': 'your_password',
                              'remote_path': '/path/on/nextcloud/file.h5ad.gz'})
    """
    # Save the AnnData object to a local .h5ad file
    adata.write(local_path, compression="gzip")
    logging.info(f"File saved locally at {local_path}")

    # Upload the file to Nextcloud if configuration is provided
    if nextcloud_config:
        try:
            create_nested_directories(
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )

            response = upload_file_to_nextcloud(
                local_path,
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            logging.info(
                f"File uploaded to Nextcloud at {nextcloud_config['remote_path']} with status code {response.status_code}"
            )
        except Exception as e:
            logging.error(f"Failed to upload file to Nextcloud: {e}")
        if create_share_link:
            share_url = get_share_link(
                nextcloud_config["url"],
                os.getenv(nextcloud_config["username"]),
                os.getenv(nextcloud_config["password"]),
                nextcloud_config["remote_path"],
            )
            return share_url


def upload_file_to_nextcloud(file_path, nextcloud_url, username, password, remote_path):
    """
    Uploads a file to a Nextcloud server via WebDAV.

    Parameters:
        file_path (str): Path to the local file to upload.
        nextcloud_url (str): URL to your personal nextcloud instance. Can be found in the settings of your nextcloud account. (File Seetings -> WebDAV)
        username (str): Username for Nextcloud authentication.
        password (str): Password for Nextcloud authentication.
        remote_path (str): Path in Nextcloud where the file will be stored.

    Returns:
        requests.Response: The response object from the requests library.

    Example:
        response = upload_file_to_nextcloud('path/to/local/file.txt',
                                            'https://nxc-fredato.imbi.uni-freiburg.de',
                                            'your_username',
                                            'your_password',
                                            '/path/to/save/file.txt')
        print(response.status_code)
    """
    # Complete URL to access the WebDAV interface
    full_url = f"{nextcloud_url}/remote.php/dav/files/{username}/{remote_path}"

    # Open the file in binary mode
    with open(file_path, "rb") as file_content:
        # Make the PUT request to upload the file
        response = requests.put(
            full_url, data=file_content, auth=HTTPBasicAuth(username, password)
        )
    return response


def create_nested_directories(nextcloud_url, username, password, remote_path):
    """
    Creates nested directories in Nextcloud one by one.

    Parameters:
        nextcloud_url (str): URL to the Nextcloud server.
        username (str): Username for Nextcloud.
        password (str): Password for Nextcloud.
        remote_path (str): Remote directory path where directories need to be created.
    """
    base_url = f"{nextcloud_url}/remote.php/dav/files/{username}"
    segments = remote_path.strip("/").split("/")
    path = ""
    for segment in segments[:-1]:  # Exclude the last segment assuming it's a file
        path += f"/{segment}"
        folder_url = f"{base_url}{path}"
        response = requests.request(
            "MKCOL", folder_url, auth=HTTPBasicAuth(username, password)
        )
        if response.status_code == 201:
            print(f"Created directory: {segment}")
        elif response.status_code in [
            405,
            301,
        ]:  # Directory already exists or moved permanently
            print(f"Directory already exists: {segment}")
        else:
            print(
                f"Error {response.status_code} creating directory {segment}: {response.reason}"
            )
            return False
    return True


def get_share_link(nextcloud_url, username, password, remote_path):
    """
    Creates a public share link for a file on Nextcloud.

    Parameters:
        nextcloud_url (str): URL to the Nextcloud server.
        username (str): Username for Nextcloud authentication.
        password (str): Password for Nextcloud authentication.
        remote_path (str): Path in Nextcloud where the file is stored, starting from the root of the user's files directory.

    Returns:
        str: URL of the shared file if successful, None otherwise.
    """
    api_url = f"{nextcloud_url}/ocs/v2.php/apps/files_sharing/api/v1/shares"
    data = {
        "shareType": 3,  # This specifies a public link
        "path": remote_path,
        "permissions": 1,  # Read-only permissions
    }
    headers = {
        "OCS-APIRequest": "true",
        "Accept": "application/xml",  # Set header to accept XML
    }

    response = requests.post(
        api_url, data=data, headers=headers, auth=HTTPBasicAuth(username, password)
    )

    if response.status_code == 200:
        try:
            # Parse the XML response
            root = ET.fromstring(response.text)
            token = root.find(".//token").text
            share_url = (
                f"{nextcloud_url}/s/{token}/download"  # Ensure direct file download
            )
            return share_url
        except ET.ParseError:
            print("Failed to parse the XML data.")
            return None
        except AttributeError:
            print("Failed to find the required elements in the XML.")
            return None
    else:
        print(f"Failed to create share link: {response.status_code}")
        print("Response details:", response.text)
        return None


def download_file_from_share_link(share_link, save_path):
    """
    Downloads a file from a Nextcloud share link and checks if it's a valid .h5ad file.

    Parameters:
        share_link (str): The full share link URL to the file.
        save_path (str): The local path where the file should be saved.

    Returns:
        bool: True if the download was successful and the file is a valid .h5ad, False otherwise.

    Example:
        success = download_file_from_share_link('https://nxc-fredato.imbi.uni-freiburg.de/s/Zs6pAa8P5ynDTiP',
                                                'path/to/save/file.h5ad')
        print("Download successful:", success)
    """
    # Send a GET request to the share link
    response = requests.get(share_link)
    if response.status_code == 200:
        # Write the content of the response to a local file
        with open(save_path, "wb") as file:
            file.write(response.content)

        # Attempt to open the file with h5py
        try:
            with h5py.File(save_path, "r") as h5_file:
                # Check for key 'X' which is common in .h5ad files
                if "X" in h5_file:
                    print("File is a valid .h5ad file.")
                    return True
                else:
                    print("File does not appear to be a valid .h5ad file.")
        except Exception as e:
            print(f"Error while checking the file: {e}")

        # If the checks fail, consider the download unsuccessful
        return False
    else:
        # Print or log the error
        print(
            f"Failed to download the file: {response.status_code} - {response.reason}"
        )
        return False
