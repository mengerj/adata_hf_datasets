from tqdm import tqdm


def download_from_link(url, save_path):
    """
    Download a file with a progress bar.

    Parameters
    ----------
    url : str
        The direct URL to the file.
    save_path : str
        The local file path to save the downloaded file.

    Returns
    -------
    bool
        True if the download was successful, False otherwise.
    """
    try:
        # Use urllib directly to avoid any requests_cache issues
        import urllib.request
        import urllib.error

        # Download with progress bar using urllib
        with tqdm(
            desc=save_path,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:

            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    bar.total = total_size
                bar.update(block_size)

            urllib.request.urlretrieve(url, save_path, progress_hook)

        print(f"\nDownload complete: {save_path}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"\nDownload failed: {e}")
        return False
    except Exception as e:
        print(f"\nDownload failed with unexpected error: {e}")
        return False


if __name__ == "__main__":
    download_from_link(
        "https://figshare.com/ndownloader/files/24539828",
        "human_pancreas.h5ad",
    )
