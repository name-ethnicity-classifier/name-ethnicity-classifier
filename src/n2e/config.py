import urllib.request
import urllib.error
import zipfile
import tempfile
import sys
import os

GIT_URL = "https://github.com/name-ethnicity-classifier/name-ethnicity-classifier/releases/download/zipped-models"
CACHE_PATH = os.path.expanduser("~/.cache/n2e")


def _download_progress(block_number: int, block_size: int, total_size: int) -> None:
    """
    Reports download progress on stderr, used as the reporthook of urlretrieve

    :param block_number: the amount of blocks transferred so far
    :param block_size: the size of a block in bytes
    :param total_size: the total size of the file in bytes, -1 if the server didn't report it
    """

    if total_size <= 0:
        return

    downloaded = min(block_number * block_size, total_size)
    percent = 100 * downloaded / total_size
    print(
        f"\r  {percent:5.1f}%  ({downloaded / 1e6:5.1f} / {total_size / 1e6:.1f} MB)",
        end="",
        file=sys.stderr,
        flush=True
    )


def download_zip(model_name: str) -> None:
    """
    Handles retrieving the .zip files in the github releases url of a model 
    Unzips them into the cache dir

    :param model_name: the name of the model to be downloaded
    """

    url = f"{GIT_URL}/{model_name}.zip"

    # progress is only rendered on a terminal, so logs and piped output stay clean
    show_progress = sys.stderr.isatty()
    print(f"Downloading model '{model_name}' (first use, cached in {CACHE_PATH})", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "model.zip")
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=_download_progress if show_progress else None)
        except urllib.error.HTTPError:
            raise FileNotFoundError(f"No model named '{model_name}' exists.")

        # close the progress line, which is only ever written on success
        if show_progress:
            print(file=sys.stderr)

        with zipfile.ZipFile(zip_path) as z:
            z.extractall(CACHE_PATH)


def get_model_folder(model_name: str) -> str:
    """
    Checks if model is downloaded, else downloads it from github releases url

    :param model_name: the name of the model to be checked
    :return: the path_name of the destination directory, defaults to ~/.cache/n2e/{model_name}
    """

    dest = os.path.join(CACHE_PATH, model_name)
    if not os.path.isdir(dest):
        download_zip(model_name)
    return dest
