import urllib.request
import urllib.error
import zipfile
import tempfile
import os

GIT_URL = "https://github.com/name-ethnicity-classifier/name-ethnicity-classifier/releases/download/zipped-models"
CACHE_PATH = os.path.expanduser("~/.cache/n2e")

def download_zip(model_name: str) -> None:
    """
    Handles retrieving the .zip files in the github releases url of a model 
    Unzips them into the dest dir

    :param model_name: the name of the model to be downloaded
    :param dest: the path of the dir holding the extracted files, defaults to ~/.cache/n2e
    """

    url = f"{GIT_URL}/{model_name}.zip"

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "model.zip")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except urllib.error.HTTPError:
            raise FileNotFoundError(f"No model named '{model_name}' exists.")
        
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