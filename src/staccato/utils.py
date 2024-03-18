import torch
from tqdm.auto import tqdm
from .defaults import GENERATOR_WEIGHTS_FN
import requests
from tqdm.auto import tqdm
from urllib.parse import urlparse
from pathlib import Path

ZENODO_URL = "https://zenodo.org/record/1234567/files/weights.h5"


def download_default_generator_weights() -> None:
    """This function downloads the weights for the generator model from Zenodo."""
    __download(ZENODO_URL, GENERATOR_WEIGHTS_FN)


def __download(url: str, fname: str) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading",
                        dynamic_ncols=True)
    with open(fname, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception(f"Download failed: expected {total_size_in_bytes} bytes, got {progress_bar.n} bytes")


def get_device() -> torch.device:
    """This function returns the device to use for training/predicting."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
