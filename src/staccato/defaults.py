"""Defaults for the staccato package."""
import os
from .utils import __download

NZ = 100
NC = 1
NGF = 64
NDF = 64
GENERATOR_WEIGHTS_FN = os.path.join(os.path.dirname(__file__), "default_weights.pt")
BATCH_SIZE = 32

ZENODO_URL = "https://sandbox.zenodo.org/records/38501/files/stellar_core_collapse_signal_generator_dcgans_normalised.pt?download=1"


def get_default_weights_path():
    if not os.path.exists(GENERATOR_WEIGHTS_FN):
        __download(ZENODO_URL, GENERATOR_WEIGHTS_FN, msg="Downloading default weights...")
    return GENERATOR_WEIGHTS_FN
