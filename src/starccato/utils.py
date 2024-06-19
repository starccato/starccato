import os
import random
import time
from typing import Optional

import numpy as np
import torch

from .defaults import (
    DEVICE,
    GENERATOR_WEIGHTS_FN,
    NZ,
    get_default_weights_path,
)
from .logger import logger
from .nn import Generator, load_model


def _load_generator(weights_file: str = None) -> Generator:
    """This function loads the generator model from the weights file.
    See https://pytorch.org/tutorials/beginner/saving_loading_models.html for more details.
    """
    if weights_file is None or weights_file == GENERATOR_WEIGHTS_FN:
        weights_file = get_default_weights_path()
    last_updated = time.ctime(os.path.getmtime(weights_file))
    logger.debug(
        f"Loading generator model from {weights_file} (last updated: {last_updated})."
    )
    return load_model(Generator, weights_file)


def generate_signals(
    n: Optional[int] = 1,
    weights_file: Optional[str] = None,
    seed: Optional[int] = None,
    filename: Optional[str] = None,
    latent_vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """This function generates signals using the trained generator model.

    The signals are generated with fs=4096Hz


    Args:
        n: The number of signals to generate.
        weights_file: The path to the weights file for the generator model.
        seed: The random seed to use for generating the signals. Random if None.
        filename: The name of the txt file to save the generated signals to.
        latent_vector: The latent vector to use for generating the signals.

    Returns:
        The generated signals as a numpy array.
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if latent_vector is not None:
        n, nz = latent_vector.shape
        assert nz == NZ, f"Expected latent vector of size {NZ}, got {nz}."
        latent_vector = torch.tensor(latent_vector, device=DEVICE).unsqueeze(
            -1
        )
    else:
        latent_vector = torch.randn((n, NZ, 1), device=DEVICE)

    signal_generator = _load_generator(weights_file)

    t0 = time.time()
    with torch.no_grad():
        signals = signal_generator(latent_vector).detach().numpy()
        signals = signals[:, 0, :]  # from (n, 1, nz) to (n, nz)
    t = time.time() - t0
    msg = f"Generated {n} signals [{t:.2f}s]."

    if filename:
        np.savetxt(filename, signals)
        msg += f" Saved to {filename}."

    logger.info(msg)
    return signals
