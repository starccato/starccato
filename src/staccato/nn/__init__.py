from .generator import Generator
from .discriminator import Discriminator

from ..defaults import get_default_weights_path
from ..utils import get_device
from typing import Optional
import random
import torch


def _load_generator(weights_file: str = None) -> Generator:
    """This function loads the generator model from the weights file."""
    if weights_file is None:
        weights_file = get_default_weights_path()
    gen = torch.load(weights_file)
    return gen


def generate_signals(
        n: int = 1,
        weights_file: Optional[str] = None,
        seed: Optional[int] = None,
        filename: Optional[str] = 'signals.txt',
):
    """This function generates signals using the trained generator model.

    Args:
        n: The number of signals to generate.
        weights_file: The path to the weights file for the generator model.
        seed: The random seed to use for generating the signals. Random if None.
        filename: The name of the txt file to save the generated signals to.
    """

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    gen = _load_generator(weights_file)
    signals = gen.generate(n)

    with open(filename, 'w') as f:
        for signal in signals:
            f.write(f"{signal}\n")
    print(f"Generated {n} signals and saved to {filename}")
