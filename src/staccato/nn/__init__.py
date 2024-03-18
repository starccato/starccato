from .generator import Generator
from .discriminator import Discriminator

from ..utils import get_device
from ..defaults import GENERATOR_WEIGHTS_FN
from typing import Optional
import random
import torch


def generate_signals(
        n: int = 1,
        weights_file: Optional[str] = GENERATOR_WEIGHTS_FN,
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

    device = get_device()
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(weights_file))

    signals = netG.generate(n)

    with open(filename, 'w') as f:
        for signal in signals:
            f.write(f"{signal}\n")
    print(f"Generated {n} signals and saved to {filename}")
