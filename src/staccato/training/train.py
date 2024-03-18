import numpy as np
import torch

from .discriminator import Discriminator
from .generator import Generator


def __set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    return seed


def train(seed: int = 99):
    __set_seed(seed)

    pass
