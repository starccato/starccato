import argparse
import math
import os
import random
import time

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils

from scipy import signal
from scipy.stats import entropy

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from .defaults import NC, NGF, NZ

__all__ = ["Generator"]


class Generator(nn.Module):
    def __init__(self, nz: int = NZ, ngf: int = NGF, nc: int = NC):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(
                nz, ngf * 32, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm1d(ngf * 32),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(
                ngf * 32,
                ngf * 16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 16),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(
                ngf * 16,
                ngf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 8),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(
                ngf * 8,
                ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 4),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(
                ngf * 4,
                ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 2),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(
                ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(ngf),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(
                ngf, nc, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )

    def forward(self, z):
        z = self.main(z)
        return z




def generate_signals(
        weights_file: str,
        n: int,
        filename: str = None,
):
    """This function generates signals using the trained generator model."""

