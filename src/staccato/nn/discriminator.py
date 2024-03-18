# critical
import argparse
import math
import os
import random
import time

# plotting
import matplotlib.pyplot as plt

# data
import numpy as np
import pandas as pd
import seaborn as sns

# deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as vutils

# signal processing
from scipy import signal
from scipy.stats import entropy

# machine learning
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from ..defaults import NC, NDF, NZ


class Discriminator(nn.Module):
    def __init__(self, nz: int = NZ, ndf: int = NDF, nc: int = NC):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.2),
            nn.Conv1d(
                ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.2),
            nn.Conv1d(
                ndf * 2,
                ndf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.2),
            nn.Conv1d(
                ndf * 4,
                ndf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(0.2),
            ### Can increase model complexity here ###
            # nn.Conv1d(ndf * 8, ndf * 16, kernel_size=4,
            #         stride=2, padding=1, bias=False),
            # nn.BatchNorm1d(ndf * 16),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.2),
            # nn.Conv1d(ndf * 16, ndf * 32, kernel_size=4,
            #         stride=2, padding=1, bias=False),
            # nn.BatchNorm1d(ndf * 32),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.2),
            # nn.Conv1d(ndf * 32, nc, kernel_size=4,
            #         stride=2, padding=0, bias=False),
            # nn.BatchNorm1d(ndf * 64),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout1d(0.2),
            # nn.Conv1d(ndf * 64, nc, kernel_size=4,
            #         stride=2, padding=0, bias=False)
        )

        self.fc = nn.Sequential(nn.LazyLinear(1), nn.Sigmoid())

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.fc(x)
        return x
