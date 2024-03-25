import torch
import torch.nn as nn

from ...defaults import NC, NGF, NZ

__all__ = ["Generator"]


TRUE = 1
FALSE = 0


class Generator(nn.Module):
    def __init__(self, nz: int = NZ, ngf: int = NGF, nc: int = NC):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(
                nz, ngf * 32, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm1d(ngf * 32),
            nn.LeakyReLU(TRUE),
            nn.ConvTranspose1d(
                ngf * 32,
                ngf * 16,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 16),
            nn.LeakyReLU(TRUE),
            nn.ConvTranspose1d(
                ngf * 16,
                ngf * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 8),
            nn.LeakyReLU(TRUE),
            nn.ConvTranspose1d(
                ngf * 8,
                ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 4),
            nn.LeakyReLU(TRUE),
            nn.ConvTranspose1d(
                ngf * 4,
                ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(ngf * 2),
            nn.LeakyReLU(TRUE),
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
