import torch.nn as nn

from ...defaults import NC, NDF, NZ


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
        )

        self.fc = nn.Sequential(nn.LazyLinear(1), nn.Sigmoid())

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = self.fc(x)
        return x
