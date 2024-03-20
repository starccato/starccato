import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..nn import Discriminator, Generator
from ..utils import init_weights, get_device
from .training_data import TrainingData
from ..plotting import plot_signals_from_latent_vector, plot_gradients, plot_loss

from tqdm.auto import trange, tqdm
import time
from torch import nn, optim
from typing import List, NamedTuple
from ..logger import logger

from torch.optim import lr_scheduler
from ..defaults import NC, NGF, NZ, BATCH_SIZE, NDF

from collections import namedtuple


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    return seed


TRAIN_METADATA = namedtuple("TrainMetadata", "epoch g_loss d_loss g_gradient d_gradient")


class Trainer:
    def __init__(
            self,
            nz: int = NZ,
            nc: int = NC,
            ngf: int = NGF,
            ndf: int = NDF,
            seed: int = 99,
            batch_size: int = BATCH_SIZE,
            num_epochs=128,
            lr_g=0.00002,
            lr_d=0.00002,
            beta1=0.5,
            outdir: str = "outdir"
    ):
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.beta1 = beta1
        self.outdir = outdir
        self.device = get_device()
        self.dataset = TrainingData(batch_size=batch_size)

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup networks
        self.netG = Generator(nz=nz, ngf=ngf, nc=nc).to(self.device)
        self.netG.apply(init_weights)
        self.netD = Discriminator(nz=nz, ndf=ndf, nc=nc).to(self.device)
        self.netD.apply(init_weights)

        # setup optimisers
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr_g, betas=(self.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999))
        sched_kwargs = dict(start_factor=1.0, end_factor=0.5, total_iters=32)
        self.schedulerG = lr_scheduler.LinearLR(self.optimizerG, **sched_kwargs)
        self.schedulerD = lr_scheduler.LinearLR(self.optimizerD, **sched_kwargs)
        self.criterion = nn.BCELoss()

        # cache a latent-vector for visualisation/testing
        self.fixed_noise = torch.randn(batch_size, nz, 1, device=self.device)

        # Lists to keep track of progress
        self.train_metadata: List[TRAIN_METADATA] = []

    @property
    def plt_kwgs(self):
        return dict(
            scaling_factor=self.dataset.scaling_factor, mean=self.dataset.mean, std=self.dataset.std,
            num_cols=4, num_rows=4
        )

    def plot_signals(self, label):
        plot_signals_from_latent_vector(
            self.netG, self.fixed_noise,
            f"{self.outdir}/{label}.png",
            **self.plt_kwgs
        )

    def train(self):
        self.plot_signals("before_training")
        t0 = time.time()
        logger.info(
            f"Starting Training Loop "
            f"[Epochs: {self.num_epochs}, "
            f"Train Size: {self.dataset.shape}, "
            f"Learning Rate: ({self.lr_g}, {self.lr_d})]"
        )

        dataloader = self.dataset.get_loader()
        # For each epoch
        epoch_str = "Loss_D: %.4f, Loss_G: %.4f, Epochs:"
        epoch_bar = trange(self.num_epochs, desc=epoch_str.format(np.nan.np.nan), position=0, leave=True)
        for epoch in epoch_bar:
            for (i, data) in tqdm(enumerate(dataloader, 0), desc="Batch", position=1, leave=False,
                                  total=len(dataloader)):
                errD, D_x, D_G_z1, _dgrad, fake, b_size = self._update_discriminator(data)
                errG, D_G_z2, _ggrad = self._update_generator(b_size, fake)
                if i % 50 == 0:
                    epoch_bar.set_description(epoch_str % (errD.item(), errG.item()))
                self.train_metadata.append(TRAIN_METADATA(epoch, errG.item(), errD.item(), _ggrad, _dgrad))

            # learning-rate decay
            self._decay_learning_rate(self.optimizerD, self.schedulerD, "Discriminator")
            self._decay_learning_rate(self.optimizerG, self.schedulerG, "Generator")

            # intermediate plot
            self.plot_signals(f"signals_epoch_{epoch}")

        runtime = (time.time() - t0) / 60
        logger.info(f"Training Time: {runtime:.2f}min")


        #
        # plot_gradients(D_gradients, "tab:red", "Discriminator", f"{self.outdir}/discriminator_gradients.png")
        # plot_gradients(G_gradients, "tab:blue", "Generator", f"{self.outdir}/generator_gradients.png")
        # plot_loss(G_losses, D_losses, f"{self.outdir}/losses.png")
        generator_weights_fn = f"{self.outdir}/generator_weights.pt"

        torch.save(self.netG.state_dict(), generator_weights_fn)
        logger.info(f"Saved generator weights+state to {generator_weights_fn}")
        # return netG, netD

    def _update_discriminator(self, data):
        """Update D network: maximize log(D(x)) + log(1 - D(G(z)))"""
        ## Train with all-real batch
        self.netD.zero_grad()
        # Format batch
        real_gpu = data.to(self.device)
        b_size = real_gpu.size(0)
        label_real = torch.FloatTensor(b_size).uniform_(1.0, 1.0).to(self.device)
        # Forward pass real batch through D
        output = self.netD(real_gpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label_real)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.nz, 1, device=self.device)
        # Generate fake signal batch with G
        fake = self.netG(noise)
        label_fake = torch.FloatTensor(b_size).uniform_(0.0, 0.25).to(self.device)
        # label_fake = torch.FloatTensor(b_size).uniform_(0.0, 0.0).to(device)
        # Classify all fake batch with D
        output = self.netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.criterion(output, label_fake)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizerD.step()
        # Calculate gradients of discriminator parameters
        _dgrad = [param.grad.norm().item() for param in self.netD.parameters()]
        return errD, D_x, D_G_z1, _dgrad, fake, b_size

    def _update_generator(self, b_size, fake):
        """Update G network: maximize log(D(G(z)))"""
        self.netG.zero_grad()
        label_real = torch.FloatTensor(b_size).uniform_(1.0, 1.0).to(self.device)
        # label_real = 1.0 - label_fake
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.criterion(output, label_real)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        self.optimizerG.step()
        # Calculate gradients of generator parameters
        _ggrad = [param.grad.norm().item() for param in self.netG.parameters()]
        return errG, D_G_z2, _ggrad

    def _decay_learning_rate(self, optimizer, scheduler, label):
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"SGD {label} lr {before_lr:.7f} -> {after_lr:.7f}")
        return after_lr
