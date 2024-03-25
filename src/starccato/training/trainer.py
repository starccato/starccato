import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm.auto import tqdm, trange

from ..defaults import BATCH_SIZE, DEVICE, NC, NDF, NGF, NZ
from ..logger import logger
from ..nn import Discriminator, Generator, save_model
from ..plotting import (
    plot_gradients,
    plot_loss,
    plot_signals_from_latent_vector,
)
from .training_data import TrainingData


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    return seed


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
        checkpoint_interval=16,
        outdir: str = "outdir",
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
        self.dataset = TrainingData(batch_size=batch_size)
        self.checkpoint_interval = checkpoint_interval

        os.makedirs(outdir, exist_ok=True)
        _set_seed(self.seed)

        # setup networks
        self.netG = Generator(nz=nz, ngf=ngf, nc=nc).to(DEVICE)
        self.netG.apply(_init_weights)
        self.netD = Discriminator(nz=nz, ndf=ndf, nc=nc).to(DEVICE)
        self.netD.apply(_init_weights)

        # setup optimisers
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.lr_g, betas=(self.beta1, 0.999)
        )
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=self.lr_d, betas=(self.beta1, 0.999)
        )
        sched_kwargs = dict(start_factor=1.0, end_factor=0.5, total_iters=32)
        self.schedulerG = lr_scheduler.LinearLR(
            self.optimizerG, **sched_kwargs
        )
        self.schedulerD = lr_scheduler.LinearLR(
            self.optimizerD, **sched_kwargs
        )
        self.criterion = nn.BCELoss()

        # cache a latent-vector for visualisation/testing
        self.fixed_noise = torch.randn(batch_size, nz, 1, device=DEVICE)

        # Lists to keep track of progress
        self.train_metadata: TrainMetadata = TrainMetadata()

    @property
    def plt_kwgs(self):
        return dict(
            scaling_factor=self.dataset.scaling_factor,
            mean=self.dataset.mean,
            std=self.dataset.std,
            num_cols=4,
            num_rows=4,
        )

    def plot_signals(self, label):
        plot_signals_from_latent_vector(
            self.netG,
            self.fixed_noise,
            f"{self.outdir}/{label}.png",
            **self.plt_kwgs,
        )

    def _prog_dict(self, loss_g, loss_d, lr_g, lr_d):
        return {
            "Loss(d,g)": f"[{loss_d:.2E}, {loss_g:.2E}]",
            "LR(d,g)": f"[{lr_d:.2E}, {lr_g:.2E}]",
        }

    def train(self):
        self.plot_signals("before_training")
        t0 = time.time()
        logger.info(
            f"\nStarting Training Loop "
            f"[Epochs: {self.num_epochs}, "
            f"Train Size: {self.dataset.shape}, "
            f"Learning Rate: ({self.lr_g}, {self.lr_d})]"
        )

        dataloader = self.dataset.get_loader()
        epoch_bar = trange(
            self.num_epochs, desc="Epochs", position=0, leave=True
        )
        epoch_bar.set_postfix(self._prog_dict(0, 0, self.lr_g, self.lr_d))
        for epoch in epoch_bar:
            for (i, data) in tqdm(
                enumerate(dataloader, 0),
                desc="Batch",
                position=1,
                leave=False,
                total=len(dataloader),
            ):
                (
                    errD,
                    D_x,
                    D_G_z1,
                    _dgrad,
                    fake,
                    b_size,
                ) = self._update_discriminator(data)
                errG, D_G_z2, _ggrad = self._update_generator(b_size, fake)
                if i % 50 == 0:
                    epoch_bar.set_postfix(
                        self._prog_dict(
                            errG.item(), errD.item(), self.lr_g, self.lr_d
                        )
                    )
                itr = epoch * len(dataloader) + i
                self.train_metadata.append(
                    itr, errG.item(), errD.item(), _ggrad, _dgrad
                )

            # learning-rate decay
            self._decay_learning_rate(
                self.optimizerD, self.schedulerD, "Discriminator"
            )
            self._decay_learning_rate(
                self.optimizerG, self.schedulerG, "Generator"
            )

            # @TODO save model+plots every N epochs
            if epoch % self.checkpoint_interval == 0:
                self.plot_signals(f"signals_epoch_{epoch}")
                self.train_metadata.plot(
                    f"{self.outdir}/training_metrics_epoch_{epoch}.png"
                )

        runtime = (time.time() - t0) / 60
        logger.info(f"Training Time: {runtime:.2f}min")
        self.plot_signals(f"signals_epoch_end")
        self.train_metadata.plot(
            f"{self.outdir}/training_metrics_epoch_end.png"
        )
        self.save_models()

    @property
    def save_fname(self):
        return f"{self.outdir}/generator_weights.pt"

    def save_models(self):
        save_model(self.netG, self.save_fname)
        logger.info(f"Saved model to {self.save_fname}")
        # @TODO save discriminator ?

    def _update_discriminator(self, data):
        """Update D network: maximize log(D(x)) + log(1 - D(G(z)))"""
        ## Train with all-real batch
        self.netD.zero_grad()
        # Format batch
        real_gpu = data.to(DEVICE)
        b_size = real_gpu.size(0)
        label_real = torch.FloatTensor(b_size).uniform_(1.0, 1.0).to(DEVICE)
        # Forward pass real batch through D
        output = self.netD(real_gpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.criterion(output, label_real)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, self.nz, 1, device=DEVICE)
        # Generate fake signal batch with G
        fake = self.netG(noise)
        label_fake = torch.FloatTensor(b_size).uniform_(0.0, 0.25).to(DEVICE)
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
        label_real = torch.FloatTensor(b_size).uniform_(1.0, 1.0).to(DEVICE)
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
        logger.debug(f"SGD {label} lr {before_lr:.7f} -> {after_lr:.7f}")
        return after_lr


class TrainMetadata:
    def __init__(self):
        self.iter: List[int] = []
        self.g_loss: List[float] = []
        self.d_loss: List[float] = []
        self.g_gradient: List[float] = []
        self.d_gradient: List[float] = []

    def append(self, iter, g_loss, d_loss, g_gradient, d_gradient):
        self.iter.append(iter)
        self.g_loss.append(g_loss)
        self.d_loss.append(d_loss)
        self.g_gradient.append(g_gradient)
        self.d_gradient.append(d_gradient)

    def plot(self, fname="training_metrics.png"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6))
        plot_gradients(
            self.d_gradient, "tab:red", "Discriminator", axes=axes[0]
        )
        plot_gradients(self.g_gradient, "tab:blue", "Generator", axes=axes[1])
        plot_loss(self.g_loss, self.d_loss, axes=axes[2])
        plt.tight_layout()
        plt.savefig(fname)


def _init_weights(m: torch.nn.Module) -> None:
    """This function initialises the weights of the model."""
    if type(m) == torch.nn.Conv1d or type(m) == torch.nn.ConvTranspose1d:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def train(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.train()
    return trainer
