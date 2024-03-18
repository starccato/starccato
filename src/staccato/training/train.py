import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..nn import Discriminator, Generator
from ..utils import init_weights, get_device
from .training_data import TrainingData
from ..plotting import plot_waveforms

from tqdm.auto import trange, tqdm
import time
from torch import nn, optim

from torch.optim import lr_scheduler
from ..defaults import NC, NGF, NZ, BATCH_SIZE, NDF


def __set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    return seed


def train_models(
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
    os.makedirs(outdir, exist_ok=True)

    __set_seed(seed)
    device = get_device()
    dataset = TrainingData(batch_size=batch_size)
    criterion = nn.BCELoss()

    # Setup generator network + optimiser
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    netG.apply(init_weights)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))
    schedulerG = lr_scheduler.LinearLR(optimizerG, start_factor=1.0, end_factor=0.5, total_iters=32)

    # Setup discriminator network + optimiser
    netD = Discriminator(nz=nz, ndf=ndf, nc=nc).to(device)
    netD.apply(init_weights)
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
    schedulerD = lr_scheduler.LinearLR(optimizerD, start_factor=1.0, end_factor=0.5, total_iters=32)

    # create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(batch_size, nz, 1, device=device)

    plt_kwgs = dict(
        scaling_factor=dataset.scaling_factor, mean=dataset.mean, std=dataset.std,
        num_cols=4, num_rows=4
    )

    # plot generated signals before training
    plot_fake_signals(netG, fixed_noise, f"{outdir}/before_training.png", **plt_kwgs)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    D_gradients = []
    G_gradients = []

    start = time.time()

    print(f"Starting Training Loop [Epochs: {num_epochs}, Train Size: {dataset.shape}, Learning Rate: {lr_g}]")
    dataloader = dataset.get_loader()
    # For each epoch
    for epoch in trange(num_epochs, desc="Epochs", position=0):
        # For each batch in the dataloader


        for (i, data) in tqdm(enumerate(dataloader, 0), desc="Batch", position=1, leave=False, total=len(dataloader)):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_gpu = data.to(device)
            b_size = real_gpu.size(0)
            label_real = torch.FloatTensor(b_size).uniform_(1.0, 1.0).to(device)
            # Forward pass real batch through D
            output = netD(real_gpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label_real)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, device=device)
            # Generate fake signal batch with G
            fake = netG(noise)
            label_fake = torch.FloatTensor(b_size).uniform_(0.0, 0.25).to(device)
            # label_fake = torch.FloatTensor(b_size).uniform_(0.0, 0.0).to(device)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label_fake)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            # Calculate gradients of discriminator parameters
            _dgrad = [param.grad.norm().item() for param in netD.parameters()]

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label_real = torch.FloatTensor(b_size).uniform_(1.0, 1.0).to(device)
            # label_real = 1.0 - label_fake
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label_real)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            # Calculate gradients of generator parameters
            _ggrad = [param.grad.norm().item() for param in netG.parameters()]

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save data for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_gradients.append(_dgrad)
            G_gradients.append(_ggrad)

        # learning-rate decay
        before_lr = optimizerD.param_groups[0]["lr"]
        schedulerD.step()
        after_lr = optimizerD.param_groups[0]["lr"]
        print("Epoch %d: SGD Discriminator lr %.7f -> %.7f" % (epoch, before_lr, after_lr))

        before_lr = optimizerG.param_groups[0]["lr"]
        schedulerG.step()
        after_lr = optimizerG.param_groups[0]["lr"]
        print("Epoch %d: SGD Generator lr %.7f -> %.7f" % (epoch, before_lr, after_lr))

        plot_fake_signals(netG, fixed_noise, f"{outdir}/signals_epoch_{epoch}.png", **plt_kwgs)

    end = time.time()

    runtime = (end - start) / 60
    print(f"Training Time: {runtime:.2f}min")

    plot_gradients(D_gradients, "tab:red", "Discriminator", f"{outdir}/discriminator_gradients.png")
    plot_gradients(G_gradients, "tab:blue", "Generator", f"{outdir}/generator_gradients.png")
    plot_loss(G_losses, D_losses, f"{outdir}/losses.png")
    generator_weights_fn = f"{outdir}/generator_weights.pt"
    torch.save(netG, generator_weights_fn)
    print(f"Saved generator weights to {generator_weights_fn}")
    return netG, netD


def plot_gradients(gradients, color, label, fname):
    # Get the total number of layers in the discriminator
    gradients = np.array(gradients)
    num_layers = gradients.shape[1]

    # Plot the gradients over training epochs
    plt.figure(figsize=(10, 6))
    for i in range(num_layers):
        # Calculate alpha value based on layer index
        alpha = 1 - (i / num_layers)  # Higher layers are more transparent
        plt.plot(gradients[:, i], label=f'Layer {i}', alpha=alpha, color=color)

    plt.xlabel('Batches')
    plt.ylabel('Gradient Magnitude')
    plt.title(f'{label} Gradients')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)


def plot_loss(G_losses, D_losses, fname):
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Discriminator Loss Convergence Point')
    plt.xlabel("Batch", size=20)
    plt.ylabel("Loss", size=20)
    plt.ylim(0, 5)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(fname)


def plot_fake_signals(generator, latent_vector, fname, **plt_kwgs):
    # plot generated signals before training
    with torch.no_grad():
        fake_signals = generator(latent_vector).detach().cpu()
        plot_waveforms(fake_signals, fname=fname, **plt_kwgs)
