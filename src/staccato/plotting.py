import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from matplotlib import rcParams
import torch
from .nn import Generator


def _config_rc_params():
    rcParams['axes.grid'] = True
    rcParams['axes.grid.which'] = 'both'


def plot_waveform_grid(
        signals: np.ndarray, scaling_factor: float, mean: float, std: float,
        num_cols: int = 4, num_rows: int = 4,
        fname: str = None
) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(num_cols, num_rows, figsize=(num_cols * 4, num_rows * 3))

    axes = axes.flatten()

    # plot each signal on a separate subplot
    for i, ax in enumerate(axes):
        x = [i / 4096 for i in range(0, 256)]
        x = [value - (53 / 4096) for value in x]
        y = signals[i, :, :].flatten()
        y = y * scaling_factor
        y = y * std + mean
        ax.set_ylim(-600, 300)
        ax.plot(x, y, color='red')

        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True)

        # Remove y-axis ticks for the right-hand column
        if i % num_cols == num_cols - 1:
            ax.yaxis.set_ticklabels([])

        # Remove x-axis tick labels for all but the bottom two plots
        if i <= 11:
            ax.xaxis.set_ticklabels([])

    for i in range(512, 8 * 4):
        fig.delaxes(axes[i])

    fig.supxlabel('time (s)', fontsize=24)
    fig.supylabel('distance x strain (cm)', fontsize=24)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    return fig, axes


def plot_gradients(gradients: List[float], color: str, label: str, fname: str):
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


def plot_loss(G_losses: List[float], D_losses: List[float], fname: str):
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


def plot_signals_from_latent_vector(generator: Generator, latent_vector: torch.Tensor, fname: str, **plt_kwgs):
    with torch.no_grad():
        fake_signals = generator(latent_vector).detach().cpu()
        plot_waveform_grid(fake_signals, fname=fname, **plt_kwgs)
