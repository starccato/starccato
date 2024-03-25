from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rcParams

# 1, 2, 3, sigmas
SIGMA_QUANTS = [0.68, 0.95, 0.99]


def _config_rc_params():
    rcParams["axes.grid"] = False
    # rcParams['axes.grid.which'] = 'both'


def plot_waveform_grid(
    signals: np.ndarray,
    scaling_factor: float,
    mean: float,
    std: float,
    num_cols: int = 4,
    num_rows: int = 4,
    fname: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(
        num_cols, num_rows, figsize=(num_cols * 4, num_rows * 3)
    )

    axes = axes.flatten()

    # plot each signal on a separate subplot
    for i, ax in enumerate(axes):
        x = [i / 4096 for i in range(0, 256)]
        x = [value - (53 / 4096) for value in x]
        y = signals[i, :, :].flatten()
        y = y * scaling_factor
        y = y * std + mean
        ax.set_ylim(-600, 300)
        ax.plot(x, y, color="red")

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.grid(True)

        # Remove y-axis ticks for the right-hand column
        if i % num_cols == num_cols - 1:
            ax.yaxis.set_ticklabels([])

        # Remove x-axis tick labels for all but the bottom two plots
        if i <= 11:
            ax.xaxis.set_ticklabels([])

    for i in range(512, 8 * 4):
        fig.delaxes(axes[i])

    fig.supxlabel("time (s)", fontsize=24)
    fig.supylabel("distance x strain (cm)", fontsize=24)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_gradients(
    gradients: List[float],
    color: str,
    label: str,
    fname: str = None,
    axes: plt.Axes = None,
):
    # Get the total number of layers in the discriminator
    gradients = np.array(gradients)
    num_layers = gradients.shape[1]

    # Plot the gradients over training epochs
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
    for i in range(num_layers):
        # Calculate alpha value based on layer index
        alpha = 1 - (i / num_layers)  # Higher layers are more transparent
        axes.plot(
            gradients[:, i], label=f"Layer {i}", alpha=alpha, color=color
        )

    axes.set_xlabel("Batches")
    axes.set_ylabel("Gradient Magnitude")
    axes.set_title(f"{label} Gradients")
    axes.legend()
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    return axes.get_figure()


def plot_loss(
    G_losses: List[float],
    D_losses: List[float],
    fname: str = None,
    axes: plt.Axes = None,
):
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
    axes.plot(G_losses, label="Generator Loss")
    axes.plot(D_losses, label="Discriminator Loss")
    axes.axhline(
        y=0.5,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Discriminator Loss Convergence Point",
    )
    axes.set_xlabel("Batch", size=20)
    axes.set_ylabel("Loss", size=20)
    axes.set_ylim(0, 5)
    axes.legend(fontsize=16)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    return axes.get_figure()


def plot_signals_from_latent_vector(
    generator: "Generator", latent_vector: torch.Tensor, fname: str, **plt_kwgs
):
    with torch.no_grad():
        fake_signals = generator(latent_vector).detach().cpu()
        plot_waveform_grid(fake_signals, fname=fname, **plt_kwgs)


def plot_signals_ci(
    signals: np.ndarray,
    color: str,
    quantiles: Optional[Tuple[float, float]] = [0.68, 0.95, 0.99],
):
    """Plot the mean and confidence interval of the signals."""
    quant_pairs = [(0.5 - 0.5 * q, 0.5 + 0.5 * q) for q in quantiles]
    x = np.arange(signals.shape[1])

    fig = plt.figure(figsize=(10, 6))
    for i, q in enumerate(quant_pairs):
        quants = np.quantile(signals, q, axis=0)
        plt.fill_between(x, quants[0], quants[1], alpha=0.3 * i, color=color)
    return fig


def overplot_signals(signals: np.ndarray, axes: plt.Axes = None, **kwgs):
    """Overplot the signals."""
    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = axes.get_figure()

    kwgs["alpha"] = kwgs.get("alpha", 0.002)
    kwgs["color"] = kwgs.get("color", "k")
    kwgs["linewidth"] = kwgs.get("linewidth", 0.1)

    for s in signals:
        axes.plot(s, **kwgs)
    return fig


def plot_stacked_signals(
    signals: np.ndarray,
    cmap: str = "inferno",
    axes: plt.Axes = None,
    norm="linear",
):
    """Plot the stacked signals."""
    if axes is None:
        fig, axes = plt.subplots()
    else:
        fig = axes.get_figure()

    im = axes.imshow(signals, cmap=cmap, aspect="auto", norm=norm)
    cbar = fig.colorbar(im, ax=axes)
    # xlabel time
    # ylabel signal number
    axes.set_xlabel("Time")
    axes.set_ylabel("Signal Number")
    cbar.set_label("distance x strain (cm)")
    return fig
