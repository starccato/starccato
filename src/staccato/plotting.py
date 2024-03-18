import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib import rcParams


def _config_rc_params():
    rcParams['axes.grid'] = True
    rcParams['axes.grid.which'] = 'both'




def plot_waveforms(
        signals: np.ndarray, scaling_factor: float, mean: float, std: float, num_cols: int = 4, num_rows: int = 4,
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
