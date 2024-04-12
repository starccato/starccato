import os

import numpy as np

from starccato import generate_signals
from starccato.cli import cli_generate

def test_cli_generate(cli_runner, tmpdir):
    fname = f"{tmpdir}/generated_signals.txt"
    n = 5
    result = cli_runner.invoke(cli_generate, ["-n", n, "--filename", fname])
    print(result)
    assert result.exit_code == 0
    assert os.path.exists(fname)
    signals = np.loadtxt(fname)
    assert signals.shape[0] == n


def test_python_generate(tmpdir):
    fname = f"{tmpdir}/generated_signals.txt"
    n = 10
    signals = generate_signals(n=n, filename=fname)
    assert os.path.exists(fname)
    assert signals.shape[0] == n



from starccato.plotting import overplot_signals
import matplotlib.pyplot as plt
from starccato import generate_signals

signals = generate_signals(n=1684)

fig = overplot_signals(signals, color="k", alpha=0.01, linewidth=0.2)
fig.axes[0].set_axis_off()
fig.axes[0].grid(False)
# mean_signal = signals.mean(axis=0)
# plt.plot(mean_signal, c="k", lw=2)
# transparent background when saving
# fig.savefig(
#     "signal_generation.png", bbox_inches="tight", transparent=True, dpi=500
# )
_ = fig.suptitle("Stellar Core Collapse Signals [GAN-Generated]")
#%%
fig.savefig("signal_generation.png", bbox_inches="tight", transparent=True, dpi=500)
