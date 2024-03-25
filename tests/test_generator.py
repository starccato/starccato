import os

import numpy as np

from starccato.cli import cli_generate
from starccato.defaults import NZ
from starccato.generate_signals import generate_signals


def test_cli_generate(cli_runner, tmpdir):
    fname = f"{tmpdir}/generated_signals.txt"
    n = 5
    result = cli_runner.invoke(cli_generate, ["-n", n, "--filename", fname])
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
