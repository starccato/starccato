import os
from unittest.mock import Mock

import pytest

from starccato import generate_signals
from starccato.training import TrainingData, train


def test_training_data(tmpdir):
    training_data = TrainingData()
    fname = f"{tmpdir}/training_waveforms.png"
    training_data.plot_waveforms(fname)
    assert os.path.exists(fname)
    fname = f"{tmpdir}/training_waveform_standardised.png"
    training_data.plot_waveforms(fname, standardised=True)
    assert os.path.exists(fname)


@pytest.fixture
def mock_training_data(monkeypatch):
    mock = Mock()
    monkeypatch.setattr("starccato.training.trainer.TrainingData", mock)
    mock.return_value = TrainingData(frac=0.002, batch_size=1)
    return mock


def test_training(mock_training_data, tmpdir):
    train_outdir = f"{tmpdir}/train_outdir"
    train(outdir=train_outdir, num_epochs=1)
    assert os.path.exists(train_outdir)
    weights_fn = f"{train_outdir}/generator_weights.pt"
    assert os.path.exists(weights_fn)

    gen_signals = f"{tmpdir}/generated_signals.txt"
    generate_signals(
        n=10,
        weights_file=weights_fn,
        filename=gen_signals,
    )

    assert os.path.exists(gen_signals)
