from staccato.training.training_data import TrainingData
import os


def test_training_data(tmpdir):
    training_data = TrainingData()
    fname = f"{tmpdir}/training_waveforms.png"
    training_data.plot_waveforms(fname)
    assert os.path.exists(fname)
    fname = f"{tmpdir}/training_waveform_standardised.png"
    training_data.plot_waveforms(fname, standardised=True)
    assert os.path.exists(fname)
