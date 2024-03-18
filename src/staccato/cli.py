"""This module provides the command line interface for the staccato package."""
import click

from .nn import generate_signals
from .defaults import GENERATOR_WEIGHTS_FN


@click.option(
    "--n",
    default=1,
    help="The number of signals to generate.",
)
@click.option(
    "--weights_file",
    default=GENERATOR_WEIGHTS_FN,
    help="The file containing the weights for the generator model.",
)
@click.option(
    "--seed",
    default=None,
    help="The random seed to use when generating the signals.",
)
@click.option(
    "--filename",
    default="signals.txt",
    help="The name of the file to save the generated signals to.",
)
def cli_generate(n, weights_file, seed, filename):
    """This function generates signals using the trained generator model."""
    generate_signals(n, weights_file, seed, filename)

