"""This module provides the command line interface for the starccato package."""
import click

from .defaults import GENERATOR_WEIGHTS_FN
from .utils import generate_signals


@click.command("starccato_generate")
@click.option(
    "-n",
    "--n",
    default=1,
    type=int,
    help="The number of signals to generate.",
)
@click.option(
    "--weights_file",
    "-w",
    type=str,
    default=GENERATOR_WEIGHTS_FN,
    help="The file containing the weights for the generator model.",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    help="The random seed to use when generating the signals.",
)
@click.option(
    "--filename",
    "-f",
    type=str,
    default=None,
    help="The name of the file to save the generated signals to.",
)
def cli_generate(n, weights_file, seed, filename):
    """This function generates signals using the trained generator model."""
    generate_signals(
        n=n, weights_file=weights_file, seed=seed, filename=filename
    )


# @TODO: Add cli-train command (@tarin-e)
