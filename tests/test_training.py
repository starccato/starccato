import torch
from torchsummary import summary

from staccato.defaults import NC, NZ
from staccato.discriminator import Discriminator
from staccato.generator import Generator


def test_generation_of_discriminator_generator():
    device = torch.device("mps")
    disc = Discriminator().to(device)
    gen = Generator().to(device)
    # summary(gen, input_size=(NZ, NC))
    # summary(disc, input_size=(NZ, NC))
