"""Defaults for the staccato package."""
import os

NZ = 100
NC = 1
NGF = 64
NDF = 64
GENERATOR_WEIGHTS_FN = os.path.join(os.path.dirname(__file__), "default_weights.pt")
BATCH_SIZE  = 32