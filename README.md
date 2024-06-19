[![Coverage Status](https://coveralls.io/repos/github/starccato/starccato/badge.svg?branch=main)](https://coveralls.io/github/starccato/starccato?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPi version](https://pypip.in/v/starccato/badge.png)](https://crate.io/packages/starccato/)

# starccato (Stellar Core Collapse GW generator)

Starccato is a fast stellar core-collapse gravitational wave generator.

## Installation

```
pip install starccato
```

## Training with starccato

Train the model with the following command:
```bash
starccato_train --outdir weights/ --epochs 8
```

From within python:
```python
import starccato
from starccato.training import train


train(outdir="weights/", epochs=128)
starccato.generate_signals(n=10, weights_file="weights/generator_weights.pth")

```


## Development

```
git clone https://github.com/tarin-e/starccato.git
cd starccato
pip install -e ".[dev]"
pre-commit install
```

Ensure unit tests are passing locally and on the CI!
```
pytest tests/
```

## Releasing to PyPI

1. Manually change the version number in `pyproject.toml`  (has to be higher than previous)
1. Create a tagged commit with the version number
2. Push the tag to GitHub

```
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```
