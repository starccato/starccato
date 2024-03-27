[![Coverage Status](https://coveralls.io/repos/github/starccato/starccato/badge.svg?branch=main)](https://coveralls.io/github/starccato/starccato?branch=main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# starccato (Stellar Core Collapse GW generator)

FILL ME DESCRIPTION @TODO


## Installation

```
pip install starccato
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
