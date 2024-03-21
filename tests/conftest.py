import os
import pytest
import sys
from staccato import logger

logger.configure(handlers=[{"sink": sys.stdout, "level": "DEBUG"}])

TEST_DIR = "out_test"


@pytest.fixture
def tmpdir() -> str:
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR, exist_ok=True)
    return TEST_DIR