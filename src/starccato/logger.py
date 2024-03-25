import sys

import loguru
from tqdm import tqdm


def config_logger(level="INFO"):
    logger_format = "{time:DD-MM-YY HH:mm:ss}|<green>starccato</green>|{level}| <level>{message}</level>"
    loguru.logger.configure(
        handlers=[
            dict(
                sink=lambda msg: tqdm.write(msg, end=""),
                format=logger_format,
                colorize=True,
            )
        ]
    )
    # set the log level to info
    logger = loguru.logger
    logger.configure(handlers=[{"sink": sys.stdout, "level": level}])
    return logger


logger = config_logger()
