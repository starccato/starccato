from loguru import logger
from tqdm import tqdm

logger_format = "<green>STACCATO</green>|<level>{time:DD-MM-YY HH:mm:ss}</level>|[<level>{level: ^12}</level>] <level>{message}</level>"
logger.configure(handlers=[dict(sink=lambda msg: tqdm.write(msg, end=''), format=logger_format, colorize=True)])
