import importlib.metadata

from .plotting import _config_rc_params
from .utils import generate_signals

__name__ = "starccato"
__version__ = importlib.metadata.version(__name__)

_config_rc_params()
