"""
configs/__init__.py
===================
Re-exports all configuration constants so that:
    from configs import DEVICE, HIDDEN_SIZE, ...
works as a single import.
"""

from configs.config_base   import *  # noqa: F401, F403
from configs.config_mgu    import *  # noqa: F401, F403
from configs.config_quant  import *  # noqa: F401, F403
from configs.config_power2 import *  # noqa: F401, F403
