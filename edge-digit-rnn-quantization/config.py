"""
config.py
=========
Backward-compatibility shim.  All values now live in the configs/ package.

Importing from this module still works exactly as before:
    from config import HIDDEN_SIZE, N_MFCC, DEVICE, ...
    import config; config.RECORDINGS_DIR
"""

from configs import *  # noqa: F401, F403

# -- Backward-compat alias --------------------------------------------------
# Old config.py had CHECKPOINT_B2 = "best_model_b2_quantized.pt" before B2
# was implemented.  The new canonical name (from config_quant.py) is
# "best_model_b2_int8.pt".  We expose the new name via the star-import above.
# Nothing in the codebase used the old name, so no alias is needed here.
