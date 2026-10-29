"""
configs/config_base.py
======================
Shared audio + dataset constants used across all tasks.
"""

import torch

# -- Paths ------------------------------------------------------------------
RECORDINGS_DIR  = "free-spoken-digit-dataset/recordings"

# -- Audio / Feature extraction ---------------------------------------------
SAMPLE_RATE = 8000   # FSDD recordings are 8 kHz
HOP_LENGTH  = 80     # ~10 ms per frame  ->  100 frames ~= 1 s
N_FFT       = 512
MAX_LEN     = 100    # fixed sequence length (frames) after pad / truncate

# -- Shared model constants -------------------------------------------------
NUM_CLASSES = 10

# -- Data loading -----------------------------------------------------------
BATCH_SIZE  = 64

# -- Device -----------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
