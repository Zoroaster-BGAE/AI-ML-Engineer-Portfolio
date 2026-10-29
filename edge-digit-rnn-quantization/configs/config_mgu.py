"""
configs/config_mgu.py
=====================
Hyperparameters for Task A (baseline GRU) and Task B1 (constrained MGU).
"""

# -- Task A — unconstrained GRU baseline ------------------------------------
N_MFCC          = 40     # MFCC coefficients; total feature dim = 3 * N_MFCC = 120
HIDDEN_SIZE     = 128
NUM_LAYERS      = 1
DROPOUT         = 0.3
LR              = 1e-3
EPOCHS          = 50
CHECKPOINT_PATH = "best_model.pt"

# -- Task B1 — 36 kB per-layer constrained MGU (float32) -------------------
# MGU param count formula (2 gates, not 3):
#   2 * H * (I + H + 2)   where I=INPUT_PROJ=32, H=50
#   -> 2 * 50 * (32 + 50 + 2) = 8,400 params = 33,600 bytes  < 36 kB OK
#
# GRU equiv comment (kept from original config.py):
#   H=20 -> 3*20*(120+20+2) = 8,520 params -> 34,080 bytes  fits
HIDDEN_SIZE_B1  = 20   # legacy GRU hidden (kept for backward compat)
NUM_LAYERS_B1   = 1
DROPOUT_B1      = 0.1
CHECKPOINT_B1   = "best_model_b1_constrained.pt"
