"""
configs/config_power2.py
========================
Hyperparameters for Task C — Power-of-Two (PoT) Quantization-Aware Training.

PoT weights are constrained to {0} ∪ {±2^k | k ∈ [POT_MIN_EXP, POT_MAX_EXP]}.
This means multiply-accumulate operations can be replaced by bit-shifts on hardware.

Zero threshold: weights with |w| < POT_ZERO_THRESH are snapped to zero.
Convention: POT_ZERO_THRESH = 2^(POT_MIN_EXP - 1) = midpoint between 0 and 2^POT_MIN_EXP.
"""

# -- Task C — Power-of-Two fake quantization (QAT) --------------------------
EPOCHS_C        = 60
LR_C            = 5e-4
POT_MIN_EXP     = -8         # smallest non-zero exponent:  2^(-8) ≈ 0.0039
POT_MAX_EXP     = 3          # largest exponent:            2^3    = 8.0
POT_ZERO_THRESH = 2.0 ** -9  # = 2^(POT_MIN_EXP - 1); weights below this → 0
CHECKPOINT_C    = "best_model_c_pow2.pt"
