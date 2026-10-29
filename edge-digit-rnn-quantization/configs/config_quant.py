"""
configs/config_quant.py
=======================
Hyperparameters for Task B2 — INT8 Quantization-Aware Training (QAT).

At INT8 (1 byte/param) the 36 kB constraint allows 36,000 params per layer,
so the MGU architecture that passed the float32 audit trivially passes here too.
"""

# -- Task B2 — INT8 fake quantization (QAT) ---------------------------------
EPOCHS_B2    = 60
LR_B2        = 5e-4
NUM_BITS     = 8
CHECKPOINT_B2 = "best_model_b2_int8.pt"
