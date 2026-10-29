"""
utils/quant_utils.py
====================
Quantization utilities for Tasks B2 (INT8 QAT) and C (Power-of-Two QAT).

Design principles:
  - STE (Straight-Through Estimator) via torch.autograd.Function, NOT .detach()
  - LayerNorm parameters are NEVER quantized (training instability)
  - Biases are NEVER quantized (standard practice)
  - FakeQuantizeLinear / FakePoTLinear wrap nn.Linear as self.linear, so that
    audit_model_memory (which uses parameters(recurse=False)) audits the inner
    nn.Linear and sees the same param count as Task B1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────── INT8 STE ──────────────────────────────────────

class _STERound(torch.autograd.Function):
    """
    Straight-Through Estimator for symmetric INT8 quantization.

    Forward : round-and-clamp to [-127, 127] grid, then rescale to float.
    Backward: gradient passes straight through (identity).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.round(x / scale), -127.0, 127.0) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None   # STE: identity for x, no grad for scale


def fake_quantize_int8(w: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """
    Per-tensor symmetric INT8 fake quantization.

    scale = max(|w|) / 127
    Forward: clamp(round(w / scale), -127, 127) * scale
    Backward: STE (gradient = grad_output)
    """
    max_val = w.detach().abs().max().clamp(min=1e-8)
    scale   = max_val / (2 ** (num_bits - 1) - 1)   # = max_val / 127
    return _STERound.apply(w, scale)


# ─────────────────────────── FakeQuantizeLinear ─────────────────────────────

class FakeQuantizeLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with INT8 fake quantization on weights.

    The weight matrix is fake-quantized during forward (STE in backward).
    The bias is left in float32 (standard practice).

    The inner nn.Linear is stored as self.linear so that:
      - audit_model_memory(model, bytes_per_param=1) audits self.linear at
        exactly the same param count as the corresponding nn.Linear in B1.
      - Warm-start key remapping maps "X.weight" -> "X.linear.weight".
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fake_quantize_int8(self.linear.weight)
        return F.linear(x, w_q, self.linear.bias)


# ─────────────────────────── Power-of-Two STE ───────────────────────────────

# Default PoT parameters (match configs/config_power2.py).
# Defined here to avoid importing config at module level.
_DEFAULT_POT_MIN_EXP     = -8
_DEFAULT_POT_MAX_EXP     = 3
_DEFAULT_POT_ZERO_THRESH = 2.0 ** -9   # = 2^(MIN_EXP - 1)


class _STEPoT(torch.autograd.Function):
    """
    Straight-Through Estimator for Power-of-Two quantization.

    Forward : snap each weight to the nearest power of 2 (or zero if |w| is
              below the zero threshold).
    Backward: gradient passes straight through (identity).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        min_exp: float,
        max_exp: float,
        zero_thresh: float,
    ) -> torch.Tensor:
        sign    = x.sign()
        abs_x   = x.abs()
        log2x   = torch.log2(abs_x.clamp(min=1e-30))
        exp     = torch.clamp(torch.round(log2x), min_exp, max_exp)
        q       = sign * (2.0 ** exp)
        return torch.where(abs_x < zero_thresh, torch.zeros_like(q), q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None   # STE for x; no grads for params


def fake_quantize_pot(
    w: torch.Tensor,
    min_exp: float = _DEFAULT_POT_MIN_EXP,
    max_exp: float = _DEFAULT_POT_MAX_EXP,
    zero_thresh: float = _DEFAULT_POT_ZERO_THRESH,
) -> torch.Tensor:
    """
    Per-weight Power-of-Two fake quantization with STE.

    Each weight is snapped to the nearest value in
      {0} ∪ {±2^k | k ∈ [min_exp, max_exp]}
    Weights with |w| < zero_thresh are mapped to 0.
    """
    return _STEPoT.apply(w, min_exp, max_exp, zero_thresh)


# ─────────────────────────── FakePoTLinear ──────────────────────────────────

class FakePoTLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with Power-of-Two fake quantization.

    Mirrors FakeQuantizeLinear in structure: inner nn.Linear stored as self.linear.
    Stores min_exp / max_exp / zero_thresh as instance attributes so that
    snap_weights_to_pot can read them without needing to import config again.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool  = True,
        min_exp:      float = _DEFAULT_POT_MIN_EXP,
        max_exp:      float = _DEFAULT_POT_MAX_EXP,
        zero_thresh:  float = _DEFAULT_POT_ZERO_THRESH,
    ):
        super().__init__()
        self.linear      = nn.Linear(in_features, out_features, bias=bias)
        self.min_exp     = min_exp
        self.max_exp     = max_exp
        self.zero_thresh = zero_thresh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fake_quantize_pot(
            self.linear.weight, self.min_exp, self.max_exp, self.zero_thresh
        )
        return F.linear(x, w_q, self.linear.bias)


# ─────────────────────────── Snapping utilities ─────────────────────────────

def _pot_snap_tensor(
    w: torch.Tensor,
    min_exp: float,
    max_exp: float,
    zero_thresh: float,
) -> torch.Tensor:
    """Pure tensor op (no autograd graph). Snaps w to nearest PoT value."""
    eps   = 1e-10
    sign  = w.sign()
    abs_w = w.abs()
    exp   = torch.clamp(torch.round(torch.log2(abs_w + eps)), min_exp, max_exp)
    q     = sign * (2.0 ** exp)
    return torch.where(abs_w < zero_thresh, torch.zeros_like(q), q)


def snap_weights_to_int8(model: nn.Module) -> None:
    """
    In-place: snap every FakeQuantizeLinear.linear.weight to the INT8 float
    grid (i.e. values representable as round(w/scale)*scale with scale=max/127).
    Call this AFTER training and BEFORE final evaluation.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, FakeQuantizeLinear):
                w     = m.linear.weight.data
                scale = w.abs().max() / 127.0 + 1e-8
                m.linear.weight.data = (
                    torch.clamp(torch.round(w / scale), -128, 127) * scale
                )


def snap_weights_to_pot(model: nn.Module) -> None:
    """
    In-place: snap every FakePoTLinear.linear.weight to its nearest
    Power-of-Two value using the instance's stored exponent bounds.
    Call this AFTER training and BEFORE final evaluation.
    """
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, FakePoTLinear):
                m.linear.weight.data = _pot_snap_tensor(
                    m.linear.weight.data,
                    m.min_exp,
                    m.max_exp,
                    m.zero_thresh,
                )


# ─────────────────────────── Warm-start key remapper ────────────────────────

def remap_b1_to_quant(
    b1_state_dict: dict,
    quant_model:   nn.Module,
    verbose:       bool = True,
) -> None:
    """
    Load B1 float32 weights into a B2/C quantized model.

    Remapping rule (applied with rsplit to handle deeply nested keys):
      B1 key  "prefix.weight"  ->  try "prefix.linear.weight" first
                                   fall back to "prefix.weight" (LayerNorm)
      B1 key  "prefix.bias"   ->  same logic

    Example mappings:
      proj.weight          -> proj.linear.weight
      norm.weight          -> norm.weight          (LayerNorm unchanged)
      mgu.cell.W_f.weight  -> mgu.cell.W_f.linear.weight

    CRITICAL: use rsplit('.', 1) not split('.', 1) so that
      "mgu.cell.W_f.weight" splits as ["mgu.cell.W_f", "weight"]
      NOT as ["mgu", "cell.W_f.weight"].
    """
    quant_params = set(quant_model.state_dict().keys())
    remapped: dict = {}

    for k, v in b1_state_dict.items():
        parts     = k.rsplit(".", 1)                           # max 1 split from right
        candidate = f"{parts[0]}.linear.{parts[1]}" if len(parts) == 2 else k
        if candidate in quant_params:
            remapped[candidate] = v
        elif k in quant_params:
            remapped[k] = v    # LayerNorm: norm.weight, norm.bias

    missing_keys, unexpected_keys = quant_model.load_state_dict(
        remapped, strict=False
    )
    if verbose:
        n_loaded = len(remapped)
        print(
            f"  [warm-start] Remapped {n_loaded} tensors from B1 checkpoint.  "
            f"missing={len(missing_keys)}  unexpected={len(unexpected_keys)}"
        )
        if missing_keys:
            print(f"    missing  : {missing_keys[:5]}")
        if unexpected_keys:
            print(f"    unexpected: {unexpected_keys[:5]}")
