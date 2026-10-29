"""
utils/memory_utils.py
=====================
Helpers to audit and enforce the per-layer memory constraint.

The hardware constraint: ALL parameters of any one layer must fit
simultaneously into 36 kB of on-chip SRAM.

  36,000 bytes / 4 bytes per float32 = 9,000 parameters maximum per layer
"""

import torch.nn as nn

MEMORY_LIMIT_BYTES  = 36_000
BYTES_PER_FLOAT32   = 4
BYTES_PER_INT8      = 1
MAX_PARAMS_FLOAT32  = MEMORY_LIMIT_BYTES // BYTES_PER_FLOAT32   # 9,000
MAX_PARAMS_INT8     = MEMORY_LIMIT_BYTES // BYTES_PER_INT8      # 36,000


def gru_param_count(input_size: int, hidden_size: int) -> int:
    """
    Compute the exact number of parameters in a single GRU layer.

    PyTorch GRU stores four tensors per layer:
      weight_ih  (3*H, I)   — input-to-hidden weights for 3 gates
      weight_hh  (3*H, H)   — hidden-to-hidden weights for 3 gates
      bias_ih    (3*H,)
      bias_hh    (3*H,)

    Total = 3*H*I  +  3*H*H  +  3*H  +  3*H
          = 3 * H * (I + H + 2)
    """
    H, I = hidden_size, input_size
    return 3 * H * (I + H + 2)


def audit_model_memory(model: nn.Module, bytes_per_param: int = BYTES_PER_FLOAT32) -> None:
    """
    Print a per-layer parameter count and memory footprint table,
    and assert that every layer fits within the 36 kB constraint.

    Parameters
    ----------
    model           : nn.Module
    bytes_per_param : 4 for float32, 1 for int8
    """
    limit = MEMORY_LIMIT_BYTES // bytes_per_param
    dtype_label = "float32" if bytes_per_param == 4 else "int8"

    print(f"\n{'='*65}")
    print(f"  Memory audit  |  dtype={dtype_label}  |  limit={MEMORY_LIMIT_BYTES} bytes ({limit} params)")
    print(f"{'='*65}")
    print(f"  {'Layer':<30}  {'Params':>8}  {'Bytes':>9}  {'Status'}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*9}  {'-'*6}")

    all_pass = True
    for name, module in model.named_modules():
        if not list(module.parameters(recurse=False)):
            continue                          # skip container modules
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        n_bytes  = n_params * bytes_per_param
        status   = "OK" if n_params <= limit else "EXCEEDS LIMIT"
        if n_params > limit:
            all_pass = False
        print(f"  {name:<30}  {n_params:>8,}  {n_bytes:>8,}B  {status}")

    total = sum(p.numel() for p in model.parameters())
    print(f"{'='*65}")
    print(f"  Total model params: {total:,}  ({total * bytes_per_param:,} bytes)")
    print(f"  Constraint: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*65}\n")

    assert all_pass, "Model violates the 36 kB per-layer memory constraint!"
