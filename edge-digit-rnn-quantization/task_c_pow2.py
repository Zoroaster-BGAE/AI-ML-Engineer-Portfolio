"""
task_c_pow2.py
==============
Task C: Power-of-Two (PoT) Quantization-Aware Training for Spoken Digit Classification

Architecture
------------
PoTConstrainedMGU — mirrors QuantizedConstrainedMGU from Task B2 exactly, but
uses FakePoTLinear instead of FakeQuantizeLinear.  Weights are constrained to
  {0} ∪ {±2^k | k ∈ [-8, 3]}
enabling multiply-accumulate ops to be replaced by bit-shifts on hardware.

LayerNorm parameters are NOT quantized (training instability).
Biases are NOT quantized (standard practice: left in float32).

Warm-start chain
----------------
1. Try to load Task B2 checkpoint (best_model_b2_int8.pt).
   B2 and C have the same key layout ("X.linear.weight"), so load_state_dict
   is used directly with strict=False (LayerNorm keys have the same names).
2. Fall back to Task B1 checkpoint (best_model_b1_constrained.pt) via
   remap_b1_to_quant (inserts ".linear." into each key).
3. Fall back to random initialization (log warning).

Post-training analysis
----------------------
After training:
  - copy.deepcopy preserves the trained model.
  - snap_weights_to_pot is applied to the deep copy for evaluation.
  - A weight distribution table shows % of weights at each PoT value.
"""

import collections
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report

import config
from utils.data_loader import build_loaders
from utils.memory_utils import audit_model_memory
from utils.quant_utils import (
    FakePoTLinear,
    snap_weights_to_pot,
    remap_b1_to_quant,
    _pot_snap_tensor,
)
from main import DigitGRU
from task_b1_constrained import distillation_loss


# ─────────────────────────── Constants ─────────────────────────────────────

N_MFCC_STUDENT = 13
N_MFCC_TEACHER = 40
HIDDEN         = 50
INPUT_PROJ     = 32
NUM_CLASSES    = 10

ALPHA          = 0.3
TEMPERATURE    = 3.0

EPOCHS         = config.EPOCHS_C        # 60
LR             = config.LR_C            # 5e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 3
MAX_GRAD_NORM  = 1.0

POT_MIN_EXP     = config.POT_MIN_EXP      # -8
POT_MAX_EXP     = config.POT_MAX_EXP      # 3
POT_ZERO_THRESH = config.POT_ZERO_THRESH  # 2^(-9)

DEVICE       = config.DEVICE
HISTORY_PATH = "history_c.json"


# ─────────────────────────── PoT MGU ────────────────────────────────────────

class PoTMGUCell(nn.Module):
    """
    MGU cell with FakePoTLinear on all weight matrices.
    Mirrors QuantizedMGUCell but uses Power-of-Two fake quantization.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        pot_kw = dict(min_exp=POT_MIN_EXP, max_exp=POT_MAX_EXP, zero_thresh=POT_ZERO_THRESH)
        # forget gate
        self.W_f = FakePoTLinear(input_size,  hidden_size, bias=True, **pot_kw)
        self.U_f = FakePoTLinear(hidden_size, hidden_size, bias=True, **pot_kw)
        # new gate
        self.W_h = FakePoTLinear(input_size,  hidden_size, bias=True, **pot_kw)
        self.U_h = FakePoTLinear(hidden_size, hidden_size, bias=True, **pot_kw)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        f       = torch.sigmoid(self.W_f(x) + self.U_f(h))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(f * h))
        return (1.0 - f) * h + f * h_tilde


class PoTMGU(nn.Module):
    """Unrolls PoTMGUCell over a sequence."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = PoTMGUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        h = h0
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return h


class PoTConstrainedMGU(nn.Module):
    """
    Power-of-Two QAT version of ConstrainedMGU (Task C).

    Input  : (B, T=100, F=39)
    Output : (B, 10)

    proj and fc use FakePoTLinear.
    norm (LayerNorm) stays in float32 — NOT quantized.
    """

    def __init__(
        self,
        n_mfcc:      int = N_MFCC_STUDENT,
        hidden:      int = HIDDEN,
        input_proj:  int = INPUT_PROJ,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        feature_dim = 3 * n_mfcc
        pot_kw = dict(min_exp=POT_MIN_EXP, max_exp=POT_MAX_EXP, zero_thresh=POT_ZERO_THRESH)
        self.proj   = FakePoTLinear(feature_dim, input_proj, **pot_kw)   # 39->32
        self.norm   = nn.LayerNorm(input_proj)                            # float32
        self.mgu    = PoTMGU(input_proj, hidden)                          # 32->50
        self.fc     = FakePoTLinear(hidden, num_classes, **pot_kw)        # 50->10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = F.relu(self.norm(x))
        h = self.mgu(x)
        return self.fc(h)


# ─────────────────────────── Training helpers ───────────────────────────────

def train_epoch(
    student:   nn.Module,
    teacher:   nn.Module,
    s_loader,
    t_loader,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    student.train()
    total_loss, correct, total = 0.0, 0, 0

    for (x_s, y), (x_t, _) in zip(s_loader, t_loader):
        x_s = x_s.to(DEVICE)
        x_t = x_t.to(DEVICE)
        y   = y.to(DEVICE)

        optimizer.zero_grad()
        s_logits = student(x_s)
        with torch.no_grad():
            t_logits = teacher(x_t)

        loss = distillation_loss(s_logits, t_logits, y, alpha=ALPHA, T=TEMPERATURE)
        loss.backward()
        clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (s_logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds       = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def print_pot_distribution(model: nn.Module) -> None:
    """
    Collect all weights from FakePoTLinear layers (after snapping),
    count occurrences of each PoT value, and print a sorted table.
    """
    all_weights: list = []
    for m in model.modules():
        if isinstance(m, FakePoTLinear):
            all_weights.append(m.linear.weight.data.flatten())

    if not all_weights:
        return

    flat = torch.cat(all_weights).cpu()
    counter: dict = collections.Counter()
    for v in flat.tolist():
        # Round to avoid float noise: nearest representable PoT
        if abs(v) < 1e-12:
            key = 0.0
        else:
            key = round(v, 10)
        counter[key] += 1

    total = flat.numel()
    print(f"\n{'='*55}")
    print(f"  PoT Weight Distribution  ({total:,} weights total)")
    print(f"{'='*55}")
    print(f"  {'Value':>12}  {'Count':>8}  {'%':>6}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*6}")
    for val, cnt in sorted(counter.items()):
        pct = 100.0 * cnt / total
        print(f"  {val:>12.6f}  {cnt:>8,}  {pct:>6.2f}%")
    print(f"{'='*55}\n")


# ─────────────────────────── Main ──────────────────────────────────────────

def main():
    print(f"Using device : {DEVICE}")
    print(f"N_MFCC  student={N_MFCC_STUDENT}  teacher={N_MFCC_TEACHER}")
    print(f"PoT exponents: [{POT_MIN_EXP}, {POT_MAX_EXP}]  zero_thresh={POT_ZERO_THRESH:.2e}\n")

    # ── 1. Dual dataloaders ──────────────────────────────────────────────────
    print("Building student loaders (N_MFCC=13) ...")
    s_train, s_val, s_test, s_feat_dim = build_loaders(
        n_mfcc=N_MFCC_STUDENT, random_state=42
    )

    print("Building teacher loaders (N_MFCC=40) ...")
    t_train, _, _, t_feat_dim = build_loaders(
        n_mfcc=N_MFCC_TEACHER, random_state=42
    )
    print()

    # ── 2. Student model + memory audit (INT8: bytes_per_param=1) ────────────
    student = PoTConstrainedMGU().to(DEVICE)
    print(f"Student total params: {sum(p.numel() for p in student.parameters()):,}")
    audit_model_memory(student, bytes_per_param=1)

    # ── 3. Warm-start chain: B2 -> B1 -> random ─────────────────────────────
    if os.path.exists(config.CHECKPOINT_B2):
        print(f"Warm-starting from B2 checkpoint '{config.CHECKPOINT_B2}' ...")
        b2_sd = torch.load(config.CHECKPOINT_B2, map_location=DEVICE)
        # B2 and C share the same key layout (both use .linear.), so load directly
        missing, unexpected = student.load_state_dict(b2_sd, strict=False)
        print(
            f"  [warm-start B2] missing={len(missing)}  unexpected={len(unexpected)}\n"
        )
    elif os.path.exists(config.CHECKPOINT_B1):
        print(
            f"B2 checkpoint not found; warm-starting from B1 checkpoint "
            f"'{config.CHECKPOINT_B1}' ..."
        )
        b1_sd = torch.load(config.CHECKPOINT_B1, map_location=DEVICE)
        remap_b1_to_quant(b1_sd, student, verbose=True)
        print()
    else:
        print(
            "WARNING: Neither B2 nor B1 checkpoints found. "
            "Training from random initialization.\n"
        )

    # ── 4. Teacher model (eval + frozen) ────────────────────────────────────
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(
            f"\nERROR: Teacher checkpoint not found at '{config.CHECKPOINT_PATH}'.\n"
            "       Run Task A first:  python main.py\n",
            file=sys.stderr,
        )
        sys.exit(1)

    teacher = DigitGRU(input_size=t_feat_dim, hidden_size=128).to(DEVICE)
    teacher.load_state_dict(
        torch.load(config.CHECKPOINT_PATH, map_location=DEVICE)
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher loaded from '{config.CHECKPOINT_PATH}' (eval, frozen)\n")

    # ── 5. Optimiser / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=PATIENCE
    )
    criterion = nn.CrossEntropyLoss()

    # ── 6. Training loop (60 epochs) ─────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start           = time.time()
        tr_loss, tr_acc       = train_epoch(
            student, teacher, s_train, t_train, optimizer
        )
        va_loss, va_acc, _, _ = evaluate(student, s_val, criterion)
        epoch_secs            = time.time() - epoch_start
        scheduler.step(va_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        ckpt_flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(student.state_dict(), config.CHECKPOINT_C)
            ckpt_flag = " +"

        elapsed = time.time() - train_start
        eta     = (elapsed / epoch) * (EPOCHS - epoch)
        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"[{epoch_secs:.1f}s | elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m]"
            f"{ckpt_flag}"
        )

    with open(HISTORY_PATH, "w") as fh:
        json.dump(history, fh)
    print(f"\nTraining history saved to '{HISTORY_PATH}'")

    total_time = time.time() - train_start
    print(
        f"Training complete in {total_time/60:.1f} min  "
        f"|  best val acc: {best_val_acc:.3f}"
    )

    # ── 7. Load best checkpoint + post-snap evaluation ───────────────────────
    student.load_state_dict(
        torch.load(config.CHECKPOINT_C, map_location=DEVICE)
    )

    # Pre-snap accuracy (fake PoT weights during training)
    _, pre_snap_acc, preds, trues = evaluate(student, s_test, criterion)

    # Post-snap accuracy (deep copy — preserves trained model, snaps the copy)
    snapped = copy.deepcopy(student)
    snap_weights_to_pot(snapped)
    _, post_snap_acc, _, _ = evaluate(snapped, s_test, criterion)

    print(f"\n{'='*60}")
    print(f"Task C  Power-of-Two QAT — Test results")
    print(f"{'='*60}")
    print(f"  Pre-snap  accuracy : {pre_snap_acc:.4f}  ({pre_snap_acc*100:.1f}%)")
    print(f"  Post-snap accuracy : {post_snap_acc:.4f}  ({post_snap_acc*100:.1f}%)")
    print(f"{'='*60}\n")
    print(classification_report(trues, preds, target_names=[str(i) for i in range(10)]))

    # ── 8. PoT weight distribution table ─────────────────────────────────────
    print_pot_distribution(snapped)


if __name__ == "__main__":
    main()
