"""
task_b2_int8.py
===============
Task B2: INT8 Quantization-Aware Training (QAT) for Spoken Digit Classification

Architecture
------------
QuantizedConstrainedMGU — mirrors ConstrainedMGU from Task B1 exactly, but
every nn.Linear is replaced by FakeQuantizeLinear (which wraps nn.Linear as
self.linear and applies per-tensor symmetric INT8 fake quantization via STE).

LayerNorm parameters are NOT quantized (training instability).
Biases are NOT quantized (standard practice: left in float32).

Memory constraint
-----------------
At INT8 (1 byte/param) the 36 kB limit = 36,000 params/layer.
The same per-layer param counts as B1 (which passed the float32 audit) are
well within this INT8 limit:
  proj   : 1,280 params  * 1 byte =  1,280 bytes  < 36,000  OK
  norm   :    64 params  * 1 byte =     64 bytes  < 36,000  OK
  mgu.W_f:  1,650 params * 1 byte =  1,650 bytes  < 36,000  OK
  mgu.U_f:  2,550 params * 1 byte =  2,550 bytes  < 36,000  OK
  mgu.W_h:  1,650 params * 1 byte =  1,650 bytes  < 36,000  OK
  mgu.U_h:  2,550 params * 1 byte =  2,550 bytes  < 36,000  OK
  fc     :    510 params * 1 byte =    510 bytes  < 36,000  OK

Training
--------
Warm-start from Task B1 checkpoint (best_model_b1_constrained.pt) via
remap_b1_to_quant, which inserts ".linear." into each key:
  "proj.weight" -> "proj.linear.weight"
  "norm.weight" -> "norm.weight"   (LayerNorm key unchanged)
  "mgu.cell.W_f.weight" -> "mgu.cell.W_f.linear.weight"

Knowledge distillation (imported from task_b1_constrained) is used with
the same DigitGRU teacher as in B1.  History is saved to history_b2.json.
"""

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
    FakeQuantizeLinear,
    snap_weights_to_int8,
    remap_b1_to_quant,
)
from main import DigitGRU                             # teacher architecture
from task_b1_constrained import distillation_loss     # reuse, do not re-implement


# ─────────────────────────── Constants ─────────────────────────────────────

N_MFCC_STUDENT = 13      # student compact features: 13 MFCC + Δ + ΔΔ = 39
N_MFCC_TEACHER = 40      # teacher full features:    40 MFCC + Δ + ΔΔ = 120
HIDDEN         = 50      # MGU hidden dimension (same as B1)
INPUT_PROJ     = 32      # input projection bottleneck (same as B1)
NUM_CLASSES    = 10

ALPHA          = 0.3     # weight on hard-label CE loss
TEMPERATURE    = 3.0     # softmax temperature for distillation

EPOCHS         = config.EPOCHS_B2       # 60
LR             = config.LR_B2           # 5e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 3
MAX_GRAD_NORM  = 1.0

DEVICE         = config.DEVICE
HISTORY_PATH   = "history_b2.json"


# ─────────────────────────── Quantized MGU ─────────────────────────────────

class QuantizedMGUCell(nn.Module):
    """
    MGU cell with FakeQuantizeLinear on all weight matrices.
    Mirrors MGUCell from task_b1_constrained but uses INT8 fake quantization.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # forget gate — quantized
        self.W_f = FakeQuantizeLinear(input_size,  hidden_size, bias=True)
        self.U_f = FakeQuantizeLinear(hidden_size, hidden_size, bias=True)
        # new gate — quantized
        self.W_h = FakeQuantizeLinear(input_size,  hidden_size, bias=True)
        self.U_h = FakeQuantizeLinear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        f       = torch.sigmoid(self.W_f(x) + self.U_f(h))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(f * h))
        return (1.0 - f) * h + f * h_tilde


class QuantizedMGU(nn.Module):
    """Unrolls QuantizedMGUCell over a sequence."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = QuantizedMGUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        h = h0
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return h


class QuantizedConstrainedMGU(nn.Module):
    """
    INT8 QAT version of ConstrainedMGU (Task B2).

    Input  : (B, T=100, F=39)
    Output : (B, 10)

    proj and fc use FakeQuantizeLinear (INT8 fake quant on weights).
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
        feature_dim = 3 * n_mfcc                                     # 39
        self.proj   = FakeQuantizeLinear(feature_dim, input_proj)     # 39->32
        self.norm   = nn.LayerNorm(input_proj)                        # 32 (float32)
        self.mgu    = QuantizedMGU(input_proj, hidden)                # 32->50
        self.fc     = FakeQuantizeLinear(hidden, num_classes)         # 50->10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)           # (B, T, 32)
        x = F.relu(self.norm(x))   # (B, T, 32)
        h = self.mgu(x)            # (B, 50)
        return self.fc(h)          # (B, 10)


# ─────────────────────────── Training helpers ───────────────────────────────

def train_epoch(
    student:   nn.Module,
    teacher:   nn.Module,
    s_loader,
    t_loader,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    """One epoch of distillation QAT. Returns (avg_loss, accuracy)."""
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
    """Returns (avg_loss, accuracy, preds, labels)."""
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


# ─────────────────────────── Main ──────────────────────────────────────────

def main():
    print(f"Using device : {DEVICE}")
    print(f"N_MFCC  student={N_MFCC_STUDENT}  teacher={N_MFCC_TEACHER}\n")

    # ── 1. Dual dataloaders ──────────────────────────────────────────────────
    print("Building student loaders (N_MFCC=13) ...")
    s_train, s_val, s_test, s_feat_dim = build_loaders(
        n_mfcc=N_MFCC_STUDENT, random_state=42
    )
    print(f"  student feature_dim = {s_feat_dim}")

    print("Building teacher loaders (N_MFCC=40) ...")
    t_train, _, _, t_feat_dim = build_loaders(
        n_mfcc=N_MFCC_TEACHER, random_state=42
    )
    print(f"  teacher feature_dim = {t_feat_dim}\n")

    # ── 2. Student model + memory audit (INT8: bytes_per_param=1) ────────────
    student = QuantizedConstrainedMGU().to(DEVICE)
    print(f"Student total params: {sum(p.numel() for p in student.parameters()):,}")
    audit_model_memory(student, bytes_per_param=1)   # 36,000 params/layer limit

    # ── 3. Warm-start from B1 checkpoint ─────────────────────────────────────
    if os.path.exists(config.CHECKPOINT_B1):
        print(f"Loading B1 weights from '{config.CHECKPOINT_B1}' and remapping ...")
        b1_sd = torch.load(config.CHECKPOINT_B1, map_location=DEVICE)
        remap_b1_to_quant(b1_sd, student, verbose=True)
        print()
    else:
        print(
            f"WARNING: B1 checkpoint '{config.CHECKPOINT_B1}' not found. "
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
            torch.save(student.state_dict(), config.CHECKPOINT_B2)
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

    # ── 7. Load best checkpoint + final test evaluation ──────────────────────
    student.load_state_dict(
        torch.load(config.CHECKPOINT_B2, map_location=DEVICE)
    )

    # Pre-snap accuracy (fake-quantized training weights, float32 repr)
    _, pre_snap_acc, preds, trues = evaluate(student, s_test, criterion)

    # Post-snap accuracy (weights snapped to INT8 float grid in-place on a copy)
    snapped = copy.deepcopy(student)
    snap_weights_to_int8(snapped)
    _, post_snap_acc, _, _ = evaluate(snapped, s_test, criterion)

    print(f"\n{'='*60}")
    print(f"B2 INT8 QAT — Test results")
    print(f"{'='*60}")
    print(f"  Pre-snap  accuracy : {pre_snap_acc:.4f}  ({pre_snap_acc*100:.1f}%)")
    print(f"  Post-snap accuracy : {post_snap_acc:.4f}  ({post_snap_acc*100:.1f}%)")
    print(f"{'='*60}\n")
    print(classification_report(trues, preds, target_names=[str(i) for i in range(10)]))


if __name__ == "__main__":
    main()
