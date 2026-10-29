"""
task_b1_constrained.py
======================
Task B1: Memory-Constrained Spoken Digit Classifier with Knowledge Distillation

Architecture
------------
Custom MGU (Minimal Gated Unit) replacing GRU's 3-gate design with 2 gates.
This saves ~33% parameters at the same hidden size, allowing hidden=50 while
staying within the 36 kB per-layer memory constraint.

Layer breakdown (float32, 4 bytes/param):
  proj  Linear(39->32)   1,280 params =  5,120 bytes  < 36 kB  OK
  norm  LayerNorm(32)       64 params =    256 bytes  < 36 kB  OK
  mgu   MGU(32->50)      8,400 params = 33,600 bytes  < 36 kB  OK
          W_f Linear(32,50)  1,650 params  6,600 bytes  OK
          U_f Linear(50,50)  2,550 params 10,200 bytes  OK
          W_h Linear(32,50)  1,650 params  6,600 bytes  OK
          U_h Linear(50,50)  2,550 params 10,200 bytes  OK
  fc    Linear(50->10)     510 params =  2,040 bytes  < 36 kB  OK

MGU recurrence (Minimal Gated Unit):
  f_t  = sigmoid( W_f(x_t) + U_f(h_{t-1}) )          <- forget gate
  h~_t = tanh(    W_h(x_t) + U_h(f_t * h_{t-1}) )    <- new gate
  h_t  = (1 - f_t) * h_{t-1} + f_t * h~_t            <- update rule

Knowledge Distillation
----------------------
Teacher: DigitGRU (Task A, hidden=128, N_MFCC=40) loaded from best_model.pt.
Student: ConstrainedMGU (this file, hidden=50, N_MFCC=13).
Teacher and student see DIFFERENT feature representations of the same audio.

Loss: alpha * CE(student_logits, hard_labels)
    + (1-alpha) * KL( log_softmax(s/T) || softmax(t/T) ) * T^2
where alpha=0.3, T=3.0.

Data pipeline bugs already fixed:
  Bug 1: n_mfcc passed explicitly — no import-time value copy from config
  Bug 2: normalise first, then pad — zero-pad frames stay near-zero
  Bug 3: LayerNorm instead of BatchNorm — no running-stats mismatch at eval
"""

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
from main import DigitGRU                          # teacher architecture


# ─────────────────────────── Constants ─────────────────────────────────────

N_MFCC_STUDENT = 13      # student compact features: 13 MFCC + delta + delta2 = 39
N_MFCC_TEACHER = 40      # teacher full features:    40 MFCC + delta + delta2 = 120
HIDDEN         = 50      # MGU hidden dimension
INPUT_PROJ     = 32      # input projection bottleneck
NUM_CLASSES    = 10

ALPHA          = 0.3     # weight on hard-label CE loss
TEMPERATURE    = 3.0     # softmax temperature for distillation

EPOCHS         = 60
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 3       # ReduceLROnPlateau patience
MAX_GRAD_NORM  = 1.0

DEVICE         = config.DEVICE


# ─────────────────────────── MGU Implementation ────────────────────────────

class MGUCell(nn.Module):
    """
    Single-step Minimal Gated Unit cell.

    Parameters for input_size=I, hidden_size=H:
      W_f  Linear(I, H)  ->  I*H + H  params
      U_f  Linear(H, H)  ->  H*H + H  params
      W_h  Linear(I, H)  ->  I*H + H  params
      U_h  Linear(H, H)  ->  H*H + H  params
      Total = 2 * H * (I + H + 2)

    For I=32, H=50: 2*50*(32+50+2) = 8,400 params = 33,600 bytes < 36 kB OK
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # forget gate
        self.W_f = nn.Linear(input_size,  hidden_size, bias=True)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=True)
        # new gate
        self.W_h = nn.Linear(input_size,  hidden_size, bias=True)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x : (B, input_size)
        h : (B, hidden_size)
        returns h_new : (B, hidden_size)
        """
        f       = torch.sigmoid(self.W_f(x) + self.U_f(h))
        h_tilde = torch.tanh(self.W_h(x) + self.U_h(f * h))
        return (1.0 - f) * h + f * h_tilde


class MGU(nn.Module):
    """Unrolls MGUCell over a sequence of length T."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = MGUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        x  : (B, T, input_size)
        h0 : (B, hidden_size)  — zeros if not supplied
        returns final hidden state (B, hidden_size)
        """
        B, T, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        h = h0
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return h                                   # (B, hidden_size)


# ─────────────────────────── Student Model ─────────────────────────────────

class ConstrainedMGU(nn.Module):
    """
    Memory-constrained spoken digit classifier (Task B1).

    Input  : (B, T=100, F=39)   F = 3 * N_MFCC_STUDENT = 39
    Output : (B, 10)             class logits
    """

    def __init__(
        self,
        n_mfcc:      int = N_MFCC_STUDENT,
        hidden:      int = HIDDEN,
        input_proj:  int = INPUT_PROJ,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        feature_dim = 3 * n_mfcc                        # 39
        self.proj   = nn.Linear(feature_dim, input_proj)   # 39->32
        self.norm   = nn.LayerNorm(input_proj)              # 32
        self.mgu    = MGU(input_proj, hidden)               # 32->50
        self.fc     = nn.Linear(hidden, num_classes)        # 50->10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 39)
        x = self.proj(x)           # (B, T, 32) — vectorised across all timesteps
        x = F.relu(self.norm(x))   # (B, T, 32)
        h = self.mgu(x)            # (B, 50)
        return self.fc(h)          # (B, 10)


# ─────────────────────────── Distillation Loss ─────────────────────────────

def distillation_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    labels:   torch.Tensor,
    alpha:    float = ALPHA,
    T:        float = TEMPERATURE,
) -> torch.Tensor:
    """
    Combined hard-label CE + soft-target KL distillation loss.

      loss = alpha * CE(s_logits, labels)
           + (1-alpha) * KL( log_softmax(s/T) || softmax(t/T) ) * T^2

    The T^2 factor compensates for the T^2 reduction in gradient magnitude
    caused by temperature scaling (Hinton et al., 2015).
    """
    ce = F.cross_entropy(s_logits, labels)
    kl = F.kl_div(
        F.log_softmax(s_logits / T, dim=1),
        F.softmax(t_logits  / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * ce + (1.0 - alpha) * kl


# ─────────────────────────── Training helpers ──────────────────────────────

def train_epoch(
    student:   nn.Module,
    teacher:   nn.Module,
    s_loader,
    t_loader,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    """One epoch of distillation training. Returns (avg_loss, accuracy)."""
    student.train()
    total_loss, correct, total = 0.0, 0, 0

    for (x_s, y), (x_t, _) in zip(s_loader, t_loader):
        x_s = x_s.to(DEVICE)
        x_t = x_t.to(DEVICE)
        y   = y.to(DEVICE)

        optimizer.zero_grad()

        s_logits = student(x_s)
        with torch.no_grad():
            t_logits = teacher(x_t)       # teacher is frozen, no grad needed

        loss = distillation_loss(s_logits, t_logits, y)
        loss.backward()
        clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (s_logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module) -> tuple:
    """Evaluate on student logits. Returns (avg_loss, accuracy, preds, labels)."""
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
    # Same random_state=42 guarantees identical train/val/test splits.
    # Student sees 39-dim features, teacher sees 120-dim features.
    print("Building student loaders (N_MFCC=13) ...")
    s_train, s_val, s_test, s_feat_dim = build_loaders(
        n_mfcc=N_MFCC_STUDENT, random_state=42
    )
    print(f"  student feature_dim = {s_feat_dim}  (3 x {N_MFCC_STUDENT})")

    print("Building teacher loaders (N_MFCC=40) ...")
    t_train, _, _, t_feat_dim = build_loaders(
        n_mfcc=N_MFCC_TEACHER, random_state=42
    )
    print(f"  teacher feature_dim = {t_feat_dim}  (3 x {N_MFCC_TEACHER})\n")

    # ── 2. Student model + memory audit ─────────────────────────────────────
    student = ConstrainedMGU().to(DEVICE)
    print(f"Student total params: {sum(p.numel() for p in student.parameters()):,}")
    # audit_model_memory iterates named_modules with recurse=False, so each
    # individual Linear and LayerNorm is checked against the 36 kB limit.
    audit_model_memory(student, bytes_per_param=4)   # aborts if any layer fails

    # ── 3. Teacher model (eval + frozen) ────────────────────────────────────
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

    # ── 4. Optimiser / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=PATIENCE
    )
    criterion = nn.CrossEntropyLoss()      # used for validation/test CE only

    # ── 5. Training loop (60 epochs) ─────────────────────────────────────────
    best_val_acc = 0.0
    train_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start           = time.time()
        tr_loss, tr_acc       = train_epoch(
            student, teacher, s_train, t_train, optimizer
        )
        va_loss, va_acc, _, _ = evaluate(student, s_val, criterion)
        epoch_secs            = time.time() - epoch_start
        scheduler.step(va_acc)

        ckpt_flag = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            # Save ONLY student state_dict — teacher weights never stored here
            torch.save(student.state_dict(), config.CHECKPOINT_B1)
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

    total_time = time.time() - train_start
    print(
        f"\nTraining complete in {total_time/60:.1f} min  "
        f"|  best val acc: {best_val_acc:.3f}"
    )

    # ── 6. Final test evaluation ──────────────────────────────────────────────
    student.load_state_dict(
        torch.load(config.CHECKPOINT_B1, map_location=DEVICE)
    )
    _, test_acc, preds, trues = evaluate(student, s_test, criterion)

    print(f"\n{'='*55}")
    print(f"B1 Constrained — Test accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"{'='*55}\n")
    print(classification_report(trues, preds, target_names=[str(i) for i in range(10)]))


if __name__ == "__main__":
    main()
