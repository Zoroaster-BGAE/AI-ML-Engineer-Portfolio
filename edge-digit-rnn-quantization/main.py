"""
main.py
=======
Task A: Free Spoken Digits — GRU Classifier

Entry point.  Orchestrates data loading, model training, and evaluation.

Usage
-----
  # 1. Clone the dataset
  git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git

  # 2. Install dependencies
  pip install torch torchaudio librosa scikit-learn numpy

  # 3. Train
  python main.py
"""

import time

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import (
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, NUM_CLASSES,
    LR, EPOCHS, DEVICE, CHECKPOINT_PATH,
)
from utils.data_loader import build_loaders


# ─────────────────────────── Model ─────────────────────────────────────────

class DigitGRU(nn.Module):
    """
    GRU-based classifier for spoken digit recognition.

    Architecture
    ------------
    Input  : (batch, T, feature_dim)
    GRU    : 2-layer GRU that processes the spectrogram frame-by-frame.
             At each step t the GRU computes:
               h_t = GRU(x_t, h_{t-1})
             The update gate decides how much of h_{t-1} to carry forward;
             the reset gate controls how much past context influences the
             candidate activation.  This means h_T is a *compressed memory*
             of the entire sequence — not just the last frame.
    h_T    : final hidden state of the last GRU layer  →  sequence embedding
    Linear : maps the 128-dim embedding to 10 class logits.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int  = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float   = DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x    : (B, T, F)
        # out  : (B, T, hidden_size)  — all hidden states h_1 … h_T
        # h_n  : (num_layers, B, hidden_size)
        out, h_n = self.gru(x)

        # Use the last layer's final hidden state as the sequence summary
        last_hidden = h_n[-1]               # (B, hidden_size)
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)         # (B, num_classes)


# ─────────────────────────── Training helpers ──────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
) -> tuple[float, float, list, list]:
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
    print(f"Using device: {DEVICE}\n")

    # 1. Data
    train_loader, val_loader, test_loader, feature_dim = build_loaders()

    # 2. Model
    model     = DigitGRU(input_size=feature_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input dim        : {feature_dim}  (MFCC + Δ + ΔΔ)")
    print(f"Hidden size      : {HIDDEN_SIZE}  |  Layers: {NUM_LAYERS}\n")

    # 3. Training loop
    best_val_acc = 0.0
    train_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start             = time.time()
        tr_loss, tr_acc         = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, _, _   = evaluate(model, val_loader, criterion)
        epoch_secs              = time.time() - epoch_start
        scheduler.step(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            ckpt_flag = " ✓"
        else:
            ckpt_flag = ""

        elapsed_total = time.time() - train_start
        remaining     = (elapsed_total / epoch) * (EPOCHS - epoch)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.3f}  "
            f"[{epoch_secs:.1f}s/epoch | elapsed={elapsed_total/60:.1f}m | eta={remaining/60:.1f}m]"
            f"{ckpt_flag}"
        )

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time/60:.1f} min  |  best val acc: {best_val_acc:.3f}")

    # 4. Test evaluation
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    _, test_acc, preds, trues = evaluate(model, test_loader, criterion)

    print(f"\n{'='*55}")
    print(f"Test accuracy: {test_acc:.4f}  ({test_acc * 100:.1f}%)")
    print(f"{'='*55}\n")
    print(classification_report(trues, preds, target_names=[str(i) for i in range(10)]))


if __name__ == "__main__":
    main()
