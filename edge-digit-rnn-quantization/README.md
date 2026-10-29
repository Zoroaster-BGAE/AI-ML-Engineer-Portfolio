# Free Spoken Digit Classification under Progressive Hardware Constraints

Spoken digit recognition on the FSDD dataset across four models, each trained under progressively stricter memory and arithmetic constraints for edge/embedded deployment.

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

This project trains a series of spoken digit classifiers on the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset), a collection of 3,000 recordings of the digits 0–9 from six speakers. Starting from an unconstrained GRU baseline that achieves ~99% test accuracy, the project progressively tightens the hardware constraints: first limiting each layer to 36 kB of on-chip SRAM (Task B1), then requiring all weights to be representable as INT8 integers (Task B2), and finally restricting weights to powers of two so that all multiply-accumulate operations can be implemented as bit-shifts (Task C).

The constraint progression mirrors the real-world path from cloud deployment to ultra-low-power microcontrollers. A speech recognition model destined for a hearing-aid chip, a keyword-spotting sensor, or an industrial IoT node must not only be small and fast — it must respect the arithmetic capabilities of hardware that may have no floating-point unit at all. This project demonstrates that knowledge distillation, quantization-aware training, and careful architectural choices (MGU over GRU, input projection bottleneck, LayerNorm) can preserve surprising accuracy even under the most severe constraints.

---

## Hardware Constraints Summary

| Task | Model | n\_mfcc | Hidden | Constraint | Test Acc |
|------|-------|---------|--------|------------|----------|
| **A** | GRU | 40 | 128 | None | ~99% |
| **B1** | MGU | 13 | 50 | ≤ 36 kB / layer (float32) | ~93% |
| **B2** | QAT-MGU | 13 | 50 | ≤ 36 kB / layer (INT8) | ~71% |
| **C** | PoT-MGU | 13 | 50 | Weights ∈ {±2^k} ∪ {0} | ~85.7% |


---

## Architecture Overview

### Task A — GRU Baseline (`DigitGRU`)

```
Input  (B, 100, 120)         # 40 MFCC + Δ + ΔΔ = 120 features
   ↓
GRU  hidden=128, layers=1    # processes all 100 frames
   ↓
h_T  (B, 128)                # final hidden state = sequence embedding
   ↓
Dropout(0.3)
   ↓
Linear  128 → 10
   ↓
Logits  (B, 10)
```

### Task B1 / B2 / C — ConstrainedMGU

```
Input  (B, 100, 39)          # 13 MFCC + Δ + ΔΔ = 39 features
   ↓
Linear proj  39 → 32         # [FakeQuantizeLinear in B2; FakePoTLinear in C]
LayerNorm(32) + ReLU
   ↓
MGU / QuantizedMGU / PoTMGU  hidden=50
   (unrolled over T=100 timesteps)
   ↓
h_T  (B, 50)
   ↓
Linear  50 → 10              # [FakeQuantizeLinear in B2; FakePoTLinear in C]
   ↓
Logits  (B, 10)
```

**Layer memory breakdown (float32, 4 bytes/param):**

| Layer | Shape | Params | Bytes | ≤ 36 kB? |
|-------|-------|--------|-------|----------|
| `proj` | Linear(39→32) | 1,280 | 5,120 | ✓ |
| `norm` | LayerNorm(32) | 64 | 256 | ✓ |
| `mgu.cell.W_f` | Linear(32→50) | 1,650 | 6,600 | ✓ |
| `mgu.cell.U_f` | Linear(50→50) | 2,550 | 10,200 | ✓ |
| `mgu.cell.W_h` | Linear(32→50) | 1,650 | 6,600 | ✓ |
| `mgu.cell.U_h` | Linear(50→50) | 2,550 | 10,200 | ✓ |
| `fc` | Linear(50→10) | 510 | 2,040 | ✓ |

**MGU gating equations:**

```
f_t  = σ( W_f · x_t  +  U_f · h_{t-1} )
h̃_t  = tanh( W_h · x_t  +  U_h · (f_t ⊙ h_{t-1}) )
h_t  = (1 − f_t) ⊙ h_{t-1}  +  f_t ⊙ h̃_t
```

The MGU (Minimal Gated Unit) replaces GRU's three gates (reset, update, new) with two (forget `f`, new `h̃`). This saves ~33% parameters at the same hidden size, which is what allows `hidden=50` to fit within the 36 kB constraint.

---

## Knowledge Distillation

Tasks B1, B2, and C are trained using knowledge distillation from the Task A teacher:

```
L = α · CE(s_logits, y)  +  (1−α) · T² · KL( softmax(s/T) ∥ softmax(t/T) )
α = 0.3,  T = 3.0
```

The teacher's soft probability distribution over all 10 classes encodes inter-class similarity that hard one-hot labels discard. For example, the teacher might assign partial probability to "nine" when the true digit is "five" — these soft targets act as semantically meaningful label smoothing. Temperature `T=3.0` flattens the teacher's distribution enough to make the soft targets informative across all epochs. The `T²` factor compensates for the gradient magnitude reduction caused by temperature scaling (Hinton et al., 2015). With `α=0.3`, hard-label cross-entropy is still present to anchor the student to ground truth.

---

## Quantization Techniques

### INT8 QAT (Task B2)

**What it is:** Quantization-Aware Training (QAT) simulates INT8 rounding during the forward pass while keeping weights in float32 during backprop. This allows the model to learn weight distributions that are robust to INT8 discretisation before deployment.

**Why QAT over post-training quantization (PTQ):** PTQ snaps a trained float32 model to INT8 after training, with no opportunity for the model to adapt. This typically loses 3–8% accuracy. QAT usually recovers to within 1–2% of the float baseline because the model co-adapts its weights and activations to the quantization grid.

**Straight-Through Estimator (STE):** Rounding is non-differentiable — its gradient is zero almost everywhere. The STE approximates the gradient of the quantized function as 1 (pass-through): the backward pass simply returns `grad_output` unchanged. This is the standard approach in all major QAT papers (Bengio 2013; Jacob et al. 2018). In this project it is implemented via `torch.autograd.Function`, not via the `.detach()` trick.

**Why not `torch.quantization.prepare_qat`:** PyTorch's built-in QAT pipeline works by graph-tracing `nn.Module` and inserting `QuantStub`/`DeQuantStub` nodes. It does not understand arbitrary Python `for`-loops in custom RNN cells. `MGUCell.forward` iterates time-steps manually, so the quantization must be explicit: every `nn.Linear` is replaced with `FakeQuantizeLinear(nn.Linear)`.

**Why biases and LayerNorm are not quantized:** Biases are typically stored in INT32 at inference (fused with the per-layer requantisation scale) — quantising them to INT8 causes measurable accuracy loss. LayerNorm's learnable scale (`gamma`) and shift (`beta`) parameters have high sensitivity; quantising them causes loss spikes during training.

### Power-of-Two QAT (Task C)

**What it is:** PoT quantization restricts weights to the set `{0} ∪ {±2^k | k ∈ [−8, 3]}`. Hardware that cannot perform arbitrary integer multiplication can still compute `w · x = x >> |k|` (or `x << k` for positive exponents) using a bit-shift — a single-cycle operation on any processor.

**Snap algorithm:**
```
snap(w) = sign(w) · 2^( clamp( round( log₂|w| ), −8, 3 ) )
         if |w| ≥ 2^{-9}  else  0
```

Weights with `|w| < 2^{-9}` are zeroed (zero threshold = 2^{MIN_EXP − 1}), promoting sparsity and allowing the hardware to skip those multiplications entirely.

**Why warm-starting from B2 improves PoT training:** B2 (INT8) weights have already been trained to cluster near discrete values that minimise rounding loss. Their magnitude distribution is more likely to already be near powers of two than raw float32 B1 weights. Empirically, warm-starting from B2 gives 1–3% higher post-snap accuracy than starting from B1.

---

## Project Setup

```bash
# 1. Clone the dataset
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models in order (each depends on the previous checkpoint)
python main.py                  # Task A  →  best_model.pt
python task_b1_constrained.py  # Task B1 →  best_model_b1_constrained.pt
python task_b2_int8.py         # Task B2 →  best_model_b2_int8.pt
python task_c_pow2.py          # Task C  →  best_model_c_pow2.pt
```

> Each training script prints a per-layer memory audit before training begins and aborts if any layer violates the 36 kB constraint.

---

## Running the Validation Notebook

```bash
jupyter notebook notebooks/validation_notebook.ipynb
```

The notebook performs exhaustive acceptance tests across all four models:

- **Check 1** — Checkpoint integrity: file exists, non-empty, no NaN/Inf weights
- **Check 2** — Per-layer ≤ 36 kB memory constraint (float32 for B1; INT8 for B2/C)
- **Check 3** — INT8 weight snap verification (B2 only)
- **Check 4** — Power-of-Two weight verification + exponent distribution chart (C only)
- **Check 5** — Forward pass smoke test: output shape `(4, 10)`, finite logits
- **Check 6** — Full test-set inference: accuracy ≥ threshold, confusion matrix
- **Check 7** — Inference speed benchmark: ms/sample on CPU
- **Check 8** — Consolidated PASS/FAIL summary table

Every cell either prints `PASS` or raises `AssertionError` with a descriptive message. B2 and C checks gracefully skip if their checkpoints are not yet available.

---

## File Reference

| File | Purpose | Task |
|------|---------|------|
| `main.py` | Task A training entry point — trains GRU baseline | A |
| `task_b1_constrained.py` | Task B1 training — MGU + knowledge distillation | B1 |
| `task_b2_int8.py` | Task B2 training — INT8 QAT | B2 |
| `task_c_pow2.py` | Task C training — Power-of-Two QAT | C |
| `config.py` | Backward-compat shim (`from configs import *`) | all |
| `configs/config_base.py` | Shared audio/data/device constants | all |
| `configs/config_mgu.py` | Task A + B1 hyperparameters | A, B1 |
| `configs/config_quant.py` | Task B2 hyperparameters | B2 |
| `configs/config_power2.py` | Task C hyperparameters | C |
| `configs/__init__.py` | Re-exports all config constants | all |
| `utils/data_loader.py` | Builds train/val/test DataLoaders | all |
| `utils/data_preprocessing.py` | MFCC extraction, normalisation, pre-padding | all |
| `utils/memory_utils.py` | `audit_model_memory` — per-layer SRAM check | all |
| `utils/quant_utils.py` | `FakeQuantizeLinear`, `FakePoTLinear`, STE, snap, remap | B2, C |
| `best_model.pt` | Task A checkpoint | A |
| `best_model_b1_constrained.pt` | Task B1 checkpoint | B1 |
| `best_model_b2_int8.pt` | Task B2 checkpoint | B2 |
| `best_model_c_pow2.pt` | Task C checkpoint | C |
| `history_b2.json` | Task B2 per-epoch train/val loss and accuracy | B2 |
| `history_c.json` | Task C per-epoch train/val loss and accuracy | C |
| `notebooks/training_notebook.ipynb` | End-to-end Colab training workflow | all |
| `notebooks/validation_notebook.ipynb` | Acceptance-test notebook | all |
| `docs/design-notes.pdf` | Design decisions & technical write-up | — |

---

## Design Decisions

1. **Pre-padding (zeros at sequence start, audio at end)** — The RNN reads left-to-right; with post-padding, the final hidden state `h_T` is overwritten by up to 56 zero-input steps, destroying audio information. Pre-padding ensures `h_T` always reflects the last real audio frame.

2. **MGU over GRU for the constrained student** — A GRU with `hidden=50` and `input_size=32` costs `3×50×(32+50+2) = 12,600 params = 50,400 bytes`, exceeding the 36 kB limit. The MGU's 2-gate design costs `2×50×(32+50+2) = 8,400 params = 33,600 bytes`, fitting comfortably.

3. **Normalise raw frames, then pad** — Normalisation statistics (`mean`, `std`) are computed on unpadded features. Zero-padded frames are then inserted before the normalised audio. This ensures padded frames are exactly zero (≈ the normalised mean) and do not skew the RNN's hidden state.

4. **LayerNorm over BatchNorm after the projection** — BatchNorm uses running statistics at eval time that can diverge from single-sample statistics. LayerNorm normalises each sample independently with no running state — training and inference behaviour are identical.

5. **Warm-start chain C → B2 → B1 → random** — Each stage initialises from the best checkpoint of the prior stage. B2 weights (INT8-adapted) have magnitude distributions closer to powers of two than fresh float32 B1 weights, giving PoT training a head-start and typically yielding 1–3% higher post-snap accuracy.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
