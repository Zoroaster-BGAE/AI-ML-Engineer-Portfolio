# Turbine modal analysis ML pipeline (portfolio release)

Reference code for a predictive-maintenance pipeline on structural vibration data: a data-concatenation pipeline and a
**synthetic signal generator with known ground truth**, so estimators can be tested without any real turbine data.

> **About this repository**
> This is a **sanitised portfolio release** of pipeline code originally written for an employer, published with the
> permission of its rights holders. It is **not** a clean-room reimplementation: the pipeline code under `pipelines/` and
> `src/oma/date_binning.py` derives from that work, with identifying details removed or replaced (`COMPANY`, `StructureN`,
> `example-bucket-N`). The signal generator (`src/synth.py`) and its tests were **written from scratch** for this repository,
> contain no third-party code or data, and are MIT-licensed.
>
> **No real data is included.** Everything the tests use is synthetic and generated with a fixed seed. Architecture and
> deployment details have been **generalised to protect proprietary information**.
>
> Apart from the generator and its tests, files are published for portfolio review and no licence for reuse is granted.
> Not affiliated with or endorsed by any third party.

## What is here

| Path | What it is |
|---|---|
| `src/synth.py` | Synthetic multi-channel vibration at 2600 Hz: damped modes with known frequency, damping and mode shapes, rotor harmonics, drift, noise at a requested SNR |
| `tests/test_synth.py` | Tests that the generator is deterministic and that its known modes are recoverable from the signal |
| `pipelines/concatenation/` | Container pipeline that selects healthy files per day and speed bin and concatenates them |
| `src/oma/date_binning.py` | Date and speed-bin helpers |
| `scripts/fetch_third_party.sh` | Fetches upstream open-source code into a git-ignored folder; nothing is vendored |
| `THIRD_PARTY.md` | Upstream attribution and licence status |

## What is NOT here

- Real sensor data, real configuration, or any cloud identifier.
- The SageMaker / Lambda deployment code. It is not in this repository.
- Upstream open-source code (fetched, not vendored).
- The mode-clustering pipeline: it adapts functions from an open-source library, so it is withheld until the licence is confirmed.

## Run the tests

```bash
pip install -r requirements.txt
pytest
```

```python
from synth import SynthConfig, make_signal

X, truth = make_signal(SynthConfig(duration_s=120, snr_db=15, seed=1))
# X: (n_channels, n_samples) at 2600 Hz; truth: known frequencies, damping ratios, mode shapes, rotor harmonics
```

## Licence

`src/synth.py` and `tests/test_synth.py` are released under the MIT licence (`LICENSES/MIT.txt`). All other files are
published for portfolio review only, and no licence for reuse is granted.
