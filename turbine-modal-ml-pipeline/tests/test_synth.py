# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nikos Alexoudis

"""Tests for the synthetic generator: determinism, validation, SNR, and recovery of the KNOWN modes from the signal."""

import numpy as np
import pytest
from scipy import signal

from synth import SynthConfig, decimate_to, make_signal

FS = 2600.0
TARGET_FS = 50.0


def _psd(x, fs, nperseg):
    return signal.welch(x, fs=fs, nperseg=nperseg, axis=-1)


@pytest.fixture(scope="module")
def clean_record():
    """High SNR, stationary, no rotor harmonics: the modes are the only lines in the spectrum."""
    cfg = SynthConfig(duration_s=600.0, snr_db=30.0, rotor_rpm=None, drift=0.0, seed=7)
    x, truth = make_signal(cfg)
    xd, fs_d = decimate_to(x, cfg.fs, TARGET_FS)
    return cfg, truth, xd, fs_d


def test_deterministic_for_a_seed_and_different_across_seeds():
    cfg = SynthConfig(duration_s=60.0, seed=3)
    a, _ = make_signal(cfg)
    b, _ = make_signal(cfg)
    c, _ = make_signal(SynthConfig(duration_s=60.0, seed=4))
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_shape_dtype_and_truth_are_consistent():
    cfg = SynthConfig(duration_s=30.0, n_channels=6)
    x, truth = make_signal(cfg)
    assert x.shape == (6, int(FS * 30.0)) and x.dtype == np.float64 and np.isfinite(x).all()
    assert truth.n_samples == x.shape[1] and truth.fs == FS
    assert truth.mode_shapes.shape == (6, len(cfg.modes_hz))
    assert np.allclose(np.linalg.norm(truth.mode_shapes, axis=0), 1.0)
    assert np.allclose(truth.mode_shapes.T @ truth.mode_shapes, np.eye(len(cfg.modes_hz)), atol=1e-12)  # orthonormal


@pytest.mark.parametrize("kwargs", [
    {"modes_hz": (2000.0,), "damping": (0.01,), "amplitudes": (1.0,)},      # above Nyquist
    {"modes_hz": (1.0, 2.0), "damping": (0.01,), "amplitudes": (1.0, 1.0)},  # length mismatch
    {"damping": (0.01, -0.02, 0.01, 0.01)},                                 # negative damping
    {"duration_s": 5.0},                                                   # shorter than a drift block
    {"rotor_rpm": 100000.0},                                               # harmonic above Nyquist
])
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SynthConfig(**kwargs)


def test_requested_snr_is_met():
    cfg = SynthConfig(duration_s=60.0, snr_db=12.0, seed=2)
    x, truth = make_signal(cfg, return_components=True)
    sig, noise = truth.components["signal"], truth.components["noise"]
    assert np.allclose(x, sig + noise)
    measured = 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
    assert measured == pytest.approx(cfg.snr_db, abs=0.05)


def test_noiseless_when_snr_is_infinite():
    x, truth = make_signal(SynthConfig(duration_s=30.0, snr_db=float("inf")), return_components=True)
    assert truth.noise_power == 0.0 and np.array_equal(x, truth.components["signal"])


def test_every_known_mode_is_a_peak_at_the_right_frequency(clean_record):
    cfg, truth, xd, fs_d = clean_record
    f, pxx = _psd(xd, fs_d, nperseg=8192)  # ~0.006 Hz resolution
    total = pxx.sum(axis=0)
    for f_true in truth.freqs_hz:
        band = (f > f_true - 0.12) & (f < f_true + 0.12)
        f_peak = f[band][np.argmax(total[band])]
        assert abs(f_peak - f_true) < 0.02, f"peak at {f_peak:.3f} Hz, truth {f_true:.3f} Hz"
        around = (f > f_true - 1.0) & (f < f_true + 1.0)
        assert total[band].max() > 5 * np.median(total[around]), "mode is not clearly above the local floor"


def test_mode_shapes_are_recoverable_from_the_cross_spectrum(clean_record):
    """At each true frequency the dominant singular vector of the cross-spectral matrix is the mode shape (FDD)."""
    cfg, truth, xd, fs_d = clean_record
    n_ch = xd.shape[0]
    f, _ = signal.csd(xd[0], xd[0], fs=fs_d, nperseg=8192)
    g = np.empty((n_ch, n_ch, f.size), dtype=complex)
    for i in range(n_ch):
        for j in range(n_ch):
            g[i, j] = signal.csd(xd[i], xd[j], fs=fs_d, nperseg=8192)[1]
    for k, f_true in enumerate(truth.freqs_hz):
        idx = np.argmin(np.abs(f - f_true))
        u, _, _ = np.linalg.svd(g[:, :, idx])
        est, ref = np.real(u[:, 0]), truth.mode_shapes[:, k]
        mac = (est @ ref) ** 2 / ((est @ est) * (ref @ ref))
        assert mac > 0.9, f"mode {k} ({f_true} Hz): MAC {mac:.3f}"


def test_rotor_harmonics_are_lines_and_are_reported():
    cfg = SynthConfig(duration_s=600.0, snr_db=30.0, rotor_rpm=14.0, harmonics=(1, 3), drift=0.0, seed=5)
    x, truth = make_signal(cfg)
    assert np.allclose(truth.harmonics_hz, [14 / 60, 3 * 14 / 60])
    xd, fs_d = decimate_to(x, cfg.fs, TARGET_FS)
    f, pxx = _psd(xd, fs_d, nperseg=8192)
    total = pxx.sum(axis=0)
    for fh in truth.harmonics_hz:
        band = (f > fh - 0.05) & (f < fh + 0.05)
        assert abs(f[band][np.argmax(total[band])] - fh) < 0.02

    _, truth_off = make_signal(SynthConfig(duration_s=30.0, rotor_rpm=None))
    assert truth_off.harmonics_hz.size == 0


def test_drift_widens_the_frequency_range_and_stays_finite():
    stationary = make_signal(SynthConfig(duration_s=60.0, drift=0.0))[1]
    drifting_x, drifting = make_signal(SynthConfig(duration_s=60.0, drift=0.04))
    assert np.array_equal(stationary.freq_min_hz, stationary.freq_max_hz)
    assert np.all(drifting.freq_max_hz - drifting.freq_min_hz == pytest.approx(0.04 * drifting.freqs_hz))
    assert np.isfinite(drifting_x).all()


def test_decimate_to_rejects_a_non_integer_ratio():
    with pytest.raises(ValueError):
        decimate_to(np.zeros((1, 2600)), 2600.0, 77.0)
