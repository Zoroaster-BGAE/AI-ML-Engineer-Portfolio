# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Nikos Alexoudis

"""Synthetic structural-vibration generator with KNOWN ground truth.

Why this exists: modal-analysis and clustering code can only be tested honestly if the right answer is known. Real
measurements never come with one, and real turbine data cannot be shared. This module builds multi-channel acceleration
records at 2600 Hz whose natural frequencies, damping ratios and mode shapes are chosen by the caller, plus the things
that make real records hard:

* ambient (broadband) excitation of every mode,
* deterministic rotor harmonics (1P, 3P, ...), which look like undamped modes and are the classic false positive,
* slow frequency drift (temperature-like), so a record is not stationary,
* white plus coloured measurement noise at a requested SNR.

Model. Mode i is the response of  q'' + 2*zeta*omega*q' + omega^2*q = u(t)  to white excitation u, observed as
acceleration, i.e. through  H(s) = s^2 / (s^2 + 2*zeta*omega*s + omega^2).  H is discretised exactly (zero-order hold)
and run as a second-order section, so the identified frequency and damping match the request. Channel c sees
sum_i phi[c, i] * a_i(t), with phi an orthonormal set of mode shapes when n_channels >= n_modes.

The default frequencies, damping ratios and amplitudes are ILLUSTRATIVE ONLY. They are not taken from any real structure.
Units are arbitrary acceleration units. Everything is reproducible from `seed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

FS_DEFAULT = 2600.0


@dataclass(frozen=True)
class SynthConfig:
    fs: float = FS_DEFAULT
    duration_s: float = 600.0
    n_channels: int = 4
    modes_hz: tuple[float, ...] = (0.43, 1.37, 3.11, 6.29)
    damping: tuple[float, ...] = (0.012, 0.018, 0.009, 0.025)
    amplitudes: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)  # RMS of each mode's acceleration before mixing
    rotor_rpm: float | None = 14.0  # None disables the harmonics
    harmonics: tuple[int, ...] = (1, 3)  # multiples of the rotor frequency (1P, 3P)
    harmonic_amplitude: float = 0.5
    drift: float = 0.0  # fractional peak-to-peak change of every modal frequency over the record, e.g. 0.01 = 1 %
    snr_db: float = 20.0  # total signal power over total noise power; float("inf") means noiseless
    colored_noise_fraction: float = 0.5  # share of the noise power that is low-pass (coloured) instead of white
    block_s: float = 10.0  # drift is realised in blocks of this length
    seed: int = 0

    def __post_init__(self) -> None:
        n = len(self.modes_hz)
        if not (len(self.damping) == len(self.amplitudes) == n and n >= 1):
            raise ValueError("modes_hz, damping and amplitudes must have the same, non-zero length")
        if self.n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if self.fs <= 0 or self.duration_s <= self.block_s:
            raise ValueError("need fs > 0 and duration_s > block_s")
        nyq = self.fs / 2
        if any(not (0 < f < nyq) for f in self.modes_hz):
            raise ValueError(f"every mode frequency must be in (0, {nyq}) Hz")
        if any(not (0 < z < 0.5) for z in self.damping):
            raise ValueError("damping ratios must be in (0, 0.5)")
        if any(a <= 0 for a in self.amplitudes):
            raise ValueError("amplitudes must be positive")
        if not (0.0 <= self.colored_noise_fraction <= 1.0):
            raise ValueError("colored_noise_fraction must be in [0, 1]")
        if not (0.0 <= self.drift < 0.5):
            raise ValueError("drift must be in [0, 0.5)")
        if self.rotor_rpm is not None:
            if self.rotor_rpm <= 0 or any(h < 1 for h in self.harmonics):
                raise ValueError("rotor_rpm must be positive and harmonic orders >= 1")
            if any(h * self.rotor_rpm / 60 >= nyq for h in self.harmonics):
                raise ValueError("a rotor harmonic is above the Nyquist frequency")


@dataclass(frozen=True)
class Truth:
    """The right answer for a generated record."""

    fs: float
    n_samples: int
    freqs_hz: np.ndarray  # nominal modal frequencies
    freq_min_hz: np.ndarray  # lowest instantaneous frequency reached (equals nominal when drift = 0)
    freq_max_hz: np.ndarray
    damping: np.ndarray
    mode_shapes: np.ndarray  # (n_channels, n_modes), unit-norm columns, largest entry positive
    harmonics_hz: np.ndarray  # rotor harmonic frequencies (empty when disabled)
    signal_power: float  # mean over channels of mean(signal^2)
    noise_power: float
    components: dict[str, np.ndarray] = field(default_factory=dict, repr=False)  # filled when return_components=True


def _mode_shapes(rng: np.random.Generator, n_ch: int, n_modes: int) -> np.ndarray:
    g = rng.standard_normal((n_ch, n_modes))
    if n_ch >= n_modes:
        phi, _ = np.linalg.qr(g)  # orthonormal columns
    else:
        phi = g / np.linalg.norm(g, axis=0, keepdims=True)
    flip = np.sign(phi[np.argmax(np.abs(phi), axis=0), np.arange(n_modes)])
    return phi * flip


def _modal_acceleration(rng: np.random.Generator, cfg: SynthConfig, f0: float, zeta: float, phase: float,
                        n_total: int, warmup: int) -> np.ndarray:
    """One mode's acceleration, with block-wise frequency drift and the start-up transient discarded."""
    dt = 1.0 / cfg.fs
    block = int(round(cfg.block_s * cfg.fs))
    u = rng.standard_normal(n_total)
    out = np.empty(n_total)
    zi = None
    for start in range(0, n_total, block):
        stop = min(start + block, n_total)
        t_mid = (start + stop) / 2 * dt
        f = f0 * (1.0 + 0.5 * cfg.drift * np.sin(2 * np.pi * t_mid / (n_total * dt) + phase))
        w = 2 * np.pi * f
        num, den, _ = signal.cont2discrete(([1.0, 0.0, 0.0], [1.0, 2 * zeta * w, w * w]), dt, method="zoh")
        sos = signal.tf2sos(np.squeeze(num), den)
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        out[start:stop], zi = signal.sosfilt(sos, u[start:stop], zi=zi)
    return out[warmup:]


def make_signal(cfg: SynthConfig = SynthConfig(), *, return_components: bool = False) -> tuple[np.ndarray, Truth]:
    """Generate (X, truth). X has shape (n_channels, n_samples) and dtype float64."""
    n = int(round(cfg.fs * cfg.duration_s))
    n_modes = len(cfg.modes_hz)
    # Independent random streams: mode shapes, harmonics, noise, drift phases, and one excitation stream per mode.
    streams = np.random.SeedSequence(cfg.seed).spawn(4 + n_modes)
    rng_shape, rng_harm, rng_noise, rng_phase = (np.random.default_rng(s) for s in streams[:4])
    exc_streams = [np.random.default_rng(s) for s in streams[4:]]
    # Discard the start-up transient: about five decay times of the slowest mode, capped to keep runs short.
    tau_max = max(1.0 / (z * 2 * np.pi * f) for f, z in zip(cfg.modes_hz, cfg.damping))
    warmup = int(round(min(5.0 * tau_max, 300.0) * cfg.fs))
    n_total = n + warmup

    phi = _mode_shapes(rng_shape, cfg.n_channels, n_modes)
    drift_phase = rng_phase.uniform(0, 2 * np.pi, n_modes)

    modal = np.empty((n_modes, n))
    for i, (f0, zeta, amp) in enumerate(zip(cfg.modes_hz, cfg.damping, cfg.amplitudes)):
        a = _modal_acceleration(exc_streams[i], cfg, f0, zeta, drift_phase[i], n_total, warmup)
        modal[i] = a * (amp / np.sqrt(np.mean(a * a)))
    x_modal = phi @ modal

    t = np.arange(n) / cfg.fs
    harmonics_hz = np.array([])
    x_harm = np.zeros_like(x_modal)
    if cfg.rotor_rpm is not None and cfg.harmonics:
        harmonics_hz = np.array([h * cfg.rotor_rpm / 60.0 for h in cfg.harmonics])
        for fh in harmonics_hz:
            gain = rng_harm.uniform(0.5, 1.5, (cfg.n_channels, 1))
            ph = rng_harm.uniform(0, 2 * np.pi, (cfg.n_channels, 1))
            x_harm += cfg.harmonic_amplitude * gain * np.sin(2 * np.pi * fh * t + ph)

    x_signal = x_modal + x_harm
    p_signal = float(np.mean(x_signal**2))

    if np.isinf(cfg.snr_db):
        noise = np.zeros_like(x_signal)
        p_noise = 0.0
    else:
        p_noise = p_signal / 10 ** (cfg.snr_db / 10)
        white = rng_noise.standard_normal(x_signal.shape)
        coloured = signal.sosfilt(signal.butter(2, 5.0, fs=cfg.fs, output="sos"), rng_noise.standard_normal(x_signal.shape), axis=-1)
        coloured /= coloured.std()
        mix = np.sqrt(1 - cfg.colored_noise_fraction) * white + np.sqrt(cfg.colored_noise_fraction) * coloured
        noise = np.sqrt(p_noise) * mix / mix.std()
        p_noise = float(np.mean(noise**2))

    f0 = np.asarray(cfg.modes_hz, dtype=float)
    spread = 0.5 * cfg.drift * f0
    truth = Truth(
        fs=cfg.fs, n_samples=n, freqs_hz=f0, freq_min_hz=f0 - spread, freq_max_hz=f0 + spread,
        damping=np.asarray(cfg.damping, dtype=float), mode_shapes=phi, harmonics_hz=harmonics_hz,
        signal_power=p_signal, noise_power=p_noise,
        components={"signal": x_signal, "noise": noise, "modal": x_modal, "harmonics": x_harm} if return_components else {},
    )
    return x_signal + noise, truth


def decimate_to(x: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    """Zero-phase Butterworth low-pass + decimation, in stages of at most 13 (the realistic first pipeline stage)."""
    ratio = fs / target_fs
    q_total = int(round(ratio))
    if abs(ratio - q_total) > 1e-9 or q_total < 1:
        raise ValueError(f"fs/target_fs must be an integer, got {ratio}")
    stages, q = [], q_total
    for p in (13, 11, 7, 5, 4, 3, 2):
        while q % p == 0 and q > 1:
            stages.append(p)
            q //= p
    if q > 1:
        stages.append(q)
    for s in sorted(stages, reverse=True):
        x = signal.decimate(x, s, ftype="iir", zero_phase=True, axis=-1)
    return x, fs / q_total
