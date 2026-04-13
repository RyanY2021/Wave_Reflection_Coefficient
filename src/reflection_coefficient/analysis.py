"""Spectral analysis utilities — dispersion solver, group velocity, FFT.

Implements the dispersion and spectral steps of
``docs/reflection_processing_pipeline.md`` §4–§5.
"""

from __future__ import annotations

import numpy as np

from .utils import GRAVITY


def solve_dispersion(
    frequency: float,
    depth: float,
    g: float = GRAVITY,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Return ``(k, L)`` solving ω² = g·k·tanh(k·d) for one frequency."""
    omega = 2.0 * np.pi * float(frequency)
    k = omega * omega / g  # deep-water initial guess
    for _ in range(max_iter):
        k_new = omega * omega / (g * np.tanh(k * depth))
        if abs(k_new - k) / k < tol:
            k = k_new
            break
        k = k_new
    return float(k), float(2.0 * np.pi / k)


def solve_dispersion_array(
    frequencies: np.ndarray,
    depth: float,
    g: float = GRAVITY,
    max_iter: int = 100,
) -> np.ndarray:
    """Vectorised dispersion solver. ``frequencies`` must be strictly > 0."""
    f = np.asarray(frequencies, dtype=float)
    omega = 2.0 * np.pi * f
    k = omega * omega / g
    for _ in range(max_iter):
        k = omega * omega / (g * np.tanh(k * depth))
    return k


def group_velocity(frequency: float, depth: float, g: float = GRAVITY) -> float:
    """Group velocity c_g at a single frequency via linear theory."""
    omega = 2.0 * np.pi * float(frequency)
    k, _ = solve_dispersion(frequency, depth, g=g)
    return float((omega / (2.0 * k)) * (1.0 + (2.0 * k * depth) / np.sinh(2.0 * k * depth)))


def positive_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided complex FFT.

    Returns ``(freqs_pos, B)`` where ``B = np.fft.fft(signal)[:n_pos]``
    (unnormalised NumPy convention — downstream code applies the 2/N factor).
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    n_pos = n // 2 + 1
    freqs = np.fft.fftfreq(n, d=1.0 / fs)[:n_pos]
    # Ensure the Nyquist bin for even-N records is non-negative
    freqs = np.abs(freqs)
    B = np.fft.fft(x)[:n_pos]
    return freqs, B
