"""Goda & Suzuki (1976) two-probe incident/reflected separation.

Formulae follow ``docs/reflection_processing_pipeline.md`` §6A.4 (regular) and
§6B.2 (irregular). Inputs are already-computed single-sided FFT coefficients
plus the wavenumber(s) at those bins.
"""

from __future__ import annotations

import numpy as np


def goda_separation(
    B1: np.ndarray,
    B3: np.ndarray,
    k: np.ndarray,
    spacing: float,
    sin2_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate incident and reflected Fourier coefficients at each bin.

    Parameters
    ----------
    B1, B3 : complex ndarray
        Single-sided FFT coefficients at probes 1 and 3, aligned with ``k``.
    k : ndarray
        Wavenumbers at each bin (same length as ``B1``/``B3``).
    spacing : float
        Probe 1–probe 3 distance Δ = X13, in metres.
    sin2_threshold : float
        Minimum ``sin²(kΔ)`` considered resolvable. Bins below are set to NaN.

    Returns
    -------
    Z_I, Z_R : complex ndarray
        Separated incident and reflected Fourier coefficients.
    valid : bool ndarray
        Mask of bins passing the singularity check.
    """
    B1 = np.asarray(B1)
    B3 = np.asarray(B3)
    k = np.asarray(k, dtype=float)
    kd = k * spacing

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 - np.exp(-2j * kd)
        Z_I = (B1 - B3 * np.exp(-1j * kd)) / denom
        denom_r = 1.0 - np.exp(2j * kd)
        Z_R = (B1 - B3 * np.exp(1j * kd)) / denom_r

    valid = np.sin(kd) ** 2 > sin2_threshold
    Z_I = np.where(valid, Z_I, np.nan + 0j)
    Z_R = np.where(valid, Z_R, np.nan + 0j)
    return Z_I, Z_R, valid
