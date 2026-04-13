"""Mansard & Funke (1980) three-probe least-squares separation.

Formulae follow ``docs/reflection_processing_pipeline.md`` §6A.5 (regular) and
§6B.3 (irregular).
"""

from __future__ import annotations

import numpy as np


def mansard_funke_separation(
    B1: np.ndarray,
    B2: np.ndarray,
    B3: np.ndarray,
    k: np.ndarray,
    X12: float,
    X13: float,
    D_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate incident/reflected Fourier coefficients at each bin.

    Parameters
    ----------
    B1, B2, B3 : complex ndarray
        Single-sided FFT coefficients at probes 1, 2, 3 (same length as ``k``).
    k : ndarray
        Wavenumbers at each bin.
    X12, X13 : float
        Probe spacings (wp1↔wp2 and wp1↔wp3) in metres.
    D_threshold : float
        Minimum value of the denominator ``D`` for a bin to be considered
        resolvable. Bins with ``D <= D_threshold`` are returned as NaN.

    Returns
    -------
    Z_I, Z_R : complex ndarray
    valid : bool ndarray
    """
    # The Mansard & Funke coefficients below are derived using the
    # mathematician's Fourier convention (kernel e^{+iωt}). NumPy's FFT uses
    # e^{-iωt}, so we conjugate the inputs to match. Without this,
    # the incident and reflected outputs come out swapped.
    B1 = np.conj(np.asarray(B1))
    B2 = np.conj(np.asarray(B2))
    B3 = np.conj(np.asarray(B3))
    k = np.asarray(k, dtype=float)

    beta = k * X12
    gamma = k * X13

    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)
    sgb = np.sin(gamma - beta)
    cgb = np.cos(gamma - beta)

    D = 2.0 * (sb * sb + sg * sg + sgb * sgb)

    R1 = sb * sb + sg * sg
    Q1 = sb * cb + sg * cg
    R2 = sg * sgb
    Q2 = sg * cgb - 2.0 * sb
    R3 = -sb * sgb
    Q3 = sb * cgb - 2.0 * sg

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_D = 1.0 / D
        Z_I = inv_D * (B1 * (R1 + 1j * Q1) + B2 * (R2 + 1j * Q2) + B3 * (R3 + 1j * Q3))
        Z_R = inv_D * (B1 * (R1 - 1j * Q1) + B2 * (R2 - 1j * Q2) + B3 * (R3 - 1j * Q3))

    valid = D > D_threshold
    Z_I = np.where(valid, Z_I, np.nan + 0j)
    Z_R = np.where(valid, Z_R, np.nan + 0j)
    return Z_I, Z_R, valid
