"""Spectral analysis utilities (FFT, cross-spectra, dispersion)."""

from __future__ import annotations


def wave_number(frequency, depth, g: float = 9.81):
    """Solve the linear dispersion relation for the wave number k."""
    raise NotImplementedError


def cross_spectrum(x, y, fs):
    """Compute the cross-spectral density between two signals."""
    raise NotImplementedError
