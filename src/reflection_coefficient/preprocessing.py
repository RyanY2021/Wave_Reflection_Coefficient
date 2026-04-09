"""Signal preprocessing: detrending, filtering, windowing."""

from __future__ import annotations


def detrend(signal):
    """Remove linear trend from a probe signal."""
    raise NotImplementedError


def bandpass(signal, fs, low, high):
    """Apply a band-pass filter to a probe signal."""
    raise NotImplementedError
