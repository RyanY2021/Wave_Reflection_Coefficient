"""Mansard & Funke three-probe least-squares method."""

from __future__ import annotations


def mansard_funke(etas, spacings, depth: float, fs: float):
    """Separate incident and reflected spectra via least squares fit.

    Parameters
    ----------
    etas : sequence of array-like
        Surface elevation time-series for each probe (>= 3).
    spacings : sequence of float
        Probe positions relative to the first probe [m].
    depth : float
        Still water depth [m].
    fs : float
        Sampling frequency [Hz].
    """
    raise NotImplementedError
