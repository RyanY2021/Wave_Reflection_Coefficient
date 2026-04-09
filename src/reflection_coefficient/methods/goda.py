"""Goda & Suzuki two-probe method for incident/reflected wave separation."""

from __future__ import annotations


def goda_suzuki(eta1, eta2, spacing: float, depth: float, fs: float):
    """Separate incident and reflected spectra using two probes.

    Parameters
    ----------
    eta1, eta2 : array-like
        Surface elevation time-series at the two probes.
    spacing : float
        Distance between probes [m].
    depth : float
        Still water depth [m].
    fs : float
        Sampling frequency [Hz].
    """
    raise NotImplementedError
