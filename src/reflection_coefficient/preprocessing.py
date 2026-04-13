"""Signal preprocessing — clipping, mean removal, detrending, windowing.

Implements ``docs/reflection_processing_pipeline.md`` §2.5 and §3.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps


def clip_window(
    t: np.ndarray,
    *signals: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, ...]:
    """Clip ``t`` and each signal to ``[t_start, t_end]`` (inclusive)."""
    t = np.asarray(t)
    mask = (t >= t_start) & (t <= t_end)
    return (t[mask], *(np.asarray(s)[mask] for s in signals))


def remove_mean(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    return x - x.mean()


def detrend(signal: np.ndarray) -> np.ndarray:
    """Linear detrend (wraps ``scipy.signal.detrend``)."""
    return sps.detrend(np.asarray(signal, dtype=float), type="linear")


def hanning_window(n: int) -> np.ndarray:
    """Hanning window of length n; ``mean(w**2)`` is the energy correction."""
    return np.hanning(n)
