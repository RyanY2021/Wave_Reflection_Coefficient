"""Linear re-calibration of wave probe time series.

The acquisition software stored each probe's raw sensor signal as

    eta_old = scale_old * raw + offset_old

A later re-calibration yields

    eta_new = scale_new * raw + offset_new

Given only the recorded ``eta_old`` we cannot recover ``raw``, but we can map
``eta_old`` directly to ``eta_new`` in closed form by eliminating ``raw``:

    eta_new = (scale_new / scale_old) * (eta_old - offset_old) + offset_new

This is a pure side transformation on already-loaded probe data. It is *not*
applied by ``load_probe_data``; callers opt in (``run_analysis.py`` does so via
the persisted ``--recalibrate`` flag).

Unit conventions:

* ``eta_old``, ``eta_new``, ``offset_old``, ``offset_new`` must share the same
  physical unit (metres, since ``load_probe_data`` returns metres).
* ``scale_old`` and ``scale_new`` appear only as a ratio, so their units
  cancel — use any unit that is consistent across old/new (e.g. mm per volt).
"""

from __future__ import annotations

import numpy as np


def apply_calibration_transfer(
    eta_old: np.ndarray,
    scale_old: float,
    offset_old: float,
    scale_new: float,
    offset_new: float,
) -> np.ndarray:
    """Map an elevation series from an old linear calibration to a new one.

    Returns ``eta_old`` unchanged when ``scale_new == scale_old`` and
    ``offset_new == offset_old``. Raises ``ValueError`` if ``scale_old`` is
    zero (the old calibration cannot be inverted).
    """
    if scale_old == 0.0:
        raise ValueError("scale_old must be non-zero to invert eta_old -> raw.")
    return (scale_new / scale_old) * (eta_old - offset_old) + offset_new


_PROBE_KEYS: tuple[str, ...] = ("wp1", "wp2", "wp3")
_FIELDS: tuple[str, ...] = ("scale_old", "offset_old", "scale_new", "offset_new")


def _probe_params(config: dict, probe: str) -> tuple[float, float, float, float]:
    entry = config.get(probe) or {}
    missing = [k for k in _FIELDS if k not in entry]
    if missing:
        raise KeyError(
            f"probes config entry {probe!r} is missing fields: {missing}. "
            f"Expected all of {list(_FIELDS)}."
        )
    return tuple(float(entry[k]) for k in _FIELDS)  # type: ignore[return-value]


def recalibrate_probes(
    eta1: np.ndarray,
    eta2: np.ndarray,
    eta3: np.ndarray,
    probes_config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply :func:`apply_calibration_transfer` to each of three probe series.

    ``probes_config`` must have top-level keys ``wp1``, ``wp2``, ``wp3``, each
    mapping to a dict with ``scale_old``, ``offset_old``, ``scale_new``,
    ``offset_new``. Extra top-level keys (e.g. ``_doc``) are ignored.
    """
    out: list[np.ndarray] = []
    for eta, probe in zip((eta1, eta2, eta3), _PROBE_KEYS):
        out.append(apply_calibration_transfer(eta, *_probe_params(probes_config, probe)))
    return out[0], out[1], out[2]
