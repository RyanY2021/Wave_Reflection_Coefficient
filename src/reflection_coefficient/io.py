"""Data loading utilities for wave probe measurements."""

from __future__ import annotations

from pathlib import Path


def load_probe_data(path: str | Path):
    """Load wave probe time-series from an experiment file.

    Returns time vector and probe channels.
    """
    raise NotImplementedError
