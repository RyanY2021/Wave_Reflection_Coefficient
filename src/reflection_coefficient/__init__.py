"""Reflection coefficient analysis for wave tank experiments."""

from .io import load_probe_data
from .pipeline import (
    IrregularResult,
    RegularResult,
    analyse,
    analyse_irregular,
    analyse_regular,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "load_probe_data",
    "analyse",
    "analyse_regular",
    "analyse_irregular",
    "RegularResult",
    "IrregularResult",
]
