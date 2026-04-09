# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Compute the reflection coefficient of a wave tank from experimental wave-probe
time-series. The repository is an early-stage scaffold — most modules are stubs
that raise `NotImplementedError` and are meant to be filled in over time.

The two reference methods (PDFs in `docs/`) the code is structured around:
- **Goda & Suzuki** — two-probe spectral separation (`methods/goda.py`)
- **Mansard & Funke** — three-probe least-squares separation (`methods/least_squares.py`)

Both methods take surface elevation time-series from multiple probes plus geometry
(probe spacing, water depth) and return separated incident/reflected spectra,
from which the reflection coefficient `Kr = |a_r| / |a_i|` is derived.

## Commands

```bash
# Editable install (src-layout package)
pip install -e .
pip install -e ".[dev]"      # with pytest + ruff

# Run tests
pytest
pytest tests/test_smoke.py::test_version   # single test

# Lint
ruff check .

# Run analysis CLI
python scripts/run_analysis.py --input experiment_data/<file> --method least_squares
```

## Architecture

The intended data flow is a one-directional pipeline; keep new code consistent
with this layering rather than mixing concerns:

```
io.load_probe_data           →  raw time-series + sampling info
preprocessing (detrend/filter) →  cleaned signals
analysis (FFT, dispersion k)   →  spectra, wave numbers
methods.{goda,least_squares}   →  separated incident / reflected spectra
scripts/run_analysis.py        →  orchestrates the above + writes results/
```

Key conventions baked into the scaffold:

- **`src/` layout.** The package lives at `src/reflection_coefficient/`; tests
  and scripts import it as `reflection_coefficient` and require the editable
  install to resolve.
- **Method modules are interchangeable.** Each function in `methods/` should
  accept probe time-series + geometry and return the same shape of result so the
  CLI's `--method` switch can pick between them without special-casing.
- **Dispersion solving lives in `analysis.wave_number`.** Both separation methods
  depend on `k(f, h)`; implement it once there rather than inline in each method.
- **`experiment_data/` and `docs/` are inputs, not code.** `docs/` holds the
  reference papers; consult them when implementing the methods. `results/` is
  generated output and is gitignored.
