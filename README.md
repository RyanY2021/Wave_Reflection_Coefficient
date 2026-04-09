# Reflection Coefficient

Calculate the reflection coefficient of a wave tank from experimental wave probe data.

## Overview

This project implements methods for separating incident and reflected wave spectra
from multi-probe wave gauge measurements in a wave flume / towing tank, and
computing the resulting reflection coefficient.

Reference methods (see `docs/`):
- Goda & Suzuki — two-probe spectral separation
- Mansard & Funke — three-probe least-squares method

## Project Structure

```
Reflection_Coefficient/
├── src/reflection_coefficient/   # main package
│   ├── io.py                     # load experimental data
│   ├── preprocessing.py          # detrend / filter / window
│   ├── analysis.py               # FFT / spectral utilities
│   ├── methods/                  # separation methods
│   │   ├── goda.py
│   │   └── least_squares.py
│   └── utils.py
├── scripts/                      # runnable entry points
├── tests/                        # unit tests
├── notebooks/                    # exploratory analysis
├── experiment_data/              # raw probe data (gitignored if large)
├── results/                      # generated outputs
└── docs/                         # reference papers / notes
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows bash
pip install -e .
```

## Usage

```bash
python scripts/run_analysis.py --input experiment_data/<run>.csv
```
