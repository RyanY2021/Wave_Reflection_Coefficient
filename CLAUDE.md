# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Compute the reflection coefficient of a wave tank from experimental wave-probe
time-series. The core algorithm is implemented end-to-end: loading raw probe
data, clipping by travel-time window, preprocessing, FFT, dispersion solving,
and incident/reflected separation via two independent methods.

The two reference methods (PDFs in `docs/`):
- **Goda & Suzuki** — two-probe spectral separation (`methods/goda.py`)
- **Mansard & Funke** — three-probe least-squares separation (`methods/least_squares.py`)

Both take surface elevation time-series from multiple probes plus geometry
(probe spacing, water depth) and return separated incident/reflected Fourier
coefficients, from which `Kr = |a_r| / |a_i|` (regular) or
`Kr_overall = sqrt(m0_R / m0_I)` (irregular) is derived.

## Canonical Processing Pipeline

`docs/reflection_processing_pipeline.md` is the authoritative, step-by-step
implementation guide for this project. When implementing or modifying any stage
(clipping, detrending, FFT, dispersion, Goda/Mansard-Funke separation, quality
checks), follow its conventions, formulae, and singularity-mask thresholds
rather than re-deriving from the PDFs. The reference PDFs in `docs/` remain the
theoretical source; the markdown file is the operational spec.

Key conventions fixed by that document:
- Probe ordering: **probe 1 nearest the wave maker, probe 3 nearest the structure**.
- Water depth `d = 2.0 m`; spacings `X12`, `X13` are geometry inputs.
- Elevations are stored in **mm** and must be converted to metres before analysis.
- Clipping window comes from group-velocity travel times
  (`t_start = (x_paddle + 2*x_struct + 2*X13)/cg`,
  `t_end = (3*x_paddle + 2*x_struct + 3*X13)/cg`).
- Dispersion: fixed-point iteration `k = ω² / (g·tanh(k·d))`.
- Singularity masks: `sin²(kΔ) > 0.05` (two-probe), `D > 0.1` (three-probe).
- Regular-wave path uses a single FFT bin; irregular-wave path vectorises over
  all bins and band-averages before computing `Kr(f)` and an energy-based
  `Kr_overall = sqrt(m0_R / m0_I)`.

## Experimental Data Layout

`experiment_data/` is **gitignored**, so a fresh clone has no tank config or
metadata CSVs. `scripts/init_project.py` scaffolds it: writes a placeholder
`tank_config.json`, empty `metadata/{rw,wn,js}.csv` with headers, and the
`{rw,wn,js}/` subfolders. Existing files are preserved unless `--force`. The
same `--tank-config` / `--metadata-dir` / `--data-dir` overrides apply and are
persisted for subsequent `run_analysis.py` calls.

The loader treats three inputs as **independently configurable paths** rather
than one monolithic root:

| Input         | CLI flag          | Default                             |
|---------------|-------------------|-------------------------------------|
| tank config   | `--tank-config`   | `experiment_data/tank_config.json`  |
| metadata CSVs | `--metadata-dir`  | `experiment_data/metadata/`         |
| raw txt files | `--data-dir`      | `experiment_data/`                  |

Each path, once supplied on the CLI, is persisted to
`~/.reflection_coefficient.json` and reused on subsequent runs unless
overridden again. Resolution priority for each: **CLI arg → stored config →
built-in default.**

`--data-dir` accepts two shapes and the loader tries both:

```
<data_dir>/<scheme>/<TEST_ID>.txt     (scheme subfolders)
<data_dir>/<TEST_ID>.txt              (flat — the folder IS one scheme)
```

So you can point `--data-dir` at the parent (`experiment_data/`) or at a
specific flat folder (`experiment_data/rw/middle_tank/`) without restructuring
the files on disk. Tank geometry is still single-layout per `tank_config.json`
— when you switch to a physically different probe array, also switch
`--tank-config`.

`notebooks/` is reserved for exploratory Jupyter work and should **not** hold
canonical metadata. All inputs required to reproduce an analysis live under
`experiment_data/`. The richer historical test matrices now in
`docs/{rw,wn}_test_matrix/` are reference-only — not loaded by the code.

Each test file (`RW###.txt` / `WN###.txt`) is **tab-separated** with a two-row
header:

```
Time    31 Keyboard    3 wp3    2 wp2    1 wp1
Units                  mm       mm       mm
0.00    0                                -96.02
0.01    0              27.05    63.26    -96.21
...
```

Notes for the loader (`io.load_probe_data`):
- Delimiter is a tab; column order is `time, keyboard, wp3, wp2, wp1`
  (note `wp3` comes **before** `wp1` in the file — do not assume left-to-right
  matches probe index).
- Skip the two header rows; units are mm, convert to m.
- Early rows may have blank `wp1/wp2/wp3` cells before the probes are armed —
  treat as NaN or drop.
- Sampling is 100 Hz (Δt = 0.01 s) in the shipped files; derive `fs` from the
  time column rather than hard-coding.
- The `Keyboard` column is an event marker from acquisition and is not used by
  the analysis.

## Metadata Schema

`experiment_data/metadata/{rw,wn}.csv` is the canonical per-test manifest.
Two rules keep the schema clean (see `experiment_data/metadata/README.md`):

1. **Only store what varies per test and cannot be derived.** `k`, `L`, `cg`,
   `t_start`, `t_end`, `N_periods`, etc. are computed by the pipeline from
   `(f, water_depth, array_geometry)` — they must not be duplicated into the
   CSV.
2. **Tank-wide constants live in `tank_config.json`**, not in the per-test
   rows.

Current columns:

- `rw.csv` — regular waves:
  `test_id, f_Hz, a_target_m, t_gen_s, notes`
  (`a_target_m` = commanded amplitude, for traceability only; the reflection
  algorithm doesn't consume it.)
- `wn.csv` — pure white-noise (flat PSD) irregular waves:
  `test_id, S0_m2_Hz, f_min_Hz, f_max_Hz, t_gen_s, notes`
- `js.csv` — JONSWAP irregular waves:
  `test_id, Hs_target_m, Tp_target_s, gamma, f_min_Hz, f_max_Hz, t_gen_s, notes`

Scheme is inferred from the test-id prefix: `RW###` → rw, `WN###` → wn,
`JS###` → js.

## Accessing a Test in Code

`io.load_probe_data` is the single entry point. It merges tank config + test
matrix + raw file and returns elevations (in metres) plus a `TestMeta`:

```python
from reflection_coefficient.io import load_probe_data

# Uses stored/default paths for tank_config, metadata_dir, data_dir:
t, eta1, eta2, eta3, meta = load_probe_data("RW005")

# Any of the three can be overridden per-call without affecting the saved config:
t, eta1, eta2, eta3, meta = load_probe_data(
    "RW005", data_dir="experiment_data/rw/middle_tank"
)
# meta.water_depth_m, meta.X13_m, meta.f_Hz, ...
```

Campaign (`rw`/`wn`/`js`) is inferred from the test-id prefix. Add new per-test
fields by extending `TestMeta` and/or the metadata CSV; unknown CSV columns
are preserved in `meta.extra` so downstream code doesn't need to change in
lockstep. `pipeline._require_geometry` raises if any of `x_paddle_to_wp1_m`,
`X12_m`, `X13_m`, or the derived `x_wp3_to_struct_m` is `None`, so partially
populated `tank_config.json` files fail loudly rather than silently.

Geometry keys in `tank_config.json → probe_geometry`:

- `tank_length_m` — distance from the wave-paddle face to the reflecting
  structure.
- `x_paddle_to_wp1_m` — distance from the wave-paddle face to probe 1 (the
  probe nearest the paddle). This is the `x_paddle` quantity in
  `docs/reflection_processing_pipeline.md` §2.1.
- `X12_m`, `X13_m` — probe spacings wp1↔wp2 and wp1↔wp3.

The `x_struct` quantity from the pipeline doc (wp3 → structure) is **not**
stored; it's derived via
`x_struct = tank_length_m - x_paddle_to_wp1_m - X13_m` and exposed on
`TestMeta` as the `x_wp3_to_struct_m` property.

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

# First-time scaffold of experiment_data/ (gitignored, so blank on fresh clone)
python scripts/init_project.py

# Run analysis CLI — three independent path inputs (each persisted when set)
python scripts/run_analysis.py \
    --tank-config  path/to/tank_config.json \
    --metadata-dir path/to/metadata \
    --data-dir     path/to/txt_files \
    --scheme rw --test RW005 --method least_squares

# After the first run with paths set, they're remembered — subsequent calls
# can omit any/all of them:
python scripts/run_analysis.py --scheme rw --test all

# Irregular-wave FFT windowing (persisted per-user like --method):
python scripts/run_analysis.py --scheme wn --test WN003 \
    --window hann --bandwidth 0.04

# Inspect which paths / choices are currently resolved:
python scripts/run_analysis.py --show-paths

# List discoverable tests for a scheme
python scripts/run_analysis.py --scheme rw --list
```

Persisted per-user choices (stored in `~/.reflection_coefficient.json` alongside
the three paths): `--method` (goda / least_squares, default least_squares),
`--window` (none / hann, default hann), `--bandwidth` (Hz, default 0.04),
`--head-drop` and `--tail-drop` (seconds trimmed from the start/end of the
travel-time clip to skip ramp transients, default 3.0 s each), and
`--freq-source` (regular-wave only, bin / target, default bin; `target`
evaluates a single-point DFT at exactly `meta.f_Hz` instead of snapping to
the nearest FFT bin).

`--window-mode` (canonical / noref) is **not** persisted — it is intended
as an explicit diagnostic mode (pre-reflection baseline check where Kr
should be ≈ 0). Output filenames gain a `_noref` suffix when used.

`--test all` is only supported for `--scheme rw`. For irregular schemes
(`wn` / `js`) you must pass an explicit test id. `--scheme` is prompted
interactively if omitted.

### Outputs

Every run creates a timestamped subfolder under `--output` (default
`<project>/results/YYYYMMDD_HHMMSS/`) so multiple runs never overwrite each
other. Contents depend on the scheme:

- **Irregular (`wn` / `js`, single test):** `<TEST>_<method>_spectrum.csv`
  plus a self-contained `<TEST>_<method>_report.html` via
  `irregular_report.write_irregular_report` (spectra, Kr(f), diagnostics).
- **Regular, `--test all` with ≥2 tests:** aggregates into
  `rw_kr_vs_freq_<method>.csv` and `rw_report_<method>.html` via
  `rw_report.write_rw_report` (Kr vs f, singularity points flagged).
- **Regular, single test:** console summary only.

HTML reports embed **Chart.js** for interactive plots (clip-window Gantt,
Kr(f), incident/reflected spectra, singularity metric) and are self-contained.
```

## Architecture

One-directional pipeline — keep new code consistent with this layering:

```
io.load_probe_data                      →  raw time-series (m) + TestMeta
preprocessing.clip_window               →  clipped signals (travel-time window)
preprocessing.remove_mean/detrend/hanning_window
analysis.solve_dispersion[_array], group_velocity, positive_fft
methods.goda.goda_separation            →  (Z_I, Z_R, valid)    [2-probe]
methods.least_squares.mansard_funke_separation →  (Z_I, Z_R, valid) [3-probe]
pipeline.analyse  (dispatches to _regular | _irregular on meta.campaign)
                                        →  RegularResult / IrregularResult
rw_report.write_rw_report               →  HTML summary (regular, group runs)
irregular_report.write_irregular_report →  HTML summary (wn / js)
utils.py                                →  shared helpers (formatting, plots)
scripts/run_analysis.py                 →  CLI orchestration + persisted config
```

Key conventions:

- **`src/` layout.** Package lives at `src/reflection_coefficient/`; tests and
  scripts require the editable install (`pip install -e .`) to resolve.
- **Separation functions take FFT bins, not time-series.** `goda_separation`
  and `mansard_funke_separation` accept pre-computed `B1/B2/B3` arrays plus
  wavenumbers `k` (one per bin). This lets the same function serve both the
  single-bin regular-wave path (§6A) and the vectorised irregular path (§6B).
- **FFT sign convention.** NumPy uses `e^{-iωt}`. The Goda formula is
  consistent with this. The Mansard–Funke formulae in the pipeline doc are
  derived in the `e^{+iωt}` convention, so `mansard_funke_separation`
  conjugates its `B` inputs before applying them — do **not** remove this
  conjugation without re-running `tests/test_pipeline.py`.
- **Dispersion solving lives in `analysis.solve_dispersion` /
  `solve_dispersion_array`.** Both separation methods depend on `k(f, h)`;
  keep it implemented once, not inlined per method.
- **`experiment_data/` and `docs/` are inputs, not code.** `results/` is
  generated output and is gitignored.

## Testing

`tests/test_pipeline.py` exercises the algorithm end-to-end against synthetic
three-probe signals with known `a_I`, `a_R`, `φ_R`: both methods must recover
the amplitudes within 2 %, and the singularity mask must flag `kΔ = π`. If
you change any spectral-convention detail (FFT scaling, conjugation, sign of
Q coefficients), this test is the tripwire.
