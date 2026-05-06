# Reflection Coefficient

Compute the reflection coefficient of a wave tank from experimental wave-probe
time-series. The pipeline loads raw probe data, clips by travel-time window,
preprocesses, runs FFT + dispersion solving, and separates incident from
reflected spectra using either the **Goda & Suzuki** two-probe method or the
**Mansard & Funke** three-probe least-squares method.

Regular-wave tests give a single `Kr = |a_r|/|a_i|` per frequency;
irregular-wave tests (white-noise / JONSWAP) give a band-averaged `Kr(f)` curve
plus an energy-based `Kr_overall = sqrt(m0_R / m0_I)`.

## Project structure

```
Reflection_Coefficient/
├── src/reflection_coefficient/
│   ├── io.py                  # load tank config, metadata, raw txt
│   ├── calibration.py         # linear probe re-calibration (side module)
│   ├── preprocessing.py       # clip / detrend / Hann window
│   ├── analysis.py            # FFT + dispersion solver
│   ├── methods/
│   │   ├── goda.py            # two-probe separation
│   │   └── least_squares.py   # three-probe Mansard–Funke
│   ├── pipeline.py            # analyse() dispatcher (regular/irregular)
│   ├── rw_report.py           # HTML report — regular-wave aggregate
│   ├── irregular_report.py    # HTML report — single irregular test
│   └── utils.py
├── scripts/run_analysis.py    # CLI entry point
├── scripts/streamlit_app.py   # Streamlit companion webpage
├── tests/                     # pytest suite
├── experiment_data/           # tank_config.json, probes.json, metadata/*.csv, raw RW###.txt / WN###.txt
└── results/                   # generated output (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows bash / Git Bash
# or: .venv\Scripts\activate          # Windows cmd / PowerShell

pip install -e .                      # editable install
pip install -e ".[dev]"               # + pytest + ruff
```

Tests / lint:

```bash
pytest
pytest tests/test_smoke.py::test_version    # single test
ruff check .
```

### Quicker setup with `uv` (optional)

[`uv`](https://docs.astral.sh/uv/) (`brew install uv` on macOS) is a drop-in
replacement for `pip` + `venv` that's 10–100× faster and skips the activation
step:

```bash
uv venv                              # create .venv/
uv pip install -e ".[dev]"           # editable install + dev extras
uv run pytest                        # run any command without activating
uv run python scripts/run_analysis.py --show-paths
```

The rest of this README uses the canonical `python` / `pip` invocations that
work on any machine; if you've gone the uv route, prefix the analysis
commands with `uv run` (e.g. `uv run python scripts/run_analysis.py ...`).

## Project initialization

`experiment_data/` is gitignored, so a fresh clone has no tank config or
metadata CSVs. Generate the input scaffold before the first analysis run:

```bash
python scripts/init_project.py
```

This creates `experiment_data/tank_config.json` (placeholder geometry + water
depth), `experiment_data/probes.json` (per-probe re-calibration with identity
defaults — see [Linear probe re-calibration](#linear-probe-re-calibration)),
empty `experiment_data/metadata/{rw,wn,js}.csv` with headers, and the
`experiment_data/{rw,wn,js}/` subfolders for raw probe txt files. Existing
files are preserved unless you pass `--force`. Any of `--tank-config`,
`--metadata-dir`, `--data-dir`, `--probes-config` can redirect the scaffold
elsewhere, and those paths are persisted for subsequent `run_analysis.py`
calls.

Then fill in the geometry fields in `tank_config.json`, add per-test rows to
the metadata CSVs, and drop raw `<TEST_ID>.txt` files into `experiment_data/`
(flat or under the scheme subfolder).

## CLI — `scripts/run_analysis.py`

Path inputs and analysis choices are **persisted per-user** in
`~/.reflection_coefficient.json`. Priority for each: **CLI arg → stored value
→ built-in default**. Pass `--show-paths` to see what will be resolved.

### Full flag reference

| Flag | Values / default | Persisted? | Description |
|---|---|---|---|
| `--tank-config PATH` | default: `experiment_data/tank_config.json` | ✔ | Tank geometry + water depth JSON. |
| `--metadata-dir PATH` | default: `experiment_data/metadata/` | ✔ | Folder containing `rw.csv` / `wn.csv` / `js.csv`. |
| `--data-dir PATH` | default: `experiment_data/` | ✔ | Folder holding raw `<TEST_ID>.txt` files. Accepts either `<data_dir>/<scheme>/<TEST_ID>.txt` (per-scheme subfolders) or `<data_dir>/<TEST_ID>.txt` (flat). |
| `--probes-config PATH` | default: `experiment_data/probes.json` | ✔ | Per-probe linear re-calibration JSON. Consumed only when `--recalibrate` is on. |
| `--scheme {rw,wn,js}` | *prompted if omitted* | — | Wave scheme. `rw` = regular, `wn` = white-noise irregular, `js` = JONSWAP irregular. |
| `--test TEST_ID \| all` | default: `all` | — | Test id (e.g. `RW005`, `WN003`) or `all` to iterate every test in the scheme. `all` is only supported for `--scheme rw`. |
| `--method {goda,least_squares}` | default: `least_squares` | ✔ | Separation method. Goda uses probes 1 & 3; Mansard–Funke uses all three. |
| `--window {none,hann}` | default: `hann` | ✔ | Spectral window for the irregular-wave FFT. Ignored for `rw`. |
| `--bandwidth HZ` | default: `0.04` | ✔ | Target resolution bandwidth in Hz for band-averaging (only meaningful with `--window hann`). |
| `--head-drop SEC` | default: `3.0` | ✔ | Seconds to trim from the **start** of the clean analysis window, so the FFT skips ramp-up transients. |
| `--tail-drop SEC` | default: `3.0` | ✔ | Seconds to trim from the **end** of the clean analysis window, so the FFT skips ramp-down transients. |
| `--recalibrate` / `--no-recalibrate` | default: on | ✔ | Apply the per-probe linear re-calibration from `probes.json` after loading. Use `--no-recalibrate` to disable. See [Linear probe re-calibration](#linear-probe-re-calibration). |
| `--window-mode {canonical,noref}` | default: `canonical` | — | Clip-window selection. `canonical` uses the full reflection-inclusive travel-time window from `docs/reflection_processing_pipeline.md` §2.1; `noref` uses a pre-reflection window ending before the first reflected front returns to probe 1, for a baseline sanity check where the true `Kr` should be zero. Explicit diagnostic — not persisted. Outputs gain a `_noref` filename suffix. |
| `--freq-source {bin,target}` | default: `target` | ✔ | Regular-wave only. `target` (default) evaluates the DFT at exactly the target frequency from metadata via a single-point `Σ x[n]·exp(-i·2π·f·n/fs)`, eliminating bin-quantisation error — relies on the wave maker holding the commanded frequency exactly across the test. `bin` snaps to the nearest FFT bin (fastest, bin-quantised). Ignored for `wn` / `js`. |
| `--goda-pair {13,12,23}` | default: `13` | ✔ | Goda-only. Which two probes feed the two-probe separation: `13` = wp1 + wp3 (widest spacing, default), `12` = wp1 + wp2, `23` = wp2 + wp3 (spacing `X13 − X12`). Changing Δ moves the `kΔ = nπ` singularities in frequency — useful when the default pair sits on a near-singular bin for a given test. Ignored when `--method least_squares`. |
| `--cn-alpha-mode {scalar,dynamic}` | default: `dynamic` | ✔ | How $\alpha$ is evaluated when applying $C_n$. `dynamic` linearly interpolates the per-bin $\alpha$ table stored in `probes_refined.json` (`per_bin.alpha`) over frequency, with the scalar masked-mean $\alpha$ as fallback for bins outside the table's $f$-range. `scalar` uses the masked-mean $\alpha$ at every frequency. The scalar's frequency mask is editable in `probes_refined.json` under `fit_mask` (`f_min_Hz`, `f_max_Hz`); user edits survive a re-fit. Ignored when `--cn-apply` is off or `--cn-mode phase`. See `docs/cn_fit_formulas.md` §2. |
| `--output PATH` | default: `<project>/results` | — | Parent output dir. A timestamped subfolder `YYYYMMDD_HHMMSS/` is created per run. |
| `--list` | flag | — | List discoverable tests for the chosen scheme and exit. |
| `--show-paths` | flag | — | Print the resolved `tank_config` / `metadata_dir` / `data_dir` / `probes_config` / `method` / `window` / `bandwidth` / drops / `freq_source` / `goda_pair` and exit. |
| `-h`, `--help` | flag | — | Show help. |

### Typical workflows

```bash
# First run — set the three paths (remembered for next time)
python scripts/run_analysis.py \
    --tank-config  experiment_data/tank_config.json \
    --metadata-dir experiment_data/metadata \
    --data-dir     experiment_data/ \
    --scheme rw --test RW005 --method least_squares

# Subsequent runs — paths & method are remembered
python scripts/run_analysis.py --scheme rw --test all

# Point at a flat scheme folder without restructuring on disk
python scripts/run_analysis.py \
    --data-dir experiment_data/rw/middle_tank \
    --scheme rw --test all

# Single irregular-wave test with an explicit window/bandwidth
python scripts/run_analysis.py --scheme wn --test WN003 \
    --window hann --bandwidth 0.04

# Widen the head/tail drop to exclude longer ramp transients
python scripts/run_analysis.py --scheme rw --test all \
    --head-drop 5 --tail-drop 5

# Turn on linear probe re-calibration (remembered until --no-recalibrate)
python scripts/run_analysis.py --scheme rw --test all --recalibrate

# Pre-reflection baseline sanity check — clip before any reflected front arrives.
# True Kr should be ~0; residual Kr shows the separation method's intrinsic bias.
# Outputs gain a `_noref` filename suffix so they don't clobber the normal run.
python scripts/run_analysis.py --scheme rw --test all --window-mode noref

# --freq-source target is now the default — evaluates the DFT at exactly
# meta.f_Hz (no FFT bin snapping). Use --freq-source bin to fall back to the
# nearest-bin behaviour for diagnostic comparison; the choice is persisted.
python scripts/run_analysis.py --scheme rw --test RW005 --freq-source bin

# Swap the Goda probe pair when the default (wp1 & wp3) sits on a singularity.
# Persisted; ignored when --method least_squares.
python scripts/run_analysis.py --scheme rw --test all \
    --method goda --goda-pair 12

# Inspect resolved configuration
python scripts/run_analysis.py --show-paths

# List tests available for a scheme
python scripts/run_analysis.py --scheme rw  --list
python scripts/run_analysis.py --scheme wn  --list
python scripts/run_analysis.py --scheme js  --list

# Override output directory for one run
python scripts/run_analysis.py --scheme rw --test all \
    --output /tmp/reflection_runs
```

### Outputs

Every run creates `results/YYYYMMDD_HHMMSS/` (or under `--output`). Contents
depend on scheme and selection:

| Scheme / selection | Files written |
|---|---|
| `rw` single test | console summary only |
| `rw` with `--test all` (≥2 tests) | `rw_kr_vs_freq_<method>.csv`, `rw_report_<method>.html` |
| `wn` or `js` (single test) | `<TEST>_<method>_spectrum.csv`, `<TEST>_<method>_report.html` |

When `--window-mode noref` is set, a `_noref` suffix is inserted before the
extension (e.g. `rw_kr_vs_freq_<method>_noref.csv`,
`<TEST>_<method>_noref_report.html`) so baseline runs don't overwrite the
reflection-inclusive ones.

The HTML reports use **Chart.js** for interactive, tooltipped plots (Gantt of
time windows, `Kr` vs frequency, incident/reflected spectra, singularity
metric). They are self-contained — open directly in a browser.

## Streamlit companion app

`scripts/streamlit_app.py` is a browser UI over the same pipeline — pick
scheme / test / method / window / head-tail drops / window-mode / freq-source
from dropdowns, run the analysis in-process, and view or download the
generated CSV + HTML report. The flag semantics mirror the CLI exactly, and it
reads and writes the same persisted `~/.reflection_coefficient.json`.

```bash
pip install streamlit                           # optional dep, not in [dev]
streamlit run scripts/streamlit_app.py
```

The app opens on `http://localhost:8501` by default. Pass `--server.port` or
`--server.headless true` for remote / headless hosting (see `streamlit run
--help`).

## Linear probe re-calibration

A side-module (`reflection_coefficient.calibration`) that corrects probe
elevations when a new gain/offset calibration supersedes the one used during
acquisition. It is **not** applied by `load_probe_data`; turn it on with
`--recalibrate` (persisted), and fill in `experiment_data/probes.json`.

The acquisition system recorded `eta_old = scale_old · raw + offset_old`;
the re-calibration target is `eta_new = scale_new · raw + offset_new`.
Eliminating `raw` gives the closed-form transfer applied per probe:

```
eta_new = (scale_new / scale_old) · (eta_old − offset_old) + offset_new
```

Identity values (`scale_new == scale_old`, `offset_new == offset_old`) disable
the transform for that probe. `probes.json` holds four numbers per probe:

```jsonc
{
  "wp1": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
  "wp2": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
  "wp3": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0}
}
```

Unit conventions: offsets share units with `eta` as returned by
`load_probe_data` (metres); scale units cancel through the ratio, so use any
unit consistent across old/new (e.g. mm per volt straight from a calibration
sheet).

Library entry points:

```python
from reflection_coefficient.calibration import (
    apply_calibration_transfer,   # pure math on a single series
    recalibrate_probes,           # applies it to eta1/eta2/eta3 using probes.json
)
```

## Library usage

```python
from reflection_coefficient.io import load_probe_data
from reflection_coefficient.pipeline import analyse

t, eta1, eta2, eta3, meta = load_probe_data("RW005")
result = analyse(t, eta1, eta2, eta3, meta, method="least_squares")

print(result.Kr)          # regular: scalar Kr
# or result.Kr_overall    # irregular: energy-based Kr
```

Any of the three paths can be overridden per-call (without touching the saved
config):

```python
load_probe_data("RW005", data_dir="experiment_data/rw/middle_tank")
```

## Testing

`tests/test_pipeline.py` drives the pipeline end-to-end on synthetic
three-probe signals with known `a_I`, `a_R`, `φ_R`. Both methods must recover
the amplitudes within 2 %, and the singularity mask must flag `kΔ = π`. If
you change any spectral-convention detail (FFT scaling, conjugation, sign of
Q coefficients), this is the tripwire.

## References

- *Isolating incident and reflected wave spectra* — Goda & Suzuki.
- *Separation of incident and reflected spectra in wave flumes.*
- *The measurement of incident and reflected spectra using a least squares
  method* — Mansard & Funke.
- *The propagation of the waves in the CTO SA towing tank* — tank context.

## License

MIT License — see [`LICENSE`](LICENSE).

Copyright © 2026 Ryan You, Kelvin Hydrodynamics Laboratory.
