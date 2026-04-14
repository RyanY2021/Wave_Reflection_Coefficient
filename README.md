# Reflection Coefficient

Compute the reflection coefficient of a wave tank from experimental wave-probe
time-series. The pipeline loads raw probe data, clips by travel-time window,
preprocesses, runs FFT + dispersion solving, and separates incident from
reflected spectra using either the **Goda & Suzuki** two-probe method or the
**Mansard & Funke** three-probe least-squares method.

Regular-wave tests give a single `Kr = |a_r|/|a_i|` per frequency;
irregular-wave tests (white-noise / JONSWAP) give a band-averaged `Kr(f)` curve
plus an energy-based `Kr_overall = sqrt(m0_R / m0_I)`.

> The operational spec lives in **`docs/reflection_processing_pipeline.md`**.
> Agent/project conventions live in **`CLAUDE.md`** — read those for anything
> the README doesn't cover (data layout, metadata schema, FFT sign convention,
> singularity masks, etc.).

## Project structure

```
Reflection_Coefficient/
├── src/reflection_coefficient/
│   ├── io.py                  # load tank config, metadata, raw txt
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
├── tests/                     # pytest suite
├── experiment_data/           # tank_config.json, metadata/*.csv, raw RW###.txt / WN###.txt
├── docs/                      # reference PDFs + processing pipeline spec
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

## CLI — `scripts/run_analysis.py`

Three path inputs and three analysis choices are **persisted per-user** in
`~/.reflection_coefficient.json`. Priority for each: **CLI arg → stored value
→ built-in default**. Pass `--show-paths` to see what will be resolved.

### Full flag reference

| Flag | Values / default | Persisted? | Description |
|---|---|---|---|
| `--tank-config PATH` | default: `experiment_data/tank_config.json` | ✔ | Tank geometry + water depth JSON. |
| `--metadata-dir PATH` | default: `experiment_data/metadata/` | ✔ | Folder containing `rw.csv` / `wn.csv` / `js.csv`. |
| `--data-dir PATH` | default: `experiment_data/` | ✔ | Folder holding raw `<TEST_ID>.txt` files. Accepts either `<data_dir>/<scheme>/<TEST_ID>.txt` (per-scheme subfolders) or `<data_dir>/<TEST_ID>.txt` (flat). |
| `--scheme {rw,wn,js}` | *prompted if omitted* | — | Wave scheme. `rw` = regular, `wn` = white-noise irregular, `js` = JONSWAP irregular. |
| `--test TEST_ID \| all` | default: `all` | — | Test id (e.g. `RW005`, `WN003`) or `all` to iterate every test in the scheme. `all` is only supported for `--scheme rw`. |
| `--method {goda,least_squares}` | default: `least_squares` | ✔ | Separation method. Goda uses probes 1 & 3; Mansard–Funke uses all three. |
| `--window {none,hann}` | default: `hann` | ✔ | Spectral window for the irregular-wave FFT. Ignored for `rw`. |
| `--bandwidth HZ` | default: `0.04` | ✔ | Target resolution bandwidth in Hz for band-averaging (only meaningful with `--window hann`). |
| `--output PATH` | default: `<project>/results` | — | Parent output dir. A timestamped subfolder `YYYYMMDD_HHMMSS/` is created per run. |
| `--list` | flag | — | List discoverable tests for the chosen scheme and exit. |
| `--show-paths` | flag | — | Print the resolved `tank_config` / `metadata_dir` / `data_dir` / `method` / `window` and exit. |
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
| `rw` with `--test all` (≥2 tests) | `rw_kr_vs_freq_<method>.csv`, `rw_kr_vs_freq_<method>.png`, `rw_report_<method>.html` |
| `wn` or `js` (single test) | `<TEST>_<method>_spectrum.csv`, `<TEST>_<method>_report.html` |

The HTML reports use **Chart.js** for interactive, tooltipped plots (Gantt of
time windows, `Kr` vs frequency, incident/reflected spectra, singularity
metric). They are self-contained — open directly in a browser.

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

Papers in `docs/`:

- *Isolating incident and reflected wave spectra* — Goda & Suzuki.
- *Separation of incident and reflected spectra in wave flumes.*
- *The measurement of incident and reflected spectra using a least squares
  method* — Mansard & Funke.
- *The propagation of the waves in the CTO SA towing tank* — tank context.

`docs/reflection_processing_pipeline.md` is the operational spec that the code
follows — refer to it for the canonical step ordering, formulae, and
singularity-mask thresholds.

## License

MIT License — see [`LICENSE`](LICENSE).

Copyright © 2026 Ryan You, Kelvin Hydrodynamics Laboratory.
