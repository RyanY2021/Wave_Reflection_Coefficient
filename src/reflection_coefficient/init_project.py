"""Generate the input scaffold required before the first analysis run.

Creates, under the resolved ``tank_config`` / ``metadata_dir`` / ``data_dir``:

* ``tank_config.json`` — tank-wide constants + probe geometry (placeholders
  the user must fill in before running the pipeline).
* ``metadata/rw.csv`` / ``wn.csv`` / ``js.csv`` — per-test manifests with
  headers only (one commented example row).
* ``data_dir`` itself, plus ``rw/``, ``wn/``, ``js/`` subfolders for raw
  probe txt files.

Existing files are never overwritten unless ``force=True``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .io import (
    resolve_cn_config,
    resolve_data_dir,
    resolve_metadata_dir,
    resolve_probes_config,
    resolve_tank_config,
)

_TANK_CONFIG_TEMPLATE: dict = {
    "tank": {
        "water_depth_m": 2.0,
        "gravity_m_s2": 9.81,
    },
    "probe_geometry": {
        "tank_length_m": None,
        "x_paddle_to_wp1_m": None,
        "X12_m": None,
        "X13_m": None,
    },
}

_PROBES_CONFIG_TEMPLATE: dict = {
    "_doc": (
        "Per-probe linear re-calibration. Acquisition recorded "
        "eta_old = scale_old*raw + offset_old; the corrected calibration is "
        "eta_new = scale_new*raw + offset_new. Transfer: "
        "eta_new = (scale_new/scale_old)*(eta_old - offset_old) + offset_new. "
        "Enabled by `run_analysis.py --recalibrate` (persisted). "
        "Offsets share units with eta returned by load_probe_data (METRES); "
        "scale units cancel — use any unit consistent across old/new. "
        "Identity values (scale_new=scale_old, offset_new=offset_old) disable "
        "the transform for that probe."
    ),
    "wp1": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
    "wp2": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
    "wp3": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
}

_CN_CONFIG_TEMPLATE: dict = {
    "_doc": (
        "Per-probe complex correction C_n(f) = alpha_n * "
        "exp(-i k Delta_x_n - i omega Delta_t_n). Probe 1 is the reference "
        "(C_1 = 1). Convention: numpy e^{-iwt}. PLACEHOLDER identity values; "
        "run `run_analysis.py --scheme rw --test all --window-mode noref "
        "--cn-fit` to populate from a noref-window fit. Apply with --cn-apply."
    ),
    "convention": {"fft_sign": "numpy_minus_iwt", "reference_probe": "wp1"},
    "fit_meta": {"_note": "placeholder; not fitted from data"},
    "wp1": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0,
            "_note": "reference probe; identity by definition"},
    "wp2": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
    "wp3": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
}

_METADATA_HEADERS: dict[str, list[str]] = {
    "rw": ["test_id", "f_Hz", "a_target_m", "t_gen_s", "notes"],
    "wn": ["test_id", "S0_m2_Hz", "f_min_Hz", "f_max_Hz", "t_gen_s", "notes"],
    "js": ["test_id", "Hs_target_m", "Tp_target_s", "gamma",
           "f_min_Hz", "f_max_Hz", "t_gen_s", "notes"],
}


def _write_if_absent(path: Path, content: str, force: bool) -> str:
    if path.exists() and not force:
        return f"skip   {path} (exists)"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"write  {path}"


def init_project(
    tank_config: Path | str | None = None,
    metadata_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    probes_config: Path | str | None = None,
    cn_config: Path | str | None = None,
    force: bool = False,
) -> list[str]:
    """Create the input scaffold. Returns a list of human-readable actions."""
    cfg_path = resolve_tank_config(tank_config)
    meta_dir = resolve_metadata_dir(metadata_dir)
    dat_dir = resolve_data_dir(data_dir)
    probes_path = resolve_probes_config(probes_config)
    cn_path = resolve_cn_config(cn_config)

    actions: list[str] = []

    actions.append(_write_if_absent(
        cfg_path, json.dumps(_TANK_CONFIG_TEMPLATE, indent=2) + "\n", force,
    ))

    actions.append(_write_if_absent(
        probes_path, json.dumps(_PROBES_CONFIG_TEMPLATE, indent=2) + "\n", force,
    ))

    actions.append(_write_if_absent(
        cn_path, json.dumps(_CN_CONFIG_TEMPLATE, indent=2) + "\n", force,
    ))

    for scheme, cols in _METADATA_HEADERS.items():
        csv_path = meta_dir / f"{scheme}.csv"
        actions.append(_write_if_absent(csv_path, ",".join(cols) + "\n", force))

    for sub in ("", "rw", "wn", "js"):
        d = dat_dir / sub if sub else dat_dir
        d.mkdir(parents=True, exist_ok=True)
    actions.append(f"mkdir  {dat_dir} (with rw/ wn/ js/ subfolders)")

    return actions
