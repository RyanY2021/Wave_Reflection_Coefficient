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

from .io import resolve_data_dir, resolve_metadata_dir, resolve_tank_config

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
    force: bool = False,
) -> list[str]:
    """Create the input scaffold. Returns a list of human-readable actions."""
    cfg_path = resolve_tank_config(tank_config)
    meta_dir = resolve_metadata_dir(metadata_dir)
    dat_dir = resolve_data_dir(data_dir)

    actions: list[str] = []

    actions.append(_write_if_absent(
        cfg_path, json.dumps(_TANK_CONFIG_TEMPLATE, indent=2) + "\n", force,
    ))

    for scheme, cols in _METADATA_HEADERS.items():
        csv_path = meta_dir / f"{scheme}.csv"
        actions.append(_write_if_absent(csv_path, ",".join(cols) + "\n", force))

    for sub in ("", "rw", "wn", "js"):
        d = dat_dir / sub if sub else dat_dir
        d.mkdir(parents=True, exist_ok=True)
    actions.append(f"mkdir  {dat_dir} (with rw/ wn/ js/ subfolders)")

    return actions
