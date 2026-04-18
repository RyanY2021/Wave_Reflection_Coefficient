"""Header-driven column lookup for raw probe txt files.

The acquisition system emits files whose first row labels columns like
``Time \t <instrument> \t 3 wp3 \t 2 wp2 \t 1 wp1``, where the instrument
column is ``Keyboard`` in some sessions and ``sonic`` in others, and where
extra redundant instruments may appear in any order. ``load_probe_data``
must locate Time and wp1/wp2/wp3 by header name, not by position.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reflection_coefficient.io import _parse_probe_header, load_probe_data


def _write_tank_config(root: Path) -> Path:
    cfg = root / "tank_config.json"
    cfg.write_text(
        '{"tank": {"water_depth_m": 2.0, "gravity_m_s2": 9.81}, '
        '"probe_geometry": {"tank_length_m": 10.0, '
        '"x_paddle_to_wp1_m": 3.0, "X12_m": 0.5, "X13_m": 1.0}}',
        encoding="utf-8",
    )
    return cfg


def _write_metadata(root: Path, test_id: str) -> Path:
    d = root / "metadata"
    d.mkdir()
    (d / "rw.csv").write_text(
        "test_id,f_Hz,a_target_m,t_gen_s,notes\n"
        f"{test_id},0.5,0.02,60,\n",
        encoding="utf-8",
    )
    return d


def _write_txt(path: Path, header: list[str], rows: list[list[str]]) -> None:
    lines = ["\t".join(header), "\t".join(["Units"] + ["mm"] * (len(header) - 1))]
    for r in rows:
        lines.append("\t".join(r))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_header_keyboard_shape(tmp_path: Path):
    p = tmp_path / "RW001.txt"
    _write_txt(
        p,
        header=["Time", "31 Keyboard", "3 wp3", "2 wp2", "1 wp1"],
        rows=[["0.00", "0", "", "", ""], ["0.01", "0", "1.0", "2.0", "3.0"]],
    )
    t_col, pc = _parse_probe_header(p)
    assert t_col == 0
    assert pc == {3: 2, 2: 3, 1: 4}


def test_parse_header_sonic_shape(tmp_path: Path):
    p = tmp_path / "RW001.txt"
    _write_txt(
        p,
        header=["Time", "4 sonic", "3 wp3", "2 wp2", "1 wp1"],
        rows=[["0.00", "", "1.0", "2.0", "3.0"]],
    )
    t_col, pc = _parse_probe_header(p)
    assert t_col == 0
    assert pc == {3: 2, 2: 3, 1: 4}


def test_parse_header_tolerates_redundant_columns_anywhere(tmp_path: Path):
    # wp columns not in consecutive positions, and an extra redundant column.
    p = tmp_path / "RW001.txt"
    _write_txt(
        p,
        header=[
            "Time", "31 Keyboard", "4 sonic", "1 wp1", "99 extra", "2 wp2", "3 wp3",
        ],
        rows=[["0.0", "0", "", "3.0", "xx", "2.0", "1.0"]],
    )
    t_col, pc = _parse_probe_header(p)
    assert t_col == 0
    assert pc == {1: 3, 2: 5, 3: 6}


def test_parse_header_missing_probe_raises(tmp_path: Path):
    p = tmp_path / "bad.txt"
    _write_txt(
        p,
        header=["Time", "31 Keyboard", "3 wp3", "2 wp2"],
        rows=[["0.0", "0", "1.0", "2.0"]],
    )
    with pytest.raises(ValueError, match="wp1"):
        _parse_probe_header(p)


def test_load_probe_data_matches_header_columns_not_position(tmp_path: Path):
    tank_cfg = _write_tank_config(tmp_path)
    meta_dir = _write_metadata(tmp_path, "RW001")

    data = tmp_path / "data"
    test_path = data / "rw" / "RW001.txt"
    # wp1/wp2/wp3 shuffled with redundant columns interleaved; loader must
    # still return the correct series keyed to probe index.
    _write_txt(
        test_path,
        header=[
            "Time", "1 wp1", "99 garbage", "3 wp3", "4 sonic", "2 wp2",
        ],
        rows=[
            ["0.00", "", "0", "", "", ""],            # dropped (NaN probes)
            ["0.01", "11.0", "0", "33.0", "x", "22.0"],
            ["0.02", "12.0", "0", "34.0", "y", "23.0"],
        ],
    )

    t, e1, e2, e3, meta = load_probe_data(
        "RW001",
        tank_config=tank_cfg, metadata_dir=meta_dir, data_dir=data,
    )

    assert np.allclose(t, [0.01, 0.02])
    assert np.allclose(e1, [0.011, 0.012])  # mm -> m
    assert np.allclose(e2, [0.022, 0.023])
    assert np.allclose(e3, [0.033, 0.034])
    assert meta.test_id == "RW001"
