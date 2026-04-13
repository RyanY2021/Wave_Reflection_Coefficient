"""Data loading utilities for wave probe measurements.

Three sources of metadata are merged for each test:

* Tank-wide constants and per-array geometry — ``experiment_data/tank_config.json``
* Per-test varying parameters — ``experiment_data/metadata/{rw,wn}.csv``
* Raw probe time-series — ``experiment_data/{rw,wn}/{array}/{TEST_ID}.txt``

Derived quantities (``k``, ``L``, ``cg``, clipping window, ...) are **not**
stored in metadata; they are computed by downstream stages from
``(f, water_depth, array_geometry)``. The metadata CSVs only carry what varies
per test and cannot be derived.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Campaign = Literal["rw", "wn", "jonswap"]
CAMPAIGNS: tuple[str, ...] = ("rw", "wn", "jonswap")

# Test-id prefix → scheme. Prefixes are matched case-insensitively.
_PREFIX_TO_CAMPAIGN = {"RW": "rw", "WN": "wn", "JS": "jonswap"}

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "experiment_data"
ENV_VAR = "REFLECTION_DATA_ROOT"

EXPECTED_LAYOUT = """\
<data_root>/
├── tank_config.json
├── metadata/
│   ├── rw.csv        # regular waves
│   ├── wn.csv        # pure white-noise (flat PSD) irregular waves
│   └── jonswap.csv   # JONSWAP irregular waves
├── rw/<array>/RW###.txt
├── wn/<array>/WN###.txt
└── jonswap/<array>/JS###.txt
(<array> is any subfolder name you use for a probe-array position.)
"""


def resolve_data_root(explicit: Path | str | None = None) -> Path:
    """Pick the data root in priority order: argument → env var → default.

    Raises ``FileNotFoundError`` with the expected layout if none of the
    candidates exist or the chosen path is missing the scaffolding.
    """
    if explicit is not None:
        root = Path(explicit)
    elif os.environ.get(ENV_VAR):
        root = Path(os.environ[ENV_VAR])
    else:
        root = DEFAULT_DATA_ROOT

    if not root.exists():
        raise FileNotFoundError(
            f"Data root {root} does not exist.\n"
            f"Pass --data-root, set ${ENV_VAR}, or place data at the default "
            f"location.\nExpected layout:\n{EXPECTED_LAYOUT}"
        )
    return root


def validate_data_root(data_root: Path | str) -> None:
    """Verify that ``data_root`` matches the expected scheme. Raise on mismatch."""
    root = Path(data_root)
    required = [
        root / "tank_config.json",
        *(root / "metadata" / f"{c}.csv" for c in CAMPAIGNS),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Data root is missing expected files:\n  "
            + "\n  ".join(missing)
            + f"\nExpected layout:\n{EXPECTED_LAYOUT}"
        )


def list_tests(
    campaign: Campaign,
    array: str,
    data_root: Path | str | None = None,
) -> list[str]:
    """Return test ids for which both a data file and a metadata row exist."""
    root = resolve_data_root(data_root)
    data_dir = root / campaign / array
    if not data_dir.is_dir():
        return []
    prefix = next(p for p, c in _PREFIX_TO_CAMPAIGN.items() if c == campaign)
    on_disk = {p.stem for p in data_dir.glob(f"{prefix}*.txt")}
    in_matrix = set(load_metadata(campaign, root).index)
    return sorted(on_disk & in_matrix)


@dataclass
class TestMeta:
    """All non-derivable metadata for one test."""

    test_id: str
    campaign: Campaign
    array: str

    # Tank-wide (from tank_config.json)
    water_depth_m: float
    gravity_m_s2: float

    # Probe geometry (None until measured and filled into tank_config.json)
    tank_length_m: float | None
    x_paddle_to_wp1_m: float | None
    X12_m: float | None
    X13_m: float | None

    @property
    def x_wp3_to_struct_m(self) -> float | None:
        """Derived: distance from probe 3 to the reflecting structure."""
        parts = (self.tank_length_m, self.x_paddle_to_wp1_m, self.X13_m)
        if any(p is None for p in parts):
            return None
        return self.tank_length_m - self.x_paddle_to_wp1_m - self.X13_m

    # Per-test, varying, non-derivable
    t_gen_s: float | None = None
    notes: str | None = None

    # Regular-wave target
    f_Hz: float | None = None
    a_target_m: float | None = None

    # White-noise target
    S0_m2_Hz: float | None = None

    # JONSWAP target
    Hs_target_m: float | None = None
    Tp_target_s: float | None = None
    gamma: float | None = None

    # Band edges (white-noise and JONSWAP)
    f_min_Hz: float | None = None
    f_max_Hz: float | None = None

    # Anything in the CSV that isn't explicitly modelled above.
    extra: dict = field(default_factory=dict)


def load_tank_config(data_root: Path | str | None = None) -> dict:
    root = resolve_data_root(data_root)
    with open(root / "tank_config.json") as f:
        return json.load(f)


def load_metadata(
    campaign: Campaign, data_root: Path | str | None = None
) -> pd.DataFrame:
    """Return the per-test metadata table for a campaign, indexed by test_id."""
    root = resolve_data_root(data_root)
    path = root / "metadata" / f"{campaign}.csv"
    return pd.read_csv(path).set_index("test_id")


_KNOWN_META_COLUMNS = {
    "t_gen_s", "notes",
    "f_Hz", "a_target_m",
    "S0_m2_Hz",
    "Hs_target_m", "Tp_target_s", "gamma",
    "f_min_Hz", "f_max_Hz",
}


def _build_meta(
    test_id: str,
    campaign: Campaign,
    array: str,
    config: dict,
    row: pd.Series,
) -> TestMeta:
    tank = config["tank"]
    geom = config["probe_geometry"]
    r = row.to_dict()
    return TestMeta(
        test_id=test_id,
        campaign=campaign,
        array=array,
        water_depth_m=tank["water_depth_m"],
        gravity_m_s2=tank["gravity_m_s2"],
        tank_length_m=geom.get("tank_length_m"),
        x_paddle_to_wp1_m=geom.get("x_paddle_to_wp1_m"),
        X12_m=geom.get("X12_m"),
        X13_m=geom.get("X13_m"),
        t_gen_s=r.get("t_gen_s"),
        notes=r.get("notes"),
        f_Hz=r.get("f_Hz"),
        a_target_m=r.get("a_target_m"),
        S0_m2_Hz=r.get("S0_m2_Hz"),
        Hs_target_m=r.get("Hs_target_m"),
        Tp_target_s=r.get("Tp_target_s"),
        gamma=r.get("gamma"),
        f_min_Hz=r.get("f_min_Hz"),
        f_max_Hz=r.get("f_max_Hz"),
        extra={k: v for k, v in r.items() if k not in _KNOWN_META_COLUMNS},
    )


def load_probe_data(
    test_id: str,
    array: str,
    campaign: Campaign | None = None,
    data_root: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TestMeta]:
    """Load one test's elevations (m) plus metadata.

    Returns ``(t, eta1, eta2, eta3, meta)`` where ``eta1`` is the probe nearest
    the wave maker and ``eta3`` the probe nearest the structure. Campaign is
    inferred from the test id prefix when not given.
    """
    if campaign is None:
        up = test_id.upper()
        for prefix, c in _PREFIX_TO_CAMPAIGN.items():
            if up.startswith(prefix):
                campaign = c
                break
        else:
            raise ValueError(
                f"Cannot infer campaign from test id {test_id!r}. "
                f"Known prefixes: {sorted(_PREFIX_TO_CAMPAIGN)}"
            )

    data_root = resolve_data_root(data_root)
    path = data_root / campaign / array / f"{test_id}.txt"

    # Tab-separated; two header rows (names, units). On-disk column order is
    # time, keyboard, wp3, wp2, wp1 — note wp3 precedes wp1.
    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        header=None,
        names=["time", "keyboard", "wp3", "wp2", "wp1"],
        engine="python",
    )
    df = df.dropna(subset=["wp1", "wp2", "wp3"]).reset_index(drop=True)

    t = df["time"].to_numpy(dtype=float)
    eta1 = df["wp1"].to_numpy(dtype=float) / 1000.0
    eta2 = df["wp2"].to_numpy(dtype=float) / 1000.0
    eta3 = df["wp3"].to_numpy(dtype=float) / 1000.0

    config = load_tank_config(data_root)
    table = load_metadata(campaign, data_root)
    if test_id not in table.index:
        raise KeyError(f"{test_id} not found in metadata/{campaign}.csv")
    meta = _build_meta(test_id, campaign, array, config, table.loc[test_id])

    return t, eta1, eta2, eta3, meta
