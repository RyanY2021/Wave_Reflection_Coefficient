"""Data loading utilities for wave probe measurements.

Three independent inputs are combined for each test:

* **tank_config** — tank-wide constants and probe geometry (single JSON file).
* **metadata_dir** — folder holding the per-scheme CSVs
  (``rw.csv``, ``wn.csv``, ``js.csv``).
* **data_dir** — folder holding the raw probe time-series files. May be flat
  (``<data_dir>/<TEST_ID>.txt``) or contain per-scheme subfolders
  (``<data_dir>/<scheme>/<TEST_ID>.txt``). The loader tries both.

Any of the three may be overridden on the CLI. When overridden, the chosen
path is persisted to ``~/.reflection_coefficient.json`` so subsequent runs
pick it up automatically.

Derived quantities (``k``, ``L``, ``cg``, clipping window, ...) are **not**
stored in metadata; they are computed by downstream stages from
``(f, water_depth, array_geometry)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Campaign = Literal["rw", "wn", "js"]
CAMPAIGNS: tuple[str, ...] = ("rw", "wn", "js")

# Test-id prefix → scheme. Prefixes are matched case-insensitively.
_PREFIX_TO_CAMPAIGN = {"RW": "rw", "WN": "wn", "JS": "js"}

_PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TANK_CONFIG = _PKG_ROOT / "experiment_data" / "tank_config.json"
DEFAULT_METADATA_DIR = _PKG_ROOT / "experiment_data" / "metadata"
DEFAULT_DATA_DIR = _PKG_ROOT / "experiment_data"

USER_CONFIG_PATH = Path.home() / ".reflection_coefficient.json"

_PATH_KEYS = ("tank_config", "metadata_dir", "data_dir")


# ---------------------------------------------------------------------------
# User-level config persistence
# ---------------------------------------------------------------------------


def _load_user_config() -> dict:
    if not USER_CONFIG_PATH.exists():
        return {}
    try:
        with open(USER_CONFIG_PATH) as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_user_config(cfg: dict) -> None:
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def save_paths(
    tank_config: Path | str | None = None,
    metadata_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> None:
    """Persist any of the three paths the user provided. ``None`` = leave alone."""
    cfg = _load_user_config()
    for key, value in zip(_PATH_KEYS, (tank_config, metadata_dir, data_dir)):
        if value is not None:
            cfg[key] = str(Path(value).resolve())
    _save_user_config(cfg)


def _resolve(key: str, explicit: Path | str | None, default: Path) -> Path:
    if explicit is not None:
        return Path(explicit)
    stored = _load_user_config().get(key)
    return Path(stored) if stored else default


def resolve_tank_config(explicit: Path | str | None = None) -> Path:
    return _resolve("tank_config", explicit, DEFAULT_TANK_CONFIG)


def resolve_metadata_dir(explicit: Path | str | None = None) -> Path:
    return _resolve("metadata_dir", explicit, DEFAULT_METADATA_DIR)


def resolve_data_dir(explicit: Path | str | None = None) -> Path:
    return _resolve("data_dir", explicit, DEFAULT_DATA_DIR)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tank_config(path: Path | str | None = None) -> dict:
    p = resolve_tank_config(path)
    with open(p) as f:
        return json.load(f)


def load_metadata(
    campaign: Campaign, metadata_dir: Path | str | None = None
) -> pd.DataFrame:
    """Return the per-test metadata table for a campaign, indexed by test_id."""
    d = resolve_metadata_dir(metadata_dir)
    return pd.read_csv(d / f"{campaign}.csv").set_index("test_id")


def _data_file(data_dir: Path, campaign: Campaign, test_id: str) -> Path:
    """Locate ``<TEST_ID>.txt`` under data_dir/scheme/ or data_dir/ (flat)."""
    with_scheme = data_dir / campaign / f"{test_id}.txt"
    if with_scheme.exists():
        return with_scheme
    flat = data_dir / f"{test_id}.txt"
    if flat.exists():
        return flat
    raise FileNotFoundError(
        f"{test_id}.txt not found under {data_dir} "
        f"(tried {with_scheme} and {flat})"
    )


def list_tests(
    campaign: Campaign,
    data_dir: Path | str | None = None,
    metadata_dir: Path | str | None = None,
) -> list[str]:
    """Return test ids for which both a data file and a metadata row exist."""
    d = resolve_data_dir(data_dir)
    prefix = next(p for p, c in _PREFIX_TO_CAMPAIGN.items() if c == campaign)
    candidates: set[str] = set()
    sub = d / campaign
    if sub.is_dir():
        candidates.update(p.stem for p in sub.glob(f"{prefix}*.txt"))
    if d.is_dir():
        candidates.update(p.stem for p in d.glob(f"{prefix}*.txt"))
    try:
        in_matrix = set(load_metadata(campaign, metadata_dir).index)
    except FileNotFoundError:
        in_matrix = set()
    return sorted(candidates & in_matrix)


# ---------------------------------------------------------------------------
# TestMeta
# ---------------------------------------------------------------------------


@dataclass
class TestMeta:
    """All non-derivable metadata for one test."""

    test_id: str
    campaign: Campaign

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
    config: dict,
    row: pd.Series,
) -> TestMeta:
    tank = config["tank"]
    geom = config["probe_geometry"]
    r = row.to_dict()
    return TestMeta(
        test_id=test_id,
        campaign=campaign,
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
    campaign: Campaign | None = None,
    tank_config: Path | str | None = None,
    metadata_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TestMeta]:
    """Load one test's elevations (m) plus metadata.

    Returns ``(t, eta1, eta2, eta3, meta)`` where ``eta1`` is the probe nearest
    the wave maker and ``eta3`` the probe nearest the structure. Campaign is
    inferred from the test id prefix when not given. Any of ``tank_config``,
    ``metadata_dir``, ``data_dir`` may be overridden; otherwise the stored
    user-config value or built-in default is used.
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

    path = _data_file(resolve_data_dir(data_dir), campaign, test_id)

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

    config = load_tank_config(tank_config)
    table = load_metadata(campaign, metadata_dir)
    if test_id not in table.index:
        raise KeyError(f"{test_id} not found in {campaign}.csv")
    meta = _build_meta(test_id, campaign, config, table.loc[test_id])

    return t, eta1, eta2, eta3, meta
