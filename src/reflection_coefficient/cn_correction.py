"""Per-probe complex correction $C_n(f) = \\alpha_n \\cdot e^{-i\\beta_n(f)}$.

Mathematical background in ``bin/bn_gain_correction_derivation.md`` and
[issue #4](https://github.com/RyanY2021/Wave_Reflection_Coefficient/issues/4).

NumPy FFT convention $e^{-i\\omega t}$ throughout. Probe 1 is the reference
($C_1 \\equiv 1$). Convention chain:

* For a single incident wave with probe 1 reference,
  $B_n^{\\text{pred}}(f) = B_1^{\\text{meas}}(f) \\cdot e^{-i k (x_n - x_1)}$.
* Measured bin includes the probe-side error:
  $B_n^{\\text{meas}} = \\alpha_n \\cdot B_n^{\\text{pred}} \\cdot
   e^{-i k \\Delta x_n - i \\omega \\Delta t_n}$,
  where $\\Delta x_n$ is the signed offset of probe $n$ from its nominal
  position (positive = further from paddle) and $\\Delta t_n$ is the channel
  time-sync offset.
* Therefore the correction is
  $C_n(f) = \\alpha_n \\cdot e^{-i k \\Delta x_n - i \\omega \\Delta t_n}$,
  fitted on a noref window where the field is single-incident.
* Apply by dividing the canonical-window FFT bins:
  $B_n^{\\text{corr}} = B_n^{\\text{meas}} / C_n(f)$, then run the standard
  Goda / Mansard–Funke separation.

This module is pure NumPy + JSON I/O; no pipeline coupling.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

CnMode = Literal["amp", "phase", "both"]
CnAlphaMode = Literal["scalar", "dynamic"]
_PROBE_KEYS: tuple[str, ...] = ("wp1", "wp2", "wp3")
_FFT_SIGN_TAG = "numpy_minus_iwt"
_REFERENCE_PROBE = "wp1"
_FIT_MASK_DOC = (
    "Frequency range used when computing the scalar alpha (the fallback "
    "for bins outside per_bin's frequency support, and the only alpha "
    "consulted in --cn-alpha-mode scalar). Edit f_min_Hz / f_max_Hz to "
    "narrow the band; null = no bound. The per_bin table itself is built "
    "from every bin and is unaffected by this mask."
)


# ---------------------------------------------------------------------------
# Convention helpers
# ---------------------------------------------------------------------------


def predicted_bins(B1: np.ndarray, k: np.ndarray, dx_nominal: float) -> np.ndarray:
    """Single-incident-wave prediction at probe n: $B_n = B_1 e^{-i k \\Delta x}$."""
    return np.asarray(B1) * np.exp(-1j * np.asarray(k, dtype=float) * float(dx_nominal))


def measured_C(
    B1: np.ndarray, Bn: np.ndarray, k: np.ndarray, dx_nominal: float,
) -> np.ndarray:
    """Per-bin $C_n(f) = B_n^{\\text{meas}} / B_n^{\\text{pred}}$ in the noref window."""
    return np.asarray(Bn) / predicted_bins(B1, k, dx_nominal)


def _eval_alpha(
    f_arr: np.ndarray,
    alpha_scalar: float,
    alpha_mode: CnAlphaMode,
    alpha_table: dict | None,
) -> np.ndarray:
    """Return $\\alpha(f)$ as an array of the same shape as ``f_arr``.

    In ``scalar`` mode every element is ``alpha_scalar``. In ``dynamic`` mode,
    in-range bins are linearly interpolated from ``alpha_table = {"f_Hz":
    [...], "alpha": [...]}`` (assumed sorted ascending), and bins outside the
    table's $[f_{\\min}, f_{\\max}]$ fall back to ``alpha_scalar``. If
    ``alpha_table`` is missing or empty, dynamic silently degrades to scalar.
    """
    out = np.full_like(f_arr, float(alpha_scalar), dtype=float)
    if alpha_mode != "dynamic" or alpha_table is None:
        return out
    f_table = np.asarray(alpha_table.get("f_Hz", []), dtype=float)
    a_table = np.asarray(alpha_table.get("alpha", []), dtype=float)
    if f_table.size == 0 or f_table.size != a_table.size:
        return out
    interp = np.interp(f_arr, f_table, a_table)
    # Tolerance shields the in-range check against float roundoff at the
    # boundary — the table's f comes from FFT-snapped or target-frequency
    # quantisation, which can drift by a few ULPs from a hand-typed apply-f.
    span = float(f_table[-1] - f_table[0]) if f_table.size > 1 else 1.0
    tol = max(1e-9 * max(span, 1.0), 1e-12)
    in_range = (f_arr >= f_table[0] - tol) & (f_arr <= f_table[-1] + tol)
    return np.where(in_range, interp, float(alpha_scalar))


def evaluate_C(
    f: np.ndarray, k: np.ndarray,
    alpha: float, delta_x: float, delta_t: float,
    mode: CnMode = "both",
    *,
    alpha_mode: CnAlphaMode = "scalar",
    alpha_table: dict | None = None,
) -> np.ndarray:
    """Evaluate the parametric $C_n(f) = \\alpha(f) \\cdot e^{-i k \\Delta x - i \\omega \\Delta t}$.

    ``mode='amp'`` returns $\\alpha(f)$ only (phase identity).
    ``mode='phase'`` returns $e^{-i k \\Delta x - i \\omega \\Delta t}$ only ($\\alpha = 1$).
    ``mode='both'`` returns the full product.

    ``alpha_mode='scalar'`` (default) uses the supplied ``alpha`` for every $f$;
    ``alpha_mode='dynamic'`` interpolates ``alpha_table`` and falls back to the
    scalar for $f$ outside the table's frequency support.
    """
    f_arr = np.asarray(f, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    omega = 2.0 * np.pi * f_arr
    alpha_arr = _eval_alpha(f_arr, float(alpha), alpha_mode, alpha_table)
    if mode == "amp":
        out = alpha_arr.astype(complex)
    elif mode == "phase":
        out = np.exp(-1j * (k_arr * float(delta_x) + omega * float(delta_t)))
    elif mode == "both":
        out = alpha_arr * np.exp(
            -1j * (k_arr * float(delta_x) + omega * float(delta_t))
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'amp', 'phase', or 'both'.")
    return out


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_probe_cn_parametric(
    f: np.ndarray,
    k: np.ndarray,
    B1: np.ndarray,
    Bn: np.ndarray,
    dx_nominal: float,
    *,
    mask_f_min: float | None = None,
    mask_f_max: float | None = None,
) -> dict:
    """Fit $(\\alpha, \\Delta x, \\Delta t)$ for probe $n$ from noref bins.

    Two flavours of $\\alpha$ are produced from the same bin set:

    * **Dynamic** $\\alpha_n(f_i) = |C_n^{\\text{obs}}(f_i)|$ — every bin, no
      mask. Returned in ``per_bin`` (sorted by frequency); used by
      ``evaluate_C`` in ``alpha_mode='dynamic'`` via linear interpolation.
    * **Scalar** $\\alpha_n = \\langle |C_n^{\\text{obs}}| \\rangle$ — arithmetic
      mean over bins inside ``[mask_f_min, mask_f_max]``; ``None`` on either
      bound = no constraint that side. Returned as ``alpha``; used as the
      out-of-range fallback for the dynamic mode and as the only $\\alpha$ in
      ``alpha_mode='scalar'``.

    $(\\Delta x, \\Delta t)$ = least-squares fit of
    $\\arg(C_n / \\alpha) = -(k \\Delta x + \\omega \\Delta t)$ across all bins
    after phase unwrapping (mask is **not** applied to the phase regression —
    LSQ benefits from every bin).
    """
    f_arr = np.asarray(f, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    if f_arr.size == 0:
        raise ValueError("fit_probe_cn_parametric: no bins supplied.")
    omega = 2.0 * np.pi * f_arr
    C = measured_C(B1, Bn, k_arr, dx_nominal)
    abs_C = np.abs(C)

    # Scalar α: masked mean. The mask narrows to a frequency band the user
    # trusts (avoiding e.g. low-SNR edges or harmonics outside the rig's
    # reliable range). Default = whole data range, i.e. no-op mask.
    f_lo = float(mask_f_min) if mask_f_min is not None else float(np.min(f_arr))
    f_hi = float(mask_f_max) if mask_f_max is not None else float(np.max(f_arr))
    mask = (f_arr >= f_lo) & (f_arr <= f_hi)
    if not np.any(mask):
        raise ValueError(
            "fit_probe_cn_parametric: scalar-alpha mask "
            f"[{f_lo}, {f_hi}] Hz excludes every bin in fit data "
            f"[{float(np.min(f_arr))}, {float(np.max(f_arr))}] Hz."
        )
    alpha = float(np.mean(abs_C[mask]))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "fit_probe_cn_parametric: masked mean |C_n| is non-positive or "
            f"non-finite ({alpha!r}); fit cannot proceed."
        )

    # Sort by frequency before unwrapping so the unwrap follows a coherent phase
    # progression rather than the arbitrary record order.
    order = np.argsort(f_arr)
    f_s = f_arr[order]
    k_s = k_arr[order]
    omega_s = omega[order]
    C_s = C[order]
    abs_C_s = abs_C[order]
    beta = np.unwrap(np.angle(C_s / alpha))

    # Solve [[k, omega]] @ [delta_x, delta_t] = -beta in least-squares sense.
    # Single-bin case (n_bins == 1) is rank-deficient; lstsq still produces a
    # value but Δx and Δt cannot be separated. Caller is expected to warn.
    A = np.column_stack([k_s, omega_s])
    sol, *_ = np.linalg.lstsq(A, -beta, rcond=None)
    delta_x, delta_t = float(sol[0]), float(sol[1])

    # Residual: how well the parametric form fits the unwrapped phase.
    beta_fit = -(k_s * delta_x + omega_s * delta_t)
    residual_rms = float(np.sqrt(np.mean((beta - beta_fit) ** 2)))

    return {
        "alpha": alpha,
        "delta_x_m": delta_x,
        "delta_t_s": delta_t,
        "fit_diagnostics": {
            "residual_rms_rad": residual_rms,
            "n_bins_used": int(f_arr.size),
            "n_bins_in_mask": int(np.count_nonzero(mask)),
            "mask_f_min_Hz": f_lo,
            "mask_f_max_Hz": f_hi,
        },
        "per_bin": {
            "f_Hz": [float(x) for x in f_s],
            "alpha": [float(x) for x in abs_C_s],
            "beta_rad": [float(x) for x in beta],
        },
    }


def fit_cn_from_records(
    records: Iterable[dict],
    X12: float,
    X13: float,
    *,
    existing_fit_mask: dict | None = None,
) -> dict:
    """Aggregate-fit $C_n$ across one or more noref records.

    Each ``record`` is a dict with keys ``f`` (1-D float), ``k`` (1-D float),
    ``B1``, ``B2``, ``B3`` (1-D complex) all the same length. Every bin in
    every record is included in the aggregate fit — pre-filter records before
    calling if you want to exclude harmonics, near-singular spacings, or
    low-SNR bins.

    ``existing_fit_mask`` (if given) carries ``f_min_Hz`` / ``f_max_Hz``
    pulled from a prior ``probes_refined.json``. Those bounds are reused
    verbatim so user edits to the JSON survive a re-fit. ``None`` keys (or
    a missing block) default to the data range, i.e. a no-op mask.

    Returns ``{wp1: {...identity...}, wp2: {...fit...}, wp3: {...fit...},
    fit_mask: {...}}`` suitable for ``save_cn_config``. Probe 1 is the
    reference and gets identity values. wp2 is fitted against $X_{12}$ as
    the nominal spacing; wp3 against $X_{13}$.
    """
    rec_list = list(records)
    if not rec_list:
        raise ValueError("fit_cn_from_records: at least one record required.")

    f_acc: list[np.ndarray] = []
    k_acc: list[np.ndarray] = []
    B1_acc: list[np.ndarray] = []
    B2_acc: list[np.ndarray] = []
    B3_acc: list[np.ndarray] = []
    for rec in rec_list:
        f_acc.append(np.asarray(rec["f"], dtype=float))
        k_acc.append(np.asarray(rec["k"], dtype=float))
        B1_acc.append(np.asarray(rec["B1"]))
        B2_acc.append(np.asarray(rec["B2"]))
        B3_acc.append(np.asarray(rec["B3"]))

    f_all = np.concatenate(f_acc)
    k_all = np.concatenate(k_acc)
    B1_all = np.concatenate(B1_acc)
    B2_all = np.concatenate(B2_acc)
    B3_all = np.concatenate(B3_acc)

    f_data_min = float(np.min(f_all))
    f_data_max = float(np.max(f_all))
    if existing_fit_mask:
        m_min = existing_fit_mask.get("f_min_Hz")
        m_max = existing_fit_mask.get("f_max_Hz")
        f_lo = float(m_min) if m_min is not None else f_data_min
        f_hi = float(m_max) if m_max is not None else f_data_max
    else:
        f_lo, f_hi = f_data_min, f_data_max

    fit2 = fit_probe_cn_parametric(
        f_all, k_all, B1_all, B2_all, X12,
        mask_f_min=f_lo, mask_f_max=f_hi,
    )
    fit3 = fit_probe_cn_parametric(
        f_all, k_all, B1_all, B3_all, X13,
        mask_f_min=f_lo, mask_f_max=f_hi,
    )

    return {
        "wp1": {
            "alpha": 1.0,
            "delta_x_m": 0.0,
            "delta_t_s": 0.0,
            "_note": "reference probe; identity by definition",
        },
        "wp2": fit2,
        "wp3": fit3,
        "fit_mask": {"f_min_Hz": f_lo, "f_max_Hz": f_hi},
    }


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _probe_alpha_table(probe_entry: dict) -> dict | None:
    """Pull the dynamic-$\\alpha$ interpolation table from a probe JSON entry.

    The table source is ``per_bin.{f_Hz, alpha}``: per-bin observed magnitudes
    in fit-frequency order. Returns ``None`` when the entry is identity-only
    (e.g. ``wp1``, or the placeholder written by ``init_project``), in which
    case dynamic mode silently degrades to scalar.
    """
    pb = probe_entry.get("per_bin")
    if not pb:
        return None
    f_table = pb.get("f_Hz")
    a_table = pb.get("alpha")
    if not f_table or not a_table or len(f_table) != len(a_table):
        return None
    return {"f_Hz": f_table, "alpha": a_table}


def apply_cn_to_bins(
    B1: np.ndarray,
    B2: np.ndarray,
    B3: np.ndarray,
    f: np.ndarray,
    k: np.ndarray,
    cn_config: dict,
    mode: CnMode = "both",
    *,
    alpha_mode: CnAlphaMode = "scalar",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide $B_2$, $B_3$ by $C_n(f)$ in place-safe fashion. $B_1$ unchanged.

    Works on scalar, 0-D, and 1-D inputs uniformly via NumPy broadcasting; the
    regular-wave path passes single-element arrays, the irregular path passes
    per-bin arrays.

    ``alpha_mode='dynamic'`` uses each probe's ``per_bin`` table (linear
    interpolation, scalar fallback outside the table's frequency support);
    ``'scalar'`` uses the stored ``alpha`` everywhere.
    """
    p2 = cn_config.get("wp2", {})
    p3 = cn_config.get("wp3", {})
    table2 = _probe_alpha_table(p2) if alpha_mode == "dynamic" else None
    table3 = _probe_alpha_table(p3) if alpha_mode == "dynamic" else None
    C2 = evaluate_C(
        f, k, p2["alpha"], p2["delta_x_m"], p2["delta_t_s"], mode=mode,
        alpha_mode=alpha_mode, alpha_table=table2,
    )
    C3 = evaluate_C(
        f, k, p3["alpha"], p3["delta_x_m"], p3["delta_t_s"], mode=mode,
        alpha_mode=alpha_mode, alpha_table=table3,
    )
    return np.asarray(B1), np.asarray(B2) / C2, np.asarray(B3) / C3


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

_DOC = (
    "Per-probe complex correction C_n(f) = alpha_n * "
    "exp(-i k Delta_x_n - i omega Delta_t_n). Fitted on the noref window "
    "(single-incident field, a_R = 0); applied to FFT bins of the canonical "
    "window before Goda / Mansard-Funke separation. Probe 1 is the reference "
    "(C_1 = 1). Convention: numpy e^{-iwt}. See "
    "src/reflection_coefficient/cn_correction.py for the math."
)


def save_cn_config(
    path: Path | str,
    cn_dict: dict,
    *,
    fit_meta: dict | None = None,
) -> None:
    """Write a probes_refined.json with a metadata header.

    ``cn_dict`` is the return value of :func:`fit_cn_from_records` (or any dict
    with ``wp1``/``wp2``/``wp3`` entries). ``fit_meta`` (if given) is merged
    into the top-level ``fit_meta`` block so apply-time can detect a
    ``--recalibrate`` mismatch, etc.
    """
    out = {
        "_doc": _DOC,
        "convention": {
            "fft_sign": _FFT_SIGN_TAG,
            "reference_probe": _REFERENCE_PROBE,
        },
        "fit_meta": dict(fit_meta or {}),
    }
    out["fit_meta"].setdefault(
        "fit_date_utc",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if "fit_mask" in cn_dict:
        fm = dict(cn_dict["fit_mask"])
        fm.setdefault("_doc", _FIT_MASK_DOC)
        out["fit_mask"] = fm
    for key in _PROBE_KEYS:
        if key in cn_dict:
            out[key] = cn_dict[key]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)


def load_cn_config(path: Path | str) -> dict:
    """Read probes_refined.json and validate the convention block.

    Raises ``ValueError`` if ``convention.fft_sign`` or ``reference_probe``
    differ from this module's expectations — a flipped sign would silently
    corrupt the separation, so we fail loudly.
    """
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        cfg = json.load(fh)
    conv = cfg.get("convention", {})
    if conv.get("fft_sign") != _FFT_SIGN_TAG:
        raise ValueError(
            f"{p}: convention.fft_sign is {conv.get('fft_sign')!r}, "
            f"expected {_FFT_SIGN_TAG!r}. Refusing to apply a correction with "
            "an unknown FFT sign convention — it would silently flip the "
            "separation."
        )
    if conv.get("reference_probe") != _REFERENCE_PROBE:
        raise ValueError(
            f"{p}: convention.reference_probe is "
            f"{conv.get('reference_probe')!r}, expected {_REFERENCE_PROBE!r}."
        )
    for key in _PROBE_KEYS:
        if key not in cfg:
            raise ValueError(f"{p}: missing required probe entry {key!r}.")
        for field in ("alpha", "delta_x_m", "delta_t_s"):
            if field not in cfg[key]:
                raise ValueError(
                    f"{p}: probe {key!r} is missing field {field!r}."
                )
    return cfg


def identity_cn_config() -> dict:
    """Return a unit-correction config (every probe identity).

    Useful for the placeholder file written by ``init_project`` and for
    short-circuit testing.
    """
    return {key: {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0}
            for key in _PROBE_KEYS}
