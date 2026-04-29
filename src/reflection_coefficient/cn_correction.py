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
_PROBE_KEYS: tuple[str, ...] = ("wp1", "wp2", "wp3")
_FFT_SIGN_TAG = "numpy_minus_iwt"
_REFERENCE_PROBE = "wp1"


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


def evaluate_C(
    f: np.ndarray, k: np.ndarray,
    alpha: float, delta_x: float, delta_t: float,
    mode: CnMode = "both",
) -> np.ndarray:
    """Evaluate the parametric $C_n(f) = \\alpha \\cdot e^{-i k \\Delta x - i \\omega \\Delta t}$.

    ``mode='amp'`` returns $\\alpha$ only (phase identity).
    ``mode='phase'`` returns $e^{-i k \\Delta x - i \\omega \\Delta t}$ only ($\\alpha = 1$).
    ``mode='both'`` returns the full product.
    """
    f_arr = np.asarray(f, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    omega = 2.0 * np.pi * f_arr
    if mode == "amp":
        out = np.full_like(omega, float(alpha), dtype=complex)
    elif mode == "phase":
        out = np.exp(-1j * (k_arr * float(delta_x) + omega * float(delta_t)))
    elif mode == "both":
        out = float(alpha) * np.exp(
            -1j * (k_arr * float(delta_x) + omega * float(delta_t))
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'amp', 'phase', or 'both'.")
    return out


# ---------------------------------------------------------------------------
# Bin-mask builder (paper-grounded constraints from issue #4)
# ---------------------------------------------------------------------------


def build_fit_mask(
    f: np.ndarray,
    k: np.ndarray,
    X13: float,
    *,
    f_peak_Hz: float | None = None,
    harmonic_halfwidth_Hz: float = 0.02,
    delta_l_over_L_lo: float = 0.05,
    delta_l_over_L_hi: float = 0.45,
) -> np.ndarray:
    """Boolean mask over bins, applying Goda-effective range and harmonic notches.

    Bin retained when ``delta_l_over_L_lo < X13 / L < delta_l_over_L_hi`` (Goda
    Eq 7 effective range, with $L = 2\\pi / k$). When ``f_peak_Hz`` is given,
    bins within ``±harmonic_halfwidth_Hz`` of $2 f_p$ and $3 f_p$ are also
    excluded — nonlinear harmonics travel at the fundamental celerity, so
    linear $k(\\omega)$ is wrong there and would corrupt the phase fit.
    """
    f_arr = np.asarray(f, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    delta_l_over_L = k_arr * float(X13) / (2.0 * np.pi)
    mask = (delta_l_over_L > float(delta_l_over_L_lo)) & (
        delta_l_over_L < float(delta_l_over_L_hi)
    )
    if f_peak_Hz is not None:
        hw = float(harmonic_halfwidth_Hz)
        for n in (2, 3):
            mask &= np.abs(f_arr - n * float(f_peak_Hz)) > hw
    return mask


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def fit_probe_cn_parametric(
    f: np.ndarray,
    k: np.ndarray,
    B1: np.ndarray,
    Bn: np.ndarray,
    dx_nominal: float,
    fit_mask: np.ndarray,
) -> dict:
    """Fit $(\\alpha, \\Delta x, \\Delta t)$ for probe $n$ from noref bins.

    $\\alpha$ = arithmetic mean of $|C_n|$ over masked bins.
    $(\\Delta x, \\Delta t)$ = least-squares fit of
    $\\arg(C_n / \\alpha) = -(k \\Delta x + \\omega \\Delta t)$ across masked bins
    after phase unwrapping.
    """
    f_arr = np.asarray(f, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    mask = np.asarray(fit_mask, dtype=bool)
    if not np.any(mask):
        raise ValueError(
            "fit_probe_cn_parametric: no bins selected by fit_mask "
            "(check Goda-effective range vs the available frequencies)."
        )
    C = measured_C(B1, Bn, k_arr, dx_nominal)
    C_m = C[mask]
    f_m = f_arr[mask]
    k_m = k_arr[mask]
    omega_m = 2.0 * np.pi * f_m

    alpha = float(np.mean(np.abs(C_m)))
    if alpha <= 0:
        raise ValueError(
            "fit_probe_cn_parametric: mean |C_n| is non-positive "
            f"({alpha:.3e}); fit cannot proceed."
        )

    # Sort by frequency before unwrapping so the unwrap follows a coherent phase
    # progression rather than the arbitrary record order.
    order = np.argsort(f_m)
    f_s = f_m[order]
    k_s = k_m[order]
    omega_s = omega_m[order]
    C_s = C_m[order]
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
            "n_bins_used": int(mask.sum()),
        },
        "per_bin": {
            "f_Hz": [float(x) for x in f_s],
            "alpha": [float(x) for x in np.abs(C_s)],
            "beta_rad": [float(x) for x in beta],
        },
    }


def fit_cn_from_records(
    records: Iterable[dict],
    X12: float,
    X13: float,
) -> dict:
    """Aggregate-fit $C_n$ across one or more noref records.

    Each ``record`` is a dict with keys ``f`` (1-D float), ``k`` (1-D float),
    ``B1``, ``B2``, ``B3`` (1-D complex) all the same length, and optional
    ``f_peak_Hz`` for the harmonic notch.

    Returns ``{wp1: {...identity...}, wp2: {...fit...}, wp3: {...fit...}}``
    suitable for ``save_cn_config``. Probe 1 is the reference and gets identity
    values. wp2 is fitted against $X_{12}$ as the nominal spacing; wp3 against
    $X_{13}$.
    """
    rec_list = list(records)
    if not rec_list:
        raise ValueError("fit_cn_from_records: at least one record required.")

    # Build the aggregated arrays per probe pair, applying each record's mask.
    f_acc: list[np.ndarray] = []
    k_acc: list[np.ndarray] = []
    B1_acc: list[np.ndarray] = []
    B2_acc: list[np.ndarray] = []
    B3_acc: list[np.ndarray] = []
    for rec in rec_list:
        f_r = np.asarray(rec["f"], dtype=float)
        k_r = np.asarray(rec["k"], dtype=float)
        mask_r = build_fit_mask(
            f_r, k_r, X13, f_peak_Hz=rec.get("f_peak_Hz"),
        )
        if not np.any(mask_r):
            continue
        f_acc.append(f_r[mask_r])
        k_acc.append(k_r[mask_r])
        B1_acc.append(np.asarray(rec["B1"])[mask_r])
        B2_acc.append(np.asarray(rec["B2"])[mask_r])
        B3_acc.append(np.asarray(rec["B3"])[mask_r])

    if not f_acc:
        raise ValueError(
            "fit_cn_from_records: no bins survived the per-record fit masks "
            "(every record's frequencies fall outside the Goda-effective range)."
        )

    f_all = np.concatenate(f_acc)
    k_all = np.concatenate(k_acc)
    B1_all = np.concatenate(B1_acc)
    B2_all = np.concatenate(B2_acc)
    B3_all = np.concatenate(B3_acc)
    full_mask = np.ones_like(f_all, dtype=bool)

    fit2 = fit_probe_cn_parametric(f_all, k_all, B1_all, B2_all, X12, full_mask)
    fit3 = fit_probe_cn_parametric(f_all, k_all, B1_all, B3_all, X13, full_mask)

    return {
        "wp1": {
            "alpha": 1.0,
            "delta_x_m": 0.0,
            "delta_t_s": 0.0,
            "_note": "reference probe; identity by definition",
        },
        "wp2": fit2,
        "wp3": fit3,
    }


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def apply_cn_to_bins(
    B1: np.ndarray,
    B2: np.ndarray,
    B3: np.ndarray,
    f: np.ndarray,
    k: np.ndarray,
    cn_config: dict,
    mode: CnMode = "both",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide $B_2$, $B_3$ by $C_n(f)$ in place-safe fashion. $B_1$ unchanged.

    Works on scalar, 0-D, and 1-D inputs uniformly via NumPy broadcasting; the
    regular-wave path passes single-element arrays, the irregular path passes
    per-bin arrays.
    """
    p2 = cn_config.get("wp2", {})
    p3 = cn_config.get("wp3", {})
    C2 = evaluate_C(f, k, p2["alpha"], p2["delta_x_m"], p2["delta_t_s"], mode=mode)
    C3 = evaluate_C(f, k, p3["alpha"], p3["delta_x_m"], p3["delta_t_s"], mode=mode)
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
