"""End-to-end reflection-coefficient pipeline.

Glues the I/O, preprocessing, spectral analysis, and separation-method modules
together following ``docs/reflection_processing_pipeline.md``.

Two public entry points:

* :func:`analyse_regular` — regular-wave path (single FFT bin, §6A)
* :func:`analyse_irregular` — irregular-wave path (band-averaged spectra, §6B)

Both accept time-series + a :class:`~reflection_coefficient.io.TestMeta` and
return a dataclass summarising the reflection coefficient, wave heights, and
quality diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .analysis import (
    group_velocity,
    positive_fft,
    solve_dispersion,
    solve_dispersion_array,
)
from .io import TestMeta
from .methods.goda import goda_separation
from .methods.least_squares import mansard_funke_separation
from .preprocessing import clip_window, hanning_window, remove_mean

MethodName = Literal["goda", "least_squares"]


# ---------------------------------------------------------------------------
# Geometry / clipping window
# ---------------------------------------------------------------------------


def _require_geometry(meta: TestMeta) -> tuple[float, float, float, float]:
    """Return ``(x_paddle, x_struct, X12, X13)`` or raise if any is missing."""
    x_paddle = meta.x_paddle_to_wp1_m
    x_struct = meta.x_wp3_to_struct_m
    X12 = meta.X12_m
    X13 = meta.X13_m
    missing = [
        name
        for name, v in [
            ("x_paddle_to_wp1_m", x_paddle),
            ("x_wp3_to_struct_m (derived)", x_struct),
            ("X12_m", X12),
            ("X13_m", X13),
        ]
        if v is None
    ]
    if missing:
        raise ValueError(
            "Probe geometry is incomplete in tank_config.json: missing "
            + ", ".join(missing)
        )
    return float(x_paddle), float(x_struct), float(X12), float(X13)


def clip_bounds(cg: float, x_paddle: float, x_struct: float, X13: float) -> tuple[float, float]:
    """Pipeline §2.3 — travel-time-based clipping window."""
    t_start = (x_paddle + 2.0 * x_struct + 2.0 * X13) / cg
    t_end = (3.0 * x_paddle + 2.0 * x_struct + 3.0 * X13) / cg
    return float(t_start), float(t_end)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class RegularResult:
    test_id: str
    method: MethodName
    f_Hz: float
    k: float
    wavelength_m: float
    a_I: float
    a_R: float
    H_I: float
    H_R: float
    Kr: float
    singularity_ok: bool


@dataclass
class IrregularResult:
    test_id: str
    method: MethodName
    f: np.ndarray                 # analysis-band frequencies (unsmoothed)
    S_I: np.ndarray               # incident spectral density [m²/Hz]
    S_R: np.ndarray               # reflected spectral density [m²/Hz]
    f_smooth: np.ndarray
    S_I_smooth: np.ndarray
    S_R_smooth: np.ndarray
    Kr_f: np.ndarray              # reflection coefficient per smoothed bin
    Kr_overall: float
    Hm0_I: float
    Hm0_R: float
    Tp_I: float
    band_mask: np.ndarray         # indices of bins used for the overall Kr
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regular-wave path (§6A)
# ---------------------------------------------------------------------------


def analyse_regular(
    t: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    eta3: np.ndarray,
    meta: TestMeta,
    method: MethodName = "least_squares",
) -> RegularResult:
    if meta.f_Hz is None:
        raise ValueError(f"{meta.test_id}: regular-wave analysis requires meta.f_Hz")
    x_paddle, x_struct, X12, X13 = _require_geometry(meta)
    depth = meta.water_depth_m
    g = meta.gravity_m_s2

    cg = group_velocity(meta.f_Hz, depth, g=g)
    t_start, t_end = clip_bounds(cg, x_paddle, x_struct, X13)

    t_c, e1, e2, e3 = clip_window(t, eta1, eta2, eta3, t_start=t_start, t_end=t_end)
    if t_c.size < 8:
        raise ValueError(
            f"{meta.test_id}: clip window [{t_start:.2f}, {t_end:.2f}] s "
            f"contains only {t_c.size} samples"
        )
    e1, e2, e3 = remove_mean(e1), remove_mean(e2), remove_mean(e3)

    fs = 1.0 / np.mean(np.diff(t_c))
    N = e1.size
    freqs, B1 = positive_fft(e1, fs)
    _, B2 = positive_fft(e2, fs)
    _, B3 = positive_fft(e3, fs)

    df = fs / N
    k_bin = int(round(meta.f_Hz / df))
    if not (1 <= k_bin < freqs.size):
        raise ValueError(
            f"{meta.test_id}: target f={meta.f_Hz} Hz falls outside FFT range"
        )

    k_val, L = solve_dispersion(meta.f_Hz, depth, g=g)

    if method == "goda":
        k_arr = np.array([k_val])
        Z_I, Z_R, valid = goda_separation(
            np.array([B1[k_bin]]), np.array([B3[k_bin]]), k_arr, X13
        )
    elif method == "least_squares":
        k_arr = np.array([k_val])
        Z_I, Z_R, valid = mansard_funke_separation(
            np.array([B1[k_bin]]),
            np.array([B2[k_bin]]),
            np.array([B3[k_bin]]),
            k_arr,
            X12,
            X13,
        )
    else:
        raise ValueError(f"Unknown method {method!r}")

    a_I = float(abs(Z_I[0])) * 2.0 / N
    a_R = float(abs(Z_R[0])) * 2.0 / N
    Kr = a_R / a_I if a_I > 0 else float("nan")

    return RegularResult(
        test_id=meta.test_id,
        method=method,
        f_Hz=float(freqs[k_bin]),
        k=k_val,
        wavelength_m=L,
        a_I=a_I,
        a_R=a_R,
        H_I=2.0 * a_I,
        H_R=2.0 * a_R,
        Kr=Kr,
        singularity_ok=bool(valid[0]),
    )


# ---------------------------------------------------------------------------
# Irregular-wave path (§6B)
# ---------------------------------------------------------------------------


def _band_average(a: np.ndarray, n_bands: int) -> np.ndarray:
    n_trim = a.size - a.size % n_bands
    return np.nanmean(a[:n_trim].reshape(-1, n_bands), axis=1)


def analyse_irregular(
    t: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    eta3: np.ndarray,
    meta: TestMeta,
    method: MethodName = "least_squares",
    f_peak: float | None = None,
    n_bands: int = 5,
    apply_window: bool = True,
    band_lo_factor: float = 0.5,
    band_hi_factor: float = 2.5,
    window: str = "hann",
    bandwidth_Hz: float | None = None,
) -> IrregularResult:
    """Irregular-wave reflection analysis (white-noise or JONSWAP).

    ``f_peak`` is used to size the clipping window (via slowest-frequency
    group velocity) and to define the valid analysis band
    ``[band_lo_factor·f_peak, band_hi_factor·f_peak]``. If omitted, it is
    inferred from metadata: ``1/Tp_target_s`` for JONSWAP, or the midpoint of
    ``[f_min_Hz, f_max_Hz]`` for white noise.
    """
    x_paddle, x_struct, X12, X13 = _require_geometry(meta)
    depth = meta.water_depth_m
    g = meta.gravity_m_s2

    if f_peak is None:
        if meta.Tp_target_s:
            f_peak = 1.0 / meta.Tp_target_s
        elif meta.f_min_Hz and meta.f_max_Hz:
            f_peak = 0.5 * (meta.f_min_Hz + meta.f_max_Hz)
        else:
            raise ValueError(
                f"{meta.test_id}: cannot infer f_peak; pass f_peak explicitly"
            )

    # Conservative window: slowest cg at f_min for t_start, fastest at f_max
    # for t_end. Fall back to cg at f_peak if band edges are unavailable.
    f_lo = meta.f_min_Hz or f_peak
    f_hi = meta.f_max_Hz or f_peak
    cg_slow = group_velocity(f_lo, depth, g=g)
    cg_fast = group_velocity(f_hi, depth, g=g)
    t_start, _ = clip_bounds(cg_slow, x_paddle, x_struct, X13)
    _, t_end = clip_bounds(cg_fast, x_paddle, x_struct, X13)

    t_c, e1, e2, e3 = clip_window(t, eta1, eta2, eta3, t_start=t_start, t_end=t_end)
    if t_c.size < 64:
        raise ValueError(
            f"{meta.test_id}: clip window too short ({t_c.size} samples)"
        )
    e1, e2, e3 = remove_mean(e1), remove_mean(e2), remove_mean(e3)

    fs = 1.0 / np.mean(np.diff(t_c))
    N_raw = e1.size
    df_raw = fs / N_raw

    win_corr = 1.0
    if window == "hann" and apply_window:
        w = hanning_window(N_raw)
        win_corr = float(np.mean(w * w))
        e1, e2, e3 = e1 * w, e2 * w, e3 * w
        # Hann noise-equivalent bandwidth (single full-record window):
        enbw_raw = 1.5 * df_raw
    elif window == "none":
        enbw_raw = df_raw
    else:
        raise ValueError(f"Unknown window {window!r}; expected 'hann' or 'none'")

    # If an explicit resolution bandwidth was requested, choose n_bands so the
    # smoothed ENBW ≈ bandwidth_Hz. Otherwise keep the caller-supplied default.
    if bandwidth_Hz is not None:
        n_bands = max(1, int(round(bandwidth_Hz / enbw_raw)))
    bandwidth_effective = n_bands * enbw_raw

    N = e1.size
    df = fs / N
    freqs, B1 = positive_fft(e1, fs)
    _, B2 = positive_fft(e2, fs)
    _, B3 = positive_fft(e3, fs)

    # Skip DC bin; dispersion solver is undefined at f=0.
    f_pos = freqs[1:]
    B1, B2, B3 = B1[1:], B2[1:], B3[1:]
    k_arr = solve_dispersion_array(f_pos, depth, g=g)

    if method == "goda":
        Z_I, Z_R, valid = goda_separation(B1, B3, k_arr, X13)
        D_or_sin2 = np.sin(k_arr * X13) ** 2
    elif method == "least_squares":
        Z_I, Z_R, valid = mansard_funke_separation(B1, B2, B3, k_arr, X12, X13)
        sb = np.sin(k_arr * X12)
        sg = np.sin(k_arr * X13)
        sgb = np.sin(k_arr * X13 - k_arr * X12)
        D_or_sin2 = 2.0 * (sb * sb + sg * sg + sgb * sgb)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # Single-sided spectral density, pipeline §6B.4
    scale = 2.0 / (N * N * df) / win_corr
    S_I = (np.abs(Z_I) ** 2) * scale
    S_R = (np.abs(Z_R) ** 2) * scale

    # Band averaging
    S_I_s = _band_average(S_I, n_bands)
    S_R_s = _band_average(S_R, n_bands)
    f_s = _band_average(f_pos, n_bands)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kr_f = np.sqrt(S_R_s / S_I_s)

    # Analysis band for the overall / integrated quantities.
    # Metadata [f_min_Hz, f_max_Hz] (WN flat band, JS cutoffs) takes precedence
    # over the peak-relative default — white-noise spectra have no energy
    # outside that interval, so integrating across it pollutes m0 with noise.
    if meta.f_min_Hz is not None and meta.f_max_Hz is not None:
        f_min, f_max = float(meta.f_min_Hz), float(meta.f_max_Hz)
    else:
        f_min = band_lo_factor * f_peak
        f_max = band_hi_factor * f_peak
    band = (f_pos >= f_min) & (f_pos <= f_max) & valid
    m0_I = float(np.nansum(S_I[band]) * df)
    m0_R = float(np.nansum(S_R[band]) * df)
    Kr_overall = float(np.sqrt(m0_R / m0_I)) if m0_I > 0 else float("nan")
    Hm0_I = 4.0 * np.sqrt(m0_I) if m0_I > 0 else 0.0
    Hm0_R = 4.0 * np.sqrt(m0_R) if m0_R > 0 else 0.0

    # Peak period from incident smoothed spectrum within the band
    band_s = (f_s >= f_min) & (f_s <= f_max)
    if np.any(band_s) and np.any(~np.isnan(S_I_s[band_s])):
        idx = int(np.nanargmax(np.where(band_s, S_I_s, -np.inf)))
        Tp_I = float(1.0 / f_s[idx]) if f_s[idx] > 0 else float("nan")
    else:
        Tp_I = float("nan")

    diagnostics = {
        "t_start_s": t_start,
        "t_end_s": t_end,
        "fs_Hz": float(fs),
        "N": int(N),
        "df_Hz": float(df),
        "window_correction": win_corr,
        "D_or_sin2_min": float(np.nanmin(D_or_sin2[band])) if np.any(band) else float("nan"),
        "n_bins_valid": int(np.count_nonzero(band)),
        "f_peak_used_Hz": float(f_peak),
        "window": window,
        "n_bands": int(n_bands),
        "bandwidth_Hz": float(bandwidth_effective),
    }

    return IrregularResult(
        test_id=meta.test_id,
        method=method,
        f=f_pos,
        S_I=S_I,
        S_R=S_R,
        f_smooth=f_s,
        S_I_smooth=S_I_s,
        S_R_smooth=S_R_s,
        Kr_f=Kr_f,
        Kr_overall=Kr_overall,
        Hm0_I=Hm0_I,
        Hm0_R=Hm0_R,
        Tp_I=Tp_I,
        band_mask=band,
        diagnostics=diagnostics,
    )


def analyse(
    t: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
    eta3: np.ndarray,
    meta: TestMeta,
    method: MethodName = "least_squares",
    window: str = "hann",
    bandwidth_Hz: float | None = None,
) -> RegularResult | IrregularResult:
    """Dispatch to the regular or irregular path based on ``meta.campaign``."""
    if meta.campaign == "rw":
        return analyse_regular(t, eta1, eta2, eta3, meta, method=method)
    return analyse_irregular(
        t, eta1, eta2, eta3, meta, method=method,
        window=window, bandwidth_Hz=bandwidth_Hz,
    )
