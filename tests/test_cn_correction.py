"""Unit tests for per-probe complex correction $C_n(f)$.

Synthetic-only (no project data). Builds noref or canonical fields with known
$(\\alpha_n, \\Delta x_n, \\Delta t_n)$ injected on probes 2 and 3 and verifies
the fit recovers them and ``apply_cn_to_bins`` undoes them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from reflection_coefficient.analysis import (
    solve_dispersion,
    solve_dispersion_array,
)
from reflection_coefficient.cn_correction import (
    apply_cn_to_bins,
    build_fit_mask,
    evaluate_C,
    fit_cn_from_records,
    fit_probe_cn_parametric,
    identity_cn_config,
    load_cn_config,
    measured_C,
    save_cn_config,
)
from reflection_coefficient.methods.goda import goda_separation
from reflection_coefficient.methods.least_squares import mansard_funke_separation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eta_with_error(
    f: float, a: float, x_nominal: float, dx: float, dt: float, alpha: float,
    t: np.ndarray, depth: float = 2.0,
) -> np.ndarray:
    """Recorded time series at a probe with injected (alpha, dx, dt) error.

    Truth field is a single incident wave $a \\cos(k x - \\omega t)$.
    Probe sits at $x_{\\text{nominal}} + \\Delta x$ and reports samples
    $\\Delta t$ later (i.e. recorded value at recording-time $t$ equals the
    physical wave value at $t - \\Delta t$), with a multiplicative gain
    $\\alpha$ on top.
    """
    k, _ = solve_dispersion(f, depth)
    omega = 2.0 * np.pi * f
    return alpha * np.cos(k * (x_nominal + dx) - omega * (t - dt))


def _record_one_freq(
    f: float, a: float, X12: float, X13: float,
    err2: tuple[float, float, float],
    err3: tuple[float, float, float],
    fs: float = 100.0, duration: float = 40.0, depth: float = 2.0,
) -> dict:
    """Synthesise one noref RW record + extract its single bin."""
    t = np.arange(0.0, duration, 1.0 / fs)
    e1 = _eta_with_error(f, a, 0.0, 0.0, 0.0, 1.0, t, depth)
    a2, dx2, dt2 = err2
    a3, dx3, dt3 = err3
    e2 = _eta_with_error(f, a, X12, dx2, dt2, a2, t, depth)
    e3 = _eta_with_error(f, a, X13, dx3, dt3, a3, t, depth)

    N = t.size
    n_pos = N // 2 + 1
    df = fs / N
    k_bin = int(round(f / df))
    B1 = np.fft.fft(e1)[:n_pos]
    B2 = np.fft.fft(e2)[:n_pos]
    B3 = np.fft.fft(e3)[:n_pos]
    f_used = float(k_bin * df)
    k_val, _ = solve_dispersion(f_used, depth)
    return {
        "f": np.array([f_used]),
        "k": np.array([k_val]),
        "B1": np.array([B1[k_bin]]),
        "B2": np.array([B2[k_bin]]),
        "B3": np.array([B3[k_bin]]),
        "f_peak_Hz": f,
    }


# ---------------------------------------------------------------------------
# 1. evaluate_C modes
# ---------------------------------------------------------------------------


def test_evaluate_C_modes():
    f = np.array([0.5, 1.0])
    k = np.array([1.0, 2.0])
    alpha, dx, dt = 1.1, 0.005, 1e-4

    amp_only = evaluate_C(f, k, alpha, dx, dt, mode="amp")
    assert np.allclose(amp_only, alpha + 0j)

    phase_only = evaluate_C(f, k, alpha, dx, dt, mode="phase")
    assert np.allclose(np.abs(phase_only), 1.0)
    omega = 2.0 * np.pi * f
    expected_phase = -k * dx - omega * dt
    assert np.allclose(np.angle(phase_only), expected_phase)

    both = evaluate_C(f, k, alpha, dx, dt, mode="both")
    assert np.allclose(both, alpha * phase_only)


# ---------------------------------------------------------------------------
# 2. RW round-trip: 8 RW noref tests, aggregate-fit, recover
# ---------------------------------------------------------------------------


def test_round_trip_synthetic_rw():
    X12, X13, depth = 0.5, 1.07, 2.0
    err2 = (0.987, 0.0042, 1.3e-5)   # alpha, dx (m), dt (s)
    err3 = (1.012, -0.0089, 0.0)
    # Frequencies snapped to the FFT grid (df = fs/N = 100/4000 = 0.025 Hz) so
    # there is no spectral leakage to corrupt the per-probe phase. All within
    # the Goda-effective range for d=2 m, X13=1.07 m (X13/L spans ~0.07-0.30).
    freqs = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]

    records = [
        _record_one_freq(float(f), 0.05, X12, X13, err2, err3, depth=depth)
        for f in freqs
    ]
    cfg = fit_cn_from_records(records, X12=X12, X13=X13)

    fit2 = cfg["wp2"]
    fit3 = cfg["wp3"]

    assert abs(fit2["alpha"] - err2[0]) < 0.005
    assert abs(fit2["delta_x_m"] - err2[1]) < 5e-4
    assert abs(fit2["delta_t_s"] - err2[2]) < 5e-5
    assert abs(fit3["alpha"] - err3[0]) < 0.005
    assert abs(fit3["delta_x_m"] - err3[1]) < 5e-4
    assert abs(fit3["delta_t_s"] - err3[2]) < 5e-5


# ---------------------------------------------------------------------------
# 3. Irregular round-trip: one broadband noref record
# ---------------------------------------------------------------------------


def test_round_trip_irregular():
    X12, X13, depth = 0.5, 1.07, 2.0
    err2 = (0.987, 0.0042, 1.3e-5)
    err3 = (1.012, -0.0089, 0.0)
    fs, duration = 100.0, 240.0
    rng = np.random.default_rng(2026)

    # Sum of 30 cosines snapped to the FFT grid (df = fs/N) so there is no
    # spectral leakage. Without snapping, every off-bin cosine bleeds phase
    # into its neighbours and the per-bin C_n is dominated by leakage.
    t = np.arange(0.0, duration, 1.0 / fs)
    N = t.size
    df = fs / N
    n_freqs = 30
    bin_lo = int(np.ceil(0.30 / df))
    bin_hi = int(np.floor(0.85 / df))
    bins = rng.choice(np.arange(bin_lo, bin_hi + 1), size=n_freqs, replace=False)
    f_band = bins * df
    amps = rng.uniform(0.005, 0.02, n_freqs)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_freqs)

    def _eta(x_nominal, dx, dt, alpha):
        eta = np.zeros_like(t)
        for fi, ai, phi in zip(f_band, amps, phases):
            ki, _ = solve_dispersion(fi, depth)
            omega = 2.0 * np.pi * fi
            eta += ai * np.cos(ki * (x_nominal + dx) - omega * (t - dt) + phi)
        return alpha * eta

    e1 = _eta(0.0, 0.0, 0.0, 1.0)
    e2 = _eta(X12, err2[1], err2[2], err2[0])
    e3 = _eta(X13, err3[1], err3[2], err3[0])

    n_pos = N // 2 + 1
    B1 = np.fft.fft(e1)[:n_pos]
    B2 = np.fft.fft(e2)[:n_pos]
    B3 = np.fft.fft(e3)[:n_pos]
    # Only the bins where we placed energy carry meaningful phase; empty bins
    # would dominate the fit with numerical noise. Real noref data is broadband
    # so this edge case is synthetic-only.
    sorted_bins = np.sort(bins)
    f_used = sorted_bins * df
    B1_used = B1[sorted_bins]
    B2_used = B2[sorted_bins]
    B3_used = B3[sorted_bins]
    k_arr = solve_dispersion_array(f_used, depth)

    cfg = fit_cn_from_records(
        [{"f": f_used, "k": k_arr,
          "B1": B1_used, "B2": B2_used, "B3": B3_used}],
        X12=X12, X13=X13,
    )
    fit2 = cfg["wp2"]
    fit3 = cfg["wp3"]

    # Looser than the RW test — irregular has bin-to-bin phase noise from
    # off-bin energy leakage even on perfectly clean synthetics.
    assert abs(fit2["alpha"] - err2[0]) < 0.01
    assert abs(fit2["delta_x_m"] - err2[1]) < 1e-3
    assert abs(fit2["delta_t_s"] - err2[2]) < 1e-4
    assert abs(fit3["alpha"] - err3[0]) < 0.01
    assert abs(fit3["delta_x_m"] - err3[1]) < 1e-3
    assert abs(fit3["delta_t_s"] - err3[2]) < 1e-4


# ---------------------------------------------------------------------------
# 4. Apply C_n^-1 then run separation: recovers known a_I, a_R
# ---------------------------------------------------------------------------


def test_apply_then_separate_recovers_known_field():
    f, depth, fs, duration = 0.6, 2.0, 100.0, 60.0
    a_I, a_R, phi_R = 0.05, 0.02, 0.6
    X12, X13 = 0.5, 1.07
    err2 = (0.987, 0.0042, 1.3e-5)
    err3 = (1.012, -0.0089, 0.0)

    k, _ = solve_dispersion(f, depth)
    omega = 2.0 * np.pi * f
    t = np.arange(0.0, duration, 1.0 / fs)

    def eta_truth(x):
        return (
            a_I * np.cos(k * x - omega * t)
            + a_R * np.cos(-k * x - omega * t + phi_R)
        )

    def eta_with_error(x_nom, dx, dt, alpha):
        x = x_nom + dx
        return alpha * (
            a_I * np.cos(k * x - omega * (t - dt))
            + a_R * np.cos(-k * x - omega * (t - dt) + phi_R)
        )

    e1 = eta_truth(0.0)
    e2 = eta_with_error(X12, err2[1], err2[2], err2[0])
    e3 = eta_with_error(X13, err3[1], err3[2], err3[0])

    N = t.size
    n_pos = N // 2 + 1
    df = fs / N
    k_bin = int(round(f / df))
    f_used = float(k_bin * df)
    k_val, _ = solve_dispersion(f_used, depth)
    B1 = np.fft.fft(e1)[:n_pos][k_bin]
    B2 = np.fft.fft(e2)[:n_pos][k_bin]
    B3 = np.fft.fft(e3)[:n_pos][k_bin]

    cn_config = {
        "wp1": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
        "wp2": {"alpha": err2[0], "delta_x_m": err2[1], "delta_t_s": err2[2]},
        "wp3": {"alpha": err3[0], "delta_x_m": err3[1], "delta_t_s": err3[2]},
    }
    f_arr = np.array([f_used])
    k_arr = np.array([k_val])
    B1c, B2c, B3c = apply_cn_to_bins(
        np.array([B1]), np.array([B2]), np.array([B3]),
        f_arr, k_arr, cn_config, mode="both",
    )

    # Goda 1-3
    Z_I, Z_R, valid = goda_separation(B1c, B3c, k_arr, X13)
    assert valid[0]
    a_I_est = abs(Z_I[0]) * 2.0 / N
    a_R_est = abs(Z_R[0]) * 2.0 / N
    assert abs(a_I_est - a_I) / a_I < 0.01
    assert abs(a_R_est - a_R) / a_R < 0.02

    # Mansard-Funke
    Z_I, Z_R, valid = mansard_funke_separation(
        B1c, B2c, B3c, k_arr, X12, X13,
    )
    assert valid[0]
    a_I_est = abs(Z_I[0]) * 2.0 / N
    a_R_est = abs(Z_R[0]) * 2.0 / N
    assert abs(a_I_est - a_I) / a_I < 0.01
    assert abs(a_R_est - a_R) / a_R < 0.02


# ---------------------------------------------------------------------------
# 5. Idempotence — fit, apply, refit → near-identity
# ---------------------------------------------------------------------------


def test_idempotence():
    X12, X13, depth = 0.5, 1.07, 2.0
    err2 = (0.987, 0.0042, 1.3e-5)
    err3 = (1.012, -0.0089, 0.0)
    freqs = np.linspace(0.30, 0.85, 8)

    records = [
        _record_one_freq(float(f), 0.05, X12, X13, err2, err3, depth=depth)
        for f in freqs
    ]
    first = fit_cn_from_records(records, X12=X12, X13=X13)

    # Apply the fitted correction to each record's bins, then refit.
    corrected = []
    for rec in records:
        B1c, B2c, B3c = apply_cn_to_bins(
            rec["B1"], rec["B2"], rec["B3"],
            rec["f"], rec["k"], first, mode="both",
        )
        corrected.append({
            "f": rec["f"], "k": rec["k"],
            "B1": B1c, "B2": B2c, "B3": B3c,
            "f_peak_Hz": rec.get("f_peak_Hz"),
        })

    second = fit_cn_from_records(corrected, X12=X12, X13=X13)
    for key in ("wp2", "wp3"):
        assert abs(second[key]["alpha"] - 1.0) < 0.005
        assert abs(second[key]["delta_x_m"]) < 2e-4
        assert abs(second[key]["delta_t_s"]) < 5e-5


# ---------------------------------------------------------------------------
# 6. Fit-mask: harmonic notches
# ---------------------------------------------------------------------------


def test_fit_mask_excludes_harmonics():
    f = np.linspace(0.1, 2.0, 191)
    k = solve_dispersion_array(f, depth=2.0)
    mask = build_fit_mask(
        f, k, X13=1.07,
        f_peak_Hz=0.5, harmonic_halfwidth_Hz=0.02,
        delta_l_over_L_lo=0.0, delta_l_over_L_hi=10.0,  # disable range filter
    )
    # Bins inside [0.98, 1.02] (2*f_p ± 0.02) should be masked out.
    in_2f = (f > 0.98) & (f < 1.02)
    assert not np.any(mask[in_2f])
    # Same for [1.48, 1.52] (3*f_p ± 0.02).
    in_3f = (f > 1.48) & (f < 1.52)
    assert not np.any(mask[in_3f])
    # A band well clear of both should be retained.
    clear = (f > 0.6) & (f < 0.9)
    assert np.all(mask[clear])


# ---------------------------------------------------------------------------
# 7. Fit-mask: Goda-effective range
# ---------------------------------------------------------------------------


def test_fit_mask_goda_range():
    f = np.linspace(0.05, 3.0, 300)
    k = solve_dispersion_array(f, depth=2.0)
    X13 = 1.07
    mask = build_fit_mask(f, k, X13=X13)
    delta_l_over_L = k * X13 / (2.0 * np.pi)
    assert np.all(mask == ((delta_l_over_L > 0.05) & (delta_l_over_L < 0.45)))


# ---------------------------------------------------------------------------
# 8. apply_cn_to_bins with identity config is a no-op
# ---------------------------------------------------------------------------


def test_apply_with_identity_config_is_identity():
    cfg = identity_cn_config()
    B1 = np.array([1 + 0.5j, 2 - 1j])
    B2 = np.array([0.5 + 0.5j, -0.2 + 0.1j])
    B3 = np.array([1.1 - 0.3j, 0.9 + 0.7j])
    f = np.array([0.5, 1.0])
    k = solve_dispersion_array(f, depth=2.0)
    B1c, B2c, B3c = apply_cn_to_bins(B1, B2, B3, f, k, cfg, mode="both")
    assert np.allclose(B1c, B1)
    assert np.allclose(B2c, B2)
    assert np.allclose(B3c, B3)


# ---------------------------------------------------------------------------
# 9. load_cn_config validates convention block
# ---------------------------------------------------------------------------


def test_load_validates_convention(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "convention": {"fft_sign": "plus_iwt", "reference_probe": "wp1"},
        "wp1": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
        "wp2": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
        "wp3": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
    }))
    with pytest.raises(ValueError, match="fft_sign"):
        load_cn_config(bad)

    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps({
        "convention": {"fft_sign": "numpy_minus_iwt", "reference_probe": "wp2"},
        "wp1": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
        "wp2": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
        "wp3": {"alpha": 1.0, "delta_x_m": 0.0, "delta_t_s": 0.0},
    }))
    with pytest.raises(ValueError, match="reference_probe"):
        load_cn_config(bad2)


# ---------------------------------------------------------------------------
# 10. measured_C is the inverse of evaluate_C on a clean field
# ---------------------------------------------------------------------------


def test_measured_C_inverts_evaluate_C():
    """If we synthesise B_n^meas = C_n · B_n^pred, then measured_C must return C_n."""
    f = np.array([0.6])
    k = solve_dispersion_array(f, depth=2.0)
    X12 = 0.5
    alpha, dx, dt = 1.05, 0.003, 2e-5

    B1 = np.array([1.0 + 0.5j])
    B2 = (
        evaluate_C(f, k, alpha, dx, dt, mode="both")
        * B1
        * np.exp(-1j * k * X12)  # the predicted phase from B1 to nominal probe-2
    )
    C_recovered = measured_C(B1, B2, k, X12)
    assert np.allclose(C_recovered, evaluate_C(f, k, alpha, dx, dt, mode="both"))


# ---------------------------------------------------------------------------
# 11. save_cn_config round-trips through load_cn_config
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    cfg = identity_cn_config()
    cfg["wp2"]["alpha"] = 1.012
    cfg["wp2"]["delta_x_m"] = 0.005
    cfg["wp3"]["alpha"] = 0.987
    p = tmp_path / "probes_refined.json"
    save_cn_config(p, cfg, fit_meta={"fitted_from_test_ids": ["RW001"]})

    loaded = load_cn_config(p)
    assert loaded["convention"]["fft_sign"] == "numpy_minus_iwt"
    assert loaded["fit_meta"]["fitted_from_test_ids"] == ["RW001"]
    assert "fit_date_utc" in loaded["fit_meta"]
    assert loaded["wp2"]["alpha"] == pytest.approx(1.012)
    assert loaded["wp3"]["alpha"] == pytest.approx(0.987)


# ---------------------------------------------------------------------------
# 12. fit_probe_cn_parametric raises on empty mask
# ---------------------------------------------------------------------------


def test_fit_raises_on_empty_mask():
    f = np.array([0.5, 1.0])
    k = np.array([1.0, 2.0])
    B1 = np.array([1 + 0j, 1 + 0j])
    Bn = np.array([1 + 0j, 1 + 0j])
    with pytest.raises(ValueError, match="no bins"):
        fit_probe_cn_parametric(f, k, B1, Bn, dx_nominal=0.5,
                                fit_mask=np.array([False, False]))
