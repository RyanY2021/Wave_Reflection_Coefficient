"""Synthetic-wave tests for the reflection-coefficient pipeline.

Builds a perfectly known incident + reflected field at three probe locations
and verifies both separation methods recover the correct amplitudes, phases,
and reflection coefficient.
"""

from __future__ import annotations

import numpy as np

from reflection_coefficient.analysis import (
    group_velocity,
    solve_dispersion,
    solve_dispersion_array,
)
from reflection_coefficient.methods.goda import goda_separation
from reflection_coefficient.methods.least_squares import mansard_funke_separation


def _three_probe_elevations(f, a_I, a_R, phi_R, X12, X13, depth, fs, duration):
    """Return (t, e1, e2, e3) for a known incident + reflected monochromatic wave.

    Probe 1 is at x=0; incident propagates in +x, reflected in -x.
    """
    k, _ = solve_dispersion(f, depth)
    omega = 2.0 * np.pi * f
    t = np.arange(0.0, duration, 1.0 / fs)

    def eta_at(x):
        return (
            a_I * np.cos(k * x - omega * t)
            + a_R * np.cos(-k * x - omega * t + phi_R)
        )

    return t, eta_at(0.0), eta_at(X12), eta_at(X13)


def test_dispersion_matches_deep_water_limit():
    # Deep water: k → ω²/g, independent of depth.
    f = 2.0
    k, L = solve_dispersion(f, depth=1e4)
    omega = 2.0 * np.pi * f
    assert abs(k - omega * omega / 9.81) / k < 1e-6
    assert abs(L - 2 * np.pi / k) < 1e-9


def test_dispersion_array_matches_scalar():
    freqs = np.array([0.3, 0.7, 1.2, 2.0])
    depth = 2.0
    k_arr = solve_dispersion_array(freqs, depth)
    for f, k_expected in zip(freqs, k_arr):
        k_scalar, _ = solve_dispersion(f, depth)
        assert abs(k_scalar - k_expected) / k_scalar < 1e-8


def test_group_velocity_positive():
    assert group_velocity(0.8, depth=2.0) > 0


def test_goda_separation_recovers_known_reflection():
    f, depth, fs, duration = 0.8, 2.0, 100.0, 40.0
    a_I, a_R, phi_R = 0.05, 0.02, 0.6
    X12, X13 = 0.35, 0.90

    t, e1, _e2, e3 = _three_probe_elevations(
        f, a_I, a_R, phi_R, X12, X13, depth, fs, duration
    )

    N = t.size
    n_pos = N // 2 + 1
    B1 = np.fft.fft(e1)[:n_pos]
    B3 = np.fft.fft(e3)[:n_pos]
    df = fs / N
    k_bin = int(round(f / df))
    k_val, _ = solve_dispersion(f, depth)

    Z_I, Z_R, valid = goda_separation(
        np.array([B1[k_bin]]), np.array([B3[k_bin]]),
        np.array([k_val]), X13,
    )
    assert valid[0]
    a_I_est = abs(Z_I[0]) * 2.0 / N
    a_R_est = abs(Z_R[0]) * 2.0 / N
    assert abs(a_I_est - a_I) / a_I < 0.02
    assert abs(a_R_est - a_R) / a_R < 0.02
    assert abs((a_R_est / a_I_est) - (a_R / a_I)) < 0.01


def test_mansard_funke_recovers_known_reflection():
    f, depth, fs, duration = 0.8, 2.0, 100.0, 40.0
    a_I, a_R, phi_R = 0.05, 0.02, 0.6
    X12, X13 = 0.35, 0.90

    t, e1, e2, e3 = _three_probe_elevations(
        f, a_I, a_R, phi_R, X12, X13, depth, fs, duration
    )

    N = t.size
    n_pos = N // 2 + 1
    B1 = np.fft.fft(e1)[:n_pos]
    B2 = np.fft.fft(e2)[:n_pos]
    B3 = np.fft.fft(e3)[:n_pos]
    df = fs / N
    k_bin = int(round(f / df))
    k_val, _ = solve_dispersion(f, depth)

    Z_I, Z_R, valid = mansard_funke_separation(
        np.array([B1[k_bin]]), np.array([B2[k_bin]]), np.array([B3[k_bin]]),
        np.array([k_val]), X12, X13,
    )
    assert valid[0]
    a_I_est = abs(Z_I[0]) * 2.0 / N
    a_R_est = abs(Z_R[0]) * 2.0 / N
    assert abs(a_I_est - a_I) / a_I < 0.02
    assert abs(a_R_est - a_R) / a_R < 0.02


def test_singularity_masks_flag_bad_bins():
    # Spacing where k*X13 ≈ π → sin(kΔ) ≈ 0, two-probe method singular.
    f, depth = 0.8, 2.0
    k_val, _ = solve_dispersion(f, depth)
    X13 = np.pi / k_val  # force the singularity

    _, _, valid = goda_separation(
        np.array([1 + 0j]), np.array([1 + 0j]),
        np.array([k_val]), X13,
    )
    assert not valid[0]
