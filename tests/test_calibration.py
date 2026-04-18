"""Tests for the linear probe re-calibration side transform."""

from __future__ import annotations

import numpy as np
import pytest

from reflection_coefficient.calibration import (
    apply_calibration_transfer,
    recalibrate_probes,
)


def test_transfer_is_identity_when_old_equals_new():
    eta = np.array([-1.2, 0.0, 3.4, 7.89])
    out = apply_calibration_transfer(eta, 2.5, 0.7, 2.5, 0.7)
    assert np.allclose(out, eta)


def test_transfer_matches_closed_form():
    raw = np.array([-0.5, 0.0, 1.0, 2.7])
    s_old, o_old = 2.0, 0.3
    s_new, o_new = 3.5, -0.1

    eta_old = s_old * raw + o_old
    eta_new_expected = s_new * raw + o_new

    eta_new = apply_calibration_transfer(eta_old, s_old, o_old, s_new, o_new)
    assert np.allclose(eta_new, eta_new_expected)


def test_transfer_rejects_zero_scale_old():
    with pytest.raises(ValueError):
        apply_calibration_transfer(np.array([1.0]), 0.0, 0.0, 1.0, 0.0)


def test_recalibrate_probes_applies_per_probe_parameters():
    e1 = np.array([1.0, 2.0])
    e2 = np.array([3.0, 4.0])
    e3 = np.array([5.0, 6.0])
    cfg = {
        "_doc": "ignored",
        "wp1": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 2.0, "offset_new": 0.1},
        "wp2": {"scale_old": 2.0, "offset_old": 1.0, "scale_new": 2.0, "offset_new": 1.0},
        "wp3": {"scale_old": 1.0, "offset_old": 0.5, "scale_new": 0.5, "offset_new": 0.0},
    }

    n1, n2, n3 = recalibrate_probes(e1, e2, e3, cfg)

    assert np.allclose(n1, 2.0 * e1 + 0.1)
    assert np.allclose(n2, e2)
    assert np.allclose(n3, 0.5 * (e3 - 0.5))


def test_recalibrate_probes_missing_field_errors():
    e = np.array([0.0])
    cfg = {
        "wp1": {"scale_old": 1.0, "offset_old": 0.0},
        "wp2": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
        "wp3": {"scale_old": 1.0, "offset_old": 0.0, "scale_new": 1.0, "offset_new": 0.0},
    }
    with pytest.raises(KeyError):
        recalibrate_probes(e, e, e, cfg)
