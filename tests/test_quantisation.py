"""
Unit tests for accumulator quantisation and the inlier criterion.

These are the places in this code base that are most easily "helpfully improved"
into a silent error: the quantisation truncates toward zero instead of flooring,
so one character in the wrong place shifts a whole bucket by one step.
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import INLIER_TOL, RHO_STEP
from wave_hough_detect.rht import (
    SIM_PRESET,
    _step_key,
    find_points_on_plane,
    get_key,
    is_point_on_plane,
    spherical_key,
)


# --- _step_key: truncation toward zero ---
@pytest.mark.parametrize("x,step,expected", [
    (9.94, 0.05, 9.90),          # positive value: trunc and floor agree
    (0.0, 0.05, 0.0),
    (-4.81374, 0.05, -4.80),     # ★ negative: trunc gives -96 steps, floor would give -97
    (-0.049, 0.05, 0.0),         # small negative value -> 0, floor would give -0.05
    (83.9, 2.0, 82.0),
    (-1.0, 2.0, 0.0),
])
def test_step_key_truncates_toward_zero(x, step, expected):
    assert _step_key(x, step) == pytest.approx(expected, abs=1e-12)


def test_step_key_negative_rho_is_not_floored():
    """What writing trunc as floor would cost: every negative-rho bucket is off by one step."""
    val = -4.81374
    assert _step_key(val, 0.05) == pytest.approx(-4.80)
    assert np.floor(val / 0.05) * 0.05 == pytest.approx(-4.85)
    assert _step_key(val, 0.05) != pytest.approx(np.floor(val / 0.05) * 0.05)


# --- get_key: the string quantisation key ---
def test_get_key_format():
    key = get_key(9.94, np.radians(84.4), np.radians(14.4))
    assert key == "9.9,84,14"


def test_get_key_rho_step_is_adjustable():
    """
    The real-data pipeline uses a rho step of 0.05 and the simulation study uses
    0.5: the two must be settable separately, otherwise the simulation would run
    against an accumulator that is wrong by a factor of 10.
    """
    rho = 12.34
    assert get_key(rho, 0.0, 0.0, rho_step=0.05).split(",")[0] == "12.3"
    assert get_key(rho, 0.0, 0.0, rho_step=0.5).split(",")[0] == "12"
    assert SIM_PRESET["rho_step"] == 0.5
    assert RHO_STEP == 0.05


def test_get_key_phi_theta_are_degrees():
    # phi = 90 deg -> second field is 90; theta = 0 deg -> third field is 0
    key = get_key(1.0, np.pi / 2, 0.0)
    _, phi_s, theta_s = key.split(",")
    assert float(phi_s) == 90.0
    assert float(theta_s) == 0.0


def test_get_key_truncation_amplifies_float_roundoff():
    """
    ★ A genuine edge effect that has to be recorded faithfully:

        np.radians(-30) -> -0.5235987755982988
        np.degrees(...) -> -29.999999999999996
        -29.999999999999996 / 2 = -14.999999999999998
        trunc -> -14        (not -15)
        quantised result = -28     (not -30)

    Truncation toward zero gives -14 here as well, so this is not a bug but the
    fact that quantisation is sensitive to floating-point error. Using floor or
    round instead would change the grouping of the buckets.
    """
    key = get_key(1.0, np.pi / 2, float(np.radians(-30.0)))
    assert key.split(",")[2] == "-28"


def test_get_key_never_produces_negative_zero():
    """
    A second floating-point detail: the quantised value is an integer, so -0.0
    must become 0 and the key must read "0", whereas formatting Python's -0.0
    with :g yields "-0". The two are equivalent as bucket labels but differ as
    key strings, so the key has to be normalised to "0".
    """
    key = get_key(-0.0, -0.0, -0.0)
    assert "-0" not in key
    assert key == "0,0,0"


# --- spherical_key: recovering a key from a normal vector ---
def test_spherical_key_matches_get_key():
    """spherical_key is only a wrapper around get_key; the two must agree."""
    normal = np.array([0.0033, 0.0117, -0.99993])
    rho = -19.66
    phi = float(np.arccos(normal[2]))
    theta = float(np.arcsin(normal[1] / np.sin(phi)))
    assert spherical_key(normal, rho) == get_key(abs(rho), phi, theta)


def test_spherical_key_handles_phi_zero():
    """
    When the normal vector is exactly parallel to the t axis, phi = 0 and theta
    is undefined (a NaN). A deterministic key must be returned here rather than
    raising an exception or producing a NaN string.
    """
    key = spherical_key(np.array([0.0, 0.0, 1.0]), 5.0)
    assert "nan" not in key.lower()
    assert key == spherical_key(np.array([0.0, 0.0, 1.0]), 5.0)


# --- is_point_on_plane / find_points_on_plane ---
def test_inlier_tolerance_default_is_real_data_value():
    assert INLIER_TOL == 0.1
    n = np.array([0.0, 0.0, 1.0])
    assert is_point_on_plane(np.array([3.0, 4.0, 1.0 + 0.05]), n, 1.0)
    assert not is_point_on_plane(np.array([3.0, 4.0, 1.0 + 0.2]), n, 1.0)


def test_inlier_tolerance_is_adjustable():
    n = np.array([0.0, 0.0, 1.0])
    p = np.array([3.0, 4.0, 1.5])
    assert not is_point_on_plane(p, n, 1.0)
    assert is_point_on_plane(p, n, 1.0, tol=1.0)


def test_find_points_on_plane_preserves_index_order():
    """Points are collected in the order of the given index array; the returned order must match."""
    pts = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    idx = np.array([7, 2, 5, 0])
    got = find_points_on_plane(pts, np.array([0.0, 0.0, 1.0]), 0.0, idx)
    assert got.tolist() == idx.tolist()


def test_find_points_on_plane_empty_when_no_hits():
    pts = np.column_stack([np.arange(5.0), np.zeros(5), np.full(5, 9.0)])
    got = find_points_on_plane(pts, np.array([0.0, 0.0, 1.0]), 0.0, np.arange(5))
    assert got.size == 0
