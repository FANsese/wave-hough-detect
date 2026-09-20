"""
Unit tests for stage 3 (wavefront model fitting).

The data are built **analytically**: (x0, y0, v, t0) are chosen first and the
exact arrival times are generated from them, and the fit then has to recover
them. Tests of this kind need no external data at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    GRID_SIDE,
    TIME_SCALE_BACK,
    build_result_array,
    circular_loss,
    fit_circular,
    fit_linear,
    linear_loss,
    plane_points,
    r_squared,
)


def circular_arrival(x0, y0, v, t0, side=GRID_SIDE):
    """Generate the (x, y, t) point set from the analytic model: t = sqrt((x-x0)^2 + (y-y0)^2)/v + t0."""
    xs, ys = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1),
                         indexing="ij")
    xs = xs.ravel().astype(float)
    ys = ys.ravel().astype(float)
    t = np.hypot(xs - x0, ys - y0) / v + t0
    return np.column_stack([xs, ys, t])


def linear_arrival(a, b, c, side=GRID_SIDE):
    """Generate the (x, y, t) point set from the analytic model: t = a_tilde*x + b_tilde*y + c_tilde."""
    xs, ys = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1),
                         indexing="ij")
    xs = xs.ravel().astype(float)
    ys = ys.ravel().astype(float)
    t = a * xs + b * ys + c
    return np.column_stack([xs, ys, t])


# --- build_result_array / plane_points ---
def test_build_result_array_scales_time_back_to_ms():
    """
    ★ The second half of the two-stage scale switch: what result.array stores must
      be milliseconds, i.e. the Hough-scale time multiplied back by 200. Missing
      this step leaves the fitted v wrong by a factor of 200.
    """
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 3.0])
    t_hough = np.array([4.0, 5.0])          # Hough scale (time / 200)
    pid = np.array([1, 1])
    arr = build_result_array(x, y, t_hough, pid, n_planes=1)
    assert arr.shape == (GRID_SIDE, GRID_SIDE, 1)
    assert arr[0, 0, 0] == pytest.approx(4.0 * TIME_SCALE_BACK)
    assert arr[1, 2, 0] == pytest.approx(5.0 * TIME_SCALE_BACK)
    assert TIME_SCALE_BACK == 200.0


def test_build_result_array_empty_slots_are_minus_one():
    arr = build_result_array(np.array([1.0]), np.array([1.0]),
                             np.array([2.0]), np.array([1]), n_planes=1)
    assert arr[0, 0, 0] == pytest.approx(400.0)
    assert np.all(arr[1:, :, :] == -1.0)     # empty slots are initialised to -1


def test_build_result_array_skips_unclassified_points():
    arr = build_result_array(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                             np.array([3.0, 4.0]), np.array([1, 0]), n_planes=1)
    assert arr[0, 0, 0] == pytest.approx(600.0)
    assert arr[1, 1, 0] == -1.0              # plane_indices == 0 -> skipped


def test_build_result_array_later_writes_win():
    """
    When the same (x, y) appears twice in the same plane, the later write wins
    over the earlier one - plain indexed assignment, neither accumulation nor a
    maximum.
    """
    arr = build_result_array(np.array([1.0, 1.0]), np.array([1.0, 1.0]),
                             np.array([3.0, 7.0]), np.array([1, 1]), n_planes=1)
    assert arr[0, 0, 0] == pytest.approx(7.0 * TIME_SCALE_BACK)


def test_plane_points_unfolds_column_major_and_drops_empty_slots():
    arr = build_result_array(np.array([1.0, 3.0]), np.array([2.0, 4.0]),
                             np.array([1.0, 2.0]), np.array([1, 1]), n_planes=1)
    p = plane_points(arr, 1)
    assert p.shape == (2, 3)
    # the array must be unfolded in column-major order, otherwise the point order changes
    assert p[:, 0].tolist() == [1.0, 3.0]
    assert p[:, 1].tolist() == [2.0, 4.0]
    assert p[:, 2].tolist() == [200.0, 400.0]


# --- loss functions ---
def test_circular_loss_is_zero_on_exact_model():
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    assert circular_loss([2.5, 3.5, 1 / 0.45, 100.0], pts) == pytest.approx(0.0)


def test_circular_loss_third_slot_is_slowness_not_speed():
    """
    ★ The slot trap: parameter slot 3 holds u = 1/v (the slowness) and the
      objective function multiplies by u. Equation (3) of the paper divides by v.
      The two are equivalent, but the code multiplies.
      Passing u where the speed belongs yields a finite but completely wrong loss
      value - it raises no error.
    """
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    assert circular_loss([2.5, 3.5, 1 / 0.45, 100.0], pts) == pytest.approx(0.0, abs=1e-9)
    assert circular_loss([2.5, 3.5, 0.45, 100.0], pts) > 1e3


def test_linear_loss_is_zero_on_exact_model():
    pts = linear_arrival(0.3, 2.1, 50.0)
    assert linear_loss([0.3, 2.1, 50.0], pts) == pytest.approx(0.0)


def test_r_squared_is_one_on_exact_model():
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    r2 = r_squared(circular_loss, [2.5, 3.5, 1 / 0.45, 100.0], pts)
    assert r2 == pytest.approx(1.0, abs=1e-12)


# --- circular wavefront fitting ---
#: The magnitude used in §3.2 of the paper: the source lies about 50 cells
#: outside the grid and t is of the order of several hundred milliseconds.
#: ★ The circular fit only works at this magnitude because the default initial
#:   guess is (u, t0) = (1, 1); at another magnitude it stalls, see
#:   test_fit_circular_needs_a_good_start_inside_grid.
PAPER_LIKE = dict(x0=-3.0, y0=-50.0, v=0.45, t0=800.0)


def test_fit_circular_recovers_known_parameters():
    """
    ★ The tolerance is not arbitrary, it reflects the stopping rule of the fit:

      Convergence requires the relative change ``||Theta - Theta_hat|| / ||Theta||
      < 1e-6`` (the default ``eps`` of the alternating minimisation), and Theta
      contains t0 ~ 800, so a change of 8e-4 in t0 is already enough to trigger
      the stop, at which point x0 is still off by about 0.008. Tightening eps to
      1e-8 converges to within 1e-4 (see
      test_fit_circular_precision_is_limited_by_the_stopping_rule).

      So "the fit differs from the truth by 0.008" is the behaviour of the
      stopping rule, not an error in the fit.
    """
    truth = PAPER_LIKE
    pts = circular_arrival(**truth)
    f = fit_circular(pts)
    assert f.n_points == GRID_SIDE * GRID_SIDE
    assert f.x0 == pytest.approx(truth["x0"], abs=0.05)
    assert f.y0 == pytest.approx(truth["y0"], abs=0.05)
    assert f.v == pytest.approx(truth["v"], rel=1e-4)
    assert f.t0 == pytest.approx(truth["t0"], abs=0.05)
    assert f.r2 == pytest.approx(1.0, abs=1e-6)


def test_fit_circular_precision_is_limited_by_the_stopping_rule():
    """Tightening the stopping rule raises the precision, which shows that the 0.008 above comes from the stopping rule and not from a wrong computation."""
    pts = circular_arrival(**PAPER_LIKE)
    loose = fit_circular(pts, eps=1e-6)     # the value used by the real-data pipeline
    tight = fit_circular(pts, eps=1e-10)
    assert abs(loose.x0 - PAPER_LIKE["x0"]) > abs(tight.x0 - PAPER_LIKE["x0"])
    assert tight.x0 == pytest.approx(PAPER_LIKE["x0"], abs=1e-4)
    assert tight.loss < loose.loss


def test_fit_circular_reports_speed_not_slowness():
    """
    ★ The slot trap of the optimiser: the slope (= 1/v) is stored back into the
      position that is named v. CircularFit.v must be the speed, so the slot must
      not be exposed directly.
    """
    pts = circular_arrival(**PAPER_LIKE)
    f = fit_circular(pts)
    assert f.v == pytest.approx(0.45, rel=1e-3)
    assert f.v != pytest.approx(1 / 0.45, rel=1e-2)


def test_fit_circular_handles_source_far_outside_grid():
    """The source lies outside the grid (y0 = -50, the situation of the real data in the paper); every wavefront must converge."""
    for t0 in (800.0, 2054.0, 7647.0):
        pts = circular_arrival(-3.0, -50.0, 0.45, t0)
        f = fit_circular(pts)
        assert f.y0 == pytest.approx(-50.0, abs=0.05)
        assert f.v == pytest.approx(0.45, rel=1e-2)


def test_fit_circular_needs_a_good_start_inside_grid():
    """
    ★★ The convergence fragility of the default start, recorded here:

      The alternating minimisation starts from (u, t0) = (1, 1), independently of
      the data. When the source lies *inside* the grid, the first grid search is
      pushed to the boundary of the search window and never climbs back.

      The default n_starts=1 (the value required to reproduce Tables 2 and 4 of
      the paper) fails on such a case; n_starts=2 additionally tries one
      data-driven start (with the centroid as a provisional centre) and then
      recovers the parameters exactly.

      This is a property of the optimiser, not an error in the implementation.
      The source of the paper lies 50 cells outside the grid, which is exactly
      the situation in which the default start works.
    """
    truth = dict(x0=2.5, y0=3.5, v=0.45, t0=100.0)
    pts = circular_arrival(**truth)

    # max_iter is limited to 200 only to keep the test fast: the default start
    # does not converge to the truth even with the full 10000 iterations (it runs
    # into the iteration cap), whereas the centroid start needs 14.
    default_start = fit_circular(pts, max_iter=200)          # n_starts=1
    assert not (abs(default_start.v - truth["v"]) < 0.01
                and abs(default_start.x0 - truth["x0"]) < 0.5), \
        "the default start should converge to a wrong local minimum at this magnitude"

    fixed = fit_circular(pts, n_starts=2, max_iter=200)
    assert fixed.x0 == pytest.approx(truth["x0"], abs=1e-2)
    assert fixed.y0 == pytest.approx(truth["y0"], abs=1e-2)
    assert fixed.v == pytest.approx(truth["v"], rel=1e-4)
    assert fixed.t0 == pytest.approx(truth["t0"], abs=1e-2)


def test_centroid_guess_does_not_break_a_case_that_already_converges():
    """
    The multi-start extension must not spoil a case in which the default start
    already converges.

    At the magnitude used here (the source slightly inside the grid, t0 = 200) the
    default start converges within 463 iterations, which makes it a good check
    that n_starts=2 does not replace a good result with a worse one.
    """
    pts = circular_arrival(4.5, 4.5, 0.45, 200.0)
    one = fit_circular(pts, n_starts=1)
    two = fit_circular(pts, n_starts=2)
    assert one.v == pytest.approx(0.45, rel=1e-3)          # the default start is already correct
    assert two.v == pytest.approx(one.v, rel=1e-6)
    assert two.x0 == pytest.approx(one.x0, abs=1e-6)
    assert two.loss <= one.loss * (1 + 1e-9)


def test_centroid_initial_guess_is_sane():
    from wave_hough_detect import centroid_initial_guess
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    u0, t0_0 = centroid_initial_guess(pts)
    assert u0 > 0
    assert t0_0 == pytest.approx(100.0, abs=10.0)


# --- linear wavefront fitting ---
def test_fit_linear_recovers_known_parameters():
    pts = linear_arrival(0.3, 2.1, 50.0)
    f = fit_linear(pts)
    assert f.n_points == GRID_SIDE * GRID_SIDE
    assert f.a == pytest.approx(0.3, abs=1e-9)
    assert f.b == pytest.approx(2.1, abs=1e-9)
    assert f.r2 == pytest.approx(1.0, abs=1e-12)


def test_fit_linear_speed_is_inverse_gradient_norm():
    """Paper formula: v = 1 / sqrt(a_tilde^2 + b_tilde^2)."""
    pts = linear_arrival(0.3, 2.1, 50.0)
    f = fit_linear(pts)
    assert f.v == pytest.approx(1.0 / np.hypot(0.3, 2.1), rel=1e-12)


def test_linear_and_circular_are_both_near_perfect_for_a_far_source():
    """
    When the source lies far outside the grid the wavefront is almost planar at
    the grid scale - this is why in §3.2 of the paper both models reach an R2
    above 0.93 and the linear model is usable.
    """
    pts = circular_arrival(**PAPER_LIKE)
    fc = fit_circular(pts)
    fl = fit_linear(pts)
    assert fc.r2 > 0.99
    assert fl.r2 > 0.99
