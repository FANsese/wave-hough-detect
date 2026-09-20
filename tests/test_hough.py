"""
Unit tests for stage 2 (randomised Hough transform).

Uncertain things such as "the outcome of a random draw" are deliberately avoided
here; what is checked instead is:

  · behaviour on degenerate input (too few points, three collinear points)
  · whether an explicitly constructed plane is found
  · ★ the tail-regrowth branch, which never triggers on the real data and only
    triggers on the simulated data
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import hough_plane
from wave_hough_detect.rht import SIM_PRESET


def _plane_points(normal, rho, side=8, t_max=40.0, t_min=0.5):
    """
    Build a point set that lies exactly on the plane ``n . p = rho``.

    Every electrode of an 8x8 grid is taken and t is solved for; only the points
    whose t falls inside [t_min, t_max] are kept.
    """
    n1, n2, n3 = normal
    xs, ys = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1),
                         indexing="ij")
    xs = xs.ravel().astype(float)
    ys = ys.ravel().astype(float)
    if n3 == 0:
        return np.empty((0, 3))
    t = (rho - n1 * xs - n2 * ys) / n3
    keep = (t >= t_min) & (t <= t_max)
    return np.column_stack([xs[keep], ys[keep], t[keep]])


# --- degenerate input ---
def test_fewer_than_three_points_returns_empty_result():
    """With fewer than 3 points the main loop breaks immediately; an empty result must be returned instead of crashing."""
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    res = hough_plane(pts, max_iter=1000, seed=0)
    assert res.n_planes == 0
    assert res.prediction.tolist() == [0, 0]
    assert res.accept_log == []


def test_all_points_collinear_does_not_crash():
    """Three points are always collinear -> the cross product is identically zero -> the loop always continues; the run must finish normally instead of looping forever."""
    t = np.linspace(1.0, 5.0, 20)
    pts = np.column_stack([t, t, t])
    res = hough_plane(pts, max_iter=2000, seed=1)
    assert res.n_planes == 0
    assert res.prediction.sum() == 0


def test_empty_input():
    res = hough_plane(np.empty((0, 3)), max_iter=10, seed=0)
    assert res.n_planes == 0
    assert res.prediction.size == 0


# --- a known plane is found ---
def test_finds_a_synthetic_plane():
    """
    Hand-build a wavefront plane: its normal vector is close to the t axis, which
    is exactly what a wavefront looks like in (x, y, t). The simulation parameter
    set is used because the coordinate scale matches §3.1.
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    pts = _plane_points(normal, rho=-10.0)
    assert len(pts) >= 40, "at least 40 electrodes worth of points are needed"

    res = hough_plane(pts, **SIM_PRESET, max_iter=20000, min_detectors=40,
                      seed=1)
    assert res.n_planes >= 1
    # at least 40 electrodes, i.e. at least 40 points, taken into the plane
    assert res.prediction.sum() >= 40
    got = np.array([res.n1[res.prediction == 1][0], res.n2[res.prediction == 1][0],
                    res.n3[res.prediction == 1][0]])
    assert abs(float(np.dot(got, normal))) == pytest.approx(1.0, abs=1e-3)


def test_two_parallel_planes_are_not_merged():
    """
    The two parallel wavefronts are 20 units apart in t, far beyond the inlier
    tolerance of 1.0, so they must be split into two planes.
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    a = _plane_points(normal, rho=-10.0, t_min=-1000, t_max=1000)
    b = _plane_points(normal, rho=-30.0, t_min=-1000, t_max=1000)
    pts = np.vstack([a, b])
    res = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                      seed=3)
    assert res.n_planes >= 2
    assert res.prediction.sum() >= 80


# --- ★ the tail-regrowth branch ---
def _tail_loop_fixture():
    """
    Build a configuration that **only the tail regrowth can collect fully**:

      · master plane A: rho = -10, 64 points;
      · parallel plane B: rho = -30, 40 points (same normal as A, 20 apart in t);
      · a few isolated points.

    The main loop is pinned by the fixed trace so that it only finds plane A; the
    points of plane B stay in the unclassified set. The "recovery" step of
    Algorithm 4 reuses the normal and the rho of A and cannot reach B. So only
    the tail regrowth can re-derive B from the master normal together with an
    unclassified point used as the anchor.
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    a = _plane_points(normal, rho=-10.0, t_min=-1000, t_max=1000)
    b = _plane_points(normal, rho=-30.0, t_min=-1000, t_max=1000)[:40]
    stray = np.array([[1.0, 1.0, 100.0], [8.0, 8.0, 101.0], [4.0, 5.0, 102.0]])
    pts = np.vstack([a, b, stray])

    # Trace: every triple is taken from plane A -> the main loop accepts A only.
    # ★ indices 0=(1,1), 1=(1,2), 8=(2,1). Do not use [0, 1, 2] - those three
    #   points share x = 1, so their cross product is exactly zero, the
    #   degenerate-triple guard skips them and not a single vote is cast.
    trace = [[0, 1, 8]] * 20
    return pts, trace, len(a)


def test_tail_regrow_is_off_by_default_when_no_plane_found():
    """When no plane is found at all, the tail regrowth must not crash (max_nv does not exist)."""
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    res = hough_plane(pts, max_iter=10, seed=0, regrow_master_plane=True)
    assert res.n_planes == 0


def test_tail_regrow_collects_the_parallel_plane():
    """Switching the tail regrowth on -> the parallel plane B is collected as a new plane."""
    pts, trace, n_a = _tail_loop_fixture()
    res = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                      trace=trace, regrow_master_plane=True)
    assert res.n_planes >= 2, "the tail regrowth should take in the parallel plane"
    assert res.prediction.sum() >= n_a + 40
    assert res.plane_indices.max() >= 2


def test_tail_regrow_can_be_disabled():
    """
    Switching the tail regrowth off -> the points of the parallel plane B stay at
    prediction == 0.

    This pair of tests is the evidence that the tail regrowth really does
    something: on the real data both settings give exactly the same result (the
    unclassified set is empty and the loop body is never entered), so only a
    construction like this one can exercise it.
    """
    pts, trace, n_a = _tail_loop_fixture()
    on = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                     trace=trace, regrow_master_plane=True)
    off = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                      trace=trace, regrow_master_plane=False)
    assert off.n_planes == 1
    assert on.n_planes > off.n_planes
    assert on.prediction.sum() > off.prediction.sum()


def test_tail_regrow_terminates_on_all_noise():
    """
    When every point is an isolated point that cannot form a plane, the tail loop
    must consume the points one by one and finish normally (it must not loop
    forever). This branch is where points are provisionally marked with
    prediction == 2 and restored to 0 once the loop is over.
    """
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(1, 8, 60), rng.uniform(1, 8, 60),
                           rng.uniform(1, 200, 60)])
    res = hough_plane(pts, **SIM_PRESET, max_iter=2000, min_detectors=40,
                      seed=5, regrow_master_plane=True)
    assert res.prediction.size == 60
    assert set(np.unique(res.prediction)).issubset({0, 1})   # the temporary 2 must be cleaned up completely


# --- the parameter sets must stay separate ---
def test_real_data_and_simulation_presets_differ():
    """
    ★ The two pipelines of the paper use different parameters (the ts scale
      differs by a factor of 10). The difference is pinned down here so that
      nobody "unifies" them into a single set.
    """
    from wave_hough_detect import INLIER_TOL, RHO_STEP
    assert (RHO_STEP, INLIER_TOL) == (0.05, 0.1)
    assert SIM_PRESET["rho_step"] == 0.5
    assert SIM_PRESET["inlier_tol"] == 1.0
    assert SIM_PRESET["strict_degenerate_check"] is True


def test_wrong_preset_loses_inliers_on_noisy_points():
    """
    Counter-proof: add a little jitter to a point set at simulation scale and then
    use the real-data tolerances (10x tighter); markedly fewer points are taken
    into the plane. This shows that "the parameter sets must not be mixed" is not
    an empty claim.

    (Note: if the points lie exactly on the plane with zero residual, both
    tolerance sets take all of them in - so "is a plane found at all" cannot
    detect a mixed-up parameter set.)
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    pts = _plane_points(normal, rho=-10.0, t_min=-1000, t_max=1000)

    rng = np.random.default_rng(0)
    # jitter of +-0.4 along the normal - below 1.0, above 0.1
    pts = pts + np.outer(rng.uniform(-0.4, 0.4, len(pts)), normal)

    loose = hough_plane(pts, **SIM_PRESET, max_iter=20000, min_detectors=40,
                        seed=1)
    tight = hough_plane(pts, rho_step=0.05, inlier_tol=0.1, max_iter=20000,
                        min_detectors=40, seed=1)
    assert loose.n_planes >= 1
    assert loose.prediction.sum() > tight.prediction.sum()
