"""
Stage 3: wavefront model fitting (section 2.3 / Algorithm 3 + linear model)

Two models:
  circular wavefront (near field)  t = sqrt((x-x0)^2+(y-y0)^2)/v + t0   -- alternating minimisation
  linear wavefront (far field)     t = a*x + b*y + c                    -- closed-form least squares

The two traps of this stage:
  ★ The u = 1/v slot: the optimiser stores the *slope* (= 1/v) in the position
    named v, i.e. the third parameter slot, and the objective is balanced by
    multiplying by that slope rather than dividing by the speed. This module
    names the variable u; the numerical behaviour is the same.
  ★ Scale: the ts used for fitting has to be multiplied back by 200, i.e. the
    original ms scale. The /200 scale used by the Hough stage only exists to
    serve the inlier tolerance of 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# -- Fixed parameters ------------------------------------------------------
GRID_SIDE = 8
SEARCH_LOWER = np.array([-50.0, -50.0])   # search window of the alternating minimisation (x0, y0)
SEARCH_UPPER = np.array([50.0, 50.0])
GRID_SPLITS = 4                            # number of splits of the grid search
GRID_EPS = 1e-8                            # convergence criterion of the grid search
GRID_MAX_ITER = 1000
HYBRID_EPS = 1e-6                          # convergence criterion of the alternating minimisation
HYBRID_MAX_ITER = 10000
TIME_SCALE_BACK = 200.0                    # multiplier that restores the original ms scale


# ==========================================================================
#  Coordinate grid
# ==========================================================================

def electrode_grid(side: int = GRID_SIDE):
    """
    x_grid[i, j] = i + 1 (row number), y_grid[i, j] = j + 1 (column number).

    The grid is filled by rows, so (x, y) = (row, column), which is consistent
    with the channel mapping of stage 1.
    """
    xs = np.arange(1, side + 1)
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")   # x_grid[i,j]=i+1
    return x_grid.astype(float), y_grid.astype(float)


def build_result_array(spike_x, spike_y, spike_ts_ms, plane_indices,
                       n_planes: int | None = None):
    """
    Pack the activation time of every plane on every electrode into an
    (8, 8, K) array.

    Points to note:
      - ts is multiplied back by 200 (back to the original ms scale) - the fit
        has to use that scale
      - unclassified points (plane_indices == 0) are skipped
      - when the same (x, y) occurs several times within one plane, the last
        write wins
      - empty slots stay at -1 (the array is initialised with -1)
    """
    if n_planes is None:
        n_planes = int(np.max(plane_indices)) if len(plane_indices) else 0

    arr = np.full((GRID_SIDE, GRID_SIDE, n_planes), -1.0)
    for x, y, ts, pid in zip(spike_x, spike_y, spike_ts_ms, plane_indices):
        if pid == 0:
            continue
        arr[int(x) - 1, int(y) - 1, int(pid) - 1] = ts * TIME_SCALE_BACK
    return arr


def plane_points(arr, plane_index: int) -> np.ndarray:
    """
    Extract the (x, y, ts) point set of one plane, keeping only ts > 0.

    The whole 8x8 grid is flattened in column-major order and then filtered by
    ts > 0.
    """
    x_grid, y_grid = electrode_grid()
    ts = arr[:, :, plane_index - 1]
    # column-major order; numpy's default is C order (row-major)
    xs = x_grid.flatten(order="F")
    ys = y_grid.flatten(order="F")
    ts_flat = ts.flatten(order="F")
    keep = ts_flat > 0
    return np.column_stack([xs[keep], ys[keep], ts_flat[keep]])


# ==========================================================================
#  Loss functions
# ==========================================================================

def circular_loss(p, points: np.ndarray) -> float:
    """
    Circular wavefront loss l_circular.

        l = Sum [ sqrt((x_i-x0)^2+(y_i-y0)^2)*u - (t_i-t0) ]^2 * 1[t_i>0]

    ★ p[2] holds u = 1/v, so the distance is *multiplied* by u. Equation (3) of
      the paper writes a division by v; the two are equivalent, but the code
      multiplies.
    """
    x0, y0, u, t0 = p
    x = points[:, 0]; y = points[:, 1]; t = points[:, 2]
    resid = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) * u - (t - t0)
    return float(np.sum(resid ** 2 * (t > 0)))


def circular_loss_xy(xy, points: np.ndarray, u: float, t0: float) -> float:
    """
    Wrapper with (u, t0) held fixed, evaluated only over (x0, y0).

    This is the callback the grid search calls on every round.
    """
    return circular_loss([xy[0], xy[1], u, t0], points)


def linear_loss(p, points: np.ndarray) -> float:
    """
    Linear wavefront loss l_linear.

        l = Sum [ (a*x_i + b*y_i + c) - t_i ]^2 * 1[t_i>0]
    """
    a, b, c = p
    x = points[:, 0]; y = points[:, 1]; t = points[:, 2]
    resid = a * x + b * y + c - t
    return float(np.sum(resid ** 2 * (t > 0)))


def r_squared(loss_fn: Callable, params, points: np.ndarray) -> float:
    """
    Coefficient of determination (R2), computed as

        R2 = 1 - SS_res / SS_tot,   SS_tot = Sum(t_i - mean(t))^2

    ★ SS_tot is computed over *all* points passed in, without the extra 1[t>0]
      factor. Since the caller has already filtered ts > 0, the two are
      equivalent.
    """
    ss_res = loss_fn(params, points)
    t = points[:, 2]
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


# ==========================================================================
#  4x4 grid tabulation search
# ==========================================================================

def _find_best_point_1d(dim: int, current: np.ndarray, n_splits: int,
                        lower: np.ndarray, upper: np.ndarray,
                        callback: Callable):
    """
    Recursively enumerate the grid points along dimension ``dim``.

    The loop runs over 0..n_spaces with n_spaces = n_splits - 1 = 3, i.e. four
    points per dimension including both ends: lower, lower + span/3,
    lower + 2*span/3, upper.
    """
    n_spaces = n_splits - 1
    best_val = 1e100
    best_pt = current.copy()

    for i in range(n_spaces + 1):                      # 0..n_spaces, both ends included
        current[dim] = lower[dim] + (upper[dim] - lower[dim]) / n_spaces * i
        if dim < len(current) - 1:
            val, pt = _find_best_point_1d(dim + 1, current, n_splits,
                                          lower, upper, callback)
        else:
            val, pt = callback(current), current.copy()
        if val < best_val:
            best_val = val
            best_pt = pt.copy()
    return best_val, best_pt


def grid_optim(lower: np.ndarray, upper: np.ndarray,
               callback: Callable, max_iter: int = GRID_MAX_ITER,
               eps: float = GRID_EPS):
    """
    4x4 grid tabulation search.

    Every round:
      1. place 4^n grid points on [lower, upper] and keep the smallest
      2. if the best point sits on the boundary, move it one cell inwards
         (★ the clamping rule)
      3. shrink the search window to the best point +/- one cell spacing
      4. stop once the window width is below eps and return the window midpoint
    """
    n_splits = GRID_SPLITS
    n_spaces = n_splits - 1
    lower = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()

    for _ in range(max_iter):
        val, best = _find_best_point_1d(0, lower.copy(), n_splits,
                                        lower, upper, callback)
        new_space = (upper - lower) / n_spaces

        # ★ Boundary clamping:
        #   if the best grid point lies exactly on the boundary of the window,
        #   the true optimum may sit outside it, so the point is moved one cell
        #   inwards; otherwise the window keeps drifting outwards.
        for k in range(len(best)):
            if abs(best[k] - lower[k]) < 1e-12:
                best[k] = lower[k] + new_space[k]
            if abs(best[k] - upper[k]) < 1e-12:
                best[k] = upper[k] - new_space[k]

        lower = best - new_space
        upper = best + new_space

        # the test is on the grid spacing, not on the displacement
        if float(np.linalg.norm(new_space)) < eps:
            break

    return (lower + upper) / 2.0


# ==========================================================================
#  Second half-step of the alternating minimisation: least-squares fit for
#  u = 1/v
# ==========================================================================

def vt_optim(p, points: np.ndarray) -> np.ndarray:
    """
    With (x0, y0) held fixed, solve for (u, t0) by least squares.

    Treating t as a linear function of the distance d:  t = u*d + t0
      -> slope u = 1/v
      -> intercept t0

    ★ The returned vector puts the *slope u* in p[2]. Callers that need the
      speed use 1/p[2].
    """
    x0, y0 = p[0], p[1]
    t = points[:, 2]
    d = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)
    keep = t > 0
    d, t = d[keep], t[keep]

    # least squares with design matrix [d, 1]
    design = np.column_stack([d, np.ones(len(d))])
    coef, *_ = np.linalg.lstsq(design, t, rcond=None)

    out = np.array(p, dtype=float)
    out[3] = coef[1]     # intercept -> t0
    out[2] = coef[0]     # slope -> u = 1/v
    return out


def hybrid_optim(points: np.ndarray,
                 lower: np.ndarray = SEARCH_LOWER,
                 upper: np.ndarray = SEARCH_UPPER,
                 max_iter: int = HYBRID_MAX_ITER,
                 eps: float = HYBRID_EPS,
                 trace: list | None = None,
                 init_u: float = 1.0, init_t0: float = 1.0) -> np.ndarray:
    """
    Alternating minimisation main loop (Algorithm 3).

    Every round:
      1. with (u, t0) fixed, find (x0, y0) with the 4x4 grid tabulation search
      2. with (x0, y0) fixed, find (u, t0) by least squares
      3. stop once the relative change is below eps

    ★ The initial values are u = 1 and t0 = 1, and the reference vector for the
      relative change is (1, 1, 1, 1) - a vector completely unrelated to the
      data, so the relative change of the *first* round is measured against
      (1, 1, 1, 1). Those defaults are reproduced here.
      ``init_u`` / ``init_t0`` are additional switches of this implementation
      whose defaults equal the values above; changing them no longer corresponds
      to the configuration behind the paper's numbers and is only meant for data
      where the fixed initial guess gets stuck (see ``n_starts`` in
      :func:`fit_circular`).

    ★★ Known convergence fragility:
      Starting the alternating minimisation at (1, 1), a source point *inside*
      the grid pushes the first grid search to the boundary of the search window,
      from where it never climbs back. Measured:
          (x0,y0,v,t0) = (2.5, 3.5, 0.45, 100)   -> converges to (-44.7, 14.6), uses all 10000 rounds
          (x0,y0,v,t0) = (-3.0, -50, 0.45, 800)  -> converges to the truth in 329 rounds ✅
      The paper's configuration (source far outside the grid, 50 cells away) is
      the second case, so the paper's numbers hold.
    """
    u, t0 = float(init_u), float(init_t0)
    p_old = np.array([1.0, 1.0, 1.0, 1.0])  # reference vector of the relative change

    p = np.array([0.0, 0.0, u, t0])
    for it in range(1, max_iter + 1):
        # 1. (x0, y0)
        xy = grid_optim(lower, upper,
                        lambda xy_: circular_loss_xy(xy_, points, u, t0))
        p = np.array([xy[0], xy[1], u, t0])

        # 2. (u, t0)
        p = vt_optim(p, points)
        u, t0 = p[2], p[3]

        if trace is not None:
            trace.append(p.copy())

        reldiff = float(np.linalg.norm(p - p_old) / np.linalg.norm(p_old))
        if reldiff < eps:
            break
        p_old = p.copy()

    return p


# ==========================================================================
#  Convenience wrappers
# ==========================================================================

@dataclass
class CircularFit:
    x0: float
    y0: float
    v: float          # already converted to a speed (not u)
    t0: float
    r2: float
    loss: float
    n_points: int
    n_iters: int


@dataclass
class LinearFit:
    a: float          # coefficient of x
    b: float          # coefficient of y
    v: float          # 1/sqrt(a^2 + b^2)
    r2: float
    loss: float
    n_points: int


def centroid_initial_guess(points: np.ndarray) -> tuple[float, float]:
    """
    Data-driven initial guess: take the (x, y) centroid of the point set as a
    temporary centre and fit (r, t) by least squares, which yields initial values
    for the slowness u and the activation time t0.

    This helper is an addition of this implementation and is only used when
    ``n_starts`` >= 2. It addresses a real failure mode: the fixed initial guess
    (u, t0) = (1, 1) converges to a wrong local minimum when the source point
    lies inside the grid. Conversely, this centroid guess is poor when the source
    is far outside the grid (the centroid is then far from the true source), so
    neither value should be the default on its own - see ``n_starts`` in
    :func:`fit_circular`.
    """
    x, y, t = points[:, 0], points[:, 1], points[:, 2]
    r = np.hypot(x - x.mean(), y - y.mean())
    design = np.column_stack([r, np.ones_like(r)])
    u0, t0 = np.linalg.lstsq(design, t, rcond=None)[0]
    return float(u0), float(t0)


def fit_circular(points: np.ndarray, trace: list | None = None,
                 n_starts: int = 1, **kw) -> CircularFit:
    """
    Fit the circular wavefront model to the (x,y,t) point set of one plane.

    ``n_starts``
        1 (default) - use only the fixed initial guess (u, t0) = (1, 1).
                      **This value is required to reproduce Tables 2 / 4 of the
                      paper.**
        2           - additionally try the initial guess given by
                      :func:`centroid_initial_guess` and keep the solution with
                      the smaller final loss.
                      Use it for data where the fixed initial guess gets stuck,
                      such as a source point inside the grid (measured to recover
                      (2.5, 3.5, 0.45, 100) from a completely wrong answer to the
                      exact solution).
    Whatever the value, the path that converges is the same alternating
    minimisation; only the starting point differs.
    """
    tr = trace if trace is not None else []
    inits = [(1.0, 1.0)]
    if n_starts >= 2:
        inits.append(centroid_initial_guess(points))

    best_p, best_loss, best_iters = None, np.inf, 0
    for k, (u0, t0_0) in enumerate(inits):
        sub = tr if (k == 0 and n_starts == 1) else []
        p = hybrid_optim(points, trace=sub, init_u=u0, init_t0=t0_0, **kw)
        loss = circular_loss(p, points)
        if loss < best_loss:
            best_p, best_loss, best_iters = p, loss, len(sub)
    if n_starts > 1:
        tr.clear()                       # with several starting points one trace has no single meaning
    p = best_p
    return CircularFit(
        x0=float(p[0]), y0=float(p[1]), v=float(1.0 / p[2]), t0=float(p[3]),
        r2=r_squared(circular_loss, p, points), loss=best_loss,
        n_points=len(points), n_iters=best_iters,
    )


def fit_linear(points: np.ndarray) -> LinearFit:
    """
    Closed-form solution of the linear wavefront model.

    ts is regressed on x and y:
        a = coefficient of x, b = coefficient of y
        v = 1/sqrt(a^2 + b^2)
    """
    x = points[:, 0]; y = points[:, 1]; t = points[:, 2]
    design = np.column_stack([x, y, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, t, rcond=None)
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    v = 1.0 / np.sqrt(a * a + b * b)
    params = np.array([a, b, c])
    return LinearFit(a=a, b=b, v=float(v),
                     r2=r_squared(linear_loss, params, points),
                     loss=linear_loss(params, points),
                     n_points=len(points))
