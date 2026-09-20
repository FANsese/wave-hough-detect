"""
Stage 4b: 96x96 circular-wavefront simulation and fit accuracy evaluation
(section 3.1.2, Figs. 6-8)

What this module contains:
  - generation of the per-electrode arrival times
  - assembling one simulated dataset
  - one "simulate + fit" run, returning ground truth and estimate
  - flattening the estimates into plot-ready series
  - a parameter sweep driven by ``plot.mode``

How it differs from section 3.1.1: this is a different simulation
-----------------------------------------------------------------
Section 3.1.1 (``simulate.py``) is a **linear wavefront** on an 8x8 grid, used to
compute FPR / FNR (Fig. 5). This section is a **circular wavefront** on a 96x96
grid, used to compute the estimation accuracy of the source position, the speed
and the activation time (Figs. 6-8). Even the sampling structure differs: the
arrival times here are **not** produced by walking along a line point by point.
They are computed independently for every electrode as
``t = t0 + distance + epsilon``.

Ground truth
------------
The truth of this simulation is ``(x, y, miux, ts) = (48, 48, 1, 2)``. Note that
the third entry is ``miux`` rather than ``v``: in this simulation miux = miuy = 1,
so one grid cell takes one time unit, i.e. v = 1. The ground truth of the
circular wavefront model is therefore

    t = sqrt((x - 48)^2 + (y - 48)^2) / 1 + 2 + epsilon

Why the fixed initial guess happens to be right here
---------------------------------------------------
The alternating minimisation starts at (u, t0) = (1, 1), with u = 1/v the
slowness, while the ground truth of this simulation is v = 1 and t0 = 2: the
initial guess is almost exactly on target. The failure mode measured in section
3.1.1 (a source inside the grid converging to a wrong local minimum) therefore
does **not** occur here. That is a property of the parameters chosen for this
simulation, not an accident of the optimiser.

RNG boundary
------------
As in ``simulate.py``, the datasets are statistically equivalent to the ones
behind the figures but not bit-identical: the draws come from numpy's generator.
This section has many and scattered random-number call sites (one normal draw per
grid cell, plus a conditional uniform draw), so no replayable index trace is
provided. This is an explicit boundary: do not describe these datasets as
bit-identical to the ones behind the figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import (
    HYBRID_MAX_ITER,
    circular_loss,
    hybrid_optim,
    r_squared,
)

# -- Fixed quantities ------------------------------------------------------
SEED_BASE_CIRCULAR = 100          # base for the seed derived from the parameters

#: Ground truth of section 3.1.2, as (x, y, miux, ts).
CIRCULAR_TRUTH = {"x0": 48.0, "y0": 48.0, "v": 1.0, "t0": 2.0}

#: Simulation parameters of section 3.1.2.
PAPER_CIRCULAR_PARAMS = dict(
    limit=96,          # 96x96 grid
    time_max=96.0,     # observation window
    ts=2.0,            # source activation time
    x=48.0, y=48.0,    # source position
    u_x=1.0, u_y=1.0,  # unused (kept in the signature, not used in the body)
    p=0.0,             # missed-detection probability (the quantity swept in Fig. 8)
    noise_freq=1.0,    # noise rate lambda_n
    miu_x=1.0, miu_y=1.0,   # anisotropic propagation: the "speed" of the truth
    sigma=1e-6,        # measurement-error standard deviation (fixed in Fig. 7, swept in Fig. 8)
    gap=1e6,           # mean of the exponential wavefront gap: so large that only one wavefront arises
    ratio=10.0,        # unused
)

#: Fitting configuration of section 3.1.2.
SEARCH_LOWER_CIRCULAR = np.array([-500.0, -500.0])
SEARCH_UPPER_CIRCULAR = np.array([500.0, 500.0])
CIRCULAR_EPS = 1e-8               # convergence criterion of the alternating minimisation

#: The three sweeps of the paper.
PLOT_MODES = {
    # variable is the quantity plotted on the horizontal axis, param its name in
    # the simulator signature
    1: {"variable": "snr", "param": "noise_freq",
        "label": "signal-to-noise ratio  λ_n (reciprocal on the axis)",
        "grid": 2.0 ** (np.arange(-6, 17) / 2.0)},
    2: {"variable": "p", "param": "p",
        "label": "missing-observation probability  p",
        "grid": np.arange(0, 9) / 10.0},
    3: {"variable": "sigma", "param": "sigma",
        "label": "measurement error  σ",
        "grid": np.arange(1, 11) / 10.0},
}


def seed_circular(*, limit, time_max, ts, x, y, u_x, u_y, p, noise_freq,
                  miu_x, miu_y, sigma, seed_shift: int = 0) -> int:
    """
    Derive the seed from the parameters:

        seed(100 - limit - time.max - ts - x - y - u.x - u.y
                - p - noise.freq - miux - miuy - sigma)

    ★ That formula contains no seed shift, so every repeated call with the same
      parameters yields **one and the same dataset**. This implementation keeps
      the formula but adds ``seed_shift`` (default 0, i.e. the same behaviour) so
      that genuinely independent replicates can be produced. For the related trap
      see :func:`accuracy_sweep`.
    """
    raw = (SEED_BASE_CIRCULAR * float(limit) * float(time_max) * float(ts)
           * float(x) * float(y) * float(u_x) * float(u_y) * float(p)
           * float(noise_freq) * float(miu_x) * float(miu_y) * float(sigma))
    return int(raw + seed_shift) % (2 ** 32)


def simulate_circular_wavefronts(seed_shift: int = 0, *,
                                 limit: int = 96, time_max: float = 96.0,
                                 ts: float = 2.0, x: float = 48.0,
                                 y: float = 48.0, u_x: float = 1.0,
                                 u_y: float = 1.0, p: float = 0.0,
                                 noise_freq: float = 1.0, miu_x: float = 1.0,
                                 miu_y: float = 1.0, sigma: float = 1e-6,
                                 gap: float = 1e6, ratio: float = 10.0,
                                 seed: int | None = None) -> pd.DataFrame:
    """
    One 96x96 circular-wavefront simulation.

    Returns a DataFrame with columns ``x, y, ts, z``; z = 1 marks a true signal
    and 0 marks noise. Rows are **stably sorted** by ts.

    Generation logic (including the order of the random draws):

    1. **Every electrode is computed independently.** For ``i, j in 1..limit``:
         - draw ``e ~ N(0, σ)`` unconditionally
         - ``t = ts + sqrt(((i-x)/miux)**2 + ((j-y)/miuy)**2) + e``
           (at the source the distance is 0, degenerating to ``t = ts + e``)
         - **only when t < time_max** draw the ``U(0,1)`` that decides the missed
           detection, and drop the point with probability p - this condition is
           essential, since it determines how much of the random stream is
           consumed
    2. After one wavefront has been generated, one ``Exp(mean=gap)`` draw
       advances the activation time. The default ``gap = 1e6`` is far larger than
       the observation window of 96, so in practice **exactly one wavefront** is
       produced.
    3. Noise: times follow a Poisson process with ``Exp(mean=1/λ_n)``
       inter-arrival times filling the observation window (first and last point
       dropped), positions are drawn uniformly and independently on the grid.
    4. Signal and noise are concatenated and stably sorted by ts.
    """
    if seed is None:
        seed = seed_circular(limit=limit, time_max=time_max, ts=ts, x=x, y=y,
                             u_x=u_x, u_y=u_y, p=p, noise_freq=noise_freq,
                             miu_x=miu_x, miu_y=miu_y, sigma=sigma,
                             seed_shift=seed_shift)
    rng = np.random.default_rng(seed)

    grid = np.arange(1, limit + 1, dtype=float)
    gi, gj = np.meshgrid(grid, grid, indexing="ij")     # i = row, j = column
    travel = np.sqrt(((gi - x) / miu_x) ** 2 + ((gj - y) / miu_y) ** 2)
    at_source = (gi == x) & (gj == y)

    sig_x: list[float] = []
    sig_y: list[float] = []
    sig_t: list[float] = []

    signal_start_ts = float(ts)
    while signal_start_ts < time_max:
        # -- Per electrode (row-major: i outer, j inner) ------------------
        for i in range(limit):
            for j in range(limit):
                e = float(rng.normal(0.0, sigma))
                arrival = signal_start_ts + (
                    e if at_source[i, j] else travel[i, j] + e)
                # ★ the uniform draw happens only when arrival < time_max -
                #   the order must not change
                if arrival < time_max and rng.random() > p:
                    sig_x.append(gi[i, j])
                    sig_y.append(gj[i, j])
                    sig_t.append(arrival)
        # -- advance to the next wavefront activation time ----------------
        signal_start_ts += float(rng.exponential(gap))

    # -- Noise ------------------------------------------------------------
    noise_t = [0.0]
    while noise_t[-1] < time_max:
        noise_t.append(noise_t[-1] + float(rng.exponential(1.0 / noise_freq)))
    noise_t = np.asarray(noise_t[1:-1], dtype=float) if len(noise_t) > 2 \
        else np.empty(0)
    n_noise = noise_t.size
    noise_x = rng.integers(1, limit + 1, size=n_noise).astype(float)
    noise_y = rng.integers(1, limit + 1, size=n_noise).astype(float)

    df = pd.DataFrame({
        "x": np.concatenate([np.asarray(sig_x, dtype=float), noise_x]),
        "y": np.concatenate([np.asarray(sig_y, dtype=float), noise_y]),
        "ts": np.concatenate([np.asarray(sig_t, dtype=float), noise_t]),
        "z": np.concatenate([np.ones(len(sig_x), dtype=int),
                             np.zeros(n_noise, dtype=int)]),
    })
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


# ==========================================================================
#  Estimation and evaluation
# ==========================================================================

@dataclass
class WavefrontEstimate:
    """
    Everything one "simulate + fit" run produces.

    ``n_signal`` / ``n_noise`` / ``mean_travel_time`` are three columns appended
    after the fit; they are used for the horizontal axis of the plots.
    """

    x0: float
    y0: float
    v: float                 # speed (already converted from the slowness)
    t0: float
    slowness: float          # 1/v
    n_signal: int
    n_noise: int
    mean_travel_time: float
    n_points: int
    n_iters: int
    loss: float
    r2: float
    truth: dict = field(default_factory=lambda: dict(CIRCULAR_TRUTH))

    # ★ Both ``r2`` and ``loss`` are computed over *all* points (signal plus
    #   noise), because the whole matrix - including the z = 0 noise rows - is
    #   handed to the optimiser, and the 1[t>0] factor inside the loss keeps
    #   every one of them.
    #
    #   Consequence: in the typical section 3.1.2 configuration noise is only
    #   about 0.9% of the points (9216 signal + 81 noise), yet it pushes R2 from
    #   1.000000 down to 0.953. Measured at the ground-truth parameters:
    #       loss (with noise) = 8.37e4,   loss (signal only) = 9.23e-9
    #       R2   (with noise) = 0.953330, R2 (signal only) = 1.0000000000
    #   So R2 ~ 0.95 here means "the model is exact and the data contains
    #   noise". **Do not** set it side by side with the 0.94 of Table 4 in
    #   section 3.2 - that is a different quantity.

    @property
    def snr(self) -> float:
        """
        The horizontal axis of Fig. 7.

        ★ Despite the name, this is **number of true signal points / number of
          noise points**:

              snr = n_signal / n_noise

          In ``accuracy_sweep`` the swept value is inserted as the first column
          of the result table, so columns 6 and 7 are ``n_signal`` and
          ``n_noise``. This is not a signal-to-noise ratio in the amplitude
          sense; keep that in mind when reading the figure.

          In the section 3.1.2 configuration n_signal = 96x96 = 9216 (every
          electrode is reached) and n_noise ~ λ_n - 96, hence
          snr ~ 96 / λ_n, which matches the range actually swept
          ([~0.375, ~768]).
        """
        return self.n_signal / self.n_noise if self.n_noise else float("inf")

    @property
    def abs_error(self) -> dict:
        """Absolute error of the four quantities (the vertical axes of Figs. 7/8
        actually plot the estimates themselves; see below)."""
        t = self.truth
        return {"x0": abs(self.x0 - t["x0"]), "y0": abs(self.y0 - t["y0"]),
                "v": abs(self.v - t["v"]), "t0": abs(self.t0 - t["t0"])}

    @property
    def rel_error(self) -> dict:
        """
        Relative error: ``|est - truth| / truth``.

        ★ In the paper's plotting this branch is never taken: all three active
          sweep configurations select absolute values, so what is plotted is the
          **absolute estimate**, not a relative error, and the name of the
          plotting routine that column comes from is therefore a misnomer. Both
          are provided here.
        """
        t = self.truth
        return {"x0": abs(self.x0 - t["x0"]) / t["x0"],
                "y0": abs(self.y0 - t["y0"]) / t["y0"],
                "v": abs(self.v - t["v"]) / t["v"],
                "t0": abs(self.t0 - t["t0"]) / t["t0"]}

    def as_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "v": self.v, "t0": self.t0,
                "slowness": self.slowness, "n_signal": self.n_signal,
                "n_noise": self.n_noise,
                "mean_travel_time": self.mean_travel_time,
                "n_points": self.n_points, "n_iters": self.n_iters,
                "loss": self.loss, "r2": self.r2, "snr": self.snr}


def estimate_circular_wavefront(df: pd.DataFrame, *, truth: dict | None = None,
                                max_iter: int = HYBRID_MAX_ITER,
                                eps: float = CIRCULAR_EPS,
                                n_starts: int = 1) -> WavefrontEstimate:
    """
    Fit the circular wavefront model to one simulation result.

    ★ The search window is **+/-500**, not the +/-50 of the real-data pipeline, and
      the convergence criterion is **1e-8**, far stricter than the 1e-6 used
      there. The two settings must not be mixed.
    """
    pts = df[["x", "y", "ts"]].to_numpy(dtype=float)
    trace: list = []            # only used to count the rounds, not part of the computation
    p = hybrid_optim(pts, lower=SEARCH_LOWER_CIRCULAR,
                     upper=SEARCH_UPPER_CIRCULAR, max_iter=max_iter, eps=eps,
                     trace=trace)
    # Multi-start (an extension of this implementation, off by default): with
    # n_starts >= 2 an additional data-driven initial guess is tried. In section
    # 3.1.2 the fixed initial guess (u, t0) = (1, 1) is already almost exactly on
    # the truth (v = 1, t0 = 2), so it is not needed here; the switch is kept for
    # data of a different order of magnitude.
    if n_starts >= 2:
        from .fit import centroid_initial_guess
        u0, t00 = centroid_initial_guess(pts)
        p2 = hybrid_optim(pts, lower=SEARCH_LOWER_CIRCULAR,
                          upper=SEARCH_UPPER_CIRCULAR, max_iter=max_iter,
                          eps=eps, init_u=u0, init_t0=t00)
        if circular_loss(p2, pts) < circular_loss(p, pts):
            p = p2

    sig = df["z"].to_numpy() == 1
    truth = dict(CIRCULAR_TRUTH if truth is None else truth)
    return WavefrontEstimate(
        x0=float(p[0]), y0=float(p[1]), v=float(1.0 / p[2]), t0=float(p[3]),
        slowness=float(p[2]),
        n_signal=int(sig.sum()), n_noise=int((~sig).sum()),
        mean_travel_time=float(df.loc[sig, "ts"].mean()) if sig.any() else np.nan,
        n_points=len(df), n_iters=len(trace), loss=circular_loss(p, pts),
        r2=r_squared(circular_loss, p, pts), truth=truth,
    )


def evaluate_circular(seed_shift: int = 0, *, params: dict | None = None,
                      **overrides) -> WavefrontEstimate:
    """
    Run one complete "simulate + fit".

    ``params`` defaults to :data:`PAPER_CIRCULAR_PARAMS`; ``overrides`` replaces
    individual entries of it (which is how the sweeps are driven).
    """
    prm = dict(PAPER_CIRCULAR_PARAMS)
    if params:
        prm.update(params)
    prm.update(overrides)

    df = simulate_circular_wavefronts(seed_shift, **prm)
    return estimate_circular_wavefront(df)


def accuracy_sweep(plot_mode: int = 1, *, n_replicates: int | None = None,
                   independent_replicates: bool = True,
                   params: dict | None = None) -> pd.DataFrame:
    """
    Parameter sweep driven by ``plot.mode``.

    ``plot_mode``
        1 - sweep the noise rate λ_n (Fig. 7), holding p = 0 and σ = 1e-6
        2 - sweep the missed-detection probability p (Fig. 8), holding λ_n = 1
            and σ = 1e-6
        3 - sweep the measurement error σ, holding λ_n = 1 and p = 0

    ``n_replicates``
        Number of replicates per grid point. Defaults to the replicate counts
        used for the paper (mode 1 -> 23, mode 2 -> 9, mode 3 -> 10).

    ``independent_replicates``
        ★★ This is an **intentional difference** from the setup behind the
        figures and has to be understood.

        The seed is ``100 - product(parameters)`` with no shift, and a sweep
        calls the simulator repeatedly with the same set of parameters - so those
        "replicates" are not replicates at all but **one and the same dataset**.
        Worse, with **p = 0** the seed is identically 0 (one factor of the product
        is p), so the entire mode 1 and mode 3 batches share a single random
        stream: the measurement error ε of every electrode is the same sequence,
        merely scaled by σ. The consequence is that those two curves are
        **unnaturally smooth**, and the error bars do not represent scatter
        between replicates.

        The default ``True`` adds a different ``seed_shift`` to every replicate,
        giving genuinely independent replicates. Set it to ``False`` to reproduce
        the smooth behaviour described above.
    """
    if plot_mode not in PLOT_MODES:
        raise ValueError(f"plot_mode must be one of {sorted(PLOT_MODES)}")
    spec = PLOT_MODES[plot_mode]
    grid = spec["grid"]
    n_rep = n_replicates if n_replicates is not None else {
        1: 23, 2: 9, 3: 10}[plot_mode]

    # each mode sweeps one quantity and holds the other two fixed
    var = spec["variable"]
    param = spec["param"]
    fixed = {"snr": {"p": 0.0, "sigma": 1e-6},
             "p": {"noise_freq": 1.0, "sigma": 1e-6},
             "sigma": {"noise_freq": 1.0, "p": 0.0}}[var]

    rows = []
    for value in grid:
        for rep in range(n_rep):
            over = dict(fixed)
            over[param] = float(value)
            shift = (rep + 1) if independent_replicates else 0
            est = evaluate_circular(shift, params=params, **over)
            rows.append({
                "plot_mode": plot_mode,
                "variable": var,
                "scan_value": float(value),
                "replicate": rep,
                **est.as_dict(),
                **{f"abs_{k}": v for k, v in est.abs_error.items()},
                **{f"rel_{k}": v for k, v in est.rel_error.items()},
            })
    return pd.DataFrame(rows)
