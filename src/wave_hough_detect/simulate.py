"""
Stage 4: simulation data and detection performance evaluation (section 3.1,
Figs. 2-8)

Why this layer is needed
------------------------
The real recording (section 3.2) comes from a collaborating laboratory and
cannot be redistributed, so the repository has to offer two paths that run
without it:

  1. ``simulate_linear_wavefronts`` - the 8x8 linear-wavefront simulation of
     section 3.1.1. It produces an (x, y, ts, z) point set, where z marks whether
     a point is a true signal or noise. Running the RHT on it yields FPR / FNR
     (Fig. 5 of the paper).

  2. ``simulate_recording`` - produces a 64-channel MEA recording CSV in exactly
     the same format as the real data, so the whole stage 1 -> 2 -> 3 pipeline
     runs end to end.

Statistics, not bit-exact reproduction
--------------------------------------
The generated datasets are statistically equivalent to the ones behind the
figures, but they are not the same numbers: the draws come from numpy's random
number generator, so a given seed determines the whole stream, yet the individual
values are specific to numpy. What this module does guarantee is:
  - the sampling structure and order are reproduced one to one (which step draws
    how many values, and from which distribution);
  - the seed derivation (the product of the parameters plus an offset) is
    reproduced, so that the property "same parameters -> same seed -> same
    dataset" holds inside Python and every user obtains the same values.

★ A bit-exact comparison is therefore impossible here. This is a known and
  explicit boundary: do not describe these datasets as identical to the
  individual draws behind the paper's figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rht import SIM_PRESET, hough_plane
from .spikes import GRID_SIDE, channel_to_xy

# -- Seed bases ------------------------------------------------------------
SEED_BASE_2D = 101        # base for the seed derived from the parameters
SEED_BASE_1D = 1000000    # base of the one-dimensional simulation variant

# -- Parameters of section 3.1.1 -------------------------------------------
PAPER_2D_PARAMS = dict(
    limit=8,          # 8x8 MEA
    time_max=80.0,    # time.max = limit*10 for Fig. 5
    ts=1.0,           # activation time of the first wavefront
    x=1, y=1,         # wavefront starting point (grid coordinates)
    u_x=1, u_y=1,     # one grid cell of space per step
    p=0.1,            # probability of a missed detection at an electrode
    noise_freq=1.0,   # temporal noise rate lambda_n
    miu=1.0,          # mean inter-electrode arrival-time difference mu
    sigma=0.1,        # standard deviation of the arrival-time difference sigma
    gap=30.0,         # gap between two wavefronts (chi-square degrees of freedom)
    ratio=2.0,        # magnification of the time difference when wrapping to a new row
)


def seed_from_params(base: float, shift: int, **factors) -> int:
    """
    Derive a seed from the parameters: ``base * product(factors) + shift``.

    The floating-point result is truncated to an integer and then reduced modulo
    2**32 so that it fits numpy's seed range.
    """
    raw = float(base)
    for v in factors.values():
        raw *= float(v)
    raw += float(shift)
    return int(raw) % (2 ** 32)


# ==========================================================================
#  Section 3.1.1: 8x8 linear-wavefront simulation
# ==========================================================================

def _simulate_one_line(rng: np.random.Generator, limit: int, time_max: float,
                       ts: float, x: float, u_x: float, p: float,
                       miu: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk one straight line along x until it reaches the boundary or runs out of
    time.

    The step is drawn *first* and the walk is tested afterwards, and a failed
    test still consumes that draw: the number of dt draws therefore equals the
    number of accepted points plus one.

    Every accepted point is then dropped with probability p (missed detection).
    """
    xs = [x]
    ts_list = [ts]
    dt = float(rng.normal(miu, sigma))
    while (xs[-1] + u_x >= 1) and (xs[-1] + u_x <= limit) and \
            (ts_list[-1] + dt <= time_max):
        xs.append(xs[-1] + u_x)
        ts_list.append(ts_list[-1] + dt)
        dt = float(rng.normal(miu, sigma))

    x_arr = np.asarray(xs, dtype=float)
    t_arr = np.asarray(ts_list, dtype=float)

    # every point survives with probability 1 - p
    keep = rng.random(x_arr.size) > p
    return x_arr[keep], t_arr[keep]


def _simulate_one_line_2d(rng: np.random.Generator, limit: int, time_max: float,
                          ts: float, x: float, y: float, u_x: float, u_y: float,
                          p: float, miu: float, sigma: float,
                          ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lay out a row of parallel lines along y (= one plane wavefront).

    The loop condition tests only y >= 1, y <= limit and ts <= time_max - there
    is *no* lower bound test on ts.
    """
    x_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    t_out: list[np.ndarray] = []

    while (y >= 1) and (y <= limit) and (ts <= time_max):
        lx, lt = _simulate_one_line(rng, limit, time_max, ts, x, u_x, p,
                                    miu, sigma)
        x_out.append(lx)
        y_out.append(np.full(lx.size, y, dtype=float))
        t_out.append(lt)
        y += u_y
        ts += float(rng.normal(ratio * miu, ratio * sigma))

    if not x_out:
        return (np.empty(0), np.empty(0), np.empty(0))
    return (np.concatenate(x_out), np.concatenate(y_out), np.concatenate(t_out))


def simulate_linear_wavefronts(seed_shift: int = 1, *,
                               limit: int = 8, time_max: float = 80.0,
                               ts: float = 1.0, x: float = 1.0, y: float = 1.0,
                               u_x: float = 1.0, u_y: float = 1.0, p: float = 0.1,
                               noise_freq: float = 1.0, miu: float = 1.0,
                               sigma: float = 0.1, gap: float = 30.0,
                               ratio: float = 2.0, seed: int | None = None
                               ) -> pd.DataFrame:
    """
    The simulation of section 3.1.1: several parallel linear wavefronts
    propagating across an 8x8 grid, plus Poisson noise.

    The correspondence between the parameters and the paper's symbols is given in
    ``PAPER_2D_PARAMS``.

    Returns
    -------
    DataFrame with columns ``x, y, ts, z``:
        x, y  - grid coordinates of the electrode (integers, 1..limit)
        ts    - activation time at that electrode
        z     - 1 = true signal, 0 = noise. **This is the reason this module
                exists**: z is what makes FPR / FNR computable.
    Rows are stably sorted by ts (the sort has to be stable, see below).
    """
    if seed is None:
        seed = seed_from_params(
            SEED_BASE_2D, seed_shift, limit=limit, time_max=time_max, ts=ts,
            x=x, y=y, u_x=u_x, u_y=u_y, p=p, noise_freq=noise_freq,
            miu=miu, sigma=sigma)
    rng = np.random.default_rng(seed)

    # -- Signal: one wavefront after another, the starting time advancing by
    #    chi-square spaced gaps -------------------------------------------
    sig_x: list[np.ndarray] = []
    sig_y: list[np.ndarray] = []
    sig_t: list[np.ndarray] = []
    signal_start_ts = ts
    while signal_start_ts < time_max:
        lx, ly, lt = _simulate_one_line_2d(
            rng, limit, time_max, signal_start_ts, x, y, u_x, u_y, p,
            miu, sigma, ratio)
        sig_x.append(lx); sig_y.append(ly); sig_t.append(lt)
        signal_start_ts += float(rng.chisquare(gap))

    x_sig = np.concatenate(sig_x) if sig_x else np.empty(0)
    y_sig = np.concatenate(sig_y) if sig_y else np.empty(0)
    t_sig = np.concatenate(sig_t) if sig_t else np.empty(0)

    # -- Noise: exponential inter-arrival times, positions uniform on the grid
    noise_ts = [0.0]
    while noise_ts[-1] < time_max:
        noise_ts.append(noise_ts[-1] + float(rng.exponential(1.0 / noise_freq)))
    # the first and last points are dropped (the first is the artificial 0, the
    # last is already past time_max)
    noise_ts = np.asarray(noise_ts[1:-1], dtype=float) if len(noise_ts) > 2 \
        else np.empty(0)
    n_noise = noise_ts.size
    noise_x = rng.integers(1, limit + 1, size=n_noise).astype(float)
    noise_y = rng.integers(1, limit + 1, size=n_noise).astype(float)

    df = pd.DataFrame({
        "x": np.concatenate([x_sig, noise_x]),
        "y": np.concatenate([y_sig, noise_y]),
        "ts": np.concatenate([t_sig, noise_ts]),
        "z": np.concatenate([np.ones(t_sig.size, dtype=int),
                             np.zeros(n_noise, dtype=int)]),
    })
    # ★ The sort has to be stable. ts is a continuous quantity that should not
    #   tie in theory, but the simulation contains a great many repeated noise
    #   times, so "stable" must be requested explicitly; otherwise the row order
    #   is not reproducible.
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


# ==========================================================================
#  Detection performance evaluation (Fig. 5 of the paper)
# ==========================================================================

@dataclass
class DetectionRates:
    """
    Detection performance of one simulation. The four counts are defined as
    TP: z=1 and prediction=1; FP: z=0 and prediction=1; FN: z=1 and
    prediction=0; TN: z=0 and prediction=0.
    """

    n_points: int
    n_true_pos: int          # z=1 and prediction=1
    n_false_pos: int         # z=0 and prediction=1 - noise called signal
    n_false_neg: int         # z=1 and prediction=0 - signal called noise
    n_true_neg: int          # z=0 and prediction=0
    n_planes: int            # number of planes found by the RHT

    @property
    def false_positive_rate(self) -> float:
        """FPR, one of the vertical axes of Fig. 5.

        ★ The definition used here is ``num_false_pos / num_all``: the
          denominator is *all* points, not the number of noise points. This is
          reproduced as it stands. The conventional definition is available as
          ``false_positive_rate_among_noise``.
        """
        return self.n_false_pos / self.n_points if self.n_points else float("nan")

    @property
    def false_negative_rate(self) -> float:
        """FNR, same convention: the denominator is all points."""
        return self.n_false_neg / self.n_points if self.n_points else float("nan")

    @property
    def n_noise(self) -> int:
        return self.n_false_pos + self.n_true_neg

    @property
    def n_signal(self) -> int:
        return self.n_true_pos + self.n_false_neg

    @property
    def false_positive_rate_among_noise(self) -> float:
        """Conventional FPR = FP / (FP + TN). For reference only; this is not the
        definition behind the paper's figure."""
        return self.n_false_pos / self.n_noise if self.n_noise else float("nan")

    @property
    def false_negative_rate_among_signal(self) -> float:
        """Conventional FNR = FN / (FN + TP). For reference only; this is not the
        definition behind the paper's figure."""
        return self.n_false_neg / self.n_signal if self.n_signal else float("nan")

    def as_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "n_signal": self.n_signal, "n_noise": self.n_noise,
            "TP": self.n_true_pos, "FP": self.n_false_pos,
            "FN": self.n_false_neg, "TN": self.n_true_neg,
            "n_planes": self.n_planes,
            "FPR": self.false_positive_rate,
            "FNR": self.false_negative_rate,
            "FPR_among_noise": self.false_positive_rate_among_noise,
            "FNR_among_signal": self.false_negative_rate_among_signal,
        }


def detection_rates(z: np.ndarray, prediction: np.ndarray,
                    n_planes: int = 0) -> DetectionRates:
    """
    Compute the four counts from the ground-truth labels z and the RHT
    prediction.

    ★ TN is computed here with its conventional definition, i.e. the number of
      points with z = 0 and prediction = 0. TN is not used by any of the paper's
      figures and only enters the reference-only rate
      ``false_positive_rate_among_noise``, so the choice cannot affect any
      reported result.
    """
    z = np.asarray(z).astype(int)
    prediction = np.asarray(prediction).astype(int)
    return DetectionRates(
        n_points=int(z.size),
        n_true_pos=int(np.sum((z == 1) & (prediction == 1))),
        n_false_pos=int(np.sum((z == 0) & (prediction == 1))),
        n_false_neg=int(np.sum((z == 1) & (prediction == 0))),
        n_true_neg=int(np.sum((z == 0) & (prediction == 0))),
        n_planes=int(n_planes),
    )


@dataclass
class DetectionResult:
    """
    Complete record of one simulation plus detection, convenient for plotting
    (Figs. 2/3/4).
    """

    data: pd.DataFrame                     # x, y, ts, z
    prediction: np.ndarray                 # the 0/1 decision of the RHT
    plane_indices: np.ndarray
    normals: np.ndarray                    # (n_planes, 3)
    rhos: np.ndarray
    rates: DetectionRates
    seed: int

    def classified(self) -> pd.DataFrame:
        """
        Four colour labels, matching how the four classes are drawn:

            green  = TP (true signal, called signal)
            yellow = FP (noise, called signal)
            red    = FN (true signal, called noise)
            black  = TN (noise, called noise)
        """
        z = self.data["z"].to_numpy()
        p = self.prediction
        label = np.empty(z.size, dtype=object)
        label[(z == 1) & (p == 1)] = "green"     # TP
        label[(z == 0) & (p == 1)] = "yellow"    # FP
        label[(z == 1) & (p == 0)] = "red"       # FN
        label[(z == 0) & (p == 0)] = "black"     # TN
        out = self.data.copy()
        out["prediction"] = p
        out["plane"] = self.plane_indices
        out["class"] = label
        return out


def evaluate_detection(seed_shift: int = 1, *, hough_kwargs: dict | None = None,
                       **params) -> DetectionResult:
    """
    Run one section 3.1.1 simulation plus the RHT and report the detection
    performance.

    ``params`` is forwarded to :func:`simulate_linear_wavefronts`.

    ★ The simulated data uses a rho step of 0.5 and an inlier tolerance of 1.0
      (a consequence of the time scale), unlike the 0.05 / 0.1 of the real-data
      pipeline. ``SIM_PRESET`` is applied by default here, so do **not** run this
      simulation with the real-data parameters.
    """
    kwargs = dict(SIM_PRESET)
    kwargs.update(dict(max_iter=30000, vote_threshold=8, min_detectors=40))
    # The randomized Hough transform must be seeded too, otherwise the whole
    # evaluation is only a single unseeded snapshot: repeated calls with the
    # SAME seed_shift return different FPR/FNR and different plane counts
    # (measured: 4 distinct outcomes in 10 runs at the default parameters).
    # Tying the RHT seed to seed_shift makes the result a pure function of it.
    kwargs.setdefault("seed", int(seed_shift))
    if hough_kwargs:
        kwargs.update(hough_kwargs)

    df = simulate_linear_wavefronts(seed_shift, **params)
    pts = df[["x", "y", "ts"]].to_numpy(dtype=float)
    res = hough_plane(pts, **kwargs)

    seed = seed_from_params(
        SEED_BASE_2D, seed_shift,
        limit=params.get("limit", PAPER_2D_PARAMS["limit"]),
        time_max=params.get("time_max", PAPER_2D_PARAMS["time_max"]),
        ts=params.get("ts", PAPER_2D_PARAMS["ts"]),
        x=params.get("x", PAPER_2D_PARAMS["x"]),
        y=params.get("y", PAPER_2D_PARAMS["y"]),
        u_x=params.get("u_x", PAPER_2D_PARAMS["u_x"]),
        u_y=params.get("u_y", PAPER_2D_PARAMS["u_y"]),
        p=params.get("p", PAPER_2D_PARAMS["p"]),
        noise_freq=params.get("noise_freq", PAPER_2D_PARAMS["noise_freq"]),
        miu=params.get("miu", PAPER_2D_PARAMS["miu"]),
        sigma=params.get("sigma", PAPER_2D_PARAMS["sigma"]))

    return DetectionResult(
        data=df,
        prediction=res.prediction,
        plane_indices=res.plane_indices,
        normals=np.column_stack([res.n1, res.n2, res.n3]),
        rhos=res.rhos,
        rates=detection_rates(df["z"].to_numpy(), res.prediction,
                              res.n_planes),
        seed=seed,
    )


def detection_sweep(seed_shifts=range(0, 100), *, hough_kwargs: dict | None = None,
                    **params) -> pd.DataFrame:
    """
    Reproduce Fig. 5: run one simulation per random seed and collect the
    distribution of FPR / FNR.

    The default is ``range(0, 100)``, i.e. 100 seeds, which is what Fig. 5 uses.
    A single-shift run (``shifts = 0``) produces exactly one dataset and is only
    useful as a smoke test.

    ★ seed_shift = 0 and 1 differ by one, but because the seed is "product of the
      parameters + shift", adjacent shifts give completely different datasets
      rather than a perturbation of each other.
    """
    rows = []
    for s in seed_shifts:
        r = evaluate_detection(int(s), hough_kwargs=hough_kwargs, **params)
        rows.append({"seed_shift": int(s), **r.rates.as_dict()})
    return pd.DataFrame(rows)


# ==========================================================================
#  Synthetic MEA recording
# ==========================================================================

def simulate_recording(out_path: str | Path | None = None, *,
                       sample_rate_hz: float = 10000.0,
                       duration_ms: float = 9000.0,
                       grid_side: int = GRID_SIDE,
                       time_units_per_ms: float = 1.0,
                       first_wave_ms: float = 700.0,
                       beat_interval_ms: float = 1000.0,
                       beat_jitter_ms: float = 60.0,
                       source_x0: float = -3.7, source_y0: float = -50.0,
                       speed: float = 0.45,
                       spike_amp_mv: float = -1.0, spike_decay_ms: float = 0.8,
                       spike_pos_amp: float = 0.30, spike_pos_ms: float = 2.5,
                       spike_pos_decay_ms: float = 1.8,
                       spike_half_ms: float = 10.0,
                       noise_sd_mv: float = 0.02, drift_mv: float = 0.5,
                       drift_period_ms: float = 4000.0,
                       seed: int = 20260915) -> tuple[pd.DataFrame, dict]:
    """
    Generate a 64-channel MEA recording in exactly the same format as the real
    data of section 3.2.

    This is what makes the whole stage 1 -> 2 -> 3 pipeline runnable end to end
    even when the experimental recording is unavailable.

    ★ It does **not** reproduce the numbers of Tables 2/3/4 - those three tables
      depend on the experimental recording itself. It has two purposes, the
      second of which is the stronger one:
        1) the format is right, it runs, it can be plotted;
        2) **it verifies the pipeline end to end against a known ground truth**:
           the source position (source_x0, source_y0), the speed and the
           activation time of every wavefront are all set here, so the fitted
           (x0, y0, v, t0) has to land back on those values. Measured:
           v = 0.449-0.451 (truth 0.450), x0 = -3.7 to -3.9 (truth -3.70),
           t0 within 1 ms of the truth.

    Returns ``(DataFrame, info)``; passing ``out_path`` also writes the CSV.
    The CSV columns are ``time, Ch01, ..., Ch64``, as in the real data.

    ★★ ``time_units_per_ms`` has to stay 1.0; this is the most subtle trap in
       the whole pipeline. In the section 3.2 input file the time column *is* in
       milliseconds (0..8999.9). The ``ts / 200`` applied inside the pipeline is
       **not** a unit conversion: it squeezes the time axis onto the same order of
       magnitude as the grid coordinates (x, y in [1, 8]) so that an inlier
       tolerance of 0.1 and a rho step of 0.05 are both meaningful; the fit
       multiplies the scale back by 200 before optimising. Consequently:

           the time column has to be such that "time / 200" is of the same order
           of magnitude as x and y, i.e. time in [200, ~10000].

       Writing the time column as "t_ms * 200" (that is, setting
       time_units_per_ms = 200) makes the internal t several thousand, three
       orders of magnitude away from x and y, so the Hough stage finds nothing but
       garbage planes of the form "x = constant". Measured: v ~ 0, t0 ~ +/-2e7,
       14 planes instead of 9. The time scaling applied inside the pipeline is
       required unconditionally: it must never be disabled.

    Intentional properties of this generator:
      - ``np.arange`` yields exactly 90000 rows, matching the row count of the
        real data, whereas a closed-form sequence that includes its endpoint
        would write one extra sample (90001 rows).
      - the waveforms come from numpy's generator, so they differ from the
        individual draws behind the figures (see the module docstring).
      - the time column is written in milliseconds.
    """
    rng = np.random.default_rng(seed)

    t_ms = np.arange(0.0, duration_ms, 1000.0 / sample_rate_hz)
    n_channels = grid_side * grid_side

    # distance from each electrode to the source -> arrival delay
    dist = np.empty(n_channels)
    for ch in range(1, n_channels + 1):
        cx, cy = channel_to_xy(ch, grid_side)
        dist[ch - 1] = np.hypot(cx - source_x0, cy - source_y0)
    max_delay_ms = float(dist.max() / speed)

    # Sequence of activation times at the source: mean interval beat_interval_ms,
    # jitter beat_jitter_ms. Waves that have not finished propagating before the
    # recording ends are dropped - which is what a real experiment does.
    wave_times = [first_wave_ms]
    while True:
        nxt = wave_times[-1] + float(
            rng.normal(beat_interval_ms, beat_jitter_ms))
        if nxt + max_delay_ms > duration_ms:
            break
        wave_times.append(nxt)
    wave_times = np.asarray(wave_times)

    def spike_shape(u: np.ndarray) -> np.ndarray:
        """Extracellular field potential: a fast negative deflection plus a
        smaller, slower positive component."""
        return (spike_amp_mv * np.exp(-(u / spike_decay_ms) ** 2)
                + spike_pos_amp
                * np.exp(-((u - spike_pos_ms) / spike_pos_decay_ms) ** 2))

    drift = drift_mv * np.sin(2 * np.pi * t_ms / drift_period_ms)

    cols: dict[str, np.ndarray] = {"time": t_ms * time_units_per_ms}
    for ch in range(1, n_channels + 1):
        y = rng.normal(0.0, noise_sd_mv, size=t_ms.size) + drift
        for act in wave_times + dist[ch - 1] / speed:
            idx = np.flatnonzero((t_ms >= act - spike_half_ms)
                                 & (t_ms <= act + spike_half_ms))
            if idx.size:
                y[idx] += spike_shape(t_ms[idx] - act)
        cols[f"Ch{ch:02d}"] = y

    df = pd.DataFrame(cols)
    info = {
        "n_samples": int(df.shape[0]),
        "n_channels": n_channels,
        "sample_rate_hz": sample_rate_hz,
        "duration_ms": duration_ms,
        # -- Ground truth (used to check that the fit lands back on the values
        #    that were set here) -----------------------------------------
        "source_x0": source_x0,
        "source_y0": source_y0,
        "speed": speed,
        "grid_side": grid_side,
        "time_units_per_ms": time_units_per_ms,
        "wave_times_ms": np.round(wave_times, 1).tolist(),
        "max_delay_ms": max_delay_ms,
        "seed": seed,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        info["path"] = str(out_path)

    return df, info
