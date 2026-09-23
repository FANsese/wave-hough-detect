"""
Tests for the circular-wavefront simulation and the accuracy evaluation of
§3.1.2 (Figures 6-8 of the paper).

Kept separate from ``test_simulate.py`` (§3.1.1): that file covers the 8x8 linear
wavefronts with FPR/FNR, this one covers the 96x96 circular wavefronts with
parameter-estimation accuracy, and the two sampling structures differ.

The strongest test here is ``test_fit_recovers_truth_without_noise``: the
simulated data are built as ``t = sqrt((x-48)^2 + (y-48)^2) + 2``, and once the
measurement error and the noise points are switched off the fit must recover
(48, 48, 1, 2) to within 1e-8.
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    CIRCULAR_TRUTH,
    PAPER_CIRCULAR_PARAMS,
    PLOT_MODES,
    accuracy_sweep,
    circular_loss,
    evaluate_circular,
    r_squared,
    seed_circular,
    simulate_circular_wavefronts,
)

NO_NOISE = dict(noise_freq=1e-9)      # lambda_n -> 0: not a single noise point is generated


# --- the simulator ---
def test_simulation_grid_and_labels():
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    assert list(df.columns) == ["x", "y", "ts", "z"]
    assert set(np.unique(df["z"])) <= {0, 1}
    assert df["x"].min() >= 1 and df["x"].max() <= 96
    assert df["y"].min() >= 1 and df["y"].max() <= 96
    assert np.allclose(df["x"], np.round(df["x"]))


def test_all_electrodes_receive_the_wavefront_when_p_is_zero():
    """
    With p = 0 and no noise, **every** one of the 96x96 = 9216 electrodes should
    receive the wavefront once.
    This is the basis of the ground-truth conversion: n_signal = 9216 is what
    makes snr = n_signal / n_noise approximately 96 / lambda_n, which matches the
    published range.
    """
    df = simulate_circular_wavefronts(0, **{**PAPER_CIRCULAR_PARAMS, **NO_NOISE})
    assert int((df["z"] == 1).sum()) == 96 * 96
    assert int((df["z"] == 0).sum()) == 0


def test_arrival_times_match_the_analytic_model():
    """
    ★ The core correctness check of the simulation: the ts of every signal point
      must equal ``t0 + sqrt((x-x0)^2 + (y-y0)^2) / v`` (sigma = 1e-6, tolerance
      relaxed to 5e-5).
    """
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    sig = df[df["z"] == 1]
    t = CIRCULAR_TRUTH
    expect = t["t0"] + np.hypot(sig["x"] - t["x0"], sig["y"] - t["y0"]) / t["v"]
    assert np.abs(sig["ts"].to_numpy() - expect.to_numpy()).max() < 5e-5


def test_noise_points_never_exceed_the_observation_window():
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    assert df["ts"].max() < PAPER_CIRCULAR_PARAMS["time_max"]
    # the time distributions of signal and noise should differ clearly: the signal is clustered, the noise fills the window
    sig_ts = df.loc[df["z"] == 1, "ts"]
    assert sig_ts.min() >= PAPER_CIRCULAR_PARAMS["ts"] - 1e-3


def test_rows_are_sorted_by_time():
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    assert df["ts"].is_monotonic_increasing


def test_missing_probability_thins_the_signal():
    n = []
    for p in (0.0, 0.3, 0.9):
        df = simulate_circular_wavefronts(1, **{**PAPER_CIRCULAR_PARAMS,
                                                **NO_NOISE, "p": p})
        n.append(int((df["z"] == 1).sum()))
    assert n[0] > n[1] > n[2]
    assert n[0] == 96 * 96


def test_noise_rate_controls_the_number_of_noise_points():
    few = simulate_circular_wavefronts(0, **{**PAPER_CIRCULAR_PARAMS,
                                             "noise_freq": 0.1})
    many = simulate_circular_wavefronts(0, **{**PAPER_CIRCULAR_PARAMS,
                                              "noise_freq": 10.0})
    assert (many["z"] == 0).sum() > (few["z"] == 0).sum() * 10


def test_simulation_is_reproducible_and_seed_shifted():
    a = simulate_circular_wavefronts(3, **PAPER_CIRCULAR_PARAMS)
    b = simulate_circular_wavefronts(3, **PAPER_CIRCULAR_PARAMS)
    c = simulate_circular_wavefronts(4, **PAPER_CIRCULAR_PARAMS)
    assert a.equals(b)
    assert not a.equals(c)


# --- the seed: p is a factor of the product ---
def test_seed_arithmetic_formula():
    """``100 * product of the parameters``. p is one factor of the product."""
    kw = dict(limit=96, time_max=96.0, ts=2.0, x=48.0, y=48.0, u_x=1.0, u_y=1.0,
              p=0.0, noise_freq=1.0, miu_x=1.0, miu_y=1.0, sigma=1e-6)
    assert seed_circular(**kw) == 0          # p = 0 makes the whole product 0
    kw["p"] = 0.5
    assert seed_circular(**kw) == int(100 * 96 * 96 * 2 * 48 * 48 * 0.5 * 1e-6)
    assert seed_circular(**{**kw, "seed_shift": 7}) == seed_circular(**kw) + 7


def test_p_zero_makes_every_replicate_identical_without_a_seed_shift():
    """
    ★★ Worth pinning down on its own: the seed formula contains p as a factor, and
       modes 1 and 3 both fix p = 0, so **the seed is identically 0**. Since the
       optimality table calls the simulator repeatedly with the same parameters,
       all those "replicates" then receive the same data set - the error bars do
       not represent the spread between replicates.

       This implementation adds a seed_shift per replicate by default to remove
       that degeneracy; ``independent_replicates=False`` restores the degenerate
       behaviour.
    """
    from wave_hough_detect.simulate_circular import simulate_circular_wavefronts as sim

    # no seed shift: all replicates are exactly identical
    a = sim(0, **{**PAPER_CIRCULAR_PARAMS, "sigma": 0.3})
    b = sim(0, **{**PAPER_CIRCULAR_PARAMS, "sigma": 0.3})
    assert a.equals(b)

    # independent replicates: a different seed_shift gives different data
    c = sim(1, **{**PAPER_CIRCULAR_PARAMS, "sigma": 0.3})
    assert not a.equals(c)


def test_accuracy_sweep_replicate_independence_flag():
    """The two replicate modes must give different results; with a shared random stream the variance between replicates is 0."""
    indep = accuracy_sweep(3, n_replicates=3, independent_replicates=True)
    shared = accuracy_sweep(3, n_replicates=3, independent_replicates=False)
    v_indep = indep.groupby("scan_value")["x0"].std(ddof=1).to_numpy()
    v_shared = shared.groupby("scan_value")["x0"].std(ddof=1).to_numpy()
    assert np.nanmax(v_shared) < 1e-12, "replicates sharing a random stream should not differ at all"
    assert v_indep.max() > 0


# --- fitting and ground truth ---
def test_fit_recovers_truth_without_noise():
    """
    ★★ The hardest check at this level.

    With both the measurement error (sigma = 0) and the noise points
    (lambda_n -> 0) switched off, the data lie exactly on
        t = sqrt((x-48)^2 + (y-48)^2)/1 + 2
    and the fit must recover all four parameters.

    This also explains why the default initial guess (u, t0) = (1, 1) is a good
    one here: the truth is v = 1 (i.e. slowness u = 1) and t0 = 2, so the start
    is almost exactly on target. The "converges to the wrong minimum when the
    source is inside the grid" problem of §3.1.1 therefore does not appear in
    §3.1.2.
    """
    est = evaluate_circular(1, sigma=0.0, **NO_NOISE)
    t = CIRCULAR_TRUTH
    assert est.x0 == pytest.approx(t["x0"], abs=1e-8)
    assert est.y0 == pytest.approx(t["y0"], abs=1e-8)
    assert est.v == pytest.approx(t["v"], abs=1e-8)
    assert est.t0 == pytest.approx(t["t0"], abs=1e-8)
    assert est.n_noise == 0


def test_measurement_error_propagates_linearly():
    """
    sigma must really enter the result, and the error must grow roughly linearly
    with sigma. This is also the evidence that the sigma sweep is not decoration.
    """
    errs = []
    for s in (0.1, 0.5, 1.0, 2.0):
        est = evaluate_circular(1, sigma=s, **NO_NOISE)
        errs.append(abs(est.x0 - CIRCULAR_TRUTH["x0"]))
    assert all(b > a for a, b in zip(errs, errs[1:])), errs
    # linear -> the ratio between the extremes should be close to the ratio of the sigmas
    assert errs[-1] / errs[0] == pytest.approx(20.0, rel=0.35)


def test_noise_points_dominate_the_absolute_error():
    """
    ★ A finding about the setup of the sweep (a consequence of the parameter
      choice, not of the implementation):

      Mode 3 sweeps sigma but at the same time fixes lambda_n = 1 (about 100 noise
      spikes). Those 100 outliers contribute far more to the loss than the
      measurement error itself, so over sigma = 0.1 ... 1.0 the estimation error
      barely moves - that curve in fact measures outlier contamination, not
      measurement error.

      The two effects are measured separately here: with noise the influence of
      sigma is swamped, without noise it shows up immediately.
    """
    with_noise = [evaluate_circular(1, sigma=s, noise_freq=1.0).rel_error["x0"]
                  for s in (0.0, 1.0)]
    no_noise = [evaluate_circular(1, sigma=s, **NO_NOISE).rel_error["x0"]
                for s in (0.0, 1.0)]

    # with noise: sigma from 0 to 1 leaves the error essentially unchanged (by less than a factor of 2)
    assert with_noise[1] / max(with_noise[0], 1e-12) < 2.0
    # without noise: the error grows clearly with sigma
    assert no_noise[1] > no_noise[0] * 10


def test_r_squared_is_computed_over_signal_and_noise():
    """
    ★ The R2 definition trap: the whole matrix, noise rows included, is handed to
      the fit, so R2 computed against the true parameters is only about 0.95
      rather than 1. Computed over the signal points alone it is 1.0000000000.

      => Do not compare this 0.95 with the 0.94 of Table 4 in §3.2 of the paper;
      that is a different quantity.
    """
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    pts = df[["x", "y", "ts"]].to_numpy(dtype=float)
    truth_p = np.array([48.0, 48.0, 1.0, 2.0])       # (x0, y0, u, t0), u = 1/v

    r2_all = r_squared(circular_loss, truth_p, pts)
    r2_sig = r_squared(circular_loss, truth_p, pts[df["z"].to_numpy() == 1])
    assert r2_sig == pytest.approx(1.0, abs=1e-9)
    assert 0.90 < r2_all < 0.99
    assert r2_all < r2_sig


# --- definition of the SNR ---
def test_snr_is_the_count_ratio_not_an_amplitude_ratio():
    """
    ★ The horizontal axis of Figure 7: ``n_signal / n_noise``.
      Despite the name it is a ratio of **counts** and has nothing to do with
      amplitude.
    """
    est = evaluate_circular(1)
    assert est.snr == pytest.approx(est.n_signal / est.n_noise)
    assert est.n_signal == 96 * 96


def test_snr_of_the_mode1_grid_matches_the_published_range():
    """
    The published snr range is [~0.375, ~768]. Under the §3.1.2 configuration
    n_signal = 9216 and n_noise ~ lambda_n * 96, so snr ~ 96 / lambda_n.
    The lambda_n grid of mode 1 is 2^((-6:16)/2) in [0.125, 256] -> [0.375, 768].
    """
    grid = PLOT_MODES[1]["grid"]
    assert grid.min() == pytest.approx(0.125)
    assert grid.max() == pytest.approx(256.0)
    assert 96 / grid.max() == pytest.approx(0.375, rel=1e-6)
    assert 96 / grid.min() == pytest.approx(768.0, rel=1e-6)


# --- sweeps ---
def test_accuracy_sweep_shape_and_columns():
    df = accuracy_sweep(2, n_replicates=2)
    assert len(df) == len(PLOT_MODES[2]["grid"]) * 2
    for col in ("plot_mode", "variable", "scan_value", "replicate", "x0", "y0",
                "v", "t0", "snr", "n_signal", "n_noise",
                "abs_x0", "rel_t0"):
        assert col in df.columns
    assert (df["plot_mode"] == 2).all()
    assert (df["variable"] == "p").all()


def test_accuracy_sweep_rejects_unknown_mode():
    with pytest.raises(ValueError):
        accuracy_sweep(99)


def test_source_position_is_more_robust_than_speed():
    """
    The degradation that Figure 7 is meant to show: as the noise grows, (x0, y0)
    stays stable while v and t0 drift together - the fit accommodates the outliers
    by slowing the wave down and delaying the excitation.
    """
    df = accuracy_sweep(1, n_replicates=2)
    lo = df[df["scan_value"] == df["scan_value"].min()]
    hi = df[df["scan_value"] == df["scan_value"].max()]
    assert hi["rel_v"].mean() > lo["rel_v"].mean() * 20
    assert hi["rel_x0"].mean() < 0.05


# -- regressions found by review ------------------------------------------

def test_evaluate_detection_is_reproducible():
    """
    The randomized transform must be seeded from ``seed_shift``; otherwise the
    whole evaluation is one unseeded snapshot and repeated calls disagree.

    Measured before the fix: 10 calls at the default parameters gave 4 distinct
    (n_planes, n_signal) outcomes, and three calls with seed_shift=2 gave FPR
    0.0694 / 0.3021 / 0.2569.
    """
    import numpy as np
    from wave_hough_detect import evaluate_detection

    runs = [evaluate_detection(2) for _ in range(3)]
    keys = {(r.rates.false_positive_rate, r.rates.false_negative_rate,
             r.rates.n_planes) for r in runs}
    assert len(keys) == 1, f"evaluate_detection is not reproducible: {sorted(keys)}"

    a = evaluate_detection(2).rates.as_dict()
    b = evaluate_detection(2).rates.as_dict()
    assert a == b


def test_evaluate_detection_differs_between_seed_shifts():
    """Seeding must not collapse every dataset onto the same result."""
    from wave_hough_detect import evaluate_detection
    a = evaluate_detection(1).rates
    b = evaluate_detection(2).rates
    assert (a.false_positive_rate, a.n_false_pos, a.n_false_neg) != \
           (b.false_positive_rate, b.n_false_pos, b.n_false_neg)


def test_ground_truth_follows_the_simulated_parameters():
    """
    Overriding the source position or the excitation time must move the ground
    truth with it. Before the fix the estimate was scored against the published
    defaults, overstating the error 142x for a source moved to (40, 30).
    """
    from wave_hough_detect import evaluate_circular
    est = evaluate_circular(1, x=40.0, y=30.0, ts=5.0)
    assert est.truth == {"x0": 40.0, "y0": 30.0, "v": 1.0, "t0": 5.0}
    assert est.abs_error["x0"] == pytest.approx(abs(est.x0 - 40.0), abs=1e-12)
    assert est.abs_error["t0"] == pytest.approx(abs(est.t0 - 5.0), abs=1e-12)
    # and the defaults are unchanged
    assert evaluate_circular(1).truth == {"x0": 48.0, "y0": 48.0,
                                          "v": 1.0, "t0": 2.0}


def test_sigma_sweep_holds_the_noise_rate_near_zero():
    """
    With noise spikes present their residuals dominate the objective and the
    sigma trend disappears: measured, |dx0| moves by under 2x across
    sigma = 0.1..1.0 at lambda_n = 1, and by 10x (proportional to sigma) at
    lambda_n = 1e-6. The sweep must therefore hold the noise rate near zero.
    """
    from wave_hough_detect import accuracy_sweep
    df = accuracy_sweep(3, n_replicates=2)
    assert (df["n_noise"] == 0).all(), "the sigma sweep must run without noise spikes"
    g = df.groupby("scan_value")["abs_x0"].mean()
    lo, hi = g.iloc[0], g.iloc[-1]
    assert hi / lo > 8.0, f"sigma trend not recovered: {lo:.5f} -> {hi:.5f}"


def test_accuracy_plot_can_show_error_bars():
    """
    Grouping must be by the swept parameter. Grouping by the realised snr gave
    one sample per group and every error bar was NaN.
    """
    from wave_hough_detect import accuracy_sweep
    df = accuracy_sweep(1, n_replicates=2)
    by_snr = df.groupby("snr")["x0"].std(ddof=1)
    by_param = df.groupby("scan_value")["x0"].std(ddof=1)
    assert by_snr.isna().all(), "grouping by snr is the broken behaviour"
    assert not by_param.isna().any()
    assert (df.groupby("scan_value").size() > 1).all()
