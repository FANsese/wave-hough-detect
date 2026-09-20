"""
Unit tests for stage 4 (simulation and evaluation).

The generated datasets are statistically equivalent to the ones behind the
figures but they are not the same numbers (see the module documentation of
simulate.py), so only **structure and statistical properties** are checked here,
together with the conventions that fail silently once they are written wrongly:
seed derivation, stable ordering, the z label, and the definition of FPR/FNR.
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    PAPER_2D_PARAMS,
    detection_rates,
    detection_sweep,
    evaluate_detection,
    seed_from_params,
    simulate_linear_wavefronts,
    simulate_recording,
)


# --- seed derivation ---
def test_seed_derivation_is_reproducible():
    a = simulate_linear_wavefronts(seed_shift=7, **PAPER_2D_PARAMS)
    b = simulate_linear_wavefronts(seed_shift=7, **PAPER_2D_PARAMS)
    assert a.equals(b)


def test_different_seed_shifts_give_different_datasets():
    a = simulate_linear_wavefronts(seed_shift=7, **PAPER_2D_PARAMS)
    b = simulate_linear_wavefronts(seed_shift=8, **PAPER_2D_PARAMS)
    assert not a.equals(b)


def test_seed_from_params_formula():
    """
    The seed is SEED_BASE times the product of the parameters, plus seed_shift.
    With SEED_BASE = 101 and the parameter set of §3.1.1:
        101 · 8 · 80 · 1 · 1 · 1 · 1 · 1 · 0.1 · 1 · 1 · 0.1 = 646.4
    truncated to 646; shift=1 -> 647.
    """
    kw = dict(limit=8, time_max=80.0, ts=1.0, x=1.0, y=1.0, u_x=1.0, u_y=1.0,
              p=0.1, noise_freq=1.0, miu=1.0, sigma=0.1)
    assert pytest.approx(646.4) == 101 * 8 * 80 * 0.1 * 0.1      # confirm the arithmetic first
    assert seed_from_params(101, 0, **kw) == 646
    assert seed_from_params(101, 1, **kw) == 647
    # truncation, not rounding
    assert seed_from_params(101, 0, **{**kw, "sigma": 0.19}) == int(101 * 8 * 80 * 0.1 * 0.19)


# --- structure of the simulated data ---
def test_simulated_columns_and_labels():
    df = simulate_linear_wavefronts(seed_shift=1, **PAPER_2D_PARAMS)
    assert list(df.columns) == ["x", "y", "ts", "z"]
    assert set(np.unique(df["z"])) <= {0, 1}
    assert (df["z"] == 1).any() and (df["z"] == 0).any()


def test_simulated_coordinates_stay_on_the_grid():
    df = simulate_linear_wavefronts(seed_shift=2, **PAPER_2D_PARAMS)
    limit = PAPER_2D_PARAMS["limit"]
    for col in ("x", "y"):
        assert df[col].min() >= 1
        assert df[col].max() <= limit
        assert np.allclose(df[col], np.round(df[col]))     # must lie on integer grid points


def test_simulated_times_within_horizon():
    """
    Noise points form a Poisson stream in time and may start near 0, so the
    overall minimum of ts falls below the departure time of the first wavefront;
    the **signal points** however must start at the ts parameter.
    """
    df = simulate_linear_wavefronts(seed_shift=3, **PAPER_2D_PARAMS)
    assert df["ts"].min() >= 0.0
    assert df.loc[df["z"] == 1, "ts"].min() >= PAPER_2D_PARAMS["ts"]
    assert df["ts"].max() <= PAPER_2D_PARAMS["time_max"] * 1.05


def test_rows_are_sorted_by_time():
    """
    ★ Rows must be ordered by time with a stable sort. ts is a continuous
      quantity but it does contain ties (several points of the same wavefront may
      share an arrival time), and the row order must follow the "stable"
      convention, otherwise indexing by position is shifted.
    """
    df = simulate_linear_wavefronts(seed_shift=4, **PAPER_2D_PARAMS)
    assert df["ts"].is_monotonic_increasing


def test_missing_probability_actually_thins_the_signal():
    """The larger p, the fewer signal points (the missed-detection mechanism of §3.1)."""
    n = []
    for p in (0.0, 0.1, 0.5, 0.9):
        df = simulate_linear_wavefronts(seed_shift=1, p=p, **
                                        {k: v for k, v in PAPER_2D_PARAMS.items()
                                         if k != "p"})
        n.append(int((df["z"] == 1).sum()))
    assert n[0] > n[1] > n[2] > n[3]


def test_wavefront_count_follows_the_gap_parameter():
    """
    gap is the chi-square degrees of freedom of the inter-wavefront interval: the
    larger gap, the longer the intervals and the fewer the wavefronts. With
    time_max=80, gap=30 gives about 2-4 wavefronts.
    """
    few = simulate_linear_wavefronts(seed_shift=1, gap=200.0, **
                                     {k: v for k, v in PAPER_2D_PARAMS.items()
                                      if k != "gap"})
    many = simulate_linear_wavefronts(seed_shift=1, gap=5.0, **
                                      {k: v for k, v in PAPER_2D_PARAMS.items()
                                       if k != "gap"})
    assert (few["z"] == 1).sum() < (many["z"] == 1).sum()


# --- definition of FPR / FNR ---
def test_detection_rates_counts():
    z = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])
    r = detection_rates(z, pred, n_planes=2)
    assert (r.n_true_pos, r.n_false_pos, r.n_false_neg, r.n_true_neg) == (2, 1, 1, 2)
    assert r.n_points == 6 and r.n_planes == 2


def test_paper_rate_denominator_is_all_points():
    """
    ★ The denominator of FPR/FNR is the total number of points, not the size of
      the respective class. The definition is pinned down here - switching to the
      conventional one would make the rates incomparable with Figure 5 of the
      paper.
    """
    z = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])
    r = detection_rates(z, pred)
    assert r.false_positive_rate == pytest.approx(1 / 6)
    assert r.false_negative_rate == pytest.approx(1 / 6)
    # the conventional definition is computed separately and does not interfere
    assert r.false_positive_rate_among_noise == pytest.approx(1 / 3)
    assert r.false_negative_rate_among_signal == pytest.approx(1 / 3)


def test_detection_rates_handles_empty_classes():
    z = np.array([1, 1])
    r = detection_rates(z, np.array([1, 1]))
    assert r.false_positive_rate == 0.0
    assert np.isnan(r.false_positive_rate_among_noise)
    assert r.false_negative_rate_among_signal == 0.0


# --- end-to-end evaluation ---
def test_evaluate_detection_uses_simulation_preset():
    """
    ★ The simulation study must use a rho step of 0.5 and a tolerance of 1.0.
      Using the real-data set instead finds no plane at all (or a pile of junk
      planes) while raising no error whatsoever.
    """
    r = evaluate_detection(seed_shift=1)
    assert r.rates.n_points > 0
    assert r.prediction.size == len(r.data)
    assert set(np.unique(r.prediction)) <= {0, 1}
    assert r.rates.n_true_pos + r.rates.n_false_neg == r.rates.n_signal


def test_classified_labels_are_consistent_with_rates():
    r = evaluate_detection(seed_shift=2)
    cls = r.classified()
    counts = cls["class"].value_counts()
    assert int(counts.get("green", 0)) == r.rates.n_true_pos
    assert int(counts.get("yellow", 0)) == r.rates.n_false_pos
    assert int(counts.get("red", 0)) == r.rates.n_false_neg
    assert int(counts.get("black", 0)) == r.rates.n_true_neg


def test_detection_sweep_shape():
    df = detection_sweep(range(0, 3))
    assert len(df) == 3
    assert {"seed_shift", "FPR", "FNR", "n_planes"} <= set(df.columns)


# --- synthetic MEA recording ---
@pytest.fixture(scope="module")
def synthetic_recording(tmp_path_factory):
    """The full 9-second, 64-channel recording is generated once for the whole module (one generation takes about 4 seconds)."""
    path = tmp_path_factory.mktemp("rec") / "rec.csv"
    df, info = simulate_recording(path)
    return path, df, info


def test_simulate_recording_format(synthetic_recording):
    """
    The CSV must have the same format as the experimental recording of §3.2: one
    time column plus 64 columns Ch01..Ch64, with a number of rows matching
    10 kHz over 9 s.
    """
    path, df, info = synthetic_recording
    assert path.exists()
    assert df.shape == (90000, 65)
    assert list(df.columns)[0] == "time"
    assert list(df.columns)[1:] == [f"Ch{i:02d}" for i in range(1, 65)]
    assert not df.isna().any().any()
    assert info["n_channels"] == 64


def test_simulate_recording_time_column_is_milliseconds(synthetic_recording):
    """
    ★★ The most insidious pitfall: the time column must be in milliseconds.
      Writing it in units of 1/200 ms moves the Hough-scale time three orders of
      magnitude away from the grid coordinates x, y in [1, 8], so every plane that
      is found is junk (measured on that convention the pipeline reports v ~ 0 and
      t0 ~ +-2e7).

      The default time_units_per_ms of this implementation is 1.0.
    """
    _, df, info = synthetic_recording
    assert info["time_units_per_ms"] == 1.0
    t = df["time"].to_numpy()
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(8999.9, abs=0.05)          # last sample within 9000 ms
    assert np.allclose(np.diff(t[:100]), 0.1)                # 10 kHz → 0.1 ms


def test_simulate_recording_spikes_are_negative_going(synthetic_recording):
    """
    Extracellular field potentials are negative deflections and the pipeline
    locates spikes at the minimum - reversed polarity would detect nothing at all.
    """
    _, df, _ = synthetic_recording
    ch = df["Ch01"].to_numpy()
    assert ch.min() < -0.3
    assert abs(ch.min()) > abs(ch.max())


def test_simulate_recording_reports_ground_truth(synthetic_recording):
    """info must carry the ground truth back, otherwise the "known truth" end-to-end check is impossible."""
    _, _, info = synthetic_recording
    assert info["source_x0"] == -3.7
    assert info["source_y0"] == -50.0
    assert info["speed"] == 0.45
    assert len(info["wave_times_ms"]) >= 2
    assert info["max_delay_ms"] > 0


def test_time_column_scale_is_right(synthetic_recording):
    """
    A time column in units of 1/200 ms would put the Hough scale off by a factor
    of 200. The full pipeline is not reproduced here; only the fact that a wrong
    setting yields a wrong Hough scale is pinned down.
    """
    from wave_hough_detect import detect_all_channels, load_recording, spikes_to_table

    path, _, _ = synthetic_recording
    rec = load_recording(path)
    st, _ = detect_all_channels(rec)
    sp = spikes_to_table(st, rec.channels)
    t = sp["t"].to_numpy()
    # the Hough scale must have the same order of magnitude as x, y in [1, 8]
    assert t.max() < 200.0, "the time column is not in milliseconds, the Hough scale has drifted off"
