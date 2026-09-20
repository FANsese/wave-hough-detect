"""
Reconciliation with the published numbers (Tables 2, 3 and 4 of the paper).

This section needs the **experimental recording** of §3.2, which comes from a
collaborating laboratory and is not distributed with the repository (see
``data/README.md``). Therefore:

  · when the recording cannot be found the whole group is skipped automatically,
    so pytest still comes out green on a clean clone;
  · dropping the recording into ``data/`` and setting ``WHD_DATA`` (or leaving it
    unset and relying on the default name) activates the group, which then checks
    Tables 2/3/4 cell by cell.

Verified passing (2026-09-18, this machine):
    all 20 numbers of Table 2 and all 10 R2 values of Table 4 agree with the
    paper to the printed number of digits.
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    build_result_array,
    detect_all_channels,
    find_recording,
    fit_circular,
    fit_linear,
    hough_plane,
    load_recording,
    plane_points,
    spikes_to_table,
)

# --- Table 2 of the paper: circular wavefront model (ordered by t0, matching the paper numbering) ---
PAPER_TABLE2 = [
    # t0,      x0,     y0,    v,     R2
    (849.75, -3.93, -50.0, 0.44, 0.9480720),
    (2054.29, 9.15, -50.0, 0.49, 0.9593685),
    (3828.32, -3.20, -50.0, 0.46, 0.9373304),
    (5667.08, -3.05, -50.0, 0.46, 0.9442630),
    (7647.11, -3.51, -50.0, 0.45, 0.9405785),
]

# --- Table 3 of the paper: linear wavefront model ---
PAPER_TABLE3 = [
    # a(tilde), b(tilde), v,     R2
    (0.34, 2.22, 0.44, 0.9426517),
    (-0.18, 2.03, 0.49, 0.9566555),
    (0.30, 2.14, 0.46, 0.9308701),
    (0.29, 2.13, 0.46, 0.9384128),
    (0.31, 2.19, 0.45, 0.9343370),
]


@pytest.fixture(scope="module")
def real_pipeline():
    try:
        path = find_recording(None)
    except FileNotFoundError:
        pytest.skip("the experimental recording of §3.2 was not found (it is not distributed with the repository, see data/README.md)")

    rec = load_recording(path)
    spike_times, _ = detect_all_channels(rec)
    sp = spikes_to_table(spike_times, rec.channels)
    pts = sp[["x", "y", "t"]].to_numpy(dtype=float)

    # ★ max_iter must be >= 200000: 30000 iterations find only 3 of the 5 planes
    res = hough_plane(pts, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)

    order = np.argsort(sp["t"].to_numpy(), kind="stable")
    arr = build_result_array(sp["x"].to_numpy()[order],
                             sp["y"].to_numpy()[order],
                             sp["t"].to_numpy()[order],
                             res.plane_indices[order], res.n_planes)

    fits = []
    for k in range(1, res.n_planes + 1):
        p = plane_points(arr, k)
        fits.append((k, fit_circular(p), fit_linear(p)))
    fits.sort(key=lambda f: f[1].t0)         # the paper numbers the planes by ascending t0
    return dict(path=path, rec=rec, res=res, fits=fits, n_spikes=len(sp))


def test_real_data_spike_count(real_pipeline):
    """§3.2 of the paper: 64 electrodes x 5 beats = 320 spikes."""
    assert real_pipeline["n_spikes"] == 320


def test_real_data_finds_five_planes(real_pipeline):
    assert real_pipeline["res"].n_planes == 5
    assert [int((real_pipeline["res"].plane_indices == k).sum())
            for k in range(1, 6)] == [64] * 5


def test_table2_circular_model(real_pipeline):
    """Table 2 of the paper: all 20 numbers checked cell by cell."""
    rows = real_pipeline["fits"]
    assert len(rows) == len(PAPER_TABLE2)
    for (t0, x0, y0, v, _), (_, c, _) in zip(PAPER_TABLE2, rows):
        # t0 is used for pairing (the paper numbering differs from the acceptance order)
        assert c.t0 == pytest.approx(t0, abs=0.02)
        assert c.x0 == pytest.approx(x0, abs=0.005)
        assert c.y0 == pytest.approx(y0, abs=0.005)
        assert c.v == pytest.approx(v, abs=0.005)


def test_table3_linear_model(real_pipeline):
    """
    Table 3 of the paper: a_tilde / b_tilde / v of the linear model.

    The tolerance is 0.006 rather than 0.005: Table 3 is printed to two decimals
    and the second decimal falls exactly on a rounding boundary, for instance
    b_tilde = 2.0249999999997605 is printed as 2.03 while Python's round gives
    2.02. The value 0.006 covers the uncertainty of "printed to two decimals"
    itself while still being a meaningful check.
    """
    rows = real_pipeline["fits"]
    for (a, b, v, _), (_, _, l) in zip(PAPER_TABLE3, rows):
        assert l.a == pytest.approx(a, abs=0.006)
        assert l.b == pytest.approx(b, abs=0.006)
        assert l.v == pytest.approx(v, abs=0.006)


def test_table4_coefficient_of_determination(real_pipeline):
    """
    Table 4 of the paper: the 10 R2 values (5 planes x 2 models),
    and the conclusion of the paper is that **the circular model is slightly
    better on every plane**.
    """
    rows = real_pipeline["fits"]
    for (_, _, _, _, r2_paper), (_, c, l) in zip(PAPER_TABLE2, rows):
        assert c.r2 == pytest.approx(r2_paper, abs=1e-6)
    for (_, _, _, r2_paper), (_, _, l) in zip(PAPER_TABLE3, rows):
        assert l.r2 == pytest.approx(r2_paper, abs=1e-6)
    # the conclusion of the paper: the circular model is better on 5/5 planes
    assert all(c.r2 > l.r2 for _, c, l in rows)


def test_paper_conclusion_source_lies_outside_the_search_window(real_pipeline):
    """
    One of the conclusions of §3.2: the y0 values given by the circular model all
    sit on the boundary -50 of the search window, which shows that the true source
    lies outside the search window and that the linear model should be used
    instead. This conclusion is pinned down here.
    """
    y0 = np.array([c.y0 for _, c, _ in real_pipeline["fits"]])
    assert np.allclose(y0, -50.0, atol=0.01)
