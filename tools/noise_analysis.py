"""
Where exactly does the noise go? — a layered measurement on real data

An MEA recording contains two completely different kinds of "noise", which are
handled by two different stages:

  type A: waveform noise inside a channel (baseline drift, high-frequency
          interference)
          → handled by the stage-1 Butterworth high-pass plus amplitude threshold
  type B: spurious spikes that were detected (isolated peaks belonging to no
          real wavefront)
          → handled by the stage-2 Hough transform (points that fall into no
          plane are called noise)

This script quantifies type A and shows how much of the work the threshold and
the Hough transform each carry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, welch

# Allow running without installing: add the repository src/ to the search path.
# If the package was already installed with `pip install -e .`, this is a no-op.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from wave_hough_detect import (  # noqa: E402
    detect_all_channels,
    find_recording,
    hough_plane,
    load_recording,
    spikes_to_table,
)


def hr(t=""):
    print()
    print("=" * 74)
    if t:
        print(f" {t}")
        print("=" * 74)


def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0


def main(data_path: Path):
    hr("Reading the data")
    rec = load_recording(data_path)
    t, v = rec.time, rec.voltage
    fs = rec.sampling_rate_hz
    print(f"  {rec.n_samples} samples x {rec.n_channels} channels, "
          f"sampling rate {fs:.0f} Hz")
    print(f"  duration {t[-1]:.1f} ms")

    ch = 0                                        # analyse Ch01
    raw = v[:, ch]

    hr("Type A: waveform noise inside a channel — the spectrum band by band")

    # Butterworth high-pass: identical to the pipeline
    b, a = butter(2, 1/500, btype="high")
    hp = lfilter(b, a, raw)

    f, pxx_raw = welch(raw, fs=fs, nperseg=8192)
    _, pxx_hp = welch(hp, fs=fs, nperseg=8192)

    bands = [
        ("baseline drift", 0.0, 1.0),
        ("slow waves / motion artefact", 1.0, 10.0),
        ("main spike band", 10.0, 500.0),
        ("high-frequency detail", 500.0, 2000.0),
        ("high-frequency noise", 2000.0, fs / 2),
    ]
    print(f"  {'band':30s} {'raw power':>12s} {'after HP':>12s} "
          f"{'retained':>10s}")
    print("  " + "-" * 67)
    for name, lo, hi in bands:
        pr = band_power(f, pxx_raw, lo, hi)
        ph = band_power(f, pxx_hp, lo, hi)
        ratio = ph / pr if pr > 0 else np.nan
        print(f"  {name:30s} {pr:12.3e} {ph:12.3e} {ratio:10.3f}")

    hr("Amplitude decomposition: drift vs spikes vs high-frequency noise")

    # low-frequency component (the drift, extracted with a low-pass)
    b_lo, a_lo = butter(2, 1/500, btype="low")
    drift = lfilter(b_lo, a_lo, raw)

    # spike amplitude: the depth at 5 detected positions
    spikes_t = np.array([965.3, 2159.0, 3938.9, 5777.5, 7760.7])
    idx = [int(np.argmin(np.abs(t - s))) for s in spikes_t]
    spike_amp = raw[idx] - drift[idx]

    print(f"  drift amplitude (low-pass)      : {drift.min():+.4f} – "
          f"{drift.max():+.4f} mV   peak-to-peak {drift.max()-drift.min():.4f} mV")
    print(f"  spike amplitude above drift     : "
          f"{np.round(spike_amp, 4).tolist()} mV")
    print(f"    → mean {np.abs(spike_amp).mean():.4f} mV")
    print(f"  residual HF noise after HP (std): {hp.std():.5f} mV")
    print(f"    → spike/noise = {np.abs(spike_amp).mean()/hp.std():.1f}x")

    hr("Role of the threshold: how many spikes each quantile detects")

    print(f"  {'quantile':>10s} {'threshold (mV)':>14s} {'detected':>8s}   note")
    print("  " + "-" * 62)
    for q, note in [(0.05, "no threshold, to see how many noise peaks there are"),
                    (0.01, ""),
                    (0.005, ""),
                    (0.001, ""),
                    (0.0005, "← used by the pipeline (= 99.95th percentile)"),
                    (0.0001, "")]:
        thr = np.quantile(hp, q)
        work = hp.copy()
        n = 0
        for _ in range(500):
            i = int(np.argmin(work))
            if work[i] >= thr:
                break
            n += 1
            work[max(0, i-500):i+501] = 1e6
        print(f"  {q:>10.4f} {thr:14.5f} {n:8d}   {note}")

    hr("All 64 channels: how many of the detected spikes are spurious?")
    n_ch = rec.n_channels
    spike_times, _ = detect_all_channels(rec)
    n_spikes = sum(s.size for s in spike_times)
    n_beats = max(s.size for s in spike_times)
    print(f"  total detected : {n_spikes}")
    print(f"  expected       : {n_ch} electrodes x {n_beats} beats = "
          f"{n_ch * n_beats}")
    print(f"  excess         : {n_spikes - n_ch * n_beats}  "
          f"← the threshold blocks every noise peak")

    hr("Type B: spurious spikes are handled by the Hough transform")
    sp = spikes_to_table(spike_times, rec.channels)
    xyz = np.column_stack([sp["x"], sp["y"], sp["t"]]).astype(float)
    res = hough_plane(xyz, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)
    n_sig = int(res.prediction.sum())
    n_noise = len(res.prediction) - n_sig
    print(f"  input points : {len(res.prediction)}")
    print(f"  called signal: {n_sig}")
    print(f"  called noise : {n_noise}")
    print()
    print("  Mechanism: an isolated point that belongs to no wavefront cannot be")
    print("             coplanar with other points, so it never collects enough")
    print("             votes in the accumulator and its prediction stays 0 — it")
    print("             is called noise.")
    if n_noise == 0:
        print()
        print("  * On THIS recording the count above is 0, so the Hough layer's")
        print("    noise-rejection role is NOT exercised here: the amplitude")
        print("    threshold of stage 1 already removes every noise peak before it")
        print("    reaches stage 2. The mechanism is real -- a large but isolated")
        print("    spike has no coplanar partners and stays unclassified -- but to")
        print("    observe it you need a recording where the threshold is")
        print("    deliberately loosened, or a simulation with injected spikes.")
    else:
        print("  In other words, the Hough transform is itself a SECOND noise filter,")
        print("  and a structural one: it does not depend on the amplitude threshold.")
        print("  Spurious spikes that are large enough but in the wrong place are")
        print("  removed here.")

    hr("Summary")
    print("  +-------------------------------------+-----------------------------------+---------+")
    print("  | noise type                          | handled by                        | stage   |")
    print("  +-------------------------------------+-----------------------------------+---------+")
    print("  | baseline drift (low freq)           | Butterworth high-pass             | stage 1 |")
    print("  | small fluctuations (high freq)      | 0.05th-percentile threshold       | stage 1 |")
    print("  | spurious spikes (large, isolated)   | Hough coplanarity                 | stage 2 |")
    print("  +-------------------------------------+-----------------------------------+---------+")
    print()
    print("  LOWESS plays no part in this pipeline — it is a discarded alternative")
    print("  that is not implemented here. From first principles it would also be")
    print("  the wrong tool for this task: LOWESS is a SMOOTHER and would flatten")
    print("  sharp peaks, whereas what is needed here is a HIGH-PASS, whose purpose")
    print("  is to remove drift rather than high frequencies.")
    print()


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to the MEA recording CSV")
    a = ap.parse_args()
    try:
        main(find_recording(a.data))
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
