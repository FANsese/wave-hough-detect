"""
Stage 1: spike detection (section 2.1 / Algorithm 1)

Three points that are easy to get wrong (each marked with a star below):
  ★1  The channel filter is a one-directional IIR filter, not a zero-phase
      forward-backward filter such as ``scipy.signal.filtfilt``.
  ★2  The spike threshold uses type-7 linear-interpolation quantiles, i.e.
      ``np.quantile(..., method="linear")``.
  ★3  ``np.argmin`` returns the first index among ties, and the detector
      depends on exactly that tie-breaking rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

# -- Fixed parameters of the pipeline --------------------------------------
GRID_SIDE = 8                 # 8 x 8 MEA
BUTTER_ORDER = 2              # filter order n
BUTTER_CUTOFF = 1 / 500       # W = fraction of the Nyquist frequency
SPIKE_QUANTILE = 0.0005       # 0.05th percentile (= upper-tail 99.95th percentile)
EXCLUSION_HALF = 500          # samples blanked on either side after a detection
MAX_SPIKES_PER_CHANNEL = 200  # upper bound on the number of detections per channel
TIME_SCALE = 200              # divisor applied to spike times before the Hough stage


def channel_to_xy(ch: int, side: int = GRID_SIDE) -> tuple[int, int]:
    """
    Channel number -> (x, y). Row-major: x is the row, y is the column.

    >>> channel_to_xy(1), channel_to_xy(8), channel_to_xy(9), channel_to_xy(64)
    ((1, 1), (1, 8), (2, 1), (8, 8))
    """
    if ch % side == 0:
        return ch // side, side
    return ch // side + 1, ch % side


@dataclass
class Recording:
    """One MEA recording."""

    time: np.ndarray          # (n_samples,) time axis in ms
    voltage: np.ndarray       # (n_samples, 64) voltage in mV
    channels: list[str]       # column names, of the form ["Ch01", ..., "Ch64"]
    sampling_rate_hz: float

    @property
    def n_samples(self) -> int:
        return self.time.size

    @property
    def n_channels(self) -> int:
        return self.voltage.shape[1]


def load_recording(path: str | Path) -> Recording:
    """
    Read a CSV export of an MEA recording.

    The raw column names look like ``T(ms), CH1(mV), ..., CH64(mV)``. Parsing a
    channel number out of such a name by taking the two characters starting at
    offset 3 is impossible: for "CH1(mV)" those characters are "1(", which is not
    a number, so the channel index would come out missing and every downstream
    indexing operation would fail. The columns are therefore renamed uniformly to
    ``Ch01..Ch64``, which makes the two characters starting at offset 3 exactly
    the two-digit channel index ("01".."64").
    """
    df = pd.read_csv(path)
    time = df.iloc[:, 0].to_numpy(dtype=float)
    voltage = df.iloc[:, 1:].to_numpy(dtype=float)

    n_ch = voltage.shape[1]
    channels = [f"Ch{i:02d}" for i in range(1, n_ch + 1)]

    # sampling rate inferred from the time-axis step (raw data: 0.1 ms -> 10 kHz)
    dt = float(np.median(np.diff(time)))
    rate = 1000.0 / dt if dt > 0 else np.nan

    return Recording(time=time, voltage=voltage, channels=channels,
                     sampling_rate_hz=rate)


def butter_highpass(order: int = BUTTER_ORDER,
                    cutoff: float = BUTTER_CUTOFF) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a Butterworth high-pass filter.

    Note that ``cutoff`` is a fraction of the Nyquist frequency, not a frequency
    in Hz. At 10 kHz sampling the Nyquist frequency is 5 kHz, so cutoff = 1/500
    corresponds to 10 Hz.
    """
    return butter(order, cutoff, btype="high")


def filter_channel(x: np.ndarray,
                   ba: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    One-directional IIR filtering.

    ★1 ``lfilter`` (one-directional) must be used here, not ``filtfilt``
       (zero-phase, forward-backward). This pipeline defines stage 1 as a causal
       filter with zero initial conditions, which is exactly what ``lfilter``
       computes. ``filtfilt`` returns different values and completely changes
       the phase response.
    """
    b, a = ba
    return lfilter(b, a, x)


def detect_spikes(filtered: np.ndarray,
                  time: np.ndarray,
                  exclusion_half: int = EXCLUSION_HALF,
                  quantile: float = SPIKE_QUANTILE,
                  max_spikes: int = MAX_SPIKES_PER_CHANNEL) -> np.ndarray:
    """
    Detect spikes in one channel's filtered signal.

    Algorithm:
      threshold = quantile(signal, 0.0005)
      loop:
        find the global minimum; stop once it is >= threshold
        record the time of that sample
        set the samples within +/- exclusion_half of it to +1e6 (blanking, so
        that the same spike is not detected twice)

    ★2 ``np.quantile(..., method="linear")`` selects the type-7
       linear-interpolation quantile: the threshold below is computed with
       exactly this convention.
    ★3 ``np.argmin`` returns the first index among ties, and the detector relies
       on that rule to break ties between equal minima.

    Extracellular field potentials are negative-going, hence argmin rather than
    argmax.
    """
    threshold = np.quantile(filtered, quantile, method="linear")

    work = filtered.copy()
    times: list[float] = []
    n = work.size

    for _ in range(max_spikes):
        idx = int(np.argmin(work))
        if work[idx] >= threshold:
            break
        times.append(float(time[idx]))
        lo = max(0, idx - exclusion_half)
        hi = min(n, idx + exclusion_half + 1)
        work[lo:hi] = 1e6

    return np.asarray(times, dtype=float)


def detect_all_channels(rec: Recording) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Run the detection on all 64 channels, one at a time.

    Returns (list of spike times per channel, filtered signal matrix).
    """
    ba = butter_highpass()
    filtered = np.empty_like(rec.voltage)
    spike_times: list[np.ndarray] = []

    for i in range(rec.n_channels):
        f = filter_channel(rec.voltage[:, i], ba)
        filtered[:, i] = f
        spike_times.append(detect_spikes(f, rec.time))

    return spike_times, filtered


def spikes_to_table(spike_times: list[np.ndarray],
                    channels: list[str],
                    time_scale: float = TIME_SCALE) -> pd.DataFrame:
    """
    Assemble the per-channel spike times into the (x, y, t) triple table that the
    Hough transform consumes.

    ``time_scale`` is the divisor applied to every spike time. This scaling is
    not optional: without it the within-plane standard deviation is about 5.35,
    while the Hough inlier tolerance is only 0.1, so not a single plane is found
    (verified by measurement).
    """
    rows = []
    for i, times in enumerate(spike_times, start=1):
        if times.size == 0:
            continue
        x, y = channel_to_xy(i)
        for t in times:
            rows.append((x, y, t / time_scale, channels[i - 1]))

    df = pd.DataFrame(rows, columns=["x", "y", "t", "channel"])

    # ★ A stable sort is mandatory here. pandas ``sort_values()`` uses quicksort
    #   (unstable) by default, while the point order of this pipeline has to keep
    #   the input order among equal t values. This dataset contains 47 tied t
    #   values (involving 98 points); an unstable sort would order those tied
    #   points differently, and any downstream operation that indexes by position
    #   (for example replaying a sampled index trace) would then point at the
    #   wrong points - without raising an error, silently.
    return df.sort_values("t", kind="stable").reset_index(drop=True)
