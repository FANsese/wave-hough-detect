# Stage 1 — Spike detection

Implements §2.1 / Algorithm 1 of the paper: Butterworth high-pass filtering of each electrode
channel, followed by global peak detection with a high-percentile threshold and a neighbourhood
exclusion window, producing the spike set `{(xᵢ, yᵢ, tᵢ)}`.

## Files

### `butterworth_design_demo.R`

A 14-line standalone demonstration of the filter design. Synthesises a 20 Hz + 2 Hz signal and
applies a high-pass Butterworth filter:

```r
bf <- butter(3, 0.3, type="high")
```

No external data required. Run it to inspect the filter response before committing to a
cut-off for a real recording:

```bash
Rscript butterworth_design_demo.R
```

### `spike_activation_time.R`

Per-channel spike detection, used to inspect one channel at a time and to produce the
raw-trace figures (Fig. 9 style, `xlab = "Time(ms)"`, `ylab = "Voltage(mV)"`).

**Input** — line 20:

```r
filename = "CM_PIN_Control_2N30_Aunor-txt.csv"
```

> ⚠️ **Note:** the `read.csv()` call at **line 21 is commented out**. The script expects the
> matrix to already exist in the session as `data` (e.g. loaded from an `.RData` file). Uncomment
> line 21, or load the recording before `source()`-ing this file.

**Before running — three switches to check:**

| Line | Current value | Meaning |
|---|---|---|
| 12 | `smooth.method = NOSMOOTH` | Overridden — see line 34 |
| 34 | `for (smooth.method in 1)` | **This loop variable overrides line 12.** `1` = LOWESS, `2` = BUTTER, `3` = NOSMOOTH |
| 37 | `for (i in 6)` | Process channel index 6 only. Change to `2:ncol(data)` for all channels |

So the file as committed exercises the **LOWESS** branch. To exercise the **Butterworth** branch
described in the paper, change line 34 to:

```r
for (smooth.method in 2)      # BUTTER = 2, defined at line 7
```

which activates lines 46–52:

```r
butter.filter = butter(2, 1/500, type="high")
smooth = filter(butter.filter, unsmooth)
upper.threshold = quantile(smooth$y, 0.9995)    # the 99.95th percentile threshold
max.index.gap = 500                              # neighbourhood exclusion window
```

**Outputs** — one PDF per detected spike plus a summary PNG, written to `plots/`:

```
plots/smoothing<m>_<channel>_<spike>.pdf
plots/smoothing<m>_<channel>.png
```

Create the directory first: `mkdir -p plots`.

> `1/500` is the cut-off as a fraction of the Nyquist frequency. At the 10 kHz acquisition rate
> implied by the plotting code (`smooth$x / 10` = milliseconds) this corresponds to a 10 Hz
> high-pass. The same value is used by `R/03_real_data/tables2to4_pipeline.R`.

The full multi-channel version of this stage, as used for the real-data study, lives in
`R/03_real_data/tables2to4_pipeline.R` (lines 18–117), where `smooth.method = BUTTER` is active
and all channels are processed.
