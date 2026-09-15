# §3.2 — Real-data study

End-to-end pipeline on the 8×8 cardiac-myocyte MEA recording: spike extraction → randomized
Hough transform → wavefront model fitting. This single script produces Figures 9 and 10 and
Tables 2, 3 and 4 of the paper.

## `tables2to4_pipeline.R`

### Running it without the experimental recording

The experimental file is not distributed with this repository (see
[`../../data/README.md`](../../data/README.md) for why). `make_synthetic_recording.R` in this
directory writes a stand-in in the same format, so the whole pipeline can be exercised:

```bash
Rscript make_synthetic_recording.R     # -> synthetic_recording.csv
# then set line 23 of tables2to4_pipeline.R to that file and run it
```

`make_synthetic_recording.R` has its parameters at the top (sampling rate, recording length, wave
schedule, source position, speed, spike shape, noise). It generates 64 electrodes, several
propagating wavefronts from a source south of the grid, and a slow baseline drift for the
high-pass filter to remove.

### Input

**Line 23:**

```r
filename = "Data_CK/heart/CM_PIN_Control_2N30_Aunor_2_input_60000.csv"
```

Point this at your own copy of the recording, or at `synthetic_recording.csv`.

Expected format:

* column 1 — time stamps
* columns 2…65 — one column per electrode, named so that `substr(name, 3, 4)` yields the channel
  number, e.g. `Ch01` … `Ch64` (parsed at lines 26–28)
* 64 channels arranged as an 8×8 grid; channel → `(x, y)` conversion at lines 82–88
* negative-going spikes (detection uses `which.min`, line 90)

### Pipeline

| Lines | Stage | Paper |
|---|---|---|
| 18–117 | Butterworth high-pass + percentile threshold + neighbourhood exclusion, all 64 channels | §2.1, Algorithm 1 |
| 248–453 | `hough.plane()` — randomized Hough transform | §2.2, Algorithms 2 & 4 |
| 540–666 | `cone.model.*()`, `grid.optim()`, `vt.optim()`, `hybrid.optim()` | §2.3, Algorithm 3 |
| 723–742 | circular-wavefront fitting → **Table 2**, **Table 4** | §2.3.1 |
| 744–779 | linear-wavefront fitting → **Table 3**, **Table 4** | §2.3.2 |

### Key settings (all active as committed)

**Spike extraction** — `smooth.method = BUTTER` at line 13 (the LOWESS alternatives at lines 11–12
are commented out):

```r
butter.filter = butter(2, 1/500, type="high")   # line 55
lower.threshold = quantile(smooth$y, 0.0005)    # line 59  -> the 0.05th percentile
min.index.gap = 500                             # line 60  -> ±500-sample exclusion window
```

The signal is negated in effect (the loop detects `which.min`), so `quantile(y, 0.0005)` is the
99.95th-percentile threshold of Algorithm 1 applied to the lower tail of the extracellular field
potential.

**Hough transform:**

| Parameter | Line | Value |
|---|---|---|
| accumulator vote threshold | 265 | `threshold = 8` |
| random 3-point sampling | 269 | `sample(unclassify.indices, 3)` |
| vote count test | 301 | `length(accumulator[[key]]) > threshold * 3` |
| inlier tolerance | 212 | `abs(dot(point, normal) - rho) < 0.1` |
| minimum electrodes per plane | 322 | `num.detectors.with.signal < 40` (of 64, i.e. ~2/3) |
| angular tolerance (Algorithm 4) | 363 | `angle.threshold = 5` (degrees) |
| tail-loop plane threshold | 407 | `small.plane.threshold = 30` points |
| `ρ` / `φ` / `θ` quantisation | 202–204 | `ρ` step `0.05`; `φ`, `θ` step `2°` |

**Model fitting** — `hybrid.optim()` at lines 668–711:

```r
pp.lower = c(-50,-50); pp.upper = c(50, 50)   # line 669 — the search window of §3.2.3
epsilon = 1e-6                                 # line 672 — relative convergence tolerance
```

Alternating minimization (Algorithm 3) is the loop at lines 674–688:

1. `grid.optim()` minimises over `(x₀, y₀)` with `(v, t₀)` held fixed — the 4×4 lattice
   tabulation of §2.3.2, with the boundary-clamping rule at lines 625–632;
2. `vt.optim.all()` re-fits `(v, t₀)` by least squares (lines 656–666) via `lm(y ~ x)`, where
   `x` is the source-to-electrode distance and `y` the arrival time — so the slope is `u = 1/v`
   and the intercept is `t₀`.

Line 733 converts back to a speed for reporting:

```r
print(list('pp'=pp.estimate, 'v'=1/pp.estimate[[3]]))
```

### Outputs

For **every** detected plane (`planes = 1:num.plane`, lines 725 and 752), the script prints to
stdout:

Circular model (lines 726–741):

```
pp       x0, y0, u = 1/v, t0          -> Table 2 (after taking 1/u for v)
v        1/pp.estimate[[3]]
grad     gradient of the loss at the estimate (stopping check)
coefficient   R^2  =  1 - SS_res/SS_tot   -> Table 4
```

Linear model (lines 753–778):

```
lm_result   a, b, v = 1/sqrt(a^2 + b^2)   -> Table 3
coefficient R^2 = 1 - SS_res/SS_tot       -> Table 4
```

`R²` comes from `compute.loss()` at lines 715–721.

Run with:

```bash
mkdir -p plots && Rscript tables2to4_pipeline.R
```

### Notes

* `stomach.plot2d.interactive()` (lines 470–538) is used for Figure 10. Passing
  `highlight.plane = -1` (the default) enters an infinite right-click loop for manual
  exploration; the figure as published uses this interactively.
* Line 115 scales the activation times: `all.lines$ts.observ = all.lines$ts.observ / 200`. Values
  are reported in the units this conversion produces. Confirm the unit of column 1 of the input
  CSV before comparing the printed `t₀` values against Table 2 — see
  [`../../docs/reproducibility-notes.md`](../../docs/reproducibility-notes.md).
