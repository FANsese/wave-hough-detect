# §3.1 — Simulation study

Reproduces the synthetic-data experiments: Figures 2–8 of the paper. All three scripts are
**self-contained** — no data files are needed.

## Files, in run order

### 1. `generate_simulation_data.R` — synthetic 8×8 MEA recording (Fig. 2)

Generates spikes propagating across an 8×8 electrode grid and writes
`stomach_simulation_data.csv` with columns `x, y, ts, z` (`z = 1` signal, `z = 0` noise).

The parameters at lines 95–107 reproduce §3.1.1 exactly:

| Argument | Value | Paper symbol |
|---|---|---|
| `limit` | 8 | 8×8 MEA grid |
| `time.max` | 80 | `t_max` |
| `ts` | 1 | `t₀` (first excitation) |
| `x`, `y` | 1, 1 | wave source at grid corner `(1,1)` |
| `p` | 0.1 | probability a spike is missed |
| `noise.freq` | 1 | `λ_n` — noise arrival rate |
| `miu` | 1 | `μ_x` — mean travel time between adjacent electrodes in x |
| `sigma` | 0.1 | `σ_x` — sd of that travel time |
| `gap` | 30 | `λ_s` — mean interval between successive waves |
| `ratio` | 2 | `μ_y/μ_x` (so `μ_y = 2`, `σ_y = 0.2`) |

```bash
Rscript generate_simulation_data.R
# -> stomach_simulation_data.csv
```

> The inter-wave interval here is drawn from `rchisq(1, gap)` (line 66). See
> `docs/reproducibility-notes.md` for a note on this distribution.

### 2. `fig5_detection_performance.R` — RHT detection performance (Figs. 3, 4, 5)

Runs the randomized Hough transform on the simulated data and evaluates how well signal spikes
are separated from noise spikes.

The simulator inside this file (`visualize2d`, lines 344–375) uses the same §3.1.1 configuration:
`limit = 8`, `time.max = limit*10 = 80`, `gap = 30`, source `(1,1)`, `t₀ = 1`, `ratio = 2`,
`λ_n = 1`, `p = 0.1`, `μ = 1`, `σ = 0.1` (lines 345–349).

**Set the number of replicates at line 683:**

```r
shifts = 1:100       # 100 independent random seeds, as reported in the paper
```

Each replicate re-seeds the simulator through the `seed.shift` argument (lines 344, 350–351).
`optimality.table()` (lines 680–705) accumulates the results and prints:

* `false.pos.rate` and `false.neg.rate` per replicate (returned by `visualize2d`, line 374)
* their **mean** (line 694) and **standard deviation** (line 695) — the quantities quoted in the
  paper's Figure 5

`visualize2d` also calls `stomach.plot2d.interactive(all.result, 0)` (line 361). Passing
`highlight.plane = 0` takes neither the `-1` (infinite right-click loop) nor the `>= 1` branch, so
it simply renders the plot and returns — safe for batch runs.

The four-colour classification plot of Figure 3 is drawn by `stomach.plot2d()` (lines 320–342):

| Colour | Meaning |
|---|---|
| green | true signal classified as signal (true positive) |
| magenta | true noise classified as signal (false positive) |
| red | true signal classified as noise (false negative) |
| black | true noise classified as noise (true negative) |

```bash
mkdir -p plots && Rscript fig5_detection_performance.R
```

### 3. `fig7_fig8_fitting_accuracy.R` — fitting accuracy (Figs. 6, 7, 8)

Simulates a circular wavefront on a 96×96 grid and fits the circular model with the alternating
minimization of Algorithm 3, sweeping one experimental factor at a time.

The ground truth is set at **line 659** and matches the paper:

```r
x = 48; y = 48; u.x = 1; u.y = 1; ts = 2; ratio = 10; miux = 1; miuy = miux
```

i.e. `(x₀, y₀, v, t₀) = (48, 48, 1, 2)`, with `limit = 96` (line 653) and `time.max = 96`
(line 656).

**Select the sweep at line 880:**

| `plot.mode` | Sweep variable | Fixed | Paper |
|---|---|---|---|
| `1` | `λ_n = 2^((-6:16)/2)` = 0.125 … 256 (line 885) | `p = 0`, `σ = 1e-6` | **Fig. 7** |
| `2` | `p = (0:8)/10` = 0 … 0.8 (line 905) | `λ_n = 1`, `σ = 1e-6` | **Fig. 8** |
| `3` | `σ = (1:10)/10` = 0.1 … 1.0 (line 924) | `p = 0`, `λ_n ≈ 0` | measurement-error analysis |

Each mode writes four PDFs to `plots/` — one per estimated parameter:

```
plots/plot_mode<k>_1.pdf   x₀
plots/plot_mode<k>_2.pdf   y₀
plots/plot_mode<k>_3.pdf   v
plots/plot_mode<k>_4.pdf   t₀
```

Plotting is done by `relerror.plot()` (lines 782–876): **red points = fitted values**,
**green line = ground truth**, matching the figure captions in the paper. In `plot.mode` 1 the
abscissa is the signal-to-noise ratio defined as *number of signal spikes ÷ number of noise
spikes* (line 788), matching §3.1.2.

```bash
mkdir -p plots && Rscript fig7_fig8_fitting_accuracy.R
```

> `plot.mode = 3` is the value currently set in the committed file. Change line 880 to `1` or `2`
> to regenerate Figures 7 and 8.
