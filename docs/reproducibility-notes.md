# Reproducibility notes

Practical notes for re-running the code: what to switch on, how randomness is controlled, and
which settings are worth double-checking against the manuscript.

---

## 1. Switches that must be set before a run

Several scripts use module-level constants to select between alternative code paths, and in a few
places a loop variable silently overrides a constant declared above it. Check these before
interpreting any output.

| File | Line | Committed value | Set to | To get |
|---|---|---|---|---|
| `R/02_simulation/fig5_detection_performance.R` | 683 | `shifts = 0` | `shifts = 1:100` | the 100-replicate study (Fig. 5) |
| `R/02_simulation/fig7_fig8_fitting_accuracy.R` | 880 | `plot.mode = 3` | `1` | accuracy vs SNR (Fig. 7) |
| | | | `2` | accuracy vs missing probability (Fig. 8) |
| `R/01_spike_detection/spike_activation_time.R` | 34 | `for (smooth.method in 1)` | `in 2` | Butterworth branch (line 12 is overridden by this loop) |
| | 37 | `for (i in 6)` | `2:ncol(data)` | all channels instead of channel index 6 |
| | 21 | `# data = read.csv(filename)` | uncomment | read the CSV; otherwise `data` must pre-exist |
| `R/03_real_data/tables2to4_pipeline.R` | 23 | `Data_CK/heart/...csv` | your path | locate the recording |

Note on line 34 of `spike_activation_time.R`: `for (smooth.method in 1)` assigns the loop
variable, so the value set at line 12 has no effect. `1 = LOWESS`, `2 = BUTTER`, `3 = NOSMOOTH`
(constants at lines 6–8).

---

## 2. Randomness

Every simulator seeds itself deterministically from its own parameters, so a single run is exactly
reproducible:

```r
SEED_BASE = 1000000                                    # 101 in generate_simulation_data.R
set.seed(SEED_BASE * limit * time.max * ts * x * y * u.x * u.y * p * noise.freq * miu * sigma
         + seed.shift)
```

`seed.shift` defaults to `1` and is the single knob used to obtain independent replicates.
The replicate study in `fig5_detection_performance.R` passes it through
`visualize2d(theta, seed.shift)` (line 344) and iterates over it at line 686.

Each simulator carries its own `SEED_BASE`, and two of them appear in the same file:

| File | Simulator | `SEED_BASE` | `seed.shift` |
|---|---|---|---|
| `generate_simulation_data.R` | `stomach.sim2d` | `101` (line 10) | no |
| `fig5_detection_performance.R` | `stomach.sim` | `1000000` (line 53) | no |
| `fig5_detection_performance.R` | `stomach.sim2d` | `101` (line 119) | **yes** (line 116) |
| `fig7_fig8_fitting_accuracy.R` | `stomach.sim` | `1000000` (line 62) | no |
| `fig7_fig8_fitting_accuracy.R` | `stomach.sim2d` | `100` (line 158) | no |

Note that `fig7_fig8_fitting_accuracy.R`'s circular simulator has **no** `seed.shift`. Independence
across the sweep points of Figures 7 and 8 comes instead from the fact that the seed expression
itself contains the swept parameter (`p`, `noise.freq` and `sigma` all appear in it), so each
setting receives a different seed. Alternative `SEED_BASE` values that were tried during
development are left as comments at `fig5_detection_performance.R:117-127`.

---

## 3. Simulation conventions

* **Parameter encoding.** The simulator uses `ratio` for the anisotropy between the two grid
  axes: the y-direction travel time is drawn as `rnorm(1, mean = ratio*miu, sd = ratio*sigma)`.
  With `miu = 1`, `sigma = 0.1`, `ratio = 2` this yields `μ_y = 2`, `σ_y = 0.2`.
* **Speed units.** The circular simulator places electrodes on an integer grid and uses
  `miux = miuy = 1`, so the speed is in *grid units per unit time* — with `limit = 96` and
  `time.max = 96`, the ground truth is `v = 1`.
* **Single wave vs many.** `fig7_fig8_fitting_accuracy.R` sets `gap = 1000000` (line 657), which
  effectively emits a single wave per run — appropriate for the fitting-accuracy study. The
  detection study uses `gap = 30` to produce several successive waves.
* **Missing-detection mechanism.** Each candidate spike is retained with probability `1 − p`:

  ```r
  if (runif(1,0,1) > p) { ... }      # generate_simulation_data.R:30
  if (arrival.time < time.max) {
    if (runif(1,0,1) > p) { ... }    # fig7_fig8_fitting_accuracy.R:132-136
  }
  ```

  In the circular simulator the check sits inside the `(i, j)` grid loop, so it is applied per
  electrode. The helper `within.range()` defined at `fig7_fig8_fitting_accuracy.R:103-105` is not
  called — the loop that used it is commented out at lines 141–150. The live bounds check is the
  `arrival.time < time.max` test at line 131, so spikes arriving after `time.max` are dropped
  rather than wrapped.
* **Inter-wave interval.** In `fig7_fig8_fitting_accuracy.R:171` successive excitation times are
  spaced by `rexp(1, 1/gap)`; `generate_simulation_data.R:66` uses `rchisq(1, gap)` instead. Both
  have mean `gap` but different dispersion, so the two scripts produce synthetically different
  wave trains. For the 96×96 study `gap = 1000000`, which in practice emits a single wave.

---

## 4. Filter design

The Butterworth high-pass filter used in the real-data pipeline is

```r
butter(2, 1/500, type="high")     # R/03_real_data/tables2to4_pipeline.R:55
```

In the `signal` package, `butter(n, W)` takes **`n` = filter order** and **`W` = cut-off as a
fraction of the Nyquist frequency**. So this designs a **2nd-order** filter with a cut-off at
Nyquist/500.

The filter order and cut-off are the two settings that most strongly affect which spikes survive,
so they are worth stating explicitly wherever the pipeline is described:

| Variant in this repository | Call | Note |
|---|---|---|
| `R/03_real_data/tables2to4_pipeline.R:55` | `butter(2, 1/500, type="high")` | used for the real-data study |
| `R/01_spike_detection/spike_activation_time.R:47` | `butter(2, 1/500, type="high")` | same design |
| `R/01_spike_detection/butterworth_design_demo.R:2` | `butter(3, 0.3, type="high")` | standalone design demo only |

To use a different order or cut-off, edit the single call at
`R/03_real_data/tables2to4_pipeline.R:55`; nothing else depends on the filter coefficients.

---

## 5. Time units

Two scalings appear in the real-data scripts:

| Location | Expression | Implication |
|---|---|---|
| `spike_activation_time.R:95,117` | `xs = smooth$x / 10`, labelled `"Time(ms)"` | sample index ÷ 10 = ms → 10 kHz acquisition |
| `spike_activation_time.R:136`, `tables2to4_pipeline.R:115` | `all.lines$ts.observ = all.lines$ts.observ / 200` | column 1 of the CSV ÷ 200 |

The excitation times `t₀` reported by `tables2to4_pipeline.R` are in whatever unit the second
conversion produces. If you are comparing printed output against published values, check the unit
of column 1 of your copy of `CM_PIN_Control_2N30_Aunor_2_input_60000.csv` first; the constant at
`tables2to4_pipeline.R:115` is the only place that scaling is applied.

---

## 6. Graphics

* Figures 3, 4 and 10 are produced with `rgl` 3-D devices. `rgl.snapshot()` (used in
  `fig7_fig8_fitting_accuracy.R:687`) and `dev.copy()` (used in `spike_activation_time.R:107,127`)
  write the files; create a `plots/` directory first.
* `stomach.plot2d.interactive()` enters an infinite right-click loop when called with
  `highlight.plane = -1` (the default). It is disabled by passing `0`, which is what
  `fig5_detection_performance.R:361` does. Set `highlight.plane = k` (k ≥ 1) to render a single
  plane statically — useful for reproducing Figure 10 without a manual session.
* Figure 5 is a boxplot of the per-replicate rates accumulated in `optimal.stats`; the script
  prints the mean and standard deviation but the plotted box is drawn separately from those
  numbers.

---

## 7. Outputs written to disk

| File | Written by |
|---|---|
| `stomach_simulation_data.csv` | `generate_simulation_data.R:110` |
| `plots/smoothing<m>_<channel>_<spike>.pdf`, `plots/smoothing<m>_<channel>.png` | `spike_activation_time.R:107,127` |
| `plots/plot_mode<k>_<1..4>.pdf` | `fig7_fig8_fitting_accuracy.R:898-900,917-919,939-941` |
| `plots/cone_simulate.png` | `fig7_fig8_fitting_accuracy.R:687` (with `plot.3d = TRUE`) |

Tables 2, 3 and 4 are printed to standard output rather than written to a file:

```bash
Rscript R/03_real_data/tables2to4_pipeline.R | tee tables234.txt
```
