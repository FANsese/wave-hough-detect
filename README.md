# wave-hough-detect

Code accompanying the paper:

> **Robust Wave Origin Detection from Sensor Array Data via Randomized Hough Transform and Model Fitting**
> Sicheng Fan, Jiayi Lu, Xiaodan Fan
> arXiv:2609.13248 [eess.SP] — https://arxiv.org/abs/2609.13248

The method estimates the origin, propagation direction, speed and excitation time of wave-like
signals recorded by a sensor array (e.g. an 8×8 microelectrode array, MEA). It is a closed-loop
**detection → separation → fitting** procedure:

| Stage | Description | Paper |
|---|---|---|
| 1. Data peak extraction | Butterworth high-pass filter + high-percentile threshold + neighbourhood exclusion → spike activation times `{(xᵢ,yᵢ,tᵢ)}` | §2.1, Algorithm 1 |
| 2. Randomized Hough transform (RHT) | Random 3-point sampling → plane normal → `(ρ,φ,θ)` accumulator → vote threshold → inlier recovery. Each plane = one propagating wavefront | §2.2, Algorithms 2 & 4 |
| 3. Model fitting | Circular (near-field) wavefront fitted by alternating minimization; linear (far-field) wavefront fitted by least squares | §2.3, Algorithm 3 |

---

## Repository layout

```
wave-hough-detect/
├── R/
│   ├── 01_spike_detection/            Stage 1 — filtering & spike activation times
│   ├── 02_simulation/                 Stages 2–3 on synthetic data  (§3.1)
│   └── 03_real_data/                  Stages 1–3 on the 8×8 MEA recording  (§3.2)
├── data/                              Data availability statement
└── docs/
    ├── code-paper-mapping.md          Item-by-item mapping: paper ↔ file ↔ line
    └── reproducibility-notes.md       Parameters, random seeds, switch settings
```

Each `R/` subdirectory has its own `README.md` describing inputs, outputs and what to change
before running.

---

## Code ↔ paper mapping (summary)

| Paper item | File | Notes |
|---|---|---|
| **Algorithm 1** | `R/01_spike_detection/spike_activation_time.R`, `R/03_real_data/tables2to4_pipeline.R` | `butter(n, W, type="high")` + `quantile(smooth$y, 0.9995)` + `±max.index.gap` exclusion |
| **Algorithm 2** | `hough.plane()` in `R/02_simulation/fig5_detection_performance.R` and `R/03_real_data/tables2to4_pipeline.R` | 8-vote accumulator threshold; ≥2/3-detector plane acceptance |
| **Algorithm 3** | `hybrid.optim()` + `grid.optim()` + `vt.optim()` in `R/03_real_data/tables2to4_pipeline.R` | 4×4 lattice tabulation on `[-50,50]²`, `ε = 1e-6`; `u = 1/v` via `lm(y ~ x)` |
| **Algorithm 4** | `hough.plane()` in `R/03_real_data/tables2to4_pipeline.R` | master normal vector + 5° angular tolerance |
| Circular model, Eq. (1)(3) | `cone.model.inverted()` / `cone.model.inverted.all()` | `Σ(√((xᵢ−x₀)²+(yᵢ−y₀)²)·u − (tᵢ−t₀))²` |
| Linear model, Eq. (2)(4)–(7) | `plane.model()` + `lm(ts.observ ~ x.observ + y.observ)` | `v = 1/√(a²+b²)` |
| R² | `compute.loss()` | `1 − SS_res/SS_tot`, computed for both models |
| Fig. 2 (8×8 synthetic ground truth) | `R/02_simulation/generate_simulation_data.R` | parameters match §3.1.1 exactly |
| Fig. 3–5 (detection performance) | `R/02_simulation/fig5_detection_performance.R` | `stomach.plot2d()` (4-colour), `optimality.table()` (FPR/FNR mean & sd) |
| Fig. 6–8 (fitting accuracy) | `R/02_simulation/fig7_fig8_fitting_accuracy.R` | `relerror.plot()`: red = estimate, green line = truth |
| Fig. 9 (raw traces) | `R/01_spike_detection/spike_activation_time.R` | `plot(..., xlab="Time(ms)", ylab="Voltage(mV)")` |
| Fig. 10, Tables 2–4 | `R/03_real_data/tables2to4_pipeline.R` | `stomach.plot2d.interactive()`, `hybrid.optim()`, `plane.model` |

Full line-level detail: **[`docs/code-paper-mapping.md`](docs/code-paper-mapping.md)**.

---

## Environment

R (tested on the versions in use in 2013–2014; any R ≥ 3.0 should work). Required packages:

```r
install.packages(c("hash", "rgl", "signal", "numDeriv"))
```

| Package | Used for |
|---|---|
| `hash` | hash-table accumulators in the Hough transform |
| `rgl` | 3-D scatter plots of the `(x, y, t)` spike clouds |
| `signal` | `butter()` / `filter()` — Butterworth filter design |
| `numDeriv` | numerical `grad()` / `hessian()` (Newton-method variants and gradient checks) |

Optional: `optimx`, `minqa` (alternative optimizers explored during development; not required by
any script in this repository).

`R/02_simulation/*` and `R/01_spike_detection/butterworth_design_demo.R` are fully self-contained
and need no external data. `R/03_real_data/tables2to4_pipeline.R` needs a recording in the
documented input format; `R/03_real_data/make_synthetic_recording.R` generates one, so the whole
pipeline runs without any download — see [`data/README.md`](data/README.md).

---

## How to run

Run each script from the directory you want its outputs written to; several scripts write into a
`plots/` subdirectory, so create it first:

```bash
mkdir -p plots
```

### 1. Synthetic data for §3.1.1 (also reproduces Fig. 2)

```bash
Rscript R/02_simulation/generate_simulation_data.R
# -> stomach_simulation_data.csv   (x, y, ts, z)
```

### 2. Detection performance — Fig. 3, 4, 5

`R/02_simulation/fig5_detection_performance.R` — set the number of replicates at **line 683**:

```r
shifts = 1:100      # 100 independent random seeds, as reported in the paper
```

`optimality.table()` then prints the mean and standard deviation of the false-positive and
false-negative rates. The 3-D classification plot (Fig. 3) is drawn by `stomach.plot2d()`.

> The interactive plane inspector is disabled when `visualize2d()` passes `highlight.plane = 0`;
> passing `-1` enters an infinite right-click loop intended for manual figure exploration only.

### 3. Fitting accuracy — Fig. 6, 7, 8

`R/02_simulation/fig7_fig8_fitting_accuracy.R` — select the sweep at **line 880**:

```r
plot.mode = 1   # Fig. 7 — noise frequency λ_n = 2^((-6:16)/2) = 0.125 … 256  (p = 0, σ = 1e-6)
plot.mode = 2   # Fig. 8 — missing probability p = 0, 0.1, …, 0.8            (λ_n = 1, σ = 1e-6)
plot.mode = 3   # measurement-error analysis, σ = 0.1, 0.2, …, 1.0
```

Each mode writes four PDFs to `plots/` (one per estimated parameter). The ground truth is
`(x₀, y₀, v, t₀) = (48, 48, 1, 2)` on a 96×96 grid, matching the paper.

### 4. Real MEA data — §3.2, Fig. 9, 10, Tables 2, 3, 4

Point **line 23** of `R/03_real_data/tables2to4_pipeline.R` at your local copy of the recording:

```r
filename = "path/to/CM_PIN_Control_2N30_Aunor_2_input_60000.csv"
```

Then:

```bash
Rscript R/03_real_data/tables2to4_pipeline.R
```

The script prints, for every detected plane:

* the circular-wavefront fit `(x₀, y₀, 1/u, t₀)` and its R²  → Table 2 and Table 4
* the linear-wavefront fit `(ã, b̃, v)` and its R²           → Table 3 and Table 4

---

## Data

Everything needed to run this code is either generated by it or included with it — no data files
need to be downloaded.

| Study | Data | How to get it |
|---|---|---|
| §3.1 simulations (Figs. 2–8) | synthetic | generated by `R/02_simulation/generate_simulation_data.R` |
| §3.2 pipeline (Tables 2–4) | synthetic stand-in | generated by `R/03_real_data/make_synthetic_recording.R` |
| §3.2 experimental recording | **not included** | third-party data; not ours to redistribute — see [`data/README.md`](data/README.md) |

The experimental cardiac-myocyte recording analysed in §3.2 was produced by a collaborating
laboratory and is not distributed here. The synthetic recorder in `R/03_real_data/` writes a file
in the exact input format the pipeline expects, so the full §3.2 code path — spike extraction,
randomized Hough transform, circular and linear wavefront fitting, R² — runs end to end without
it. Only the numerical values printed in Tables 2, 3 and 4 depend on the experimental file;
[`data/README.md`](data/README.md) documents the input format for anyone substituting their own
recording.

---

## Known limitations

* The direction-of-propagation arrows in Figures 11 and 12 were drawn outside these scripts and
  are not included here.
* Several figures are produced through `rgl` interactive 3-D devices; run with a working graphics
  device (`rgl.snapshot()` / `dev.copy()` is used for the PDF/PNG outputs).
* Scripts use `set.seed()` derived deterministically from the simulation parameters (see
  `docs/reproducibility-notes.md`), so single runs are exactly reproducible; the 100-replicate
  study varies the seed through the `seed.shift` argument.

---

## Citation

```bibtex
@article{fan2026robust,
  title  = {Robust Wave Origin Detection from Sensor Array Data via
            Randomized Hough Transform and Model Fitting},
  author = {Fan, Sicheng and Lu, Jiayi and Fan, Xiaodan},
  journal = {arXiv preprint arXiv:2609.13248},
  year   = {2026}
}
```

## Acknowledgment

We thank Professor John A. Rudd for introducing the gastric microelectrode-array data to us,
which motivated this research.

## License

To be determined by the authors before public release.
