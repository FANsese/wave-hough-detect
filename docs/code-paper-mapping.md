# Code ↔ paper mapping

Line numbers refer to the files as committed in this repository.

---

## Algorithms

### Algorithm 1 — Butterworth filtering and spike activation times (§2.1)

| Paper step | Implementation | Location |
|---|---|---|
| Apply Butterworth high-pass filter | `butter(2, 1/500, type="high")` + `filter()` | `R/03_real_data/tables2to4_pipeline.R:55-56` |
| | same, LOWESS alternative | `R/03_real_data/tables2to4_pipeline.R:40-43` |
| | filter design demo (standalone) | `R/01_spike_detection/butterworth_design_demo.R:2,8` |
| Dynamic threshold at the 99.95th percentile | `quantile(smooth$y, 0.9995)` (upper-tail form) | `R/01_spike_detection/spike_activation_time.R:51` |
| | `quantile(smooth$y, 0.0005)` (lower-tail form, equivalent on the negated signal) | `R/03_real_data/tables2to4_pipeline.R:59` |
| Locate the first candidate peak `t₀ = argmax y*` | `which.max(change.smooth$y)` | `R/01_spike_detection/spike_activation_time.R:89` |
| | `which.min(change.smooth$y)` | `R/03_real_data/tables2to4_pipeline.R:90` |
| Loop while the threshold condition holds | `while (1) { ... if (peak <= threshold) break }` | `spike_activation_time.R:63-79` |
| | | `tables2to4_pipeline.R:66-80` |
| Exclude the neighbourhood `G` of each spike | `[max.index ± max.index.gap] <- -1000000`, `max.index.gap = 500` | `spike_activation_time.R:52,110-111` |
| | `[min.index ± min.index.gap] <- 1000000`, `min.index.gap = 500` | `tables2to4_pipeline.R:60,96-98` |
| Output the spike set `{(xᵢ,yᵢ,tᵢ)}` | `all.lines$x.observ / y.observ / ts.observ` | `tables2to4_pipeline.R:82-94` |
| Time scaling | `all.lines$ts.observ / 200` | `tables2to4_pipeline.R:115`, `spike_activation_time.R:136` |

### Algorithm 2 — Randomized Hough transform (§2.2)

Reference implementation: `R/03_real_data/tables2to4_pipeline.R:248-453` (`hough.plane`).
An equivalent, simulation-oriented copy lives in `R/02_simulation/fig5_detection_performance.R:440-638`.

| Paper step | Implementation | Location (real-data file) |
|---|---|---|
| Accumulator vote threshold "8 votes" | `threshold = 8` | `:265` |
| Randomly sample 3 points | `sample.indices = sample(unclassify.indices, 3)` | `:269` |
| Plane through 3 points | `cross.product(v2 - v1, v3 - v1)` → normal, normalised | `:275-277` |
| Sign canonicalisation of the normal | `n <- n * sign(n[[1]])` | `:279-281` |
| Accumulator discretisation `(ρ, φ, θ)` | `get.key()`; `ρ` step 0.05, `φ`/`θ` step 2° | `:201-207` |
| Vote | `accumulator[[key]] <- append(accumulator[[key]], sample.indices)` | `:297-300` |
| Bucket reached the threshold | `length(accumulator[[key]]) > threshold * 3` | `:301` |
| Least-squares plane refinement | `lm.fit(cbind(x, y, 1), ts)` → `n.fit`, `rho.fit` | `:305-311` |
| Inlier test "is the point on the plane" | `abs(dot(point, normal) - rho) < 0.1` | `:209-213` |
| Collect plane points from unclassified points | `find.points.on.plane()` | `:215-225` |
| Accept only if > 2/3 of the detectors are on it | `num.unique.detectors()`; `num.detectors.with.signal < 40` | `:227-236`, `:319-322` |
| Mark cluster and remove | `keys[...] <- key; prediction[...] <- 1`; plane index incremented | `:328-348` |
| Re-assign unclassified points close to a plane | loop over accepted planes | `:386-404` |
| Unclassified points = noise | `prediction[noise.indices] = 0` | `:449-450` |

### Algorithm 3 — Alternating minimization, circular wavefront (§2.3.2)

| Paper step | Implementation | Location |
|---|---|---|
| Loss `ℓ_circular` (Eq. 3) | `cone.model.inverted.all(p, sim.matrix)` | `R/03_real_data/tables2to4_pipeline.R:550-556` |
| | grid form `cone.model.inverted(p, ts.grid)` | `:540-548` |
| 4×4 lattice tabulation, `a₀…a₃` | `grid.optim.find.best.point.1d()`, `num.splits = 4`, `for (i in 0:num.spaces)` | `:578-605` |
| Shrink to `[a_{î-1}, a_{î+1}]` | `pp.lower <- pp - new.space; pp.upper <- pp + new.space` | `:635-636` |
| `î` clamping at the boundaries | `if (abs(pp - pp.lower) < 1e-12) pp <- pp.lower + new.space`; symmetric for the upper bound | `:625-632` |
| Convergence `‖Θ−Θ̂‖/‖Θ‖ < ε` | `epsilon = 1e-6`; `reldiff = normv(pp.estimate - pp.old)/normv(pp.old)` | `:672,685-686` |
| Search window | `pp.lower = c(-50,-50); pp.upper = c(50, 50)` | `:669` |
| `(x₀, y₀)` step, `(v, t₀)` fixed | `grid.optim(..., cone.model.inverted.all.xy, sim.matrix, v, t)` | `:675-676` |
| `u = 1/v` substitution | `vt.optim.all()`: `x = distance`, `y = time`, `lm(y ~ x)` → slope `u`, intercept `t₀` | `:656-666` |
| `(v, t₀)` step, `(x₀, y₀)` fixed | `pp.estimate = vt.optim.all(pp.estimate, sim.matrix)` | `:679` |
| Outer loop until convergence | `for (iter in 1:iter.max) { ... }` | `:674-688` |
| Convert back to speed | `'v' = 1/pp.estimate[[3]]` | `:733` |
| | `relerror.v = 1/pp.estimates[,4]` | `R/02_simulation/fig7_fig8_fitting_accuracy.R:793` |

### Algorithm 4 — Single-source constrained RHT (§3.2.2)

Reference implementation: `R/03_real_data/tables2to4_pipeline.R:248-453`.

| Paper step | Implementation | Location |
|---|---|---|
| Master normal = plane with most points | `mode(positive.plane.indices)` → `max.normal.vector` | `:353-358` |
| Angle between each plane normal and the master | `acos(dot(n, p) / (‖n‖‖p‖))` → degrees | `:364-379` |
| 5° tolerance; discard planes beyond it | `angle.threshold = 5`; `min(angle.difference, 180 - angle.difference) > angle.threshold` | `:363,373` |
| Marks discarded plane points as unclassified | `prediction[noise.indices] = 0` | `:381-383` |
| Re-assign unclassified points close to a kept plane | loop over `plane.indices.guess.as.signal` | `:386-404` |
| Regrow a plane through an unclassified point along the master normal | **active**, gated by `small.plane.threshold = 30` | `:406-448` (`:407` for the threshold) |

---

## Models

| Paper equation | Implementation | Location |
|---|---|---|
| Eq. (1) circular arrival time | `sqrt((x-x0)^2 + (y-y0)^2)/v + e` | `tables2to4_pipeline.R:543` |
| Eq. (2) linear arrival time | `plane.model()`: `a*x + b*y + c` | `tables2to4_pipeline.R:744-749` |
| Eq. (3) `ℓ_circular` | `cone.model.inverted.all()` | `tables2to4_pipeline.R:550-556` |
| Eq. (4) `ℓ_linear` | `plane.model()` | `tables2to4_pipeline.R:744-749` |
| Eq. (5)–(7) least-squares solution | `lm(ts.observ ~ x.observ + y.observ)` | `tables2to4_pipeline.R:769` |
| Speed from the linear model | `v = 1/sqrt(a^2 + b^2)` | `tables2to4_pipeline.R:771` |
| Coefficient of determination `R²` | `compute.loss()`: `1 - ssres/sstot` | `tables2to4_pipeline.R:715-721` |

---

## Figures and tables

| Item | File | Producing code |
|---|---|---|
| **Fig. 2** 8×8 synthetic ground truth | `R/02_simulation/generate_simulation_data.R` | `stomach.sim2d()` and the call at `:95-107` |
| **Fig. 3** four-colour classification | `R/02_simulation/fig5_detection_performance.R` | `stomach.plot2d()` `:320-342` |
| **Fig. 4** planes found by RHT | `R/02_simulation/fig5_detection_performance.R` | `draw.plane()` `:268-277`, called from `stomach.plot2d.interactive()` `:279-319` |
| **Fig. 5** FPR/FNR boxplot | `R/02_simulation/fig5_detection_performance.R` | `optimality.table()` `:680-705`; rates returned at `:371-374`; mean/sd printed at `:694-695`. Set `shifts = 1:100` at `:683` |
| **Fig. 6** circular wavefront simulation | `R/02_simulation/fig7_fig8_fitting_accuracy.R` | `visualize2d(..., plot.3d = TRUE)` `:652-730`, 3-D render at `:676-687` (writes `plots/cone_simulate.png`) |
| **Fig. 7** accuracy vs SNR | `R/02_simulation/fig7_fig8_fitting_accuracy.R` | `optimality.table()` `plot.mode==1` `:884-902`; SNR defined at `:788`; plotting in `relerror.plot()` `:782-876` |
| **Fig. 8** accuracy vs missing probability | same file | `plot.mode==2` `:904-921` |
| σ (measurement error) analysis | same file | `plot.mode==3` `:923-943` |
| **Fig. 9** raw extracellular traces | `R/01_spike_detection/spike_activation_time.R` | `plot(..., xlab="Time(ms)", ylab="Voltage(mV)")` `:99-100,117-118`; `dev.copy(pdf/png, ...)` `:107,127` |
| **Fig. 10** five planes on real data | `R/03_real_data/tables2to4_pipeline.R` | `hough.plane()` `:248-453`; `stomach.plot2d.interactive()` `:470-534` |
| **Fig. 11** circular-model directions | — | Direction arrows were drawn outside these scripts; not included (see `README.md`, Known limitations) |
| **Fig. 12** linear-model directions | — | As above |
| **Table 2** circular fit per plane | `R/03_real_data/tables2to4_pipeline.R` | `hybrid.optim()` loop `:724-742`; `v = 1/u` at `:733` |
| **Table 3** linear fit per plane | `R/03_real_data/tables2to4_pipeline.R` | `plane.model` loop `:751-778`; `lm` at `:769`; `v` at `:771` |
| **Table 4** R² comparison | `R/03_real_data/tables2to4_pipeline.R` | `compute.loss()` `:715-721`, called at `:739` (circular) and `:776` (linear) |

---

## Simulation parameters

### §3.1.1 — 8×8 linear-wavefront simulation

`generate_simulation_data.R:95-107` and `fig5_detection_performance.R:345-349`:

| Argument | Code | Paper |
|---|---|---|
| `limit` | 8 | 8×8 MEA |
| `time.max` | 80 | `t_max = 80` |
| `ts` | 1 | `t₀ = 1` |
| `x`, `y` | 1, 1 | source at `(1,1)` |
| `p` | 0.1 | `p = 0.1` |
| `noise.freq` | 1 | `λ_n = 1` |
| `miu` | 1 | `μ_x = 1` |
| `sigma` | 0.1 | `σ_x = 0.1` |
| `ratio` | 2 | `μ_y = 2`, `σ_y = 0.2` |
| `gap` | 30 | `λ_s = 1/30` |

### §3.1.2 — 96×96 circular-wavefront simulation

`fig7_fig8_fitting_accuracy.R:653-659`:

| Argument | Code | Paper |
|---|---|---|
| `limit` | 96 | 96×96 grid |
| `time.max` | 96 | `t_max = 96` |
| `x`, `y` | 48, 48 | `x₀ = y₀ = 48` |
| `ts` | 2 | `t₀ = 2` |
| `miux`, `miuy` | 1, 1 | `v = 1` in all directions |

Sweep settings: `plot.mode==1` → `λ_n = 2^((-6:16)/2)` ∈ [0.125, 256] with `p = 0`, `σ = 1e-6`;
`plot.mode==2` → `p = (0:8)/10` ∈ [0, 0.8] with `λ_n = 1`, `σ = 1e-6`.

### §3.2 — real data

`R/03_real_data/tables2to4_pipeline.R`:

| Quantity | Value | Line |
|---|---|---|
| Butterworth high-pass | `butter(2, 1/500, type="high")` | `:55` |
| Spike threshold | `quantile(smooth$y, 0.0005)` | `:59` |
| Neighbourhood exclusion window | ±500 samples | `:60` |
| RHT vote threshold | `8` (tested as `> threshold * 3`) | `:265`, `:301` |
| Accumulator quantisation | `ρ = 0.05`, `φ = θ = 2°` | `:202-204` |
| Inlier tolerance | `0.1` | `:212` |
| Minimum electrodes per plane | `40` of 64 (≈2/3) | `:322` |
| Angular tolerance (Algorithm 4) | `5°` | `:363` |
| Tail-loop plane threshold | `small.plane.threshold = 30` points | `:407` |
| Search window | `[-50, 50]²` | `:669` |
| Convergence tolerance | `1e-6` | `:672` |
