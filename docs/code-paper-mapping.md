# Code ↔ paper mapping

Every line number refers to the files as committed in this repository. The equivalent locations in
the original R scripts are given in the third column, so the port can be audited item by item.

Unqualified line numbers (`:337`) are in the Python file named in the same row.

---

## Coverage: is anything missing?

The original code is 7 R scripts. Their **reachable** functions were extracted by walking the call
graph from each script's top-level driver (transitive closure; function-valued arguments such as the
loss passed to `grid.optim` were added by hand, since a static scan cannot see them):

| R script | Lines | Paper | Python | Status |
|---|---|---|---|---|
| `cm_hough_grid_8.R` | 779 | §3.2, Tables 2/3/4, Fig. 10 | `spikes.py`, `rht.py`, `fit.py` | ✅ bit-for-bit verified |
| `stomach_hough17_newton_square20.r` | 1012 | §3.1.2, Figs 6–8 | `simulate_circular.py`, `fit.py` | ✅ truth-verified |
| `stomach_plane_anglediff14.r` | 707 | §3.1.1, Figs 2–5 | `simulate.py` | ✅ |
| `generate_stomach_data.r` | 109 | §3.1.1 data | `simulate.py` | ✅ |
| `spike_activation_time_plot2.R` | 139 | Fig. 9 | `examples/demo_pipeline.py` fig. 1 | ✅ logic folded in |
| `BUTTER_example_plot_3.r` | 14 | filter design | `butter_highpass()` | ✅ |
| `make_synthetic_recording.R` | 147 | — | `simulate_recording()` | ✅ (time-unit bug fixed) |

All 16 / 20 / 13 / 2 reachable functions of the four computational scripts have a Python
counterpart. The complete reachable sets and their mappings are below.

### What is *not* ported, and why

| R code | Reason |
|---|---|
| `point.angle.to.line` (3 copies) | Only called by `hough2d` (an unused 2-D Hough variant) and by an interactive `select3d` debugging loop inside `stomach.plot2d.interactive`. Neither is on any path that produces a paper figure or number. |
| `hough2d`, `hough` | Dead: never called outside commented-out blocks. |
| `stomach.plot`, `visualize`, `compute.theta` | Dead: belong to an abandoned θ-estimation route. |
| `stomach.sim` (1-D variant) | Dead: superseded by the 2-D simulator. |
| `newton.cone`, `newton.cone.square`, `gra`, `hes`, `dt0`, `dx0`, … | Dead: the Newton route was abandoned in favour of `grid.optim` + `vt.optim`. Kept in R, never executed. |
| `cone.model`, `cone.model.square`, `convert.to.grid`, `norm.xy` | Dead on every active path. (Note the *speed* vs *slowness* split: the analytic gradient `gra`/`hes` is the exact derivative of `cone.model`, the **speed** form, whereas the active driver uses `cone.model.inverted`, the **slowness** form.) |
| `plane.model`, `vt.optim` (singular), `cone.model.inverted.xy`, `.all.xy2` | Dead: earlier variants replaced by `.all.xy` / `.all`. |
| `draw.plane`, `get.ts`, `stomach.plot2d`, `stomach.plot2d.interactive` | **Ported as plotting**, not as 1:1 functions: `draw_planes()` and `DetectionResult.classified()` produce the same figures without `rgl`. |
| `plot.mode ∈ {4,5,8}` (§3.1.2) | Interactive 3-D renderings needing OpenGL/`rgl`; the underlying quantity is covered by mode 3. |
| `plot.mode ∈ {6,7}` (§3.1.2) | They **error in R** (`argument "params" is missing` — the 3-D path is called with the wrong number of arguments). |
| The LOWESS branch of stage 1 | Commented out in R and never used; a high-pass filter is the right tool (LOWESS is a smoother, measured to retain 100.9 % of the baseline drift vs 17.2 % for the Butterworth high-pass). |
| The gastric slow-wave analysis | Not part of the method in the paper. |

### Reachable-function → Python symbol

`cm_hough_grid_8.R` (16 reachable)

| R | Python |
|---|---|
| `hough.plane` | `rht.hough_plane` |
| `get.key`, `get.step.key` | `rht.get_key`, `rht._step_key` |
| `is.point.on.plane`, `find.points.on.plane` | `rht.is_point_on_plane`, `rht.find_points_on_plane` |
| `num.unique.detectors`, `mode` | `rht.num_unique_detectors`, `rht.r_mode` |
| `cross.product`, `dot.product` | `rht.cross_product`, `np.dot` (inline) |
| `hybrid.optim`, `vt.optim.all`, `compute.loss` | `fit.hybrid_optim`, `fit.vt_optim`, `fit.r_squared` |
| `grid.optim`, `grid.optim.find.best.point`, `.1d` | `fit.grid_optim`, `fit._find_best_point_1d` |
| `cone.model.inverted.all.xy`, `.all` | `fit.circular_loss_xy`, `fit.circular_loss` |
| `normv` | `np.linalg.norm` (inline) |

`stomach_plane_anglediff14.r` (20 reachable) adds

| R | Python |
|---|---|
| `stomach.sim.one`, `stomach.sim.one2d`, `stomach.sim2d` | `simulate._simulate_one_line`, `._simulate_one_line_2d`, `simulate.simulate_linear_wavefronts` |
| `visualize2d`, `optimality.table` | `simulate.evaluate_detection`, `simulate.detection_sweep` |
| `last.element`, `first` | `[-1]`, `[0]` (inline) |
| `stomach.plot2d` | `simulate.DetectionResult.classified` + `examples/demo_simulation.py` fig. 3 |
| `stomach.plot2d.interactive`, `draw.plane`, `get.ts` | `draw_planes()` in `examples/demo_simulation.py` |

`stomach_hough17_newton_square20.r` (13 reachable) adds

| R | Python |
|---|---|
| `stomach.sim.one`, `stomach.sim.one2d`, `stomach.sim2d` | `simulate_circular.simulate_circular_wavefronts` (per-cell form, different from the §3.1.1 simulator) |
| `visualize2d`, `optimality.table` | `simulate_circular.evaluate_circular`, `.accuracy_sweep` |
| `relerror.plot` | `plot_accuracy()` in `examples/demo_simulation_circular.py` |
| `last` | `[-1]` (inline) |

`generate_stomach_data.r` (2 reachable): `stomach.sim2d`, `last.element` → as above.

## Stage 1 — Butterworth filtering and spike activation times (§2.1, Algorithm 1)

`src/wave_hough_detect/spikes.py` · R: `R_Codes/cm_hough_grid_8.R:40-115`, `spike_activation_time.R:51-136`

| Paper step | Implementation | R counterpart |
|---|---|---|
| Butterworth high-pass design | `butter_highpass()` :87 — `butter(2, 1/500, btype="high")` | `butter(2, 1/500, type="high")` |
| Filter order | `BUTTER_ORDER = 2` :24 | `signal::butter(2, ...)` — `n` *is* the order |
| Cutoff | `BUTTER_CUTOFF = 1/500` :25 (fraction of Nyquist → 10 Hz at 10 kHz) | same |
| Single-pass IIR filtering | `filter_channel()` :98 — `scipy.signal.lfilter` | `filter()` |
| Discarded LOWESS alternative | documented only; not implemented (it is commented out in R and was never used) | `cm_hough_grid_8.R:40-43` |
| Threshold at the 99.95th percentile | `detect_spikes()` :111 — `np.quantile(filtered, 0.0005)` | `quantile(smooth$y, 0.0005)`; equivalently the upper-tail `0.9995` on the negated signal (`spike_activation_time.R:51`) |
| Locate the first candidate peak | `np.argmin` on the running working copy :111 | `which.min` (negative-going spikes) |
| Loop while the peak is below threshold | `while True: … if peak >= threshold: break` | `while (1) { … }` |
| Neighbourhood exclusion `G` | `EXCLUSION_HALF = 500` samples :27, applied inside `detect_spikes()` | `min.index.gap = 500` |
| Cap on spikes per channel | `MAX_SPIKES_PER_CHANNEL = 200` :28 | `for (k in 1:200)` |
| Per-channel driver | `detect_all_channels()` :150 | `for (i in 2:ncol(data))` |
| Output point set `{(xᵢ,yᵢ,tᵢ)}` | `spikes_to_table()` :168 | `all.lines$x.observ / y.observ / ts.observ` |
| Grid coordinate mapping | `channel_to_xy()` :32, `GRID_SIDE = 8` :23 | `substr(name,3,4)`-based mapping, row-major |
| **Time scaling `ts / 200`** | `TIME_SCALE = 200` :29, applied in `spikes_to_table()` :184 | `cm_hough_grid_8.R:115` |

---

## Stage 2 — Randomized Hough transform (§2.2, Algorithms 2 & 4)

`src/wave_hough_detect/rht.py` · R: `R_Codes/cm_hough_grid_8.R:197-453`

| Paper step | Implementation | R counterpart |
|---|---|---|
| Plane through three sampled points | `cross_product()` :50, used at :293 | `cross.product(v2-v1, v3-v1)` :275-277 |
| Sign canonicalisation of the normal | :304 (`n̂ ← n̂·sign(n₁)`) | :279-281 |
| Degenerate-normal rejection | :307-313 (2 component pairs by default; all 3 when `strict_degenerate_check=True`) | :284-285 (2 pairs; the R *simulation* copy tests 3) |
| Spherical coordinates | :314-320 — `ρ = \|n̂·p₁\|`, `φ = acos(n₃)`, `θ = asin(n₂/sin φ)` | :289-292 |
| Accumulator discretisation `(ρ, φ, θ)` | `get_key()` :77; `RHO_STEP = 0.05`, `PHI_STEP = THETA_STEP = 2°` :26-28 | `get.key()` :201-207 |
| Truncation toward zero | `_step_key()` :59 — `np.trunc`, **not** `floor` | `as.integer()` |
| Vote | :338 — `accumulator[key].extend(sample_idx)` | :297-300 |
| **Vote threshold** | :341 — `len(...) > vote_threshold * 3`, i.e. **≥ 9 votes** with `vote_threshold = 8` | :301 |
| Least-squares plane refinement | :349-357 — `lstsq([y, t, 1], x)` → `n̂ = (1, −a, −b)`, `ρ = c` | `lm.fit(cbind(pts[,2:3],1), pts[,1])` :305-311 |
| Inlier test "is the point on the plane" | `is_point_on_plane()` :118, `INLIER_TOL = 0.1` :29 | `is.point.on.plane()` :209-213 |
| Collect inliers from the unclassified set | `find_points_on_plane()` :130 | `find.points.on.plane()` :215-225 |
| Accept only if > ⅔ of detectors are on it | `num_unique_detectors()` :146; `min_detectors = 40` (of 64) | :227-236, :319-322 |
| Mark the cluster and remove it | :368-379 | :328-348 |
| Clear the accumulator after acceptance | :391 | :348 |
| Empty-accumulator guard | :395-398 — returns an empty result | **absent in R**: `1:length(...)` degenerates to `1:0` and `if()` errors |
| Master normal = plane with most points | `r_mode()` :157, used at :413 | `mode(positive.plane.indices)` :353-358 |
| Angle between each normal and the master | :416-420 | :364-379 |
| 5° tolerance, `min(Δ, 180−Δ)` | `ANGLE_TOL_DEG = 5.0` :30, applied at :422 | :363, :373 |
| Discarded planes become unclassified | :425-426 | :381-383 |
| Re-assign unclassified points close to a kept plane | :429-446 | :386-404 |
| **Regrow a plane along the master normal** | :448-480, `regrow_master_plane=True`, `SMALL_PLANE_THRESHOLD = 30` :31 | :406-450 |
| Unclassified points are noise | `prediction == 0` throughout | :449-450 |

Two details that only matter away from the experimental data: the tail-regrow block (:448-480)
never executes on it (all 320 points are classified, so the unclassified set is empty), which is
why omitting it produces results that *look* completely correct; and the simulation copy of the R
function uses a ρ step of 0.5 and an inlier tolerance of 1.0 instead of 0.05 and 0.1, matching the
fact that its time axis is not divided by 200.

---

## Stage 3 — Wavefront model fitting (§2.3, Algorithm 3)

`src/wave_hough_detect/fit.py` · R: `R_Codes/cm_hough_grid_8.R:521-779`

| Paper step | Implementation | R counterpart |
|---|---|---|
| Assemble `result.array` per plane and electrode | `build_result_array()` :54 | :521-533 |
| **Scale time back to ms (`×200`)** | `TIME_SCALE_BACK = 200.0` :34, applied at :72 | :528 |
| Extract a plane's point set | `plane_points()` :76 (column-major flatten, keep `t > 0`) | :727-731 |
| Eq. (1) circular arrival time | `circular_loss()` :97 — `√((x−x₀)²+(y−y₀)²)·u − (t−t₀)` | `cone.model.inverted.all()` :550-556 |
| **`u = 1/v` slot convention** | `circular_loss()` :97 uses multiplication by `u`, not division by `v` | `p[[3]]` holds the slowness |
| 4×4 lattice tabulation | `_find_best_point_1d()` :151, `grid_optim()` :177; `GRID_SPLITS = 4` :29 | `grid.optim.find.best.point.1d()` :578-605 |
| Shrink to `[a_{î−1}, a_{î+1}]`, clamp at the bounds | :151-175 | :625-636 |
| `(x₀, y₀)` step, `(v, t₀)` fixed | :286 | :675-676 |
| `(v, t₀)` step by least squares | `vt_optim()` :222-247 — regression of `t` on `r` | `vt.optim.all()` :656-666 |
| Outer loop until convergence | `hybrid_optim()` :249; `HYBRID_EPS = 1e-6` :32, `HYBRID_MAX_ITER = 10000` :33 | :674-688 |
| Search window | `SEARCH_LOWER/UPPER = ±50` :27-28 | :669 |
| Convert back to speed | `fit_circular()` :377 — `v = 1/p[2]` | `'v' = 1/pp.estimate[[3]]` :733 |
| Eq. (2) linear arrival time | `linear_loss()` :120 — `ãx + b̃y + c̃` | `plane.model()` :744-749 |
| Eqs. (5)–(7) least squares | `fit_linear()` :383-393 — `lstsq([x, y, 1], t)` | `lm(ts.observ ~ x.observ + y.observ)` :769 |
| Speed from the linear model | :395 — `v = 1/√(ã²+b̃²)` | :771 |
| Eq. (3)/(4) as an objective | `r_squared()` :132 — `1 − SS_res/SS_tot` | `compute.loss()` :715-721 |

**Data-driven initial guess** — `centroid_initial_guess()` :329 and `fit_circular(..., n_starts=2)`
:346 have no R counterpart. R's alternating minimisation always starts from `(u, t₀) = (1, 1)`,
which fails when the source lies inside the grid; `n_starts=1`, the default, is exactly the R
behaviour and is what reproduces Tables 2 and 4.

---

## Stage 4 — Simulation and evaluation (§3.1, Figures 2–8)

`src/wave_hough_detect/simulate.py` · R: `figure 5` copy `fig5_detection_performance.R`,
`generate_stomach_data.r`, `make_synthetic_recording.R`

| Paper item | Implementation | R counterpart |
|---|---|---|
| Fig. 2–4 — 8×8 linear-wavefront ground truth | `simulate_linear_wavefronts()` :138 | `stomach.sim2d()` `fig5:115` / `generate_stomach_data.r:8` |
| One straight line of the wavefront | `_simulate_one_line()` :83 | `stomach.sim.one()` `fig5:23` |
| One planar wavefront | `_simulate_one_line_2d()` :111 | `stomach.sim.one2d()` `fig5:92` |
| Seed derived from the parameters | `seed_from_params()` :65 — `SEED_BASE × Π(parameters) + shift` | `set.seed(...)` `fig5:128-129` |
| Fig. 5 — FPR / FNR | `detection_rates()` :271, `evaluate_detection()` :327, `detection_sweep()` :374 | `visualize2d()` `fig5:344-375`, `optimality.table()` `fig5:680-705` |
| Four-class labels (TP/FP/FN/TN) | `DetectionResult.classified()` :293 | `stomach.plot2d()` `fig5:320-342` |
| Plane surfaces for Fig. 4 | `draw_planes()` in `examples/demo_simulation.py` | `draw.plane()` `fig5:268-277` |
| Synthetic recording in the §3.2 input format | `simulate_recording()` :397 | `make_synthetic_recording.R` — **its `TIME_UNITS_PER_MS <- 200` is wrong**, see `docs/reproducibility-notes.md` |
| §3.1.2 — 96×96 circular simulation, Figs 6–8 | `simulate_circular.py` — see the next section | `stomach_hough17_newton_square20.r` |

---

## Stage 4b — 96×96 circular-wavefront simulation and accuracy (§3.1.2, Figures 6–8)

`src/wave_hough_detect/simulate_circular.py` ·
R: `R_Codes/stomach_hough17_newton_square20.r`

| Paper item | Implementation | R counterpart |
|---|---|---|
| One 96×96 dataset | `simulate_circular_wavefronts()` :113 | `stomach.sim2d()` :156-198 |
| Per-electrode arrival time | inner double loop :161-170 | `stomach.sim.one2d()` :100-153 |
| Seed from the parameters | `seed_circular()` :94 | `set.seed(100·Πparams)` :160-161 |
| Fit one dataset | `estimate_circular_wavefront()` :283 | `hybrid.optim(sim.matrix)` :671 |
| Search window / tolerance | `SEARCH_LOWER_CIRCULAR`/`UPPER` :75-76, `CIRCULAR_EPS = 1e-8` :77 | ``c(-500,-500)``/``c(500,500)``, `epsilon = 1e-8` :632-635 |
| What is scored (signal + noise) | `WavefrontEstimate.loss` / `.r2` :217-232 | `cone.model.inverted.all` over the whole `sim.matrix` :663, 671 |
| Ground truth | `CIRCULAR_TRUTH` :57 | `list('truth'=c(x,y,miux,ts))` :728 |
| SNR on the x-axis | `WavefrontEstimate.snr` :233 | `pp[i,6]/pp[i,7]` :788 |
| Sweep modes 1/2/3 | `PLOT_MODES` :80-90, `accuracy_sweep()` :339 | `optimality.table()` :878-1010 |
| Fig. 6 — 3-D wavefront | `examples/demo_simulation_circular.py`, part ① | `plot3d`/`points3d` :676-689 (needs `rgl`) |
| Figs 7/8 — accuracy curves | `plot_accuracy()` in the same example | `relerror.plot()` :782-876 |

Three things about this section that are easy to get wrong, all measured rather than assumed:

* **R's fixed initial guess is right here by construction.** The fit starts from `(u, t₀) = (1, 1)`
  and the truth is `v = 1`, `t₀ = 2`. That is why the circular fit converges here (4 iterations)
  while the same routine fails on inside-the-grid sources in §3.1.1.
* **`r2` here is not comparable to Table 4.** R hands the fit the whole matrix *including the
  `z = 0` noise rows*, so with an exactly correct model the paper's own R² is 0.953, not 1. Using
  the signal points alone gives 1.0000000000. Both are reported.
* **The σ sweep is dominated by the noise spikes.** Mode 3 sweeps σ while holding λ_n = 1, and the
  ≈100 outlier spikes contribute far more to the loss than the measurement error does. Measured:
  relative error changes by less than 2× across σ ∈ [0, 1] with the noise present, and by more than
  10× with it switched off.

**Not ported from this file:** `plot.mode ∈ {4, 5, 8}` (interactive 3-D renderings needing
OpenGL/`rgl`; the underlying quantity is already covered by mode 3), `plot.mode ∈ {6, 7}` (they
raise `argument "params" is missing` in R — the 3-D figure path is passed the wrong number of
arguments), and the `newton.cone` / `gra` / `hes` machinery, which is dead code on every active path
(verified: the analytic gradient/Hessian are the exact derivatives of `cone.model`, i.e. of the
*speed* parametrisation, while the active driver uses `cone.model.inverted`, the *slowness* one).

---

## Simulation parameters



### §3.1.1 — 8×8 linear-wavefront simulation

`PAPER_2D_PARAMS` (`simulate.py:50`); R: `generate_stomach_data.r:95-107`, `fig5:345-349`.

| Argument | Value | Paper symbol |
|---|---|---|
| `limit` | 8 | 8×8 MEA |
| `time_max` | 80 | `t_max = 80` (= `limit × 10`) |
| `ts` | 1 | `t₀ = 1` |
| `x`, `y` | 1, 1 | source at `(1, 1)` |
| `u_x`, `u_y` | 1, 1 | one grid step per propagation step |
| `p` | 0.1 | `p = 0.1`, per-electrode miss probability |
| `noise_freq` | 1 | `λ_n = 1`, noise rate |
| `miu`, `sigma` | 1, 0.1 | `μ_x = 1`, `σ_x = 0.1` |
| `gap` | 30 | wavefront separation (χ² degrees of freedom) |
| `ratio` | 2 | `μ_y = 2`, `σ_y = 0.2` across rows |

### §3.2 — experimental recording

R: `R_Codes/cm_hough_grid_8.R`.

| Quantity | Value | Line |
|---|---|---|
| Butterworth high-pass | `butter(2, 1/500, type="high")` | :55 |
| Spike threshold | `quantile(smooth$y, 0.0005)` | :59 |
| Neighbourhood exclusion window | ±500 samples | :60 |
| RHT vote threshold | `8` (tested as `> threshold × 3` → 9 votes) | :265, :301 |
| Accumulator quantisation | `ρ = 0.05`, `φ = θ = 2°` | :202-204 |
| Inlier tolerance | `0.1` | :212 |
| Minimum electrodes per plane | `40` of 64 (≈ ⅔) | :322 |
| Angular tolerance (Algorithm 4) | `5°` | :363 |
| Regrow threshold | `small.plane.threshold = 30` points | :407 |
| Search window | `[-50, 50]²` | :669 |
| Convergence tolerance | `1e-6` | :672 |

---

### §3.1.2 — 96×96 circular-wavefront simulation

`PAPER_CIRCULAR_PARAMS` (`simulate_circular.py:60`); R: `stomach_hough17_newton_square20.r:653-662`.

| Argument | Value | Meaning |
|---|---|---|
| `limit` | 96 | 96×96 grid |
| `time_max` | 96 | observation window |
| `ts` | 2 | source excitation time (the true `t₀`) |
| `x`, `y` | 48, 48 | source position (the true `x₀`, `y₀`); note 48, not the geometric centre 48.5 — coordinates are 1-based |
| `miu_x`, `miu_y` | 1, 1 | propagation speed per grid unit (the true `v`) |
| `p` | 0 | per-electrode miss probability (swept in mode 2) |
| `noise_freq` | 1 | noise rate λ_n (swept in mode 1) |
| `sigma` | 1e-6 | measurement error (swept in mode 3) |
| `gap` | 1e6 | exponential mean of the inter-wavefront interval — large enough that only one wavefront is generated |
| `u_x`, `u_y`, `ratio` | 1, 1, 10 | present in R's signature but **unused** by this simulator |

Sweep grids: mode 1 → λ_n = 2^((−6:16)/2) ∈ [0.125, 256] with `p = 0`, `σ = 1e-6`;
mode 2 → `p = (0:8)/10` with λ_n = 1, `σ = 1e-6`; mode 3 → `σ = (1:10)/10` with λ_n = 1, `p = 0`.
R's committed replicate counts are 23, 9 and 10 respectively.

---
