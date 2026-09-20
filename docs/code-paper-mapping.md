# Code ↔ paper mapping

Every line number in this document refers to a file committed in this repository and points at the
line where the named symbol is defined: `spikes.py` 89 is the `def` line of `butter_highpass()`.
Steps that live inside a larger function are described as such rather than cited by an internal line
number, so the mapping stays readable and stays true.

---

## Coverage: what is implemented, and what is not

This package implements the paper's method end to end. Every algorithm, figure and table below is
produced by the modules in `src/wave_hough_detect/` driven by the scripts in `examples/`; nothing in
this document depends on code outside this repository.

| Paper item | Where it is implemented | Product |
|---|---|---|
| §2.1, Algorithm 1 — Butterworth high-pass and spike activation times | `src/wave_hough_detect/spikes.py` | the `(x, y, t)` point cloud every later stage consumes |
| §2.2, Algorithms 2 and 4 — randomized Hough transform over the point cloud | `src/wave_hough_detect/rht.py` | plane normals, offsets, and a signal/noise label per point |
| §2.3, Algorithm 3 — wavefront model fitting | `src/wave_hough_detect/fit.py` | circular and linear fits, their speeds, and R² |
| Fig. 9 style — spike detection on one channel (§2.1) | `examples/demo_pipeline.py` fig. 1 | `fig1_traces.png` |
| Fig. 10 style — planes detected on the experimental recording | `examples/demo_pipeline.py` fig. 2 | `fig2_planes.png` |
| Figs 11–12 style — fitted circular and linear wavefronts, and fit quality | `examples/demo_pipeline.py` figs 3–4 | `fig3_directions.png`, `fig4_fit_quality.png` |
| Table 2 — per-plane circular fit, §3.2 recording | `examples/demo_pipeline.py` | `table2_circular.csv` |
| Table 3 — per-plane linear fit | `examples/demo_pipeline.py` | `table3_linear.csv` |
| Table 4 — R² per plane | `examples/demo_pipeline.py` | `table4_r2.csv` |
| §3.1.1 — 8×8 linear-wavefront simulation, Figs 2–5 | `src/wave_hough_detect/simulate.py`, `examples/demo_simulation.py` | `sim_fig2_ground_truth.png` … `sim_fig5_fpr_fnr.png` |
| §3.1.2 — 96×96 circular-wavefront simulation, Figs 6–8 | `src/wave_hough_detect/simulate_circular.py`, `examples/demo_simulation_circular.py` | `circ_fig6_wavefront.png`, `circ_fig7_*.png`, `circ_fig8_accuracy_vs_p.png` |
| §3.2 input format — a synthetic stand-in for the experimental recording | `simulate_recording()` (`simulate.py` 422) | a CSV in the same layout, time column in milliseconds |
| Noise budget of the recording (a diagnostic, not a paper item) | `tools/noise_analysis.py` | which stage removes which kind of noise |
| Randomized vs standard dense Hough transform (a diagnostic) | `tools/compare_ht_vs_rht.py` | cost and output of both transforms on the same spike cloud |

### Deliberately out of scope

| Paper item | Why it is out of scope here |
|---|---|
| Interactive 3-D renderings of the wavefront (`plot.mode ∈ {4, 5, 8}`, §3.1.2) | They need an interactive OpenGL viewer. The quantity they show is already computed and plotted via `plot.mode = 3`; the interactive viewer itself, and the manual exploration it allows, are the parts not provided. |
| Figure-only modes (`plot.mode ∈ {6, 7}`, §3.1.2) | Their only product is a figure, and the quantities they would show are covered by modes 1–3 and by the §3.1.1 evaluation. |
| A LOWESS detrending branch as an alternative to the high-pass filter | Not implemented. A high-pass filter is the right tool for baseline drift here: LOWESS is a smoother and would flatten the sharp spikes that the next stage is meant to detect. `tools/noise_analysis.py` documents the reasoning and the frequency-band measurements behind it. |
| A standard (non-randomized) dense Hough transform as the detector | Not part of the paper's method. The dense transform is included only as a comparison, in `tools/compare_ht_vs_rht.py`. |
| A Newton/gradient-based optimiser for the wavefront fit | The analytic gradient and Hessian belong to the **speed** parametrisation, while the fit is written in the **slowness** form (`u = 1/v`), so they are not the derivatives of the function being minimised. The tabulated grid search plus least squares (`grid_optim()` and `vt_optim()`) is used instead. |
| The gastric slow-wave analysis | Not part of the method in the paper. |

## Stage 1 — Butterworth filtering and spike activation times (§2.1, Algorithm 1)

`src/wave_hough_detect/spikes.py`

| Paper step | Implementation |
|---|---|
| Butterworth high-pass design | `butter_highpass()` (`spikes.py` 89) — `scipy.signal.butter(2, 1/500, btype="high")` |
| Filter order | `BUTTER_ORDER = 2` (`spikes.py` 24) — the second argument of `butter()` is the order |
| Cutoff | `BUTTER_CUTOFF = 1/500` (`spikes.py` 25) — a fraction of Nyquist, i.e. 10 Hz at a 10 kHz sampling rate |
| Single-pass IIR filtering | `filter_channel()` (`spikes.py` 101) — `scipy.signal.lfilter`, one forward pass, no phase compensation |
| Threshold at the 99.95th percentile | `SPIKE_QUANTILE = 0.0005` (`spikes.py` 26), applied inside `detect_spikes()` as `np.quantile(filtered, 0.0005)` |
| Locate the first candidate peak | `detect_spikes()` (`spikes.py` 116) — the global minimum of the running working copy, via `np.argmin`; spikes in this recording are negative-going |
| Loop while the peak is below threshold | `detect_spikes()` — the loop stops as soon as the minimum is at or above the threshold |
| Neighbourhood exclusion `G` | `EXCLUSION_HALF = 500` samples (`spikes.py` 27), applied inside `detect_spikes()`: samples within ±500 of an accepted spike are blanked, so the same spike is not found twice |
| Cap on spikes per channel | `MAX_SPIKES_PER_CHANNEL = 200` (`spikes.py` 28) |
| Per-channel driver | `detect_all_channels()` (`spikes.py` 159) — column 1 is time, columns 2…65 are electrodes |
| Output point set `{(xᵢ,yᵢ,tᵢ)}` | `spikes_to_table()` (`spikes.py` 177) — also carries the channel index, and sorts stably |
| Grid coordinate mapping | `channel_to_xy()` (`spikes.py` 32), `GRID_SIDE = 8` (`spikes.py` 23) — row-major |
| **Time scaling `t / 200`** | `TIME_SCALE = 200` (`spikes.py` 29), applied inside `spikes_to_table()`. This is a conditioning scale, not a unit conversion — see `docs/reproducibility-notes.md` |
| Discarded LOWESS alternative | Documented only, not implemented (see the out-of-scope table above) |

---

## Stage 2 — Randomized Hough transform (§2.2, Algorithms 2 & 4)

`src/wave_hough_detect/rht.py`

| Paper step | Implementation |
|---|---|
| Plane through three sampled points | `cross_product()` (`rht.py` 51), applied inside `hough_plane()` to three distinct indices drawn without replacement |
| Sign canonicalisation of the normal | Inside `hough_plane()` — `n̂ ← n̂·sign(n₁)`. The rule is ill-conditioned when `n₁ ≈ 0`, which is why one plane can be split across the φ ≈ 0° and φ ≈ 180° buckets |
| Degenerate-normal rejection | Inside `hough_plane()` — two component pairs by default; all three when `strict_degenerate_check=True` |
| Spherical coordinates | Inside `hough_plane()` — `ρ = \|n̂·p₁\|`, `φ = acos(n₃)`, `θ = asin(n₂/sin φ)` |
| Accumulator discretisation `(ρ, φ, θ)` | `get_key()` (`rht.py` 82); `RHO_STEP = 0.05`, `PHI_STEP = THETA_STEP = 2°` (`rht.py` 25-27) |
| Truncation toward zero | `_step_key()` (`rht.py` 60) — `np.trunc`, **not** `floor`, so a negative ρ lands in the bucket it belongs to |
| Vote | Inside `hough_plane()` — three indices are appended to the bucket of the plane's key |
| **Vote threshold** | Inside `hough_plane()` — a plane is accepted when `len(bucket) > vote_threshold * 3`, and since each vote appends three indices that is **≥ 9 votes** with the default `vote_threshold = 8`. The effective threshold is 9, not the paper's 8 |
| Least-squares plane refinement | Inside `hough_plane()` — `lstsq([y, t, 1], x)` over every point in the bucket, giving `n̂ = (1, −a, −b)`, `ρ = c` |
| Inlier test "is the point on the plane" | `is_point_on_plane()` (`rht.py` 128), `INLIER_TOL = 0.1` (`rht.py` 28), applied inside `hough_plane()` |
| Collect inliers from the unclassified set | `find_points_on_plane()` (`rht.py` 140) — only points not yet assigned to a plane are considered |
| Accept only if > ⅔ of detectors are on it | `num_unique_detectors()` (`rht.py` 157), tested inside `hough_plane()` against `min_detectors = 40` of 64 |
| Mark the cluster and remove it | Inside `hough_plane()` — accepted points receive the plane index and label 1, so they leave the unclassified set |
| Log the accepted plane | Inside `hough_plane()` — iteration, votes, detector count, normal and ρ are appended to `HoughResult.accept_log` |
| Clear the accumulator after acceptance | Inside `hough_plane()` — the accumulator starts empty for the next plane |
| Empty-accumulator guard | `hough_plane()` returns an empty `HoughResult` (`rht.py` 229) when no plane is accepted, instead of raising; callers check `HoughResult.n_planes` |
| Master normal = plane with most points | `r_mode()` (`rht.py` 168) over the accepted plane indices, inside `hough_plane()` |
| Angle between each normal and the master | Inside `hough_plane()` — the angle between each accepted normal and the master normal |
| 5° tolerance, `min(Δ, 180−Δ)` | `ANGLE_TOL_DEG = 5.0` (`rht.py` 29); the smaller of the angle and its supplement is compared with it |
| Discarded planes become unclassified | Inside `hough_plane()` — their points are relabelled 0 |
| Re-assign unclassified points close to a kept plane | Inside `hough_plane()` — each kept plane re-claims the remaining points within the inlier tolerance |
| **Regrow a plane along the master normal** | Inside `hough_plane()`, `regrow_master_plane=True`, `SMALL_PLANE_THRESHOLD = 30` (`rht.py` 30) — the block that follows Algorithm 4 and is not described in the paper. See the limitations in `docs/reproducibility-notes.md` |
| Unclassified points are noise | `prediction == 0` throughout |

Two details that only matter away from the experimental recording: the tail-regrow block never
executes on it (all 320 points are classified, so the unclassified set is empty), which is why
disabling it still produces results that *look* completely correct; and the simulation preset
`SIM_PRESET` (`rht.py` 40) uses a ρ step of 0.5 and an inlier tolerance of 1.0
instead of 0.05 and 0.1, because the simulation study's time axis is not divided by 200 and its `t`
values are correspondingly larger. `evaluate_detection()` applies `SIM_PRESET` automatically.

---

## Stage 3 — Wavefront model fitting (§2.3, Algorithm 3)

`src/wave_hough_detect/fit.py`

| Paper step | Implementation |
|---|---|
| Assemble `result.array` per plane and electrode | `build_result_array()` (`fit.py` 53) |
| **Scale time back to ms (`×200`)** | `TIME_SCALE_BACK = 200.0` (`fit.py` 34), applied inside `build_result_array()` |
| Extract a plane's point set | `plane_points()` (`fit.py` 78) — column-major flatten, keeps `t > 0` |
| Eq. (1) circular arrival time | `circular_loss()` (`fit.py` 99) — `√((x−x₀)²+(y−y₀)²)·u − (t−t₀)` |
| **`u = 1/v` slot convention** | `circular_loss()` multiplies by `u`, the slowness; the speed is recovered only at the end |
| 4×4 lattice tabulation | `_find_best_point_1d()` (`fit.py` 156), `grid_optim()` (`fit.py` 183); `GRID_SPLITS = 4`, `GRID_EPS = 1e-8`, `GRID_MAX_ITER = 1000` (`fit.py` 29-31) |
| Shrink to `[a_{î−1}, a_{î+1}]`, clamp at the bounds | `grid_optim()` — each pass narrows the search interval around the best lattice point of the previous pass |
| `(x₀, y₀)` step, `(v, t₀)` fixed | Inside `hybrid_optim()` — a `grid_optim()` pass on `(x₀, y₀)` with the slowness and `t₀` held fixed |
| `(v, t₀)` step by least squares | `vt_optim()` (`fit.py` 231) — regression of `t` on the radius `r` |
| Outer loop until convergence | `hybrid_optim()` (`fit.py` 258); `HYBRID_EPS = 1e-6`, `HYBRID_MAX_ITER = 10000` (`fit.py` 32-33) |
| Search window | `SEARCH_LOWER`/`SEARCH_UPPER` = ±50 (`fit.py` 27-28) |
| Convert back to speed | `fit_circular()` (`fit.py` 364) — the fitted slowness is inverted, `v = 1/u` |
| Eq. (2) linear arrival time | `linear_loss()` (`fit.py` 124) — `ãx + b̃y + c̃` |
| Eqs. (5)–(7) least squares | `fit_linear()` (`fit.py` 405) — `lstsq([x, y, 1], t)` |
| Speed from the linear model | `fit_linear()` — `v = 1/√(ã²+b̃²)` |
| Eq. (3)/(4) as an objective | `r_squared()` (`fit.py` 136) — `1 − SS_res/SS_tot` |

**Initial guess** — the alternating minimisation always starts from `(u, t₀) = (1, 1)`, independent
of the data, which is a poor start when the source lies inside the grid.
`centroid_initial_guess()` (`fit.py` 343) is available through
`n_starts`: `fit_circular(..., n_starts=2)` tries both starts and keeps the lower final loss. The
default `n_starts=1` is the setting that reproduces Tables 2 and 4; the sensitivity to the starting
point is a property of this fit and is listed in the limitations of `docs/reproducibility-notes.md`.

---

## Stage 4 — Simulation and evaluation (§3.1, Figures 2–8)

`src/wave_hough_detect/simulate.py`

| Paper item | Implementation |
|---|---|
| Fig. 2–4 — 8×8 linear-wavefront ground truth | `simulate_linear_wavefronts()` (`simulate.py` 144) |
| One straight line of the wavefront | `_simulate_one_line()` (`simulate.py` 86) |
| One planar wavefront | `_simulate_one_line_2d()` (`simulate.py` 116) |
| Seed derived from the parameters | `seed_from_params()` (`simulate.py` 68) — `SEED_BASE × Π(parameters) + shift`, with `SEED_BASE_2D = 101` (`simulate.py` 49) |
| Fig. 5 — FPR / FNR | `detection_rates()` (`simulate.py` 287), `evaluate_detection()` (`simulate.py` 349), `detection_sweep()` (`simulate.py` 397) |
| Four-class labels (TP/FP/FN/TN) | `DetectionResult.classified()` (`simulate.py` 326) |
| Plane surfaces for Fig. 4 | `draw_planes()` in `examples/demo_simulation.py` |
| Synthetic recording in the §3.2 input format | `simulate_recording()` (`simulate.py` 422) — the time column is written in milliseconds (`time_units_per_ms = 1.0`), pinned by `tests/test_simulate.py` |
| §3.1.2 — 96×96 circular simulation, Figs 6–8 | `simulate_circular.py` — see the next section |

---

## Stage 4b — 96×96 circular-wavefront simulation and accuracy (§3.1.2, Figures 6–8)

`src/wave_hough_detect/simulate_circular.py`

| Paper item | Implementation |
|---|---|
| One 96×96 dataset | `simulate_circular_wavefronts()` (`simulate_circular.py` 127) |
| Per-electrode arrival time | Inside `simulate_circular_wavefronts()` — a row-major double loop over the 96×96 electrodes |
| Seed from the parameters | `seed_circular()` (`simulate_circular.py` 107) — `SEED_BASE_CIRCULAR × Π(parameters) + seed_shift`, `SEED_BASE_CIRCULAR = 100` (`simulate_circular.py` 66) |
| Fit one dataset | `estimate_circular_wavefront()` (`simulate_circular.py` 311) |
| Search window / tolerance | `SEARCH_LOWER_CIRCULAR` / `SEARCH_UPPER_CIRCULAR` = ∓500 (`simulate_circular.py` 87-88), `CIRCULAR_EPS = 1e-8` (`simulate_circular.py` 89) |
| What is scored (signal + noise) | `WavefrontEstimate.loss` / `.r2` (`simulate_circular.py` 219) — both are computed over every point of the dataset |
| Ground truth | `CIRCULAR_TRUTH` (`simulate_circular.py` 69) — `x₀ = y₀ = 48`, `v = 1`, `t₀ = 2` |
| SNR on the x-axis | `WavefrontEstimate.snr` (`simulate_circular.py` 256) — the ratio of signal points to noise points, not an amplitude ratio |
| Sweep modes 1/2/3 | `PLOT_MODES` (`simulate_circular.py` 92), `accuracy_sweep()` (`simulate_circular.py` 370) |
| Fig. 6 style — 3-D wavefront | `examples/demo_simulation_circular.py`, part ① |
| Figs 7/8 style — accuracy curves | `plot_accuracy()` in the same example |

Three things about this section that are easy to get wrong, all measured rather than assumed:

* **The default initial guess is right here by construction.** The fit starts from `(u, t₀) = (1, 1)`
  and the truth is `v = 1`, `t₀ = 2`. That is why the circular fit converges here (4 iterations)
  while the same routine fails on inside-the-grid sources in §3.1.1, where
  `fit_circular(..., n_starts=2)` is the remedy.
* **`r2` here is not comparable to Table 4.** The fit is handed the whole dataset *including the
  `z = 0` noise rows*, so with an exactly correct model R² is 0.953, not 1. Scoring the signal
  points alone gives 1.0000000000.
* **The σ sweep is dominated by the noise spikes.** Mode 3 sweeps σ while holding λ_n = 1, and the
  ≈100 outlier spikes contribute far more to the loss than the measurement error does. Measured:
  relative error changes by less than 2× across σ ∈ [0, 1] with the noise present, and by more than
  10× with it switched off.

**Out of scope here:** `plot.mode ∈ {4, 5, 8}` (interactive 3-D renderings needing an OpenGL viewer;
the underlying quantity is already covered by mode 3), `plot.mode ∈ {6, 7}` (figure-only modes), and
the Newton-route gradient/Hessian machinery — the analytic gradient and Hessian are the exact
derivatives of the *speed* parametrisation, while the fit here uses the *slowness* form, so the
tabulated grid search plus least squares is used instead.

---

## Simulation parameters

### §3.1.1 — 8×8 linear-wavefront simulation

`PAPER_2D_PARAMS` (`simulate.py` 53).

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

| Quantity | Value | Where |
|---|---|---|
| Butterworth high-pass | `butter(2, 1/500, btype="high")` | `BUTTER_ORDER`, `BUTTER_CUTOFF` (`spikes.py` 24-25), applied by `butter_highpass()` |
| Spike threshold | 0.05th percentile of the filtered signal | `SPIKE_QUANTILE` (`spikes.py` 26), applied inside `detect_spikes()` |
| Neighbourhood exclusion window | ±500 samples | `EXCLUSION_HALF` (`spikes.py` 27) |
| RHT vote threshold | `8`, tested as `> 8 × 3` → 9 votes | `hough_plane()` (`rht.py` 229) |
| Accumulator quantisation | `ρ = 0.05`, `φ = θ = 2°` | `RHO_STEP`, `PHI_STEP`, `THETA_STEP` (`rht.py` 25-27) |
| Inlier tolerance | `0.1` | `INLIER_TOL` (`rht.py` 28) |
| Minimum electrodes per plane | `40` of 64 (≈ ⅔) | `min_detectors` of `hough_plane()`, checked with `num_unique_detectors()` |
| Angular tolerance (Algorithm 4) | `5°` | `ANGLE_TOL_DEG` (`rht.py` 29) |
| Regrow threshold | 30 points | `SMALL_PLANE_THRESHOLD` (`rht.py` 30) |
| Search window | `[−50, 50]²` | `SEARCH_LOWER`, `SEARCH_UPPER` (`fit.py` 27-28) |
| Convergence tolerance | `1e-6` | `HYBRID_EPS` (`fit.py` 32) |
| Iteration cap | `max_iter = 200000` on this recording | `hough_plane()` (`rht.py` 229); the default is 30 000, see the limitations in `docs/reproducibility-notes.md` |

---

### §3.1.2 — 96×96 circular-wavefront simulation

`PAPER_CIRCULAR_PARAMS` (`simulate_circular.py` 72).

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
| `u_x`, `u_y`, `ratio` | 1, 1, 10 | accepted by the simulator's signature but **unused** by its body |

Sweep grids: mode 1 → λ_n = 2^(k/2) for k = −6 … 16, i.e. λ_n ∈ [0.125, 256], with `p = 0` and
`σ = 1e-6`; mode 2 → `p = 0.0, 0.1, …, 0.8` with λ_n = 1 and `σ = 1e-6`; mode 3 →
`σ = 0.1, 0.2, …, 1.0` with λ_n = 1 and `p = 0`. The default replicate counts of `accuracy_sweep()`
are 23, 9 and 10 respectively, and are the counts behind the runtimes quoted in
`docs/reproducibility-notes.md`.

---
