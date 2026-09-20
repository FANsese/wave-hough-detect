# Reproducibility notes

What can be reproduced from this repository alone, what cannot, and why.

---

## 1. Determinism

| Path | Deterministic? | Seeded by |
|---|---|---|
| Stage 1 — spike detection | **yes**, exactly | no randomness at all |
| Stage 2 — RHT | **no**, it is a randomised algorithm | `seed=` argument (numpy `default_rng`), or an explicit `trace` |
| Stage 3 — fitting | yes, given the stage-2 output | no randomness |
| §3.1.1 simulation | yes, given `seed_shift` | `SEED_BASE × Π(parameters) + seed_shift` |
| §3.1.2 simulation | yes, given `seed_shift` | `100 × Π(parameters) + seed_shift` |
| Synthetic recording | yes, given `seed` | `seed=20260915` by default |

The RHT is randomised by design: the number of iterations needed is a **random variable**. Over 30
seeds on the experimental data it has mean ≈ 13 800 and range 8 795–21 425 — a factor of 2.4. This
is a property of the method, not of the implementation. All 30 seeds found all 5 planes.

Within one Python process, passing the same `seed` reproduces the same result. Across R and Python
it does **not**, because the random number generators are different — see section 3.

---

## 2. Exact commands

```bash
# Full pipeline on a synthesised recording, with a ground-truth check  (~2 min)
python examples/demo_simulation.py --seeds 100

# Simulation study only, fewer seeds                                        (~10 s)
python examples/demo_simulation.py --seeds 5

# The paper's tables (needs the experimental recording)                      (~12 s)
python examples/demo_pipeline.py --data /path/to/recording.csv

# Where noise is removed, layer by layer                                    (~1 min)
python tools/noise_analysis.py --data /path/to/recording.csv

# Standard dense Hough transform vs RHT                                     (~1 s)
python tools/compare_ht_vs_rht.py --data /path/to/recording.csv

# Test suite, no experimental data needed                                   (~1 min)
pytest -q
```

---

## 3. R ↔ Python: statistically equivalent, never bit-wise

R's `sample()` uses Mersenne-Twister with rejection sampling; numpy uses PCG64. There is no way to
make numpy emit R's stream. Two consequences:

* **The port's logic is verified bit for bit** by having R export the exact sequence of sampled
  index triples and replaying it in Python (`dev/verify_against_r/`). Every step *except* the
  random draw is compared exactly — see `docs/porting-and-validation.md` for the results.
* **The simulation study is statistically equivalent only.** `simulate_linear_wavefronts()`
  reproduces R's *sampling structure* (which draw happens when, from which distribution) and R's
  *seed arithmetic*, so a given parameter set is reproducible within Python, but the datasets
  themselves differ from R's. The FPR/FNR distributions agree in character, not in digits.

This is a hard boundary. Do not describe the simulation layer as "matching R".

---

## 4. The two time scales, and the trap they create

The pipeline uses two different scalings of the time axis, and mixing them up produces wrong
answers **without any error message**:

```
raw time (ms)  --/200-->  Hough scale  --hough_plane-->  planes
                              |
                              +--×200-->  back to ms  -->  result.array  -->  fitting
```

The conversion points are `spikes_to_table()` (`TIME_SCALE = 200`) and `build_result_array()`
(`TIME_SCALE_BACK = 200.0`).

Why `/200`: the Hough transform treats `(x, y, t)` as a 3-D point cloud and applies a single inlier
tolerance (0.1) and a single ρ quantisation step (0.05) to all three axes. With the experimental
recording, `x, y ∈ [1, 8]` and `t ∈ [4.8, 38.9]` after division — comparable magnitudes. Without the
division, `σ` within a plane is ≈ 5.35 against a tolerance of 0.1 and **not a single plane is
found**.

Why the input's time column must be in milliseconds: because the *fitted* times are the original
column values, and `v` comes out in grid-units per column-unit. The paper reports `v ≈ 0.44`, which
is grid-units per millisecond. Writing the time column in units of 1/200 ms moves the Hough-scale
`t` to ≈ 900–8700 while `x, y` stay in `[1, 8]` — three orders of magnitude apart. Measured result:
14 planes instead of 9, every normal pointing along `x` or along a grid diagonal, `v ≈ 0`,
`t₀ ≈ ±2×10⁷`.

The released R script `make_synthetic_recording.R` sets `TIME_UNITS_PER_MS <- 200` and its comment
recommends commenting out the `/200` line when it is set to 1. **Both halves of that advice are
wrong.** The time column must be in ms and the `/200` must stay. The Python port therefore defaults
to `time_units_per_ms=1.0` and this is pinned by
`tests/test_simulate.py::test_simulate_recording_time_column_is_milliseconds`.

---

## 5. Runtimes measured on this machine

Apple M1 MacBook Air, Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5.

| Task | Time | Notes |
|---|---|---|
| Read the 55 MB / 90 000 × 65 CSV | ≈ 3 s | |
| Stage 1, all 64 channels | 0.5 s | |
| Stage 2, RHT (200 000 iteration cap) | 0.6 s | ≈ 33 000 iterations actually used |
| Stage 3, five planes, circular + linear | 9.9 s | the circular fits dominate |
| **Full pipeline, end to end** | **11.8 s** | `examples/demo_pipeline.py` |
| §3.1.1 detection evaluation, one dataset | 0.69 s | |
| §3.1.1 sweep, 100 seeds | 69 s | |
| §3.1.2 one dataset (9 297 points) + fit | 0.02 s + 0.17 s | converges in 4 iterations |
| §3.1.2 full three sweeps (R's replicate counts) | ≈ 4 min | 529 + 81 + 100 replicates |
| Synthetic recording generation | ≈ 4 s | 5.76 M normal draws |
| Test suite (84 tests) | ≈ 60 s | |

Peak memory for the real-data pipeline is dominated by the voltage matrix
(90 000 × 64 float64 = 46 MB) plus the filtered copy. The accumulator of the standard dense Hough
transform at 2° resolution is 52.6 MB (`1624 × 90 × 90` int32); RHT's is a hash table holding fewer
than 100 keys at peak.

---

## 6. The state of the original R code

The paper's experiments are spread over 6 R files, 2 760 lines. Three points matter for
reproducibility:

1. **There are two copies of `hough.plane`, with different parameters.** The real-data pipeline
   (`cm_hough_grid_8.R`) uses ρ step 0.05 and inlier tolerance 0.1; the simulation copy
   (`fig5_detection_performance.R`) uses 0.5 and 1.0, and also tests one more degenerate-normal
   condition. The factor of 10 is explained by the missing `/200` in the simulation. The Python port
   exposes both as parameter sets (`RHO_STEP`/`INLIER_TOL` and `SIM_PRESET`) and there is a test
   asserting they differ.
2. **`iter.max = 30000` in the published script is too small.** It finds 3 of the 5 planes on the
   experimental data. All results here use `max_iter=200000`.
3. **The tail-regrow block (`cm_hough_grid_8.R:406-450`) is absent from the paper's description** of
   Algorithm 4. It is in the code and runs unconditionally, but on the experimental data its loop
   body never executes (all 320 points are already classified). It *does* execute in the simulation
   study, where FNR ≠ 0. The port implements it, default-on, and there are tests that pin both the
   on and off behaviour.

### Files that exist but were not ported

| R file | Status |
|---|---|
| `stomach_hough17_newton_square20.r` (§3.1.2, Figs 6–8) | **ported** — `simulate_circular.py`. Not ported from it: `plot.mode ∈ {4,5,8}` (3-D `rgl` renderings), `plot.mode ∈ {6,7}` (they error in R), and the dead `newton.cone`/`gra`/`hes` machinery. See `docs/code-paper-mapping.md`. |
| `stomach_plane_anglediff14.r` | a different copy of the same material; not ported |
| `spike_activation_time_plot2.R`, `BUTTER_example_plot_3.r` | figure-only scripts; their logic is folded into `examples/demo_pipeline.py` |
| the gastric slow-wave analysis | not part of the method in the paper |

### Documented bugs in the original, reproduced deliberately

| # | Issue | Where | Reproduced? |
|---|---|---|---|
| 1 | Vote threshold is 9, not the 8 stated in the paper | `length > threshold*3` | yes |
| 2 | `iter.max` too small to find all planes | R:265 | exposed via `max_iter`; examples pass 200000 |
| 3 | `n̂ ← n̂·sign(n₁)` fails for `n₁ ≈ 0`, splitting a plane across φ ≈ 0° and 180° | R:279-281 | yes; documented as the main cause of high iteration counts |
| 4 | `θ = asin(n₂/sin φ)` is ill-conditioned as φ → 0; `φ = 0` gives `NaN` | R:292 | yes, except that a deterministic key is substituted for the `NaN` key (geometry unaffected) |
| 5 | `1:length(...)` degenerates to `1:0` when no plane is found; R then errors | R:353-358 | no — the port returns an empty result (documented at `rht.py:395`) |
| 6 | FNR uses the wrong index (`num.true.neg = length(red.indices)`) | `fig5:369` | no — the port computes TN correctly; the variable is unused in R so nothing depends on it |
| 7 | `make_synthetic_recording.R` time-column units | `TIME_UNITS_PER_MS <- 200` | no — corrected to ms, see section 4 |

---

## 7. Environment used for the numbers in this repository

```
Python 3.11.15
numpy   2.4.6
scipy   1.17.1
pandas  3.0.5
matplotlib 3.11.2
pytest  9.1.1
```

`pandas` is used only for tabular I/O and the spike table. One pandas default differs from R in a
way that silently corrupts results: `sort_values()` is **not stable**, while R's `order()` is. The
experimental data has 47 tied `t` values covering 98 points; `spikes_to_table()` therefore always
passes `kind="stable"`. Without it the RHT finds only 4 planes instead of 5, and replayed sampling
traces index the wrong points — with no error raised.
