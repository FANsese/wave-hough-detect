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
seeds on the experimental data it has mean 43 833, median 26 066 and range 21 915–200 000; 3 of
the 30 seeds exhaust the 200 000 iteration cap. This is a property of the method, not of the
implementation. All 30 seeds report 5 planes.

What *does* vary with the seed is whether the numbers match the published tables. Checked on the
experimental recording: 26 of 30 seeds reproduce Table 2 exactly, and seeds 6, 18, 22 and 26 do
not (worst case `|Δx₀| = 8.59`). The examples use `seed=1`, which reproduces. If you change the
seed, check the structure before trusting the numbers — every plane must cover all 64 electrodes.

Within one Python process, passing the same `seed` reproduces the same result exactly, and passing
an explicit `trace` — a pre-drawn sequence of sampled index triples — bypasses the random number
generator altogether. The stages marked deterministic above take no seed because they draw nothing.

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

## 3. The two time scales, and the trap they create

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

Why the input's time column must be in milliseconds: because the *fitted* times stay in the unit of
that column, and `v` comes out in grid-units per column-unit. The paper reports `v ≈ 0.44`, which
is grid-units per millisecond. Writing the time column in units of 1/200 ms moves the Hough-scale
`t` to ≈ 900–8700 while `x, y` stay in `[1, 8]` — three orders of magnitude apart. Measured result:
14 planes instead of 9, every normal pointing along `x` or along a grid diagonal, `v ≈ 0`,
`t₀ ≈ ±2×10⁷`.

The time column must therefore be in ms and the `/200` must stay. `simulate_recording()`
(`simulate.py` 422) defaults to `time_units_per_ms=1.0` for exactly
this reason, and the choice is pinned by
`tests/test_simulate.py::test_simulate_recording_time_column_is_milliseconds`.

---

## 4. Runtimes measured on this machine

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
| §3.1.2 full three sweeps (default replicate counts) | ≈ 4 min | 529 + 81 + 100 replicates |
| Synthetic recording generation | ≈ 4 s | 5.76 M normal draws |
| Test suite (84 tests) | ≈ 60 s | |

Peak memory for the real-data pipeline is dominated by the voltage matrix
(90 000 × 64 float64 = 46 MB) plus the filtered copy. The accumulator of the standard dense Hough
transform at 2° resolution is 52.6 MB (`1624 × 90 × 90` int32); RHT's is a hash table holding fewer
than 100 keys at peak.

---

## 5. Known limitations, and what they mean for reproduction

Each item is a property of this implementation, measured here and pinned by a test wherever a test
can reach it. They are the settings a reader has to respect when re-running the pipeline.

1. **The vote threshold is effectively 9, not the 8 stated in the paper.** `hough_plane()`
   (`rht.py` 229) accepts a plane when `len(bucket) > vote_threshold * 3`, and
   every vote appends three indices, so nine votes are needed. Counting votes against the paper's 8
   is off by one.
2. **Count planes by `len(HoughResult.accept_log)`, not by `n_planes`.** Planes can be produced by
   the voting loop or grown by the tail regrowth, and only the former is logged. At
   `max_iter=30000` the experimental recording still reports 5 planes of 64 points and still
   reproduces Tables 2 and 4, but only 2 were voted in. The example scripts pass
   `max_iter=200000`, which suffices for 26 of 30 seeds.
3. **The sign canonicalisation of the plane normal is ill-conditioned when its first component is
   near zero.** The rule `n̂ ← n̂·sign(n₁)` cannot disambiguate in that case, so one plane can be
   split between the φ ≈ 0° and φ ≈ 180° buckets. On this recording it is the main reason the
   iteration count is high, and that count is a random variable rather than a constant — see
   section 1.
4. **`θ = asin(n₂ / sin φ)` is ill-conditioned as φ → 0 or 180°**, being a ratio of two small
   quantities. Exactly φ = 0 would produce `NaN`; a deterministic key is substituted there instead,
   which is safe because the key is only a label and the geometry is unaffected. The accumulator
   keys also truncate **toward zero** rather than toward −∞, so that a negative ρ lands in the
   bucket it belongs to.
5. **An empty result is returned when no plane is accepted**, rather than an exception. Callers
   should check `HoughResult.n_planes` before using the plane normals, since every downstream stage
   assumes at least one plane.
6. **The fit's starting point is fixed, not data-driven, and the fit is sensitive to it.** The
   alternating minimisation in `fit_circular()` (`fit.py` 364) starts from
   `(u, t₀) = (1, 1)` regardless of the data. That start is nearly exact for the §3.1.2 simulation
   (true `v = 1`, `t₀ = 2`) but converges to a wrong local minimum when the source lies inside the
   grid. `n_starts=2` adds a centroid-based start and keeps the lower final loss; `n_starts=1`, the
   default, is the setting that reproduces Tables 2 and 4.
7. **The tail-regrow block is part of this implementation and is on by default.** The block that
   follows Algorithm 4 inside `hough_plane()` is not described in the paper; it is reached through
   `regrow_master_plane=True` (the default) and accepts a further plane along the master normal when
   it covers more than `SMALL_PLANE_THRESHOLD` points. On the experimental recording its loop body
   never executes, because all 320 points are already classified — which is why switching it off
   still produces results that look correct. In the §3.1 simulation study, where FNR ≠ 0, it does
   execute. Tests pin both the on and the off behaviour, so the paper's figures should be read
   against the setting they were produced with.
8. **Two parameter sets appear in the pipeline and they are not interchangeable.** The defaults
   (ρ step 0.05, inlier tolerance 0.1, two degenerate-normal component pairs) belong to the
   real-data path, whose time axis is divided by 200; the simulation study uses `SIM_PRESET` (ρ step
   0.5, inlier tolerance 1.0, all three component pairs) because its time axis is not divided by 200
   and its `t` values are correspondingly larger. `evaluate_detection()` applies `SIM_PRESET`
   automatically; a test asserts that the two sets differ.
9. **The `r2` of §3.1.2 and the R² column of Table 4 are not the same quantity.** The circular fit is
   handed the whole dataset, noise rows included, so an exactly correct model scores R² ≈ 0.953
   rather than 1; scoring the signal points alone gives 1.0000000000.
10. **The simulation replicates are independent here, and that matters.** The seed is derived from
    the parameters and `p` is one of the factors, so with `p = 0` — which sweep modes 1 and 3 both
    fix — an unshifted derivation would give every replicate the same dataset.
    `accuracy_sweep()` therefore adds a per-replicate `seed_shift` by default;
    `independent_replicates=False` removes it, and the spread between replicates is then exactly
    zero.
11. **The paper's FPR and FNR use every point as the denominator**, not the points of the relevant
    class. `DetectionRates` reports that convention in `false_positive_rate` /
    `false_negative_rate`, and the per-class rates in `false_positive_rate_among_noise` /
    `false_negative_rate_among_signal`. Which convention is meant has to be decided before a figure
    is compared with Fig. 5.
12. **The spike table must be sorted stably** — see section 6. An unstable sort reorders the 47 tied
    `t` values of the experimental recording, which changes the plane count and makes any replayed
    sampling trace index the wrong points, silently.
13. **The input's time column must be in milliseconds.** See section 3 for the measurement behind
    this.

---

## 6. Environment used for the numbers in this repository

```
Python 3.11.15
numpy   2.4.6
scipy   1.17.1
pandas  3.0.5
matplotlib 3.11.2
pytest  9.1.1
```

`pandas` is used only for tabular I/O and the spike table. One of its defaults silently corrupts
results here: `sort_values()` is **not stable**. The experimental data has 47 tied `t` values
covering 98 points, and `spikes_to_table()` therefore always passes `kind="stable"`. Without it the
RHT finds only 4 planes instead of 5, and replayed sampling traces index the wrong points — with no
error raised.
