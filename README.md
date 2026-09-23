# wave-hough-detect

Code accompanying the paper:

> **Robust Wave Origin Detection from Sensor Array Data via Randomized Hough Transform and
> Model Fitting**
> Sicheng Fan, Jiayi Lu, Xiaodan Fan
> arXiv:2609.13248 [eess.SP] — <https://arxiv.org/abs/2609.13248>

The method estimates the **origin, propagation direction, speed and excitation time** of
wave-like signals recorded by a sensor array (e.g. an 8×8 microelectrode array, MEA). It is a
closed-loop **detection → separation → fitting** procedure:

| Stage | What it does | Paper |
|---|---|---|
| **1. Spike extraction** | Butterworth high-pass filter + high-percentile threshold + neighbourhood exclusion → activation times `{(xᵢ, yᵢ, tᵢ)}` | §2.1, Algorithm 1 |
| **2. Randomized Hough transform (RHT)** | Random 3-point sampling → plane normal → `(ρ, φ, θ)` accumulator → vote threshold → inlier recovery. Each plane = one propagating wavefront | §2.2, Algorithms 2 & 4 |
| **3. Model fitting** | Circular (near-field) wavefront by alternating minimization; linear (far-field) wavefront by least squares | §2.3, Algorithm 3 |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot,test]"          # numpy, scipy, pandas (+ matplotlib, pytest)
```

Python ≥ 3.10. Runtime dependencies are only `numpy`, `scipy`, `pandas`; `matplotlib` is needed
by the examples only.

---

## Quick start

### 1. Full pipeline on a synthetic recording (no data download needed)

```bash
python examples/demo_simulation.py --seeds 20
```

Part ③ of that script generates a 64-channel recording in the **exact format** of the
experimental file, then runs stages 1→2→3 on it. Because the recording is generated with a known
source position, speed and excitation-time schedule, the fit can be checked against the truth —
and it passes:

```
Ground truth: source (x0, y0) = (-3.7, -50.0), v = 0.45 cells/ms, 9 wavefronts
    ✅ number of planes = number of wavefronts     9 vs 9
    ✅ every plane covers exactly 64 electrodes    points per plane [64, 64, 64, 64, ...]
    ✅ all spikes classified (no noise left over)  576 / 576
    ✅ speed v ≈ 0.45                              max deviation 0.0013
    ✅ source x0 ≈ -3.7                            max deviation 0.224
    ✅ source y0 ≈ -50.0                           max deviation 0.000
    ✅ firing time t0 within 2 ms of the ground truth  max deviation 0.50 ms
```

### 2. Simulation study of detection performance (paper §3.1.1, Fig. 5)

```bash
python examples/demo_simulation.py              # 100 random seeds, ≈ 2 min
```

Over 100 simulated datasets (`sim_fig5_fpr_fnr.png`, `sim_detection_rates.csv`):

| Quantity | mean | sd | median |
|---|---|---|---|
| FPR (paper's definition) | 0.0657 | 0.0493 | 0.0563 |
| FNR (paper's definition) | 0.0195 | 0.0410 | 0.0000 |
| planes found by RHT | 2.90 | 0.70 | 3 |

These are reproducible bit for bit: `evaluate_detection` seeds the randomized
transform from `seed_shift`, so a repeated sweep returns identical numbers.

> The paper's FPR/FNR use **all points** as the denominator, not the per-class counts. Both
> conventions are reported; see `DetectionRates`.

### 3. Circular-wavefront estimation accuracy (paper §3.1.2, Figs 6–8)

```bash
python examples/demo_simulation_circular.py                # default replicate counts, ≈ 4 min
python examples/demo_simulation_circular.py --replicates 2 # ≈ 30 s
```

A **different simulation** from the one above: 96×96 grid, circular wavefronts, and the figure of
merit is the accuracy of the estimated source position, speed and excitation time. The truth is set
by the code, so every estimate can be compared against it.

The default run has ≈ 100 noise spikes in the window, and it is scored over signal *and* noise
(limitation 11), so the estimates are close to but not equal to the truth:

```
truth    x0=48.0000  y0=48.0000  v=1.0000  t0=2.0000
default  x0=47.9629  y0=47.9566  v=1.0140  t0=2.6289    (94 noise spikes, converged in 4 iterations)
```

With the noise switched off, the same fit recovers the truth exactly — this is the check that the
estimator itself is right:

```python
evaluate_circular(1, noise_freq=1e-9)   # -> x0=48.0000 y0=48.0000 v=1.0000 t0=2.0000
```

The fixed initial guess `(u, t₀) = (1, 1)` happens to be almost exactly right for this simulation
— the true values are `v = 1` and `t₀ = 2` — which is why the fit converges in a handful of
iterations here. Limitation 6 describes the case where it does not.

### 4. Reproducing the paper's Tables 2, 3 and 4 (requires the experimental recording)

```bash
python examples/demo_pipeline.py --data /path/to/recording.csv
```

Takes ≈ 12 s and writes 4 figures plus `table2_circular.csv`, `table3_linear.csv`,
`table4_r2.csv` and `tables.md`. With the experimental recording in place the numbers match the
published tables digit for digit (`pytest tests/test_paper_tables.py`).

---

## Repository layout

```
wave-hough-detect/
├── src/wave_hough_detect/
│   ├── spikes.py              Stage 1 — filtering & spike activation times    (Algorithm 1)
│   ├── rht.py                 Stage 2 — randomized Hough transform            (Algorithms 2 & 4)
│   ├── fit.py                 Stage 3 — circular / linear wavefront fitting   (Algorithm 3)
│   ├── simulate.py            §3.1.1 — 8×8 linear wavefronts, FPR/FNR         (Figs 2–5)
│   ├── simulate_circular.py   §3.1.2 — 96×96 circular wavefronts, accuracy    (Figs 6–8)
│   └── paths.py               Locating the input recording
├── examples/
│   ├── demo_pipeline.py      Real-data pipeline → Figs 9–12 + Tables 2/3/4
│   ├── demo_simulation.py    §3.1.1 → Figs 2–5 + ground-truth validation
│   └── demo_simulation_circular.py  §3.1.2 → Figs 6–8
├── tools/
│   ├── noise_analysis.py     Where each kind of noise is removed (measured, layer by layer)
│   └── compare_ht_vs_rht.py  Standard dense 3-D Hough transform vs RHT, head to head
├── tests/                    pytest suite (runs without the experimental recording)
├── data/                     Data availability statement (no data files are distributed)
└── docs/
    ├── code-paper-mapping.md        Paper ↔ code, item by item
    └── reproducibility-notes.md     Parameters, seeds, runtime, what is and is not reproducible
```

---

## Verification

### Against the published numbers

Tables 2, 3 and 4 are reproduced exactly — all 20 numbers of Table 2 and all 10 R² values of
Table 4, to the precision printed in the paper. The paper's qualitative conclusion is reproduced
too: every `y₀` sits on the `-50` search-window boundary, i.e. the source lies outside the window,
so the linear model should be preferred.

```bash
python examples/demo_pipeline.py --data /path/to/recording.csv   # prints Tables 2/3/4
pytest tests/test_paper_tables.py -q                             # checks them digit by digit
```

### Against known ground truth

Two independent checks, neither of which needs experimental data:

* The synthetic-recording path (Quick start 1) recovers a source position, speed and excitation
  schedule that the generator itself chose. `tests/test_pipeline_end_to_end.py` runs the same
  check on a short recording and additionally compares every detected spike time against the
  analytic arrival time `t = t₀ + √((x−x₀)²+(y−y₀)²)/v`.
* The circular simulation (Quick start 3) recovers `(x₀, y₀, v, t₀) = (48, 48, 1, 2)` to 1e-8 when
  the measurement error is switched off, and its arrival times match the analytic model to 4e-6
  at σ = 1e-6.

### Test suite

```bash
pytest -q          # no experimental data required, ≈ 1 min
```

`tests/test_paper_tables.py` skips itself automatically when the experimental recording is absent,
so a clean clone is green out of the box.

---

## Parameters

Two parameter sets appear in the code and they are **not interchangeable**: the time axis is
divided by 200 before the Hough transform, and the two studies use different time scales, so the
accumulator quantisation and the inlier tolerance differ by a factor of 10.

| Parameter | Real data (§3.2, Tables 2–4) | Simulation (§3.1, Figs 2–8) |
|---|---|---|
| Accumulator ρ step | 0.05 | 0.5 |
| Inlier tolerance | 0.1 | 1.0 |
| Degenerate-normal test | 2 pairs of components | all 3 pairs |
| Vote threshold | `> 8×3` → **9 votes** | same |
| Minimum electrodes per plane | 40 of 64 (≈ ⅔) | same |
| Angular tolerance (Algorithm 4) | 5° | same |
| `iter.max` | **≥ 200 000** | — |

In code, the real-data values are the defaults and the simulation set is
`wave_hough_detect.rht.SIM_PRESET`; `evaluate_detection` applies it automatically.

---

## Known limitations

Each item below is a property of this implementation, documented at the point of use and pinned by
a test where a test can reach it.

1. **The vote threshold is 9, not the 8 stated in the paper.** The acceptance test is
   `len(bucket) > vote_threshold * 3` and every vote appends three indices, so nine votes are
   needed before a plane is accepted.
2. **`n_planes` alone does not tell you whether the planes came from the voting loop.** There are
   two routes to a plane — the main accumulator loop, and the tail regrowth that reuses the master
   normal — and only the first is recorded in `HoughResult.accept_log`. On the experimental
   recording at `max_iter=30000` the result still reports 5 planes of 64 points each and *still
   reproduces Tables 2 and 4*, but only 2 of them came from the voting loop; the other 3 were
   grown. Check `len(res.accept_log)` when it matters. The examples pass `max_iter=200000`, which
   is enough for all 5 to be voted in for 26 of 30 seeds.
3. **The sign canonicalisation of the plane normal fails when its first component is near zero.**
   The rule `n̂ ← n̂·sign(n₁)` cannot disambiguate in that case, so one plane can be split between
   the φ ≈ 0° and φ ≈ 180° buckets. On the experimental recording this is the main reason the
   iteration count is high, and that count is a random variable rather than a constant: mean
   over 30 seeds it has mean 43 833, median 26 066, range 21 915–200 000, and 3 of the 30 seeds
   exhaust the 200 000 iteration cap.
4. **`θ = asin(n₂ / sin φ)` is ill-conditioned** as φ → 0 or 180°: it is a ratio of two small
   quantities. Exactly φ = 0 would produce `NaN`, so a deterministic key is substituted there;
   the geometry is unaffected because the key is only a label.
5. **An empty result is returned when no plane is accepted**, rather than raising. Callers should
   check `HoughResult.n_planes` before using the plane normals.
6. **The fit's initial guess is fixed, not data-driven.** The alternating minimisation starts from
   `(u, t₀) = (1, 1)`, independent of the data. This works when the source lies well outside the
   grid, and it is almost exactly right for the §3.1.2 simulation (true values `v = 1`, `t₀ = 2`),
   but it converges to a wrong local minimum when the source is inside the grid: measured on
   synthetic data, a source at `(2.5, 3.5)` with `t₀ = 100` converges to `x₀ = −44.7`, `v = 0.96`.
   `fit_circular(..., n_starts=2)` additionally tries a centroid-based initial guess and keeps
   whichever fit has the lower final loss; the default `n_starts=1` is the setting that reproduces
   Tables 2 and 4.
7. **On this dataset the wavefront planes are nearly parallel to the `(x, y)` plane.** Within one
   plane the Hough-scale `t` spans only ≈ 0.12, against an inlier tolerance of 0.1. So a
   constant-`t` plane explains a wavefront almost as well as the true one, and all of the velocity
   information sits in the ≈ 0.6° tilt of the normal. The method works, but its separation margin
   on this recording is thin — worth knowing before reading the R² values as large margins.
8. **`/200` is a conditioning scale, not a unit conversion.** The time column of the input file must
   be in the same unit as the paper's (milliseconds, spanning a few thousand), so that `t/200` is
   comparable to the grid coordinates `x, y ∈ [1, 8]`. Feeding a time column in units of 1/200 ms
   makes the Hough geometry meaningless — the pipeline then reports `v ≈ 0` and `t₀ ≈ ±2×10⁷`.
   Measured, not theorised.
9. **The σ sweep only measures the measurement error if the noise rate is held at ~0.** The fit is
   given every point including the noise spikes (`z = 0`) and is scored with the paper's own loss,
   so with λ_n = 1 there are ≈ 100 outlier spikes whose residuals dwarf the per-electrode error:
   measured, the relative error moves by under 2× across σ = 0.1…1.0 and `|Δt₀|` even *decreases*.
   `accuracy_sweep` therefore holds λ_n = 1e-6 for the σ mode, and the error then grows in
   proportion to σ (0.00085 → 0.00849 in `|Δx₀|` for σ = 0.1 → 1.0). If you re-enable noise there,
   the plot measures outlier contamination instead.
10. **The simulation replicates are independent here, and that matters.** The seed is derived from
    the parameters, and `p` is one of the factors — so with `p = 0`, which sweep modes 1 and 3 both
    fix, a naive seed derivation gives the same seed for every replicate and the "repeats" are the
    same dataset. `accuracy_sweep` therefore adds a per-replicate `seed_shift` by default;
    `independent_replicates=False` removes it, and then the measured spread between replicates is
    exactly zero.
11. **`r2` in §3.1.2 is computed over signal *and* noise points**, because the fit is given the
    whole matrix. With an exactly correct model the value is ≈ 0.953, not 1; using the signal
    points alone gives 1.0000000000. It is not comparable to the R² column of Table 4.
12. **The SNR axis of Figure 7 is a count ratio, not an amplitude ratio.** It is
    `n_signal / n_noise`. In the §3.1.2 configuration `n_signal = 9216`, so the axis is effectively
    `96 / λ_n`, which reproduces the published range [0.375, 768].
13. **Two numerical details are easy to get wrong and are pinned by tests.** Integer truncation in
    the accumulator keys must be **toward zero**, not `floor` — otherwise negative-ρ planes land
    one step away in the wrong bucket. And the spike table must be sorted **stably**: the
    experimental recording has 47 tied `t` values covering 98 points. An unstable sort reorders
    them; on the experimental recording that was measured to leave the plane count at 5 and change
    only the iteration count, but any procedure that indexes points by position (for example feeding
    a fixed sequence of sampled indices) then addresses the wrong points, which is why the requirement
    is kept.

---

## RHT vs a standard dense Hough transform

`tools/compare_ht_vs_rht.py` runs both on the same spike cloud with the same parametrisation,
quantisation, plane-acceptance rule and refinement. Measured on the experimental data at 2°
angular resolution:

| | Standard HT | RHT |
|---|---|---|
| Vote operations | 7 776 000 | 33 565 |
| Accumulator memory | 52.6 MB (`1624 × 90 × 90` int32) | hash table, < 100 keys at peak |
| Time | 0.78 s | 0.54 s |
| Planes found | 5 | 5 |

The honest conclusion is **not** "RHT is more robust": at this size and resolution both find the
same five planes, equally well, in the same order of time. The difference is how the cost scales.
The dense accumulator is an `n_ρ × n_φ × n_θ` array — halving the angular step multiplies both its
memory and its vote count by four — whereas RHT's hash table only ever stores the keys that were
actually sampled, so its cost is essentially independent of angular resolution. On large point
clouds at fine resolution the standard transform runs into a memory wall; RHT does not.

Two things this comparison makes concrete:

* The standard transform's global accumulator peak is **not** a wavefront unless degenerate
  directions are excluded. A vertical plane such as `y = const` contains 40 *points* — one per cell
  of a grid row per wavefront — but the acceptance test counts distinct **detectors** `(x, y)`, and
  the same 8 cells repeat across the 5 wavefronts, so it has only 8. The failure is not that it
  passes the test but that it *is* the global peak, its least-squares refinement then yields no
  inliers, and a peak search with a small retry budget never reaches a real wavefront. Directions whose
  normal is orthogonal to the time axis (`φ = 90°`) must be excluded, because such a plane carries
  no time component and cannot represent a wavefront. Without that filter a standard implementation
  finds nothing, which is an artefact of the implementation rather than a property of the method.
* The wavefront planes in this dataset are almost parallel to the `(x, y)` plane: within one plane,
  the Hough-scale `t` spans only ≈ 0.12 while the inlier tolerance is 0.1. The velocity information
  therefore lives entirely in the *tiny* tilt of the normal — `n̂ ≈ (0.005, 0.010, ±0.9999)`, i.e.
  `φ ≈ 0.6°`. This is why the iteration count is high, and why the `φ → 0` ill-conditioning
  (limitation 4) is not a theoretical concern here but the dominant numerical difficulty.

---

## Data availability

The experimental recording analysed in §3.2 is **not distributed with this repository**. It was
produced by a collaborating laboratory and the authors do not hold the rights to redistribute it.
See [`data/README.md`](data/README.md) for the full statement, the recording protocol reference,
the exact input format, and how to substitute your own recording.

Everything that depends only on code is reproducible from this repository alone: the two simulation
studies, the synthetic recording, and the full three-stage pipeline. Only the numerical values of
Tables 2, 3 and 4 depend on the experimental file.

---

## Citation

```bibtex
@article{fan2026wavehough,
  title  = {Robust Wave Origin Detection from Sensor Array Data via Randomized
            Hough Transform and Model Fitting},
  author = {Fan, Sicheng and Lu, Jiayi and Fan, Xiaodan},
  journal = {arXiv preprint arXiv:2609.13248},
  year   = {2026}
}
```

## License

Not yet chosen. Until a license is added, the default is all rights reserved.
