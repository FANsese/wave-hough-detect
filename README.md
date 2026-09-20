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

This repository is the **Python implementation**. It was ported from the original R scripts and
validated **line by line** against them — see [Verification](#verification) below.

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
真值：源点 (x0, y0) = (-3.7, -50.0)，v = 0.45 格/ms，9 条波前
✅ 平面数 = 波前数              9 vs 9
✅ 每个平面恰好覆盖 64 个电极       [64, 64, 64, 64, 64, 64, 64, 64, 64]
✅ 全部尖峰被归类（无噪声）        576 / 576
✅ 速度 v ≈ 0.45               最大偏差 0.0013
✅ 源点 x0 ≈ -3.7              最大偏差 0.224
✅ 源点 y0 ≈ -50.0             最大偏差 0.000
✅ 激发时刻 t0 与真值差 < 2 ms   最大偏差 0.50 ms
```

### 2. Simulation study of detection performance (paper §3.1, Fig. 5)

```bash
python examples/demo_simulation.py              # 100 random seeds ≈ 2 min
```

Over 100 simulated datasets (`sim_fig5_fpr_fnr.png`, `sim_detection_rates.csv`):

| Quantity | mean | sd | median |
|---|---|---|---|
| FPR (paper's definition) | 0.0674 | 0.0534 | 0.0569 |
| FNR (paper's definition) | 0.0212 | 0.0434 | 0.0000 |
| planes found by RHT | 3.01 | 0.97 | 3 |

> The paper's FPR/FNR use **all points** as the denominator, not the per-class counts. Both
> conventions are reported; see `DetectionRates`.

### 3. Circular-wavefront estimation accuracy (paper §3.1.2, Figs 6–8)

```bash
python examples/demo_simulation_circular.py                # R's replicate counts, ≈ 4 min
python examples/demo_simulation_circular.py --replicates 2 # ≈ 30 s
```

This is a **different simulation** from the one above: 96×96 grid, circular wavefronts, and the
figure of merit is the accuracy of the estimated source position, speed and excitation time. The
truth is set by the code and the fit recovers it:

```
真值   x₀=48.0000  y₀=48.0000  v=1.0000  t₀=2.0000
拟合   x₀=48.0000  y₀=48.0000  v=1.0000  t₀=2.0000     ← σ=0 且无噪声时，误差 < 1e-8
```

Unlike §3.1.1, R's fixed initial guess `(u, t₀) = (1, 1)` is **almost exactly right** here — the
true values are `v = 1` and `t₀ = 2` — which is why the circular fit converges in a handful of
iterations in this section and not in others.

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
│   ├── spikes.py      Stage 1 — filtering & spike activation times      (Algorithm 1)
│   ├── rht.py         Stage 2 — randomized Hough transform              (Algorithms 2 & 4)
│   ├── fit.py         Stage 3 — circular / linear wavefront fitting     (Algorithm 3)
│   ├── simulate.py    Simulation & detection-rate evaluation          (§3.1.1, Figs 2–5)
│   ├── simulate_circular.py  96×96 circular simulation & accuracy        (§3.1.2, Figs 6–8)
│   └── paths.py       Locating the input recording
├── examples/
│   ├── demo_pipeline.py      Real-data pipeline → Figs 9–12 + Tables 2/3/4
│   ├── demo_simulation.py    §3.1.1 → Figs 2–5 + ground-truth validation
│   └── demo_simulation_circular.py  §3.1.2 → Figs 6–8
├── tools/
│   ├── noise_analysis.py     Where each kind of noise is removed (measured, layer by layer)
│   └── compare_ht_vs_rht.py  Standard dense 3-D Hough transform vs RHT, head to head
├── tests/                    pytest suite (runs without the experimental recording)
├── dev/                      Line-by-line comparison against the original R implementation
├── data/                     Data availability statement (no data files are distributed)
└── docs/
    ├── code-paper-mapping.md        Paper ↔ code, item by item
    ├── reproducibility-notes.md     Parameters, seeds, runtime, what is and is not reproducible
    ├── porting-and-validation.md    How the port was verified, and the traps found on the way
    └── spec-circular-simulation.md  Line-by-line reference for §3.1.2 (what was and was not ported)
```

---

## Verification

### Against the original R implementation

The port must behave **identically** to the R code, not merely "similarly". Since R's RNG
(Mersenne-Twister + rejection sampling) cannot be reproduced by numpy, the comparison is done by
having R export **the exact sequence of sampled index triples**, which Python then replays. Every
algorithmic step is therefore compared bit for bit.

| Stage | Result |
|---|---|
| 1 — spike detection | **320 / 320 activation times bit-identical** |
| 2 — RHT | `prediction`, `plane_indices`, `keys` **identical**; `n̂`/`ρ` differ by ~1e-15 (R's `lm.fit` uses QR, numpy's `lstsq` uses SVD); plane-acceptance iterations `18034, 25435, 29765, 30795, 30947` match one by one |
| 3 — fitting | `result.array` differs in **0 cells**; circular model max diff 5.6e-07; linear model 4.0e-12 |

Reproduce with:

```bash
python dev/verify_against_r/step1_spikes.py
python dev/verify_against_r/step2_hough.py
python dev/verify_against_r/step3_fit.py
```

These scripts need the experimental recording and the R-side reference outputs; see
`dev/verify_against_r/_paths.py`.

### Against the published numbers

Tables 2, 3 and 4 are reproduced exactly — all 20 numbers of Table 2 and all 10 R² values of
Table 4, to the precision printed in the paper. The paper's qualitative conclusion is reproduced
too: every `y₀` sits on the `-50` search-window boundary, i.e. the source lies outside the window,
so the linear model should be preferred.

### Against known ground truth

The synthetic-recording path (Quick start #1) checks the whole pipeline against a source position,
speed and excitation schedule that the code itself chose.

### Test suite

```bash
pytest -q          # 84 tests, ≈ 1 min, no experimental data required
```

`tests/test_paper_tables.py` skips itself automatically when the experimental recording is absent,
so a clean clone is green out of the box.

---

## Parameters

Two parameter sets appear in the paper's code. They are **not interchangeable** — the time axis
is scaled by 200 in the real-data pipeline but not in the simulation study, so the accumulator
quantisation and inlier tolerance differ by a factor of 10:

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

These are inherited from the published implementation and are preserved deliberately, because
the port's contract is behavioural equivalence. Each is documented at the point of use and pinned
by a test.

1. **The vote threshold is 9, not 8.** The code tests `length > threshold * 3` and appends three
   indices per vote, so 9 votes are needed. The paper says 8.
2. **`iter.max` must be ≥ 200 000.** The value in the published script (30 000) finds only 3 of the
   5 planes on the experimental data.
3. **`n̂ ← n̂·sign(n₁)` fails when n₁ ≈ 0**, splitting one plane between the φ ≈ 0° and φ ≈ 180°
   buckets. This is the main reason the iteration count is high (a random variable: mean ≈ 13 800
   over 30 seeds, range 8 795–21 425).
4. **θ = asin(n₂ / sin φ) is ill-conditioned** as φ → 0 or 180°: it is a ratio of two small
   quantities. Exactly φ = 0 would give `NaN` in R.
5. **The R code crashes when no plane is found** — `1:length(...)` degenerates to `1:0`. The port
   returns an empty result instead.
6. **The circular fit is sensitive to its starting point.** R initialises `(u, t₀) = (1, 1)`,
   independent of the data. This works when the source lies well outside the grid (the paper's
   case) but converges to a wrong local minimum when the source is inside it. `fit_circular(...,
   n_starts=2)` additionally tries a data-driven initial guess and keeps the better fit;
   `n_starts=1` (default) is the faithful R behaviour.
7. **On this dataset the wavefront planes are nearly parallel to the `(x, y)` plane.** Within one
   plane the Hough-scale `t` spans only ≈ 0.12, against an inlier tolerance of 0.1. So a constant-`t`
   plane explains a wavefront almost as well as the true one, and all of the velocity information
   sits in the ~0.6° tilt of the normal. The method works, but its separation margin on this
   recording is thin — worth knowing before reading the R² values as large margins.
8. **`/200` is a conditioning scale, not a unit conversion.** The time column of the input file must
   be in the same unit as the paper's (milliseconds, spanning a few thousand), so that `t/200` is
   comparable to the grid coordinates `x, y ∈ [1, 8]`. Feeding a time column in units of 1/200 ms
   makes the Hough geometry meaningless — the pipeline then reports `v ≈ 0` and `t₀ ≈ ±2×10⁷`.
   Measured, not theorised.

9. **In §3.1.2 the noise spikes, not the measurement error, dominate the error.** The fit is given
   every point including the noise spikes (`z = 0`), and it is scored with the paper's own loss. With
   λ_n = 1 there are ≈ 100 outlier spikes whose residuals dwarf the per-electrode measurement error,
   so sweeping σ from 0.1 to 1.0 barely moves the estimate — measured, the relative error changes by
   under 2× while with the noise switched off it grows by more than 10×. If Figure 8 is the σ sweep,
   it is measuring outlier contamination rather than measurement error. (Which mode Figure 8
   corresponds to is itself ambiguous — see `docs/spec-circular-simulation.md` §7.)
10. **R's simulations are not independent replicates.** The seed is derived as
    `100 × Π(parameters)`, and `p` is one of the factors — so with `p = 0` (which modes 1 and 3 both
    fix) the seed is **0 for every replicate**. Worse, `optimality.table` re-runs the same parameter
    set, so all its "replicates" are the *same dataset*: measured spread between them is exactly 0.
    The default here adds a `seed_shift` per replicate, giving genuinely independent repeats;
    `accuracy_sweep(..., independent_replicates=False)` restores R's behaviour so the two can be
    compared.

Two implementation details that are easy to get wrong and are pinned by tests: R's `as.integer()`
truncates **toward zero** (not `floor`), and R's `order()` is a **stable** sort while pandas'
`sort_values()` is not by default (the experimental data has 47 tied `t` values covering 98
points, and an unstable sort silently makes exported sampling traces point at the wrong points).

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
  directions are excluded. A vertical plane such as `y = const` covers exactly 40 points — one per
  cell of a grid row per wavefront — which is exactly the acceptance threshold. R's Algorithm 2
  already rejects `n ⊥ t` (`dot(n, (0,0,1)) == 0`, `cm_hough_grid_8.R:288`); without that filter a
  standard implementation finds nothing, which is an artefact of the implementation rather than a
  property of the method.
* The wavefront planes in this dataset are almost parallel to the `(x, y)` plane: within one plane,
  the Hough-scale `t` spans only ≈ 0.12 while the inlier tolerance is 0.1. The velocity information
  therefore lives entirely in the *tiny* tilt of the normal — `n̂ ≈ (0.005, 0.010, ±0.9999)`, i.e.
  `φ ≈ 0.6°`. This is why the iteration count is high, and why `φ → 0` ill-conditioning (limitation
  4 below) is not a theoretical concern here but the dominant numerical difficulty.

---

## Data availability

The experimental recording analysed in §3.2 is **not distributed with this repository**. It was
produced by a collaborating laboratory and the authors do not hold the rights to redistribute it.
See [`data/README.md`](data/README.md) for the full statement, the recording protocol reference,
the exact input format, and how to substitute your own recording.

Everything that depends only on code is reproducible from this repository alone: the simulation
study (§3.1), the synthetic recording, and the full three-stage pipeline. Only the numerical values
of Tables 2, 3 and 4 depend on the experimental file.

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

Not yet chosen — see the repository owner. Until a license is added, the default is all rights
reserved.
