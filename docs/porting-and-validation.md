# Porting and validation

This document records how the Python implementation was checked against the original R code, what
the checks found, and which behaviours are deliberate reproductions of R quirks rather than
improvements.

The porting contract was: **behave identically to the R code, not merely similarly.** Any numerical
difference had to be explained, not absorbed into a looser tolerance.

---

## 1. Why a naive comparison is impossible, and what was done instead

R's `sample()` uses Mersenne-Twister with rejection sampling. numpy's `default_rng` uses PCG64.
There is no seed, no compatibility mode and no reimplementation that makes the two emit the same
stream. A direct "run both and compare" test is therefore meaningless for any randomised step.

The workaround: **export the randomness, then replay it.**

The instrumented R script `R_work/run2_trace.R` records the index triple drawn at *every* iteration
of the RHT main loop and writes it to `trace.csv` (30 947 draws on the experimental data).
`hough_plane(..., trace=...)` then consumes that sequence instead of drawing its own. Everything
downstream of the draw — the cross product, the sign canonicalisation, the spherical coordinates,
the quantisation key, the voting, the least-squares refinement, the inlier test, the detector
coverage test, plane acceptance, accumulator clearing, the 5° rule, the unclassified-point
recovery — is then compared exactly.

For stage 1 there is no randomness at all, so the comparison is direct.

---

## 2. Results

### Stage 1 — spike detection (`dev/verify_against_r/step1_spikes.py`)

```
R       : 320 个尖峰
Python  : 320 个尖峰
✅ 总数一致: 320
✅ 逐通道计数一致 (64 个通道)
✅ 通道 → (x,y) 映射与 R 一致
逐点时刻差: 最大 0.0000000000 ms
完全相同的点: 320 / 320
✅ 全部 320 个尖峰时刻【逐位相同】
```

Zero tolerance, no rounding. The script also reports a control: replacing `lfilter` with `filtfilt`
(zero-phase, two-pass) happens to give the same spike times on this recording, but produces a
different waveform — so it would diverge on other data.

### Stage 2 — randomized Hough transform (`dev/verify_against_r/step2_hough.py`)

```
✅ prediction           最大差 0.000e+00
✅ plane_indices        最大差 0.000e+00
✅ keys                 完全一致
✅ n1 / n2 / n3         最大差 2.2e-15 / 3.7e-15 / 3.3e-16
✅ rhos                 最大差 7.1e-14
✅ 平面数                R=5  Python=5
✅ 各平面点数            P1..P5 各 64 点
平面接受时机（迭代次数）: 18034, 25435, 29765, 30795, 30947   ← 与 R 逐个吻合
```

The residual ~1e-15 in the plane normals comes from the least-squares solver: R's `lm.fit` uses a
QR decomposition, numpy's `lstsq` uses SVD. On a well-conditioned problem the two agree to machine
precision, and the difference does not propagate: the inlier sets are identical, so `prediction` and
`plane_indices` are exact.

### Stage 3 — wavefront fitting (`dev/verify_against_r/step3_fit.py`)

```
✅ 与 R 的 result.array 逐格对比: 0 处不同
圆波前模型（论文表 2 + 表 4 圆模型列）   最大差 5.57e-07
线波前模型（论文表 3 + 表 4 线模型列）   最大差 1.76e-11
论文表 2 的 20 个数字                  全部吻合 ✅
论文表 3 的 20 个数字                  全部吻合 ✅
论文表 4 的 10 个 R²                   全部吻合到 7 位小数 ✅
```

The circular-model residual (~5e-7) is the alternating minimisation's stopping rule, not a porting
error: `hybrid_optim` stops when the *relative* change of `(x₀, y₀, u, t₀)` falls below 1e-6, and
`t₀` is of order 10³, so the absolute resolution in `x₀` is bounded by roughly 1e-3. Tightening
`eps` to 1e-10 moves the fit to within 1e-4 of the analytic optimum on synthetic data — this is
pinned by `tests/test_fit.py::test_fit_circular_precision_is_limited_by_the_stopping_rule`.

### §3.1.2 — circular simulation (no R reference available, so validated against truth)

The 96×96 circular-wavefront section cannot use the trace-replay trick: its randomness is spread over
one normal draw per grid cell plus a conditional uniform, not a single sampling step. It is therefore
checked against the **analytic model the simulator is built from**, plus the structure of R's code:

```
信号点数 = 9216 = 96×96            ← p = 0 时每个电极都收到，与真值换算一致
|ts − (t₀ + 距离/v)| 最大 = 4.0e-06  ← σ = 1e-6，量级正确
σ = 0 且无噪声时：x₀/y₀/v/t₀ 全部恢复到 1e-8 以内
σ 增大时误差线性增长（0.1→2.0 使 x₀ 误差涨 20 倍）
snr 范围 [0.375, 768] 与 R 实测范围逐个吻合
```

That last line is the strongest available cross-check with R: the SNR axis of Figure 7 is
`n_signal / n_noise`, and the mode-1 grid `λ_n = 2^((−6:16)/2)` reproduces R's reported range
exactly, which pins both the definition and the sweep grid.

### Independent check: known ground truth

The verification above shows the port matches R. It does not show that R is right. The synthetic
recording closes that gap: the generator fixes the source position `(−3.7, −50)`, the speed
`0.45` grid-units/ms and the excitation schedule, and the pipeline recovers all of them.

```
✅ 平面数 = 波前数              9 vs 9
✅ 每个平面恰好覆盖 64 个电极       [64, 64, 64, 64, 64, 64, 64, 64, 64]
✅ 全部尖峰被归类（无噪声）        576 / 576
✅ 速度 v ≈ 0.45               最大偏差 0.0013
✅ 源点 x0 ≈ -3.7              最大偏差 0.224
✅ 源点 y0 ≈ -50.0             最大偏差 0.000
✅ 激发时刻 t0 与真值差 < 2 ms   最大偏差 0.50 ms
```

`tests/test_pipeline_end_to_end.py` runs the same check on a shorter recording, and adds a
sample-level comparison against the analytic arrival times
`t = t_wave + √((x−x₀)²+(y−y₀)²)/v`.

---

## 3. Traps found during the port

Each of these produced a silently wrong answer — no exception, no warning. All are now pinned by
tests.

### 3.1 pandas' `sort_values()` is not stable; R's `order()` is

The experimental data has **47 tied `t` values covering 98 points**. With an unstable sort, the row
order differs from R's, and every subsequent index-based operation — including replaying R's
exported sampling trace — points at the wrong rows. The symptom was RHT finding only 4 planes
instead of 5. Fix: `sort_values("t", kind="stable")` in `spikes_to_table()`.

### 3.2 R's `as.integer()` truncates toward zero, not toward −∞

`as.integer(-4.81374 / 0.05)` is `-96`; `floor(-96.2748)` is `-97`. A one-step difference in the
accumulator key, and negative-ρ planes land in the wrong bucket. `_step_key()` uses `np.trunc`.

### 3.3 `-0.0` formats as `"-0"` in Python but `"0"` in R

R's `as.integer()` returns an *integer*, so `-0.0` becomes `0L` and the key string is `"0"`. Python
formats `-0.0` as `"-0"` with `:g`. Bucket identity is unaffected, but the key strings differ, which
would break any string-level comparison with R. `_step_key()` normalises `-0.0` to `0.0`.

### 3.4 The tail-regrow block is missing from the paper's description

`cm_hough_grid_8.R:406-450` regrows planes along the master normal after Algorithm 4 has finished.
The paper does not describe it. On the experimental data the loop body never runs — all 320 points
are classified, so the unclassified set is empty — which means omitting it produces results that
look completely correct. It only matters when FNR ≠ 0, i.e. in the simulation study. The first
version of the port omitted it; it is now implemented and tested both ways
(`tests/test_hough.py::test_tail_regrow_*`).

### 3.5 There are two `hough.plane` copies with different parameters

The real-data pipeline uses ρ step 0.05 and inlier tolerance 0.1. The simulation copy uses **0.5 and
1.0** — a factor of 10, because its time axis is not divided by 200. Mixing them up raises no error;
it just quietly finds nothing or merges planes. Both are now explicit parameter sets with a test
asserting they differ, plus a test showing that the tight set loses inliers on simulation-scaled
data.

### 3.6 The fit's initial guess is not data-driven

R initialises the alternating minimisation with `(u, t₀) = (1, 1)`. This works for the paper's
geometry (source ≈ 50 grid units outside an 8×8 array) but converges to a wrong local minimum when
the source is inside the grid. Measured, with `max_iter` unbounded:

| Truth `(x₀, y₀, v, t₀)` | R's default | `n_starts=2` |
|---|---|---|
| `(−3.0, −50, 0.45, 800)` (paper-like) | converges, 329 rounds ✅ | same ✅ |
| `(4.5, 4.5, 0.45, 200)` | converges, 463 rounds ✅ | same ✅ |
| `(2.5, 3.5, 0.45, 100)` | `x₀ = −44.7`, `v = 0.96` ❌ | exact, 14 rounds ✅ |
| `(4.0, 4.0, 0.45, 5000)` | `x₀ = −50`, `v = 2.16` ❌ | exact, 2 rounds ✅ |

The default (`n_starts=1`) is the faithful behaviour and is required to reproduce Tables 2 and 4.
`n_starts=2` additionally tries a centroid-based initial guess and keeps whichever fit has the lower
final loss; it never degrades the paper-like cases (pinned by a test).

### 3.7 The synthetic-recording generator's time units

Covered in full in `docs/reproducibility-notes.md` §4. Short version: the released R script writes
the time column in units of 1/200 ms, which moves the Hough-scale time three orders of magnitude
away from the grid coordinates. Running the pipeline on it gives 14 nonsense planes, `v ≈ 0` and
`t₀ ≈ ±2×10⁷`. The port writes milliseconds, and part ③ of `examples/demo_simulation.py` validates
the result against the generator's own ground truth.

### 3.8 A trace of collinear points produces no votes at all

R skips any sampled triple whose cross product is zero. Three points from the same row of the grid
are exactly collinear, so a fixed trace built from them produces **zero** votes and the RHT returns
nothing. This cost some debugging time while writing the tail-regrow test; it is now noted in the
test itself, since anyone constructing a trace by hand will hit it.

### 3.9 Iteration cap and randomness

`iter.max` must be ≥ 200 000 for the experimental data; the published 30 000 finds 3 of 5 planes.
And the iteration count is a random variable, not a constant: over 30 seeds it has mean ≈ 13 800 and
range 8 795–21 425. Any claim of the form "the RHT converges in N iterations" is false; only the
distribution is meaningful.

---

## 4. Deliberate deviations from R

Everything below is a change, and each is marked in the code.

| # | Change | Reason |
|---|---|---|
| 1 | Empty result instead of an error when no plane is found | R's `1:length(...)` → `1:0` makes `if()` fail; there is no meaningful behaviour to reproduce |
| 2 | Deterministic key when `φ = 0` instead of an `NA`-bearing key | R computes `asin(n₂/sin 0) = NaN` and stores a `"NaN"` key but still reassigns the points; the key does not affect geometry, so the observable behaviour is preserved without `NaN` strings |
| 3 | Correct `n_true_neg` | R has `num.true.neg = length(red.indices)`; the variable is unused, so nothing depends on it |
| 4 | `time_units_per_ms=1.0` default in `simulate_recording()` | R's default is wrong, see 3.7 |
| 5 | `n_starts` and `centroid_initial_guess()` | Optional; default preserves R |
| 6 | `strict_degenerate_check`, `rho_step`, `inlier_tol` parameters | Needed to express both R copies with one implementation; defaults match the real-data copy |
| 7 | The 90 000-row synthetic recording | R's `seq(0, 9000, by=0.1)` writes one extra sample (90 001 rows); `np.arange` matches the real file |
| 8 | `(-0.0)` → `0.0` in quantisation keys | See 3.3 |

---

## 5. Re-running the verification

The comparison scripts live in `dev/verify_against_r/` and need three things that are not
distributed: the experimental recording, the R reference outputs, and (to regenerate those outputs)
an R installation with the original scripts.

```bash
# R side (in the paper project root, where R_Codes/ and R_work/ live)
Rscript R_work/run1_spike_detection.R   # → spikes_xyz.csv
Rscript R_work/run2_trace.R            # → trace.csv, hough_result.csv, hough_planes.csv
Rscript R_work/run3_fit.R              # → result_array_r.csv, fit_circular_r.csv, fit_linear_r.csv

# Python side (in this repository)
python dev/verify_against_r/step1_spikes.py
python dev/verify_against_r/step2_hough.py
python dev/verify_against_r/step3_fit.py
```

`dev/verify_against_r/_paths.py` documents the directory layout these scripts expect and prints an
actionable message when a reference file is missing. Each script exits non-zero on any difference.

The parts of the verification that need no R and no experimental data are in `tests/`, and run in
about a minute:

```bash
pytest -q
```
