"""
§3.1.2 圆波前仿真与精度评估的测试（论文图 6–8）。

与 ``test_simulate.py``（§3.1.1）分开：那是 8×8 线性波前 + FPR/FNR，
这是 96×96 圆波前 + 参数估计精度，两套仿真连抽样结构都不同。

最强的一条是 ``test_fit_recovers_truth_without_noise``：仿真数据是按
``t = √((x−48)²+(y−48)²) + 2`` 造的，把测量误差与噪声都关掉之后，
拟合必须把 (48, 48, 1, 2) 反解到 1e-8 以内。
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    CIRCULAR_TRUTH,
    PAPER_CIRCULAR_PARAMS,
    PLOT_MODES,
    accuracy_sweep,
    circular_loss,
    evaluate_circular,
    r_squared,
    seed_circular,
    simulate_circular_wavefronts,
)

NO_NOISE = dict(noise_freq=1e-9)      # λ_n → 0：一条噪声都不产生


# ── 仿真器 ───────────────────────────────────────────────────────────────

def test_simulation_grid_and_labels():
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    assert list(df.columns) == ["x", "y", "ts", "z"]
    assert set(np.unique(df["z"])) <= {0, 1}
    assert df["x"].min() >= 1 and df["x"].max() <= 96
    assert df["y"].min() >= 1 and df["y"].max() <= 96
    assert np.allclose(df["x"], np.round(df["x"]))


def test_all_electrodes_receive_the_wavefront_when_p_is_zero():
    """
    p = 0 且没有噪声时，96×96 = 9216 个电极**每个**都该收到一次。
    这是真值换算的基础：n_signal = 9216 才使 snr = n_signal / n_noise
    约等于 96 / λ_n，与 R 实测的范围对得上。
    """
    df = simulate_circular_wavefronts(0, **{**PAPER_CIRCULAR_PARAMS, **NO_NOISE})
    assert int((df["z"] == 1).sum()) == 96 * 96
    assert int((df["z"] == 0).sum()) == 0


def test_arrival_times_match_the_analytic_model():
    """
    ★ 核心的仿真正确性检验：每个信号点的 ts 必须等于
      ``t₀ + √((x−x₀)² + (y−y₀)²) / v``（σ = 1e-6，容差放到 5e-5）。
    """
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    sig = df[df["z"] == 1]
    t = CIRCULAR_TRUTH
    expect = t["t0"] + np.hypot(sig["x"] - t["x0"], sig["y"] - t["y0"]) / t["v"]
    assert np.abs(sig["ts"].to_numpy() - expect.to_numpy()).max() < 5e-5


def test_noise_points_never_exceed_the_observation_window():
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    assert df["ts"].max() < PAPER_CIRCULAR_PARAMS["time_max"]
    # 信噪之间的时间分布应当明显不同：信号成片、噪声铺满整个窗
    sig_ts = df.loc[df["z"] == 1, "ts"]
    assert sig_ts.min() >= PAPER_CIRCULAR_PARAMS["ts"] - 1e-3


def test_rows_are_sorted_by_time():
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    assert df["ts"].is_monotonic_increasing


def test_missing_probability_thins_the_signal():
    n = []
    for p in (0.0, 0.3, 0.9):
        df = simulate_circular_wavefronts(1, **{**PAPER_CIRCULAR_PARAMS,
                                                **NO_NOISE, "p": p})
        n.append(int((df["z"] == 1).sum()))
    assert n[0] > n[1] > n[2]
    assert n[0] == 96 * 96


def test_noise_rate_controls_the_number_of_noise_points():
    few = simulate_circular_wavefronts(0, **{**PAPER_CIRCULAR_PARAMS,
                                             "noise_freq": 0.1})
    many = simulate_circular_wavefronts(0, **{**PAPER_CIRCULAR_PARAMS,
                                              "noise_freq": 10.0})
    assert (many["z"] == 0).sum() > (few["z"] == 0).sum() * 10


def test_simulation_is_reproducible_and_seed_shifted():
    a = simulate_circular_wavefronts(3, **PAPER_CIRCULAR_PARAMS)
    b = simulate_circular_wavefronts(3, **PAPER_CIRCULAR_PARAMS)
    c = simulate_circular_wavefronts(4, **PAPER_CIRCULAR_PARAMS)
    assert a.equals(b)
    assert not a.equals(c)


# ── 种子：R 的那个 bug ───────────────────────────────────────────────────

def test_seed_arithmetic_matches_r():
    """R:160-161 的 ``100 · Π(参数)``。p 是乘积的一项。"""
    kw = dict(limit=96, time_max=96.0, ts=2.0, x=48.0, y=48.0, u_x=1.0, u_y=1.0,
              p=0.0, noise_freq=1.0, miu_x=1.0, miu_y=1.0, sigma=1e-6)
    assert seed_circular(**kw) == 0          # 含 p = 0 → 乘积为 0
    kw["p"] = 0.5
    assert seed_circular(**kw) == int(100 * 96 * 96 * 2 * 48 * 48 * 0.5 * 1e-6)
    assert seed_circular(**{**kw, "seed_shift": 7}) == seed_circular(**kw) + 7


def test_p_zero_makes_every_replicate_identical_in_r():
    """
    ★★ R 的坑，值得单独钉住：种子公式里有一项是 ``p``，而 mode 1/3 都固定 p = 0，
      于是**种子恒为 0**。``optimality.table`` 又是对同一组参数反复调用，
      所以 R 的那些「重复」拿到的是同一份数据集 —— 误差棒不代表重复间的波动。

      本实现默认给每个重复加一个 seed_shift 来修掉它；
      ``independent_replicates=False`` 可以复现 R 的行为。
    """
    from wave_hough_detect.simulate_circular import simulate_circular_wavefronts as sim

    # 复现 R：所有重复完全相同
    a = sim(0, **{**PAPER_CIRCULAR_PARAMS, "sigma": 0.3})
    b = sim(0, **{**PAPER_CIRCULAR_PARAMS, "sigma": 0.3})
    assert a.equals(b)

    # 本实现的独立重复：seed_shift 不同 → 数据不同
    c = sim(1, **{**PAPER_CIRCULAR_PARAMS, "sigma": 0.3})
    assert not a.equals(c)


def test_accuracy_sweep_replicate_independence_flag():
    """两种重复模式必须给出不同结果；共享随机流时重复间方差为 0。"""
    indep = accuracy_sweep(3, n_replicates=3, independent_replicates=True)
    shared = accuracy_sweep(3, n_replicates=3, independent_replicates=False)
    v_indep = indep.groupby("scan_value")["x0"].std(ddof=1).to_numpy()
    v_shared = shared.groupby("scan_value")["x0"].std(ddof=1).to_numpy()
    assert np.nanmax(v_shared) < 1e-12, "共享随机流时重复之间不该有任何差别"
    assert v_indep.max() > 0


# ── 拟合与真值 ───────────────────────────────────────────────────────────

def test_fit_recovers_truth_without_noise():
    """
    ★★ 本层最硬的一条检验。

    把测量误差（σ = 0）与噪声点（λ_n → 0）都关掉，数据就精确落在
        t = √((x−48)² + (y−48)²)/1 + 2
    上，拟合必须把四个参数全部反解出来。

    顺带说明为什么 R 的初值 (u, t₀) = (1, 1) 在这里是好的：
    真值是 v = 1（即慢度 u = 1）、t₀ = 2 —— 初值几乎正中靶心。
    §3.1.1 里那个「源点在网格内就收敛错」的毛病因此在 §3.1.2 不会发作。
    """
    est = evaluate_circular(1, sigma=0.0, **NO_NOISE)
    t = CIRCULAR_TRUTH
    assert est.x0 == pytest.approx(t["x0"], abs=1e-8)
    assert est.y0 == pytest.approx(t["y0"], abs=1e-8)
    assert est.v == pytest.approx(t["v"], abs=1e-8)
    assert est.t0 == pytest.approx(t["t0"], abs=1e-8)
    assert est.n_noise == 0


def test_measurement_error_propagates_linearly():
    """
    σ 必须真的进到结果里，而且误差随 σ 大致线性增长。
    这条同时是「σ 扫描不是摆设」的证据。
    """
    errs = []
    for s in (0.1, 0.5, 1.0, 2.0):
        est = evaluate_circular(1, sigma=s, **NO_NOISE)
        errs.append(abs(est.x0 - CIRCULAR_TRUTH["x0"]))
    assert all(b > a for a, b in zip(errs, errs[1:])), errs
    # 线性 → 相邻比值应当接近 σ 的比值
    assert errs[-1] / errs[0] == pytest.approx(20.0, rel=0.35)


def test_noise_points_dominate_the_absolute_error():
    """
    ★ 一个关于 R 的 setup 的发现（不是移植问题，是参数挑出来的）：

      R 的 mode 3 扫 σ，但同时又固定 λ_n = 1（约 100 个噪声尖峰）。
      那 100 个离群点对损失的贡献远超测量误差本身，所以在 σ = 0.1…1.0
      范围内估计误差几乎不动 —— 那条曲线测的其实是「离群点污染」，
      不是「测量误差」。

      这里把两件事分开量：有噪声时 σ 的影响被淹没，去掉噪声后立刻显现。
    """
    with_noise = [evaluate_circular(1, sigma=s, noise_freq=1.0).rel_error["x0"]
                  for s in (0.0, 1.0)]
    no_noise = [evaluate_circular(1, sigma=s, **NO_NOISE).rel_error["x0"]
                for s in (0.0, 1.0)]

    # 有噪声：σ 从 0 到 1，误差基本不变（差异不到 2 倍）
    assert with_noise[1] / max(with_noise[0], 1e-12) < 2.0
    # 无噪声：误差随 σ 明显增长
    assert no_noise[1] > no_noise[0] * 10


def test_r_squared_is_computed_over_signal_and_noise():
    """
    ★ R² 的口径陷阱：R 把【含噪声行】的整张矩阵交给拟合（R:663, 671），
      所以这里对真值参数算 R² 也只有 ~0.95，不是 1。
      只用信号点算则是 1.0000000000。

      ⇒ 不要把这个 0.95 与论文 §3.2 表 4 的 0.94 并列比较，那是另一回事。
    """
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    pts = df[["x", "y", "ts"]].to_numpy(dtype=float)
    truth_p = np.array([48.0, 48.0, 1.0, 2.0])       # (x0, y0, u, t0)，u = 1/v

    r2_all = r_squared(circular_loss, truth_p, pts)
    r2_sig = r_squared(circular_loss, truth_p, pts[df["z"].to_numpy() == 1])
    assert r2_sig == pytest.approx(1.0, abs=1e-9)
    assert 0.90 < r2_all < 0.99
    assert r2_all < r2_sig


# ── SNR 的定义 ───────────────────────────────────────────────────────────

def test_snr_is_the_count_ratio_not_an_amplitude_ratio():
    """
    ★ 图 7 的横轴（R:788）：``n_signal / n_noise``。
      名字叫信噪比，但它是**点数之比**，与幅度无关。
    """
    est = evaluate_circular(1)
    assert est.snr == pytest.approx(est.n_signal / est.n_noise)
    assert est.n_signal == 96 * 96


def test_snr_of_the_mode1_grid_matches_the_published_range():
    """
    R 实测的 snr 范围是 [≈0.375, ≈768]。在 §3.1.2 配置下
    n_signal = 9216、n_noise ≈ λ_n × 96，所以 snr ≈ 96 / λ_n。
    mode 1 的 λ_n 网格是 2^((-6:16)/2) ∈ [0.125, 256] → [0.375, 768]。
    """
    grid = PLOT_MODES[1]["grid"]
    assert grid.min() == pytest.approx(0.125)
    assert grid.max() == pytest.approx(256.0)
    assert 96 / grid.max() == pytest.approx(0.375, rel=1e-6)
    assert 96 / grid.min() == pytest.approx(768.0, rel=1e-6)


# ── 扫描 ─────────────────────────────────────────────────────────────────

def test_accuracy_sweep_shape_and_columns():
    df = accuracy_sweep(2, n_replicates=2)
    assert len(df) == len(PLOT_MODES[2]["grid"]) * 2
    for col in ("plot_mode", "variable", "scan_value", "replicate", "x0", "y0",
                "v", "t0", "snr", "n_signal", "n_noise",
                "abs_x0", "rel_t0"):
        assert col in df.columns
    assert (df["plot_mode"] == 2).all()
    assert (df["variable"] == "p").all()


def test_accuracy_sweep_rejects_unknown_mode():
    with pytest.raises(ValueError):
        accuracy_sweep(99)


def test_source_position_is_more_robust_than_speed():
    """
    图 7 想展示的退化方式：噪声变多时，(x₀, y₀) 仍然很稳，
    而 v 与 t₀ 会一起漂 —— 拟合靠「放慢速度 + 推迟激发」去迁就离群点。
    """
    df = accuracy_sweep(1, n_replicates=2)
    lo = df[df["scan_value"] == df["scan_value"].min()]
    hi = df[df["scan_value"] == df["scan_value"].max()]
    assert hi["rel_v"].mean() > lo["rel_v"].mean() * 20
    assert hi["rel_x0"].mean() < 0.05
