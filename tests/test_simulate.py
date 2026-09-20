"""
阶段 4（仿真与评估）的单元测试。

仿真的随机数无法与 R 逐位对齐（见 simulate.py 模块文档），所以这里只检验
**结构与统计性质**，以及那些一旦写错就会静默出错的约定：
种子推导、稳定排序、z 标签、FPR/FNR 的口径。
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    PAPER_2D_PARAMS,
    detection_rates,
    detection_sweep,
    evaluate_detection,
    seed_from_params,
    simulate_linear_wavefronts,
    simulate_recording,
)


# ── 种子推导 ─────────────────────────────────────────────────────────────

def test_seed_derivation_is_reproducible():
    a = simulate_linear_wavefronts(seed_shift=7, **PAPER_2D_PARAMS)
    b = simulate_linear_wavefronts(seed_shift=7, **PAPER_2D_PARAMS)
    assert a.equals(b)


def test_different_seed_shifts_give_different_datasets():
    a = simulate_linear_wavefronts(seed_shift=7, **PAPER_2D_PARAMS)
    b = simulate_linear_wavefronts(seed_shift=8, **PAPER_2D_PARAMS)
    assert not a.equals(b)


def test_seed_from_params_matches_r_formula():
    """
    R: set.seed(SEED_BASE * 各参数连乘 + seed.shift)。
    SEED_BASE = 101，参数取 §3.1.1 那一组：
        101 · 8 · 80 · 1 · 1 · 1 · 1 · 1 · 0.1 · 1 · 1 · 0.1 = 646.4
    截断成 646；shift=1 → 647。
    """
    kw = dict(limit=8, time_max=80.0, ts=1.0, x=1.0, y=1.0, u_x=1.0, u_y=1.0,
              p=0.1, noise_freq=1.0, miu=1.0, sigma=0.1)
    assert pytest.approx(646.4) == 101 * 8 * 80 * 0.1 * 0.1      # 先确认算式
    assert seed_from_params(101, 0, **kw) == 646
    assert seed_from_params(101, 1, **kw) == 647
    # 截断而不是四舍五入
    assert seed_from_params(101, 0, **{**kw, "sigma": 0.19}) == int(101 * 8 * 80 * 0.1 * 0.19)


# ── 仿真数据结构 ─────────────────────────────────────────────────────────

def test_simulated_columns_and_labels():
    df = simulate_linear_wavefronts(seed_shift=1, **PAPER_2D_PARAMS)
    assert list(df.columns) == ["x", "y", "ts", "z"]
    assert set(np.unique(df["z"])) <= {0, 1}
    assert (df["z"] == 1).any() and (df["z"] == 0).any()


def test_simulated_coordinates_stay_on_the_grid():
    df = simulate_linear_wavefronts(seed_shift=2, **PAPER_2D_PARAMS)
    limit = PAPER_2D_PARAMS["limit"]
    for col in ("x", "y"):
        assert df[col].min() >= 1
        assert df[col].max() <= limit
        assert np.allclose(df[col], np.round(df[col]))     # 必须落在整数格点上


def test_simulated_times_within_horizon():
    """
    噪声点在时间上是泊松流，可以从 0 附近就开始，所以整体的 ts 最小值
    小于第一条波前的出发时刻；但**信号点**必须从 ts 参数出发。
    """
    df = simulate_linear_wavefronts(seed_shift=3, **PAPER_2D_PARAMS)
    assert df["ts"].min() >= 0.0
    assert df.loc[df["z"] == 1, "ts"].min() >= PAPER_2D_PARAMS["ts"]
    assert df["ts"].max() <= PAPER_2D_PARAMS["time_max"] * 1.05


def test_rows_are_sorted_by_time():
    """
    ★ R 用 order()（稳定排序）。ts 是连续量但也有重复（同一条波前上多个点
      可能同刻），行序必须与"稳定"这一约定一致，否则按下标索引会错位。
    """
    df = simulate_linear_wavefronts(seed_shift=4, **PAPER_2D_PARAMS)
    assert df["ts"].is_monotonic_increasing


def test_missing_probability_actually_thins_the_signal():
    """p 越大，信号点越少（论文 §3.1 的漏检机制）。"""
    n = []
    for p in (0.0, 0.1, 0.5, 0.9):
        df = simulate_linear_wavefronts(seed_shift=1, p=p, **
                                        {k: v for k, v in PAPER_2D_PARAMS.items()
                                         if k != "p"})
        n.append(int((df["z"] == 1).sum()))
    assert n[0] > n[1] > n[2] > n[3]


def test_wavefront_count_follows_the_gap_parameter():
    """
    gap 是波前间隔的卡方自由度：gap 越大，间隔越大，波前越少。
    time_max=80 时 gap=30 大约给出 2–4 条波前。
    """
    few = simulate_linear_wavefronts(seed_shift=1, gap=200.0, **
                                     {k: v for k, v in PAPER_2D_PARAMS.items()
                                      if k != "gap"})
    many = simulate_linear_wavefronts(seed_shift=1, gap=5.0, **
                                      {k: v for k, v in PAPER_2D_PARAMS.items()
                                       if k != "gap"})
    assert (few["z"] == 1).sum() < (many["z"] == 1).sum()


# ── FPR / FNR 口径 ───────────────────────────────────────────────────────

def test_detection_rates_counts():
    z = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])
    r = detection_rates(z, pred, n_planes=2)
    assert (r.n_true_pos, r.n_false_pos, r.n_false_neg, r.n_true_neg) == (2, 1, 1, 2)
    assert r.n_points == 6 and r.n_planes == 2


def test_paper_rate_denominator_is_all_points():
    """
    ★ R 的 FPR/FNR 分母是【全部点】，不是各类点数（fig5_...R:371-372）。
      这里把口径固定下来 —— 换成常规定义会与论文图 5 不可比。
    """
    z = np.array([1, 1, 1, 0, 0, 0])
    pred = np.array([1, 1, 0, 1, 0, 0])
    r = detection_rates(z, pred)
    assert r.false_positive_rate == pytest.approx(1 / 6)
    assert r.false_negative_rate == pytest.approx(1 / 6)
    # 常规口径另算，互不影响
    assert r.false_positive_rate_among_noise == pytest.approx(1 / 3)
    assert r.false_negative_rate_among_signal == pytest.approx(1 / 3)


def test_detection_rates_handles_empty_classes():
    z = np.array([1, 1])
    r = detection_rates(z, np.array([1, 1]))
    assert r.false_positive_rate == 0.0
    assert np.isnan(r.false_positive_rate_among_noise)
    assert r.false_negative_rate_among_signal == 0.0


# ── 端到端评估 ───────────────────────────────────────────────────────────

def test_evaluate_detection_uses_simulation_preset():
    """
    ★ 仿真研究必须用 ρ 步长 0.5 / 容忍度 1.0。用真数据那组会一个平面都找不到
      （或找到一堆垃圾平面），而不会报任何错。
    """
    r = evaluate_detection(seed_shift=1)
    assert r.rates.n_points > 0
    assert r.prediction.size == len(r.data)
    assert set(np.unique(r.prediction)) <= {0, 1}
    assert r.rates.n_true_pos + r.rates.n_false_neg == r.rates.n_signal


def test_classified_labels_are_consistent_with_rates():
    r = evaluate_detection(seed_shift=2)
    cls = r.classified()
    counts = cls["class"].value_counts()
    assert int(counts.get("green", 0)) == r.rates.n_true_pos
    assert int(counts.get("yellow", 0)) == r.rates.n_false_pos
    assert int(counts.get("red", 0)) == r.rates.n_false_neg
    assert int(counts.get("black", 0)) == r.rates.n_true_neg


def test_detection_sweep_shape():
    df = detection_sweep(range(0, 3))
    assert len(df) == 3
    assert {"seed_shift", "FPR", "FNR", "n_planes"} <= set(df.columns)


# ── 合成 MEA 记录 ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_recording(tmp_path_factory):
    """整份 9 秒 64 通道记录只生成一次（生成一次约 4 秒）。"""
    path = tmp_path_factory.mktemp("rec") / "rec.csv"
    df, info = simulate_recording(path)
    return path, df, info


def test_simulate_recording_format(synthetic_recording):
    """
    CSV 必须与论文 §3.2 的实验记录同格式：
    1 列 time + 64 列 Ch01..Ch64，行数与 10 kHz / 9 s 对应。
    """
    path, df, info = synthetic_recording
    assert path.exists()
    assert df.shape == (90000, 65)
    assert list(df.columns)[0] == "time"
    assert list(df.columns)[1:] == [f"Ch{i:02d}" for i in range(1, 65)]
    assert not df.isna().any().any()
    assert info["n_channels"] == 64


def test_simulate_recording_time_column_is_milliseconds(synthetic_recording):
    """
    ★★ 最隐蔽的一个坑：time 列必须是【毫秒】。
      写成"毫秒 × 200"会让管线内部的霍夫尺度变成几千，与 x,y ∈ [1,8]
      差三个数量级，于是找到的全是垃圾平面（实测 v≈0、t0≈±2e7）。

      R 版 make_synthetic_recording.R 里默认的 TIME_UNITS_PER_MS <- 200
      正是这个错误；Python 版默认改成 1.0。
    """
    _, df, info = synthetic_recording
    assert info["time_units_per_ms"] == 1.0
    t = df["time"].to_numpy()
    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(8999.9, abs=0.05)          # 9000 ms 内的最后一个采样
    assert np.allclose(np.diff(t[:100]), 0.1)                # 10 kHz → 0.1 ms


def test_simulate_recording_spikes_are_negative_going(synthetic_recording):
    """
    胞外场电位是负向偏转，管线用 which.min 找尖峰 —— 极性反了就一个都检不出。
    """
    _, df, _ = synthetic_recording
    ch = df["Ch01"].to_numpy()
    assert ch.min() < -0.3
    assert abs(ch.min()) > abs(ch.max())


def test_simulate_recording_reports_ground_truth(synthetic_recording):
    """info 里必须带回真值，否则无法做"已知真值"的端到端判定。"""
    _, _, info = synthetic_recording
    assert info["source_x0"] == -3.7
    assert info["source_y0"] == -50.0
    assert info["speed"] == 0.45
    assert len(info["wave_times_ms"]) >= 2
    assert info["max_delay_ms"] > 0


def test_time_column_scale_is_right(synthetic_recording):
    """
    时间列用 1/200 ms 会让霍夫尺度错 200 倍。这里不复现完整管线，
    只固定住"错的设置会得到错的霍夫尺度"这个事实。
    """
    from wave_hough_detect import detect_all_channels, load_recording, spikes_to_table

    path, _, _ = synthetic_recording
    rec = load_recording(path)
    st, _ = detect_all_channels(rec)
    sp = spikes_to_table(st, rec.channels)
    t = sp["t"].to_numpy()
    # 霍夫尺度必须与 x, y ∈ [1, 8] 同量级
    assert t.max() < 200.0, "time 列不是毫秒，霍夫尺度已经跑偏"
