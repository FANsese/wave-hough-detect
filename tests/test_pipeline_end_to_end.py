"""
端到端管线测试：在**已知真值**的合成记录上跑完阶段 1→2→3。

这是整套测试里价值最高的一条 —— 它同时验证三个阶段的正确性：
记录由 :func:`simulate_recording` 生成，源点位置、速度、各条波前的激发时刻
都是设定的，拟合必须把它们反解回来。

用较短的记录（3000 ms）以保持测试快速。
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    build_result_array,
    detect_all_channels,
    fit_circular,
    fit_linear,
    hough_plane,
    load_recording,
    plane_points,
    simulate_recording,
    spikes_to_table,
)

#: 合成记录的真值。3 秒、4 条波前，够验算又够快。
TRUTH = dict(source_x0=-3.7, source_y0=-50.0, speed=0.45,
             duration_ms=3000.0, first_wave_ms=500.0, beat_interval_ms=800.0)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """生成一次合成记录，模块内所有测试共用。"""
    path = tmp_path_factory.mktemp("rec") / "rec.csv"
    _, info = simulate_recording(path, **TRUTH)
    return path, info


@pytest.fixture(scope="module")
def pipeline(recording):
    """跑完整管线，返回中间结果。"""
    path, info = recording
    rec = load_recording(path)
    spike_times, _ = detect_all_channels(rec)
    sp = spikes_to_table(spike_times, rec.channels)
    pts = sp[["x", "y", "t"]].to_numpy(dtype=float)

    res = hough_plane(pts, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)

    order = np.argsort(sp["t"].to_numpy(), kind="stable")
    arr = build_result_array(sp["x"].to_numpy()[order],
                             sp["y"].to_numpy()[order],
                             sp["t"].to_numpy()[order],
                             res.plane_indices[order], res.n_planes)

    fits = []
    for k in range(1, res.n_planes + 1):
        p = plane_points(arr, k)
        fits.append((k, fit_circular(p), fit_linear(p)))
    fits.sort(key=lambda f: f[1].t0)

    return dict(info=info, rec=rec, spike_times=spike_times, sp=sp, res=res,
                arr=arr, fits=fits)


# ── 阶段 1 ───────────────────────────────────────────────────────────────

def test_stage1_finds_one_spike_per_wavefront_per_electrode(pipeline):
    """每条波前在每个电极上恰好一个尖峰 —— 检出总数 = 电极数 × 波前数。"""
    info = pipeline["info"]
    counts = np.array([s.size for s in pipeline["spike_times"]])
    n_waves = len(info["wave_times_ms"])
    assert (counts == n_waves).all(), f"逐通道检出数 {np.unique(counts)}"
    assert counts.sum() == info["n_channels"] * n_waves


def test_stage1_detected_times_match_the_true_arrival_times(pipeline):
    """
    比"数量对"更硬：检出的时刻必须与解析到达时刻 t = t_wave + distance/v 一致。

    容差取 0.5 ms（约 5 个采样点）而不是 0.1 ms，原因有二：
      · 尖峰位置只能在采样点上取到，本身就有 0.1 ms 的量化；
      · 记录里有 0.02 mV 白噪声与一条 0.5 mV 的正弦基线，会使峰位小幅左右移动。
    实测最大偏差约 0.21 ms、平均偏差小一个量级；下面两条断言分别卡住这两点。
    """
    info = pipeline["info"]
    sp = pipeline["sp"]
    x0, y0, v = info["source_x0"], info["source_y0"], info["speed"]
    # 逐条波前对齐：按 t 排序后每 64 个点一组
    t_sorted = sp["t"].to_numpy() * 200.0          # 回到 ms
    xs = sp["x"].to_numpy(); ys = sp["y"].to_numpy()
    dist = np.hypot(xs - x0, ys - y0)

    worst, all_dev = 0.0, []
    for i, tw in enumerate(sorted(info["wave_times_ms"])):
        lo, hi = i * 64, (i + 1) * 64
        expect = tw + dist[lo:hi] / v
        dev = np.abs(t_sorted[lo:hi] - expect)
        all_dev.append(dev)
        worst = max(worst, float(dev.max()))
        assert dev.max() < 0.5, f"第 {i + 1} 条波前对不上（最大偏差 {dev.max():.3f} ms）"

    assert worst < 0.5
    assert np.concatenate(all_dev).mean() < 0.15


# ── 阶段 2 ───────────────────────────────────────────────────────────────

def test_stage2_finds_one_plane_per_wavefront(pipeline):
    info = pipeline["info"]
    assert pipeline["res"].n_planes == len(info["wave_times_ms"])


def test_stage2_every_spike_is_classified(pipeline):
    """无噪声的合成数据上，每个尖峰都该落进某个平面。"""
    res = pipeline["res"]
    n = len(pipeline["sp"])
    assert int(res.prediction.sum()) == n


def test_stage2_each_plane_covers_all_electrodes(pipeline):
    res = pipeline["res"]
    sizes = [int((res.plane_indices == k).sum()) for k in range(1, res.n_planes + 1)]
    assert sizes == [64] * res.n_planes


# ── 阶段 3：与真值对账 ───────────────────────────────────────────────────

def test_stage3_recovers_the_source_position(pipeline):
    info = pipeline["info"]
    x0 = np.array([c.x0 for _, c, _ in pipeline["fits"]])
    y0 = np.array([c.y0 for _, c, _ in pipeline["fits"]])
    assert np.abs(x0 - info["source_x0"]).max() < 0.5
    assert np.abs(y0 - info["source_y0"]).max() < 0.5


def test_stage3_recovers_the_propagation_speed(pipeline):
    info = pipeline["info"]
    v = np.array([c.v for _, c, _ in pipeline["fits"]])
    assert np.abs(v - info["speed"]).max() < 0.01, f"各平面速度 {np.round(v, 4)}"


def test_stage3_recovers_the_excitation_times(pipeline):
    """
    拟合出的 t₀ 必须是设定好的源点激发时刻。
    注意阶段 3 内部有 ×200 的尺度切换，少乘一步这里会差 200 倍。
    """
    info = pipeline["info"]
    truth = np.sort(np.asarray(info["wave_times_ms"]))
    t0 = np.array([c.t0 for _, c, _ in pipeline["fits"]])
    assert t0.size == truth.size
    assert np.abs(t0 - truth).max() < 2.0, f"拟合 {np.round(t0, 1)} vs 真值 {truth}"


def test_stage3_fit_quality_is_high(pipeline):
    """无噪声合成数据上 R² 应当非常接近 1。"""
    assert min(c.r2 for _, c, _ in pipeline["fits"]) > 0.99
    assert min(l.r2 for _, _, l in pipeline["fits"]) > 0.98


def test_stage3_linear_model_recovers_direction(pipeline):
    """
    线性模型的 ã, b̃ 与真实方向的关系：波前法向 ∝ (ã, b̃)。
    对源点在 (−3.7, −50) 的远场波前，方向应当主要沿 y，所以 b̃ 远大于 |ã|。
    """
    for _, c, l in pipeline["fits"]:
        assert abs(l.b) > abs(l.a)
        assert l.b > 0        # 波从 y 小的一侧来
