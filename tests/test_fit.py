"""
阶段 3（波前模型拟合）的单元测试。

用**解析构造**的数据做检验：先给定 (x0, y0, v, t0) 生成精确的到达时刻，
再看拟合能不能把它们反解回来。这类测试不需要任何外部数据。
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    GRID_SIDE,
    TIME_SCALE_BACK,
    build_result_array,
    circular_loss,
    fit_circular,
    fit_linear,
    linear_loss,
    plane_points,
    r_squared,
)


def circular_arrival(x0, y0, v, t0, side=GRID_SIDE):
    """由解析模型生成 (x, y, t) 点集：t = √((x−x0)²+(y−y0)²)/v + t0。"""
    xs, ys = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1),
                         indexing="ij")
    xs = xs.ravel().astype(float)
    ys = ys.ravel().astype(float)
    t = np.hypot(xs - x0, ys - y0) / v + t0
    return np.column_stack([xs, ys, t])


def linear_arrival(a, b, c, side=GRID_SIDE):
    """由解析模型生成 (x, y, t)：t = ãx + b̃y + c̃。"""
    xs, ys = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1),
                         indexing="ij")
    xs = xs.ravel().astype(float)
    ys = ys.ravel().astype(float)
    t = a * xs + b * ys + c
    return np.column_stack([xs, ys, t])


# ── build_result_array / plane_points ────────────────────────────────────

def test_build_result_array_scales_time_back_to_ms():
    """
    ★ 两段尺度切换的后半段：result.array 里存的必须是【乘回 200】的 ms。
      少乘这一步，拟合出来的 v 会差 200 倍。
    """
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 3.0])
    t_hough = np.array([4.0, 5.0])          # /200 尺度
    pid = np.array([1, 1])
    arr = build_result_array(x, y, t_hough, pid, n_planes=1)
    assert arr.shape == (GRID_SIDE, GRID_SIDE, 1)
    assert arr[0, 0, 0] == pytest.approx(4.0 * TIME_SCALE_BACK)
    assert arr[1, 2, 0] == pytest.approx(5.0 * TIME_SCALE_BACK)
    assert TIME_SCALE_BACK == 200.0


def test_build_result_array_empty_slots_are_minus_one():
    arr = build_result_array(np.array([1.0]), np.array([1.0]),
                             np.array([2.0]), np.array([1]), n_planes=1)
    assert arr[0, 0, 0] == pytest.approx(400.0)
    assert np.all(arr[1:, :, :] == -1.0)     # R 用 array(-1, ...) 初始化


def test_build_result_array_skips_unclassified_points():
    arr = build_result_array(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                             np.array([3.0, 4.0]), np.array([1, 0]), n_planes=1)
    assert arr[0, 0, 0] == pytest.approx(600.0)
    assert arr[1, 1, 0] == -1.0              # plane_indices == 0 → 跳过


def test_build_result_array_later_writes_win():
    """
    同一 (x, y) 在同一平面出现两次时，后写入的覆盖先写入的
    —— 复现 R 的同名下标赋值语义（不是累加、不是取最大）。
    """
    arr = build_result_array(np.array([1.0, 1.0]), np.array([1.0, 1.0]),
                             np.array([3.0, 7.0]), np.array([1, 1]), n_planes=1)
    assert arr[0, 0, 0] == pytest.approx(7.0 * TIME_SCALE_BACK)


def test_plane_points_unfolds_column_major_and_drops_empty_slots():
    arr = build_result_array(np.array([1.0, 3.0]), np.array([2.0, 4.0]),
                             np.array([1.0, 2.0]), np.array([1, 1]), n_planes=1)
    p = plane_points(arr, 1)
    assert p.shape == (2, 3)
    # R 的 as.vector 是列主序；这里必须与之一致，否则点序不同
    assert p[:, 0].tolist() == [1.0, 3.0]
    assert p[:, 1].tolist() == [2.0, 4.0]
    assert p[:, 2].tolist() == [200.0, 400.0]


# ── 损失函数 ─────────────────────────────────────────────────────────────

def test_circular_loss_is_zero_on_exact_model():
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    assert circular_loss([2.5, 3.5, 1 / 0.45, 100.0], pts) == pytest.approx(0.0)


def test_circular_loss_third_slot_is_slowness_not_speed():
    """
    ★ R 的 p[[3]] 存的是 u = 1/v（慢度），目标函数里是【乘】u。
      论文式 (3) 写的是【除 v】。两者等价，但代码里是乘。
      把 u 当成速度传进去会得到一个有限但完全错误的损失值 —— 不会报错。
    """
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    assert circular_loss([2.5, 3.5, 1 / 0.45, 100.0], pts) == pytest.approx(0.0, abs=1e-9)
    assert circular_loss([2.5, 3.5, 0.45, 100.0], pts) > 1e3


def test_linear_loss_is_zero_on_exact_model():
    pts = linear_arrival(0.3, 2.1, 50.0)
    assert linear_loss([0.3, 2.1, 50.0], pts) == pytest.approx(0.0)


def test_r_squared_is_one_on_exact_model():
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    r2 = r_squared(circular_loss, [2.5, 3.5, 1 / 0.45, 100.0], pts)
    assert r2 == pytest.approx(1.0, abs=1e-12)


# ── 圆波前拟合 ───────────────────────────────────────────────────────────

#: 论文 §3.2 的量级：源点在网格之外约 50 格，t 在几百毫秒量级。
#: ★ 圆模型拟合在这个量级上才工作 —— 这是 R 的初值 (u, t0) = (1, 1) 决定的，
#:   换量级就会卡住，见 test_fit_circular_needs_a_good_start_inside_grid。
PAPER_LIKE = dict(x0=-3.0, y0=-50.0, v=0.45, t0=800.0)


def test_fit_circular_recovers_known_parameters():
    """
    ★ 容差不是随手定的，它反映 R 的停机判据：

      R 的收敛条件是 ``‖Θ−Θ̂‖/‖Θ‖ < 1e-6``（cm_hough_grid_8.R:672），
      而 Θ 里含 t₀ ≈ 800，所以 t₀ 变化 8e-4 就足以触发停机，
      此时 x₀ 还差约 0.008。把它收紧到 1e-8 就能收敛到 1e-4 以内
      （见 test_fit_circular_precision_is_limited_by_the_stopping_rule）。

      所以"拟合结果与真值差 0.008"是 R 的行为，不是移植误差。
    """
    truth = PAPER_LIKE
    pts = circular_arrival(**truth)
    f = fit_circular(pts)
    assert f.n_points == GRID_SIDE * GRID_SIDE
    assert f.x0 == pytest.approx(truth["x0"], abs=0.05)
    assert f.y0 == pytest.approx(truth["y0"], abs=0.05)
    assert f.v == pytest.approx(truth["v"], rel=1e-4)
    assert f.t0 == pytest.approx(truth["t0"], abs=0.05)
    assert f.r2 == pytest.approx(1.0, abs=1e-6)


def test_fit_circular_precision_is_limited_by_the_stopping_rule():
    """把停机判据收紧 → 精度随之提高，证明上面的 0.008 来自停机判据而非算错。"""
    pts = circular_arrival(**PAPER_LIKE)
    loose = fit_circular(pts, eps=1e-6)     # R 真数据管线的取值
    tight = fit_circular(pts, eps=1e-10)
    assert abs(loose.x0 - PAPER_LIKE["x0"]) > abs(tight.x0 - PAPER_LIKE["x0"])
    assert tight.x0 == pytest.approx(PAPER_LIKE["x0"], abs=1e-4)
    assert tight.loss < loose.loss


def test_fit_circular_reports_speed_not_slowness():
    """
    ★ vt_optim 的槽位陷阱：R 把斜率（= 1/v）存回了名为 v 的位置。
      Python 版的 CircularFit.v 必须是【速度】，不能直接把槽位透出来。
    """
    pts = circular_arrival(**PAPER_LIKE)
    f = fit_circular(pts)
    assert f.v == pytest.approx(0.45, rel=1e-3)
    assert f.v != pytest.approx(1 / 0.45, rel=1e-2)


def test_fit_circular_handles_source_far_outside_grid():
    """源点在网格外（论文真数据的实际情况：y0 = −50），逐条波前都该收敛。"""
    for t0 in (800.0, 2054.0, 7647.0):
        pts = circular_arrival(-3.0, -50.0, 0.45, t0)
        f = fit_circular(pts)
        assert f.y0 == pytest.approx(-50.0, abs=0.05)
        assert f.v == pytest.approx(0.45, rel=1e-2)


def test_fit_circular_needs_a_good_start_inside_grid():
    """
    ★★ 继承自 R 的收敛脆弱性，必须记录下来：

      交替最小化的初值是 (u, t0) = (1, 1)，与数据无关。当源点落在网格【内部】
      时，第一轮网格搜索会被推向搜索窗边界并且再也爬不回来。

      默认 n_starts=1 = 完全复现 R（也是复现论文表 2 的前提），这时会失败；
      n_starts=2 额外试一次数据驱动的初值（以质心为临时中心），就能精确恢复。

      这不是移植引入的 bug —— R 的行为完全相同。论文的源点在网格外 50 格，
      恰好是 R 初值能工作的情形。
    """
    truth = dict(x0=2.5, y0=3.5, v=0.45, t0=100.0)
    pts = circular_arrival(**truth)

    # max_iter 限到 200 只是为了测试跑得快：R 初值那条路即使给满 10000 轮
    # 也收敛不到真值（它会一路跑到 10000 轮上限），而质心初值 14 轮就够了。
    r_only = fit_circular(pts, max_iter=200)          # n_starts=1
    assert not (abs(r_only.v - truth["v"]) < 0.01
                and abs(r_only.x0 - truth["x0"]) < 0.5), \
        "R 的初值在这个量级上本应收敛到错误的局部极小"

    fixed = fit_circular(pts, n_starts=2, max_iter=200)
    assert fixed.x0 == pytest.approx(truth["x0"], abs=1e-2)
    assert fixed.y0 == pytest.approx(truth["y0"], abs=1e-2)
    assert fixed.v == pytest.approx(truth["v"], rel=1e-4)
    assert fixed.t0 == pytest.approx(truth["t0"], abs=1e-2)


def test_centroid_guess_does_not_break_a_case_where_r_already_works():
    """
    多起点的扩展不能把"R 初值本来就能收敛"的情形弄坏。

    这里用的量级（源点在网格偏内部、t₀ = 200）R 初值能在 463 轮内收敛，
    正好用来检验 n_starts=2 不会把好结果换掉。
    """
    pts = circular_arrival(4.5, 4.5, 0.45, 200.0)
    one = fit_circular(pts, n_starts=1)
    two = fit_circular(pts, n_starts=2)
    assert one.v == pytest.approx(0.45, rel=1e-3)          # R 初值本身是对的
    assert two.v == pytest.approx(one.v, rel=1e-6)
    assert two.x0 == pytest.approx(one.x0, abs=1e-6)
    assert two.loss <= one.loss * (1 + 1e-9)


def test_centroid_initial_guess_is_sane():
    from wave_hough_detect import centroid_initial_guess
    pts = circular_arrival(2.5, 3.5, 0.45, 100.0)
    u0, t0_0 = centroid_initial_guess(pts)
    assert u0 > 0
    assert t0_0 == pytest.approx(100.0, abs=10.0)


# ── 线波前拟合 ───────────────────────────────────────────────────────────

def test_fit_linear_recovers_known_parameters():
    pts = linear_arrival(0.3, 2.1, 50.0)
    f = fit_linear(pts)
    assert f.n_points == GRID_SIDE * GRID_SIDE
    assert f.a == pytest.approx(0.3, abs=1e-9)
    assert f.b == pytest.approx(2.1, abs=1e-9)
    assert f.r2 == pytest.approx(1.0, abs=1e-12)


def test_fit_linear_speed_is_inverse_gradient_norm():
    """论文式：v = 1/√(ã² + b̃²)。"""
    pts = linear_arrival(0.3, 2.1, 50.0)
    f = fit_linear(pts)
    assert f.v == pytest.approx(1.0 / np.hypot(0.3, 2.1), rel=1e-12)


def test_linear_and_circular_are_both_near_perfect_for_a_far_source():
    """
    源点远在网格之外时，波前在网格尺度上几乎就是平面 ——
    这就是论文 §3.2 里两个模型的 R² 都超过 0.93、且线性模型可用的原因。
    """
    pts = circular_arrival(**PAPER_LIKE)
    fc = fit_circular(pts)
    fl = fit_linear(pts)
    assert fc.r2 > 0.99
    assert fl.r2 > 0.99
