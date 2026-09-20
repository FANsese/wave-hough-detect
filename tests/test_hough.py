"""
阶段 2（随机霍夫变换）的单元测试。

这里刻意避开"随机数结果"这类不确定的东西，只检验：
  · 输入退化时的行为（点数不足、共线三点）
  · 明确的平面能否被找到
  · ★ 尾部扩展（R 406-450）这条分支 —— 真数据上不触发，仿真上才触发
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import hough_plane
from wave_hough_detect.rht import SIM_PRESET


def _plane_points(normal, rho, side=8, t_max=40.0, t_min=0.5):
    """
    构造一张精确落在平面 ``n·p = rho`` 上的点集。

    在 8×8 网格上取所有电极，解出 t；只保留 t 落在 [t_min, t_max] 内的点。
    """
    n1, n2, n3 = normal
    xs, ys = np.meshgrid(np.arange(1, side + 1), np.arange(1, side + 1),
                         indexing="ij")
    xs = xs.ravel().astype(float)
    ys = ys.ravel().astype(float)
    if n3 == 0:
        return np.empty((0, 3))
    t = (rho - n1 * xs - n2 * ys) / n3
    keep = (t >= t_min) & (t <= t_max)
    return np.column_stack([xs[keep], ys[keep], t[keep]])


# ── 退化输入 ─────────────────────────────────────────────────────────────

def test_fewer_than_three_points_returns_empty_result():
    """点数 < 3 时主循环立刻 break，必须返回空结果而不是崩掉。"""
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    res = hough_plane(pts, max_iter=1000, seed=0)
    assert res.n_planes == 0
    assert res.prediction.tolist() == [0, 0]
    assert res.accept_log == []


def test_all_points_collinear_does_not_crash():
    """三点永远共线 → 叉积恒为 0 → 一直 next；必须正常跑完而不是死循环。"""
    t = np.linspace(1.0, 5.0, 20)
    pts = np.column_stack([t, t, t])
    res = hough_plane(pts, max_iter=2000, seed=1)
    assert res.n_planes == 0
    assert res.prediction.sum() == 0


def test_empty_input():
    res = hough_plane(np.empty((0, 3)), max_iter=10, seed=0)
    assert res.n_planes == 0
    assert res.prediction.size == 0


# ── 能找到已知平面 ───────────────────────────────────────────────────────

def test_finds_a_synthetic_plane():
    """
    手工造一个波前平面：法向量接近 t 轴（这正是波前在 (x,y,t) 里的样子）。
    用仿真参数集，因为坐标尺度与 §3.1 一致。
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    pts = _plane_points(normal, rho=-10.0)
    assert len(pts) >= 40, "构造的点数至少要够 40 个电极"

    res = hough_plane(pts, **SIM_PRESET, max_iter=20000, min_detectors=40,
                      seed=1)
    assert res.n_planes >= 1
    # 至少 40 个电极、也就是至少 40 个点被收进平面
    assert res.prediction.sum() >= 40
    got = np.array([res.n1[res.prediction == 1][0], res.n2[res.prediction == 1][0],
                    res.n3[res.prediction == 1][0]])
    assert abs(float(np.dot(got, normal))) == pytest.approx(1.0, abs=1e-3)


def test_two_parallel_planes_are_not_merged():
    """
    两条平行波前在 t 上隔开 20 个单位，远超内点容忍度 1.0，必须分成两个平面。
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    a = _plane_points(normal, rho=-10.0, t_min=-1000, t_max=1000)
    b = _plane_points(normal, rho=-30.0, t_min=-1000, t_max=1000)
    pts = np.vstack([a, b])
    res = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                      seed=3)
    assert res.n_planes >= 2
    assert res.prediction.sum() >= 80


# ── ★ 尾部扩展分支（R 406-450）───────────────────────────────────────────

def _tail_loop_fixture():
    """
    构造一个**只有尾部扩展才能收全**的局面：

      · 主平面 A：rho = -10，64 个点；
      · 平行平面 B：rho = -30，40 个点（与 A 同法向量，t 上隔开 20）；
      · 若干孤立点。

    主循环先用 R 的轨迹固定住，只让它找到平面 A；平面 B 的点留在未分类集里。
    Algorithm 4 的"回收"用的是 A 的法向量与 A 的 rho，够不到 B。
    于是只有尾部扩展能凭【主法向量 + 以未分类点为锚】重新定出 B。
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    a = _plane_points(normal, rho=-10.0, t_min=-1000, t_max=1000)
    b = _plane_points(normal, rho=-30.0, t_min=-1000, t_max=1000)[:40]
    stray = np.array([[1.0, 1.0, 100.0], [8.0, 8.0, 101.0], [4.0, 5.0, 102.0]])
    pts = np.vstack([a, b, stray])

    # 轨迹：全部取自平面 A → 主循环只会接受 A 这一个平面。
    # ★ 下标 0=(1,1), 1=(1,2), 8=(2,1)。不能取 [0,1,2] —— 那三点同处 x=1，
    #   叉积恰好为 0（R 的 `if (norm.n == 0) next`），一次投票都不会发生。
    trace = [[0, 1, 8]] * 20
    return pts, trace, len(a)


def test_tail_regrow_is_off_by_default_when_no_plane_found():
    """一个平面都找不到时，尾部扩展不能崩（max_nv 不存在）。"""
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    res = hough_plane(pts, max_iter=10, seed=0, regrow_master_plane=True)
    assert res.n_planes == 0


def test_tail_regrow_collects_the_parallel_plane():
    """开启尾部扩展 → 平行平面 B 被收成一个新平面。"""
    pts, trace, n_a = _tail_loop_fixture()
    res = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                      trace=trace, regrow_master_plane=True)
    assert res.n_planes >= 2, "尾部扩展应当把平行平面收进来"
    assert res.prediction.sum() >= n_a + 40
    assert res.plane_indices.max() >= 2


def test_tail_regrow_can_be_disabled():
    """
    关掉尾部扩展 → 平行平面 B 的点留在 prediction == 0。

    这一对测试是"尾部扩展真的起作用"的证据：真数据上两种设置结果完全相同
    （未分类集为空、循环体一次都不进），所以只有这种构造才能测到它。
    """
    pts, trace, n_a = _tail_loop_fixture()
    on = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                     trace=trace, regrow_master_plane=True)
    off = hough_plane(pts, **SIM_PRESET, max_iter=100000, min_detectors=40,
                      trace=trace, regrow_master_plane=False)
    assert off.n_planes == 1
    assert on.n_planes > off.n_planes
    assert on.prediction.sum() > off.prediction.sum()


def test_tail_regrow_terminates_on_all_noise():
    """
    全是无法成面的孤立点时，尾部循环必须逐点消耗、正常结束（不能死循环）。
    这一支对应 R 里 prediction 先标 2、最后统一还原成 0 的写法。
    """
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(1, 8, 60), rng.uniform(1, 8, 60),
                           rng.uniform(1, 200, 60)])
    res = hough_plane(pts, **SIM_PRESET, max_iter=2000, min_detectors=40,
                      seed=5, regrow_master_plane=True)
    assert res.prediction.size == 60
    assert set(np.unique(res.prediction)).issubset({0, 1})   # 临时的 2 必须清干净


# ── 参数集必须分开 ───────────────────────────────────────────────────────

def test_real_data_and_simulation_presets_differ():
    """
    ★ 论文的两份 R 拷贝参数不同（ts 尺度差 10 倍）。
      这里把差异固定下来，防止有人"统一"成一组。
    """
    from wave_hough_detect import INLIER_TOL, RHO_STEP
    assert (RHO_STEP, INLIER_TOL) == (0.05, 0.1)
    assert SIM_PRESET["rho_step"] == 0.5
    assert SIM_PRESET["inlier_tol"] == 1.0
    assert SIM_PRESET["strict_degenerate_check"] is True


def test_wrong_preset_loses_inliers_on_noisy_points():
    """
    反证：把仿真尺度的点集加上一点抖动，再用真数据那组（紧 10 倍）容忍度，
    能收进平面的点会明显变少。这条测试说明"参数集不能混用"不是空话。

    （注意：若点集精确落在平面上、残差为 0，两组容忍度都能全收 ——
      单看"能不能找到平面"是测不出参数集混用的。）
    """
    normal = np.array([0.02, 0.03, -1.0])
    normal = normal / np.linalg.norm(normal)
    pts = _plane_points(normal, rho=-10.0, t_min=-1000, t_max=1000)

    rng = np.random.default_rng(0)
    # 沿法向量方向加 ±0.4 的抖动 —— 小于 1.0、大于 0.1
    pts = pts + np.outer(rng.uniform(-0.4, 0.4, len(pts)), normal)

    loose = hough_plane(pts, **SIM_PRESET, max_iter=20000, min_detectors=40,
                        seed=1)
    tight = hough_plane(pts, rho_step=0.05, inlier_tol=0.1, max_iter=20000,
                        min_detectors=40, seed=1)
    assert loose.n_planes >= 1
    assert loose.prediction.sum() > tight.prediction.sum()
