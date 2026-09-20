"""
累加器量化与内点判据的单元测试。

这些是全套代码里最容易被"顺手改好"从而静默出错的地方：
R 的 ``as.integer()`` 是向零取整而不是 floor，写错一位就有一个步长的偏差。
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import INLIER_TOL, RHO_STEP
from wave_hough_detect.rht import (
    SIM_PRESET,
    _step_key,
    find_points_on_plane,
    get_key,
    is_point_on_plane,
    spherical_key,
)


# ── _step_key：向零取整 ──────────────────────────────────────────────────

@pytest.mark.parametrize("x,step,expected", [
    (9.94, 0.05, 9.90),          # 正数：trunc 与 floor 相同
    (0.0, 0.05, 0.0),
    (-4.81374, 0.05, -4.80),     # ★ 负数：trunc 给 -96 步，floor 会给 -97 步
    (-0.049, 0.05, 0.0),         # 负的小量 → 0，floor 会给 -0.05
    (83.9, 2.0, 82.0),
    (-1.0, 2.0, 0.0),
])
def test_step_key_truncates_toward_zero(x, step, expected):
    assert _step_key(x, step) == pytest.approx(expected, abs=1e-12)


def test_step_key_negative_rho_is_not_floored():
    """把 trunc 写成 floor 的后果：负 rho 的桶整体差一个步长。"""
    val = -4.81374
    assert _step_key(val, 0.05) == pytest.approx(-4.80)
    assert np.floor(val / 0.05) * 0.05 == pytest.approx(-4.85)
    assert _step_key(val, 0.05) != pytest.approx(np.floor(val / 0.05) * 0.05)


# ── get_key：字符串量化键 ────────────────────────────────────────────────

def test_get_key_format():
    key = get_key(9.94, np.radians(84.4), np.radians(14.4))
    assert key == "9.9,84,14"


def test_get_key_rho_step_is_adjustable():
    """
    真数据管线 ρ 步长 0.05，仿真研究 0.5 —— 两者必须能分别指定，
    否则仿真会去用一个错 10 倍的累加器。
    """
    rho = 12.34
    assert get_key(rho, 0.0, 0.0, rho_step=0.05).split(",")[0] == "12.3"
    assert get_key(rho, 0.0, 0.0, rho_step=0.5).split(",")[0] == "12"
    assert SIM_PRESET["rho_step"] == 0.5
    assert RHO_STEP == 0.05


def test_get_key_phi_theta_are_degrees():
    # phi = 90° → 第二个字段 90；theta = 0° → 第三个字段 0
    key = get_key(1.0, np.pi / 2, 0.0)
    _, phi_s, theta_s = key.split(",")
    assert float(phi_s) == 90.0
    assert float(theta_s) == 0.0


def test_get_key_truncation_amplifies_float_roundoff():
    """
    ★ 一个真实存在的边界效应，必须如实保留：

        np.radians(-30) → -0.5235987755982988
        np.degrees(...) → -29.999999999999996
        -29.999999999999996 / 2 = -14.999999999999998
        trunc → -14        （而不是 -15）
        量化结果 = -28     （而不是 -30）

    R 的 as.integer() 同样会给出 -14，所以这不是 bug，而是"量化对浮点误差
    敏感"这一事实本身。用 floor 或 round 反而会与 R 不一致。
    """
    key = get_key(1.0, np.pi / 2, float(np.radians(-30.0)))
    assert key.split(",")[2] == "-28"


def test_get_key_never_produces_negative_zero():
    """
    第二个浮点细节：R 的 as.integer() 返回整数，-0.0 会变成 0L，键里是 "0"；
    Python 的 -0.0 用 :g 格式化会得到 "-0"。两者分组等价但键字符串不同，
    这里必须归一成 "0" 才能与 R 逐字符比对。
    """
    key = get_key(-0.0, -0.0, -0.0)
    assert "-0" not in key
    assert key == "0,0,0"


# ── spherical_key：由法向量反算键 ────────────────────────────────────────

def test_spherical_key_matches_get_key():
    """spherical_key 只是 get_key 的一个包装，两者必须一致。"""
    normal = np.array([0.0033, 0.0117, -0.99993])
    rho = -19.66
    phi = float(np.arccos(normal[2]))
    theta = float(np.arcsin(normal[1] / np.sin(phi)))
    assert spherical_key(normal, rho) == get_key(abs(rho), phi, theta)


def test_spherical_key_handles_phi_zero():
    """
    法向量恰好平行于 t 轴时 φ=0，R 会算出 theta=NaN。
    这里必须返回一个确定性的键而不是抛异常或产生 NaN 字符串。
    """
    key = spherical_key(np.array([0.0, 0.0, 1.0]), 5.0)
    assert "nan" not in key.lower()
    assert key == spherical_key(np.array([0.0, 0.0, 1.0]), 5.0)


# ── is_point_on_plane / find_points_on_plane ─────────────────────────────

def test_inlier_tolerance_default_is_real_data_value():
    assert INLIER_TOL == 0.1
    n = np.array([0.0, 0.0, 1.0])
    assert is_point_on_plane(np.array([3.0, 4.0, 1.0 + 0.05]), n, 1.0)
    assert not is_point_on_plane(np.array([3.0, 4.0, 1.0 + 0.2]), n, 1.0)


def test_inlier_tolerance_is_adjustable():
    n = np.array([0.0, 0.0, 1.0])
    p = np.array([3.0, 4.0, 1.5])
    assert not is_point_on_plane(p, n, 1.0)
    assert is_point_on_plane(p, n, 1.0, tol=1.0)


def test_find_points_on_plane_preserves_index_order():
    """R 按 unclassify.indices 的顺序 append，返回顺序必须一致。"""
    pts = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    idx = np.array([7, 2, 5, 0])
    got = find_points_on_plane(pts, np.array([0.0, 0.0, 1.0]), 0.0, idx)
    assert got.tolist() == idx.tolist()


def test_find_points_on_plane_empty_when_no_hits():
    pts = np.column_stack([np.arange(5.0), np.zeros(5), np.full(5, 9.0)])
    got = find_points_on_plane(pts, np.array([0.0, 0.0, 1.0]), 0.0, np.arange(5))
    assert got.size == 0
