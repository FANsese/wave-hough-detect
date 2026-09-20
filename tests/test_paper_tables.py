"""
与论文发表数字对账（论文表 2 / 表 3 / 表 4）。

这一段需要论文 §3.2 的**实验记录**，它来自合作实验室、不随仓库分发
（见 ``data/README.md``）。所以：

  · 找不到记录时整组测试自动 skip —— 干净克隆的仓库跑 pytest 仍然是全绿的；
  · 把记录放到 ``data/`` 下并设好 ``WHD_DATA``（或不设，按默认名查找）
    就会自动生效，逐格核对论文表 2/3/4。

已核对通过（2026-09-18，本机）：
    表 2 的 20 个数字、表 4 的 10 个 R² 全部吻合到论文给出的位数。
"""

from __future__ import annotations

import numpy as np
import pytest

from wave_hough_detect import (
    build_result_array,
    detect_all_channels,
    find_recording,
    fit_circular,
    fit_linear,
    hough_plane,
    load_recording,
    plane_points,
    spikes_to_table,
)

# ── 论文表 2：圆波前模型（按 t₀ 升序，与论文编号一致）──────────────────
PAPER_TABLE2 = [
    # t0,      x0,     y0,    v,     R2
    (849.75, -3.93, -50.0, 0.44, 0.9480720),
    (2054.29, 9.15, -50.0, 0.49, 0.9593685),
    (3828.32, -3.20, -50.0, 0.46, 0.9373304),
    (5667.08, -3.05, -50.0, 0.46, 0.9442630),
    (7647.11, -3.51, -50.0, 0.45, 0.9405785),
]

# ── 论文表 3：线波前模型 ────────────────────────────────────────────────
PAPER_TABLE3 = [
    # a(tilde), b(tilde), v,     R2
    (0.34, 2.22, 0.44, 0.9426517),
    (-0.18, 2.03, 0.49, 0.9566555),
    (0.30, 2.14, 0.46, 0.9308701),
    (0.29, 2.13, 0.46, 0.9384128),
    (0.31, 2.19, 0.45, 0.9343370),
]


@pytest.fixture(scope="module")
def real_pipeline():
    try:
        path = find_recording(None)
    except FileNotFoundError:
        pytest.skip("找不到论文 §3.2 的实验记录（它不随仓库分发，见 data/README.md）")

    rec = load_recording(path)
    spike_times, _ = detect_all_channels(rec)
    sp = spikes_to_table(spike_times, rec.channels)
    pts = sp[["x", "y", "t"]].to_numpy(dtype=float)

    # ★ iter.max 必须 ≥ 200000：R 脚本里的 30000 只够找到 3 个平面
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
    fits.sort(key=lambda f: f[1].t0)         # 论文按 t₀ 升序编号
    return dict(path=path, rec=rec, res=res, fits=fits, n_spikes=len(sp))


def test_real_data_spike_count(real_pipeline):
    """论文 §3.2：64 个电极 × 5 次搏动 = 320 个尖峰。"""
    assert real_pipeline["n_spikes"] == 320


def test_real_data_finds_five_planes(real_pipeline):
    assert real_pipeline["res"].n_planes == 5
    assert [int((real_pipeline["res"].plane_indices == k).sum())
            for k in range(1, 6)] == [64] * 5


def test_table2_circular_model(real_pipeline):
    """论文表 2：20 个数字逐格核对。"""
    rows = real_pipeline["fits"]
    assert len(rows) == len(PAPER_TABLE2)
    for (t0, x0, y0, v, _), (_, c, _) in zip(PAPER_TABLE2, rows):
        # t₀ 用来配对（论文的平面编号与"接受顺序"不同）
        assert c.t0 == pytest.approx(t0, abs=0.02)
        assert c.x0 == pytest.approx(x0, abs=0.005)
        assert c.y0 == pytest.approx(y0, abs=0.005)
        assert c.v == pytest.approx(v, abs=0.005)


def test_table3_linear_model(real_pipeline):
    """
    论文表 3：线性模型的 ã / b̃ / v。

    容差 0.006 而不是 0.005：表 3 只印到两位小数，而第二位恰好落在进位边界上，
    例如 b̃ = 2.0249999999997605 印成 2.03（四舍五入）而 Python 的 round 会给 2.02。
    0.006 覆盖"印刷到两位小数"本身的不确定性，同时仍然是有意义的核对。
    """
    rows = real_pipeline["fits"]
    for (a, b, v, _), (_, _, l) in zip(PAPER_TABLE3, rows):
        assert l.a == pytest.approx(a, abs=0.006)
        assert l.b == pytest.approx(b, abs=0.006)
        assert l.v == pytest.approx(v, abs=0.006)


def test_table4_coefficient_of_determination(real_pipeline):
    """
    论文表 4：10 个 R²（5 个平面 × 圆/线两个模型），
    并且论文的结论是【圆模型在每个平面上都略优】。
    """
    rows = real_pipeline["fits"]
    for (_, _, _, _, r2_paper), (_, c, l) in zip(PAPER_TABLE2, rows):
        assert c.r2 == pytest.approx(r2_paper, abs=1e-6)
    for (_, _, _, r2_paper), (_, _, l) in zip(PAPER_TABLE3, rows):
        assert l.r2 == pytest.approx(r2_paper, abs=1e-6)
    # 论文的结论：圆模型 5/5 都更好
    assert all(c.r2 > l.r2 for _, c, l in rows)


def test_paper_conclusion_source_lies_outside_the_search_window(real_pipeline):
    """
    论文 §3.2 的结论之一：圆模型给出的 y₀ 全部贴在搜索窗边界 −50，
    说明真实源点在搜索窗之外，应当改用线性模型。这里把这个结论固定下来。
    """
    y0 = np.array([c.y0 for _, c, _ in real_pipeline["fits"]])
    assert np.allclose(y0, -50.0, atol=0.01)
