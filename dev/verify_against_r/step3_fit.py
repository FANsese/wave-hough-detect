"""
阶段 3 验证：Python 的波前拟合 vs R 的波前拟合

流程与 R 完全一致（注意两段尺度切换）：
    尖峰(ms) --/200--> 霍夫尺度 --重放R的轨迹--> 平面归属
                            |
                            +--×200--> 回 ms --> result.array --> 拟合

运行（在仓库根目录）：
    python dev/verify_against_r/step3_fit.py

运行前先执行：
    Rscript R_work/run2_trace.R     # trace.csv
    Rscript R_work/run3_fit.R       # result_array_r.csv 等基准
    python dev/verify_against_r/step1_spikes.py   # 生成 spikes_xyz.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _paths import OUT as MY_OUT, R_OUT, hr, require_r_outputs

from wave_hough_detect import (  # noqa: E402
    build_result_array,
    fit_circular,
    fit_linear,
    hough_plane,
    plane_points,
)


def main() -> int:
    if not require_r_outputs("step3_fit.py"):
        return 2
    if not (MY_OUT / "spikes_xyz.csv").exists():
        print(f"\n❌ 缺少 {MY_OUT / 'spikes_xyz.csv'}")
        print("   请先运行 dev/verify_against_r/step1_spikes.py\n")
        return 2

    MY_OUT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    hr("步骤 1/4  重放 R 的霍夫轨迹")
    sp = pd.read_csv(MY_OUT / "spikes_xyz.csv")          # t 已是 /200 尺度
    pts = np.column_stack([sp["x"], sp["y"], sp["t"]]).astype(float)
    trace = pd.read_csv(R_OUT / "trace.csv")[["i1", "i2", "i3"]].to_numpy(int) - 1
    res = hough_plane(pts, vote_threshold=8, max_iter=200000,
                      min_detectors=40, trace=trace)
    print(f"  点数 {len(pts)}  轨迹 {len(trace)}  找到平面 {res.n_planes}")

    # 按 R 的行序重建 all.lines（R 用 order(ts.observ) 稳定排序）
    order = np.argsort(sp["t"].to_numpy(), kind="stable")
    x_obs = sp["x"].to_numpy()[order]
    y_obs = sp["y"].to_numpy()[order]
    t_obs = sp["t"].to_numpy()[order]                     # /200 尺度
    plane_idx = res.plane_indices[order]

    hr("步骤 2/4  构造 result.array（含 ×200 回到 ms）")
    arr = build_result_array(x_obs, y_obs, t_obs, plane_idx, res.n_planes)
    print(f"  形状 {arr.shape}  非空槽 {int((arr > 0).sum())} / {arr.size}")
    print(f"  ts 范围 {arr[arr > 0].min():.2f} – {arr[arr > 0].max():.2f} ms")

    # 与 R 的 result.array 对比
    r_ra = pd.read_csv(R_OUT / "result_array_r.csv")
    mismatch = 0
    for _, row in r_ra.iterrows():
        mine = arr[int(row["x"]) - 1, int(row["y"]) - 1, int(row["plane"]) - 1]
        if abs(mine - row["ts"]) > 1e-9:
            mismatch += 1
    print(f"  {'✅' if mismatch == 0 else '❌'} 与 R 的 result.array 逐格对比: "
          f"{mismatch} 处不同")
    if mismatch:
        failures.append(f"result.array 有 {mismatch} 处不同")

    hr("步骤 3/4  逐平面拟合（圆波前 + 线波前）")
    r_circ = pd.read_csv(R_OUT / "fit_circular_r.csv")
    r_lin = pd.read_csv(R_OUT / "fit_linear_r.csv")

    rows_c, rows_l = [], []
    for k in range(1, res.n_planes + 1):
        p = plane_points(arr, k)
        fc = fit_circular(p)
        fl = fit_linear(p)
        rows_c.append(dict(plane=k, x0=fc.x0, y0=fc.y0, u=1 / fc.v, v=fc.v,
                           t0=fc.t0, r2=fc.r2, loss=fc.loss,
                           n_points=fc.n_points, n_iters=fc.n_iters))
        rows_l.append(dict(plane=k, a=fl.a, b=fl.b, v=fl.v, r2=fl.r2,
                           loss=fl.loss, n_points=fl.n_points))
    my_c, my_l = pd.DataFrame(rows_c), pd.DataFrame(rows_l)

    def cmp_table(name, mine, theirs, cols, tol):
        print(f"\n  【{name}】")
        print(f"    {'平面':>4s} " + " ".join(f"{c:>12s}" for c in cols) +
              "   最大差")
        ok_all = True
        for i in range(len(mine)):
            diffs = [abs(float(mine[c][i]) - float(theirs[c][i])) for c in cols]
            ok = max(diffs) <= tol
            ok_all &= ok
            print(f"    {int(mine['plane'][i]):>4d} " +
                  " ".join(f"{float(mine[c][i]):12.6f}" for c in cols) +
                  f"   {max(diffs):.2e} {'✅' if ok else '❌'}")
        if not ok_all:
            failures.append(f"{name} 超出容差")

    cmp_table("圆波前模型（论文表 2 + 表 4 圆模型列）", my_c, r_circ,
              ["x0", "y0", "v", "t0", "r2", "loss"], 1e-6)
    cmp_table("线波前模型（论文表 3 + 表 4 线模型列）", my_l, r_lin,
              ["a", "b", "v", "r2", "loss"], 1e-9)

    hr("步骤 4/4  与论文发表的数字对账")
    # 按 t0 匹配论文表 2（论文的平面编号与接受顺序不同）
    paper_circ = [
        (849.75, -3.93, -50.0, 0.44, 0.948072),
        (2054.29, 9.15, -50.0, 0.49, 0.9593685),
        (3828.32, -3.20, -50.0, 0.46, 0.9373304),
        (5667.08, -3.05, -50.0, 0.46, 0.944263),
        (7647.11, -3.51, -50.0, 0.45, 0.9405785),
    ]
    print(f"    {'论文 t₀':>10s} {'论文 x₀':>9s} {'论文 v':>8s} {'论文 R²':>11s}"
          f"   →  {'本次 t₀':>10s} {'本次 x₀':>9s} {'本次 v':>8s} {'本次 R²':>11s}  判定")
    for pt0, px0, py0, pv, pr2 in paper_circ:
        best = min(rows_c, key=lambda r: abs(r["t0"] - pt0))
        dt = abs(best["t0"] - pt0)
        ok = dt < 0.02
        print(f"    {pt0:10.2f} {px0:9.2f} {pv:8.2f} {pr2:11.7f}"
              f"   →  {best['t0']:10.2f} {best['x0']:9.2f} {best['v']:8.2f}"
              f" {best['r2']:11.7f}  {'✅' if ok else '❌'}")
        if not ok:
            failures.append(f"t0={pt0} 与论文不符")

    # 表 3 线性模型
    paper_lin = [(0.34, 2.22, 0.44, 0.9426517), (-0.18, 2.03, 0.49, 0.9566555),
                 (0.30, 2.14, 0.46, 0.9308701), (0.29, 2.13, 0.46, 0.9384128),
                 (0.31, 2.19, 0.45, 0.934337)]
    print()
    print(f"    {'论文 ã':>8s} {'论文 b̃':>8s} {'论文 v':>8s} {'论文 R²':>11s}"
          f"   →  {'本次 ã':>8s} {'本次 b̃':>8s} {'本次 v':>8s} {'本次 R²':>11s}  判定")
    for pa, pb, pv, pr2 in paper_lin:
        best = min(rows_l, key=lambda r: abs(r["r2"] - pr2))
        ok = abs(best["r2"] - pr2) < 1e-6
        print(f"    {pa:8.2f} {pb:8.2f} {pv:8.2f} {pr2:11.7f}"
              f"   →  {best['a']:8.4f} {best['b']:8.4f} {best['v']:8.4f}"
              f" {best['r2']:11.7f}  {'✅' if ok else '❌'}")
        if not ok:
            failures.append(f"线性 R²={pr2} 与论文不符")

    my_c.to_csv(MY_OUT / "fit_circular_py.csv", index=False)
    my_l.to_csv(MY_OUT / "fit_linear_py.csv", index=False)

    hr()
    if failures:
        print(" ❌ 存在差异：")
        for f in failures:
            print(f"    - {f}")
        return 1
    print(" ✅ 移植成功：圆/线波前拟合与 R 一致，且与论文表 2/3/4 吻合")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
