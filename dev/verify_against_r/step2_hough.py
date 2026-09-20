"""
阶段 2 验证脚本：Python 的随机霍夫变换 vs R 的随机霍夫变换

【验证方法】
随机数发生器无法跨语言复现（R 用 Mersenne-Twister + Rejection 采样，
numpy 用 PCG64）。所以这里让 R 先把【每次抽样的三元组下标】导出来，
Python 重放同一条轨迹。

这样除了随机数发生器本身，算法逻辑的每一步都被逐位比对：
  叉积 → 符号规范化 → 球坐标 → 量化键 → 投票 → 最小二乘精修
  → 内点判定 → 探测器覆盖 → 平面接受 → 累加器清空
  → 5° 容差 → 未分类点回收

运行（在仓库根目录）：
    python dev/verify_against_r/step2_hough.py

运行前需先执行 R 脚本产出轨迹与基准结果：
    Rscript R_work/run2_trace.R
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _paths import OUT as MY_OUT, R_OUT as OUT, hr, require_r_outputs

from wave_hough_detect import hough_plane  # noqa: E402


def main() -> int:
    if not require_r_outputs("step2_hough.py"):
        return 2
    if not (MY_OUT / "spikes_xyz.csv").exists():
        print(f"\n❌ 缺少 {MY_OUT / 'spikes_xyz.csv'}")
        print("   请先运行 dev/verify_against_r/step1_spikes.py\n")
        return 2

    MY_OUT.mkdir(parents=True, exist_ok=True)

    hr("步骤 1/4  读入数据与 R 的抽样轨迹")
    spikes = pd.read_csv(MY_OUT / "spikes_xyz.csv")       # t 已是 /200 尺度
    points = np.column_stack([spikes["x"], spikes["y"],
                              spikes["t"]]).astype(float)

    trace_df = pd.read_csv(OUT / "trace.csv")
    trace = trace_df[["i1", "i2", "i3"]].to_numpy(dtype=int) - 1   # R 是 1-based
    print(f"  点数      : {len(points)}")
    print(f"  坐标范围  : x {points[:,0].min():.0f}–{points[:,0].max():.0f}  "
          f"y {points[:,1].min():.0f}–{points[:,1].max():.0f}  "
          f"t {points[:,2].min():.4f}–{points[:,2].max():.4f}")
    print(f"  轨迹长度  : {len(trace)} 次抽样")

    hr("步骤 2/4  用同一条轨迹运行 Python 版")
    res = hough_plane(points, vote_threshold=8, max_iter=200000,
                      min_detectors=40, trace=trace)
    print("  " + res.summary().replace("\n", "\n  "))

    hr("步骤 3/4  与 R 的结果逐位比对")
    r_res = pd.read_csv(OUT / "hough_result.csv")
    failures: list[str] = []

    def cmp(name: str, r_col: str, p_arr, tol: float = 0.0) -> None:
        r_arr = r_res[r_col].to_numpy()
        p_arr = np.asarray(p_arr)
        if r_arr.dtype.kind in "if" and p_arr.dtype.kind in "if":
            d = np.abs(r_arr.astype(float) - p_arr.astype(float))
            mx = float(np.nanmax(d)) if d.size else 0.0
            ok = mx <= tol
            print(f"  {'✅' if ok else '❌'} {name:20s} 最大差 {mx:.3e}"
                  f"  (容差 {tol:.0e})")
            if not ok:
                failures.append(f"{name} 最大差 {mx:.3e}")
        else:
            ok = bool(np.array_equal(r_arr, p_arr))
            n_diff = int((r_arr != p_arr).sum())
            print(f"  {'✅' if ok else '❌'} {name:20s} "
                  f"{'完全一致' if ok else f'{n_diff} 处不同'}")
            if not ok:
                failures.append(f"{name} 有 {n_diff} 处不同")

    cmp("prediction", "prediction", res.prediction)
    cmp("plane_indices", "plane.indices", res.plane_indices)
    cmp("keys", "keys", res.keys)
    cmp("n1", "n1", res.n1, tol=1e-12)
    cmp("n2", "n2", res.n2, tol=1e-12)
    cmp("n3", "n3", res.n3, tol=1e-12)
    cmp("rhos", "rhos", res.rhos, tol=1e-12)

    # 平面数
    r_np = int(r_res["plane.indices"].max())
    print(f"  {'✅' if r_np == res.n_planes else '❌'} "
          f"{'平面数':20s} R={r_np}  Python={res.n_planes}")
    if r_np != res.n_planes:
        failures.append(f"平面数不同 R={r_np} Python={res.n_planes}")

    # 被接受的平面法向量
    r_pl = pd.read_csv(OUT / "hough_planes.csv")
    print()
    print("  被接受平面的法向量与 ρ：")
    print("    平面 │    R 的 n1/n2/n3                      │  Python 与 R 的最大差")
    for _, row in r_pl.iterrows():
        al = [a for a in res.accept_log if a["plane"] == int(row["plane"])]
        if not al:
            print(f"    {int(row['plane']):4d} │  (Python 未记录)")
            continue
        pn = al[0]["normal"]
        d = np.abs(np.array([row["n1"], row["n2"], row["n3"]]) - pn).max()
        dr = abs(row["rho"] - al[0]["rho"])
        flag = "✅" if (d < 1e-12 and dr < 1e-12) else "❌"
        print(f"    {int(row['plane']):4d} │ ({row['n1']:+.6f}, {row['n2']:+.6f}, "
              f"{row['n3']:+.6f})  │ {flag} n̂ {d:.2e}  ρ {dr:.2e}")
        if d >= 1e-12 or dr >= 1e-12:
            failures.append(f"平面 {int(row['plane'])} 法向量/ρ 有差异")

    # 接受时机（迭代次数）
    print()
    print("  平面接受时机（迭代次数）：")
    r_iters = None
    if "iter" in r_pl.columns:
        r_iters = r_pl["iter"].tolist()
    p_iters = [a["iter"] for a in res.accept_log]
    print(f"    R      : {[a for a in (r_iters or ['(未导出)'])]}")
    print(f"    Python : {p_iters}")
    if r_iters and r_iters != p_iters:
        failures.append(f"接受时机不同 R={r_iters} Python={p_iters}")

    hr("步骤 4/4  写出 Python 结果")
    out_df = pd.DataFrame({
        "idx": np.arange(len(points)),
        "x": points[:, 0], "y": points[:, 1], "t": points[:, 2],
        "prediction": res.prediction, "keys": res.keys,
        "n1": res.n1, "n2": res.n2, "n3": res.n3,
        "plane_indices": res.plane_indices, "rhos": res.rhos,
    })
    out_df.to_csv(MY_OUT / "hough_result_py.csv", index=False)
    print(f"  {MY_OUT / 'hough_result_py.csv'}")

    hr()
    if failures:
        print(" ❌ 移植失败，存在以下差异：")
        for f in failures:
            print(f"    - {f}")
        return 1
    print(" ✅ 移植成功：重放同一条抽样轨迹，Python 与 R 的每一步结果完全一致")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
