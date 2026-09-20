"""
仿真示例（§3.1.2）：96×96 圆波前 —— 估计精度随噪声/漏检/测量误差的变化

对应论文图 6–8。与 ``demo_simulation.py``（§3.1.1，8×8 线性波前，图 2–5）是
两套不同的仿真，不要混在一起看。

真值是代码自己设定的：源点 (48, 48)、速度 1、激发时刻 2，即

    t = √((x−48)² + (y−48)²) / 1 + 2 + ε

所以每个估计值都能和真值直接比较。

运行：
    python examples/demo_simulation_circular.py                # 按 R 的重复数（~4 分钟）
    python examples/demo_simulation_circular.py --replicates 2 # 快跑（~30 秒）

输出：
    examples/out/circ_fig6_wavefront.png      仿真数据 + 拟合出的圆波前
    examples/out/circ_fig7_accuracy_vs_snr.png
    examples/out/circ_fig8_accuracy_vs_p.png
    examples/out/circ_sigma_accuracy.png
    examples/out/circ_sweep_*.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "out"
_OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_OUT / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_OUT / ".cache"))

# 允许不安装直接运行：把仓库的 src/ 加入搜索路径。
_SRC = _HERE.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
import pandas as pd                                     # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                 # noqa: E402,F401

from wave_hough_detect import (                          # noqa: E402
    CIRCULAR_TRUTH, PAPER_CIRCULAR_PARAMS, accuracy_sweep, evaluate_circular,
    simulate_circular_wavefronts,
)

#: 要跟踪的四个量：显示名、真值、估计列名
QUANTITIES = [("x₀", "x0", CIRCULAR_TRUTH["x0"]),
              ("y₀", "y0", CIRCULAR_TRUTH["y0"]),
              ("v", "v", CIRCULAR_TRUTH["v"]),
              ("t₀", "t0", CIRCULAR_TRUTH["t0"])]


def hr(t=""):
    print()
    print("=" * 78)
    if t:
        print(f" {t}")
        print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════
#  ① 一次仿真 + 拟合（论文图 6 风格）
# ══════════════════════════════════════════════════════════════════════════

def part_one() -> pd.DataFrame:
    hr("① 一次仿真 + 圆波前拟合（论文图 6 风格）")
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    est = evaluate_circular(0)
    sig = df[df.z == 1]

    print(f"  仿真参数 λ_n={PAPER_CIRCULAR_PARAMS['noise_freq']}, "
          f"p={PAPER_CIRCULAR_PARAMS['p']}, σ={PAPER_CIRCULAR_PARAMS['sigma']}")
    print(f"  {len(df)} 点（信号 {est.n_signal} = 96×96 全部电极，噪声 {est.n_noise}）")
    print(f"  真值   x₀={CIRCULAR_TRUTH['x0']:.4f}  y₀={CIRCULAR_TRUTH['y0']:.4f}  "
          f"v={CIRCULAR_TRUTH['v']:.4f}  t₀={CIRCULAR_TRUTH['t0']:.4f}")
    print(f"  估计   x₀={est.x0:.4f}  y₀={est.y0:.4f}  "
          f"v={est.v:.4f}  t₀={est.t0:.4f}   （{est.n_iters} 轮收敛）")
    print(f"  相对误差 " + "  ".join(
        f"{k}={v:.2e}" for k, v in est.rel_error.items()))
    print(f"  R²={est.r2:.6f}   ← 注意：这是对**信号+噪声全部点**算的，"
          f"噪声占 {100*est.n_noise/est.n_points:.1f}%")
    print(f"         真值参数下 R²(仅信号) = 1.0000000000，"
          f"所以 0.95 不代表模型不准")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(sig.x, sig.y, sig.ts, s=3, c="tab:green",
               label=f"signal  n={est.n_signal}", depthshade=False)
    noi = df[df.z == 0]
    if len(noi):
        ax.scatter(noi.x, noi.y, noi.ts, s=8, c="0.4",
                   label=f"noise  n={est.n_noise}", depthshade=False)
    # 拟合出的圆波前：t = √((x−x₀)²+(y−y₀)²)/v + t₀
    g = np.arange(1, 97)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    surf = np.hypot(gx - est.x0, gy - est.y0) / est.v + est.t0
    ax.plot_surface(gx, gy, np.where(surf <= 96, surf, np.nan),
                    alpha=0.25, color="tab:red", linewidth=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")
    ax.set_title(f"96×96 circular wavefront\nfitted (x₀,y₀)=({est.x0:.2f},{est.y0:.2f}), "
                 f"v={est.v:.3f}")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=20, azim=-60)

    # 按距离重排后看残差：模型正确时应当只剩噪声
    ax2 = fig.add_subplot(122)
    d = np.hypot(sig.x - est.x0, sig.y - est.y0)
    pred = d / est.v + est.t0
    ax2.scatter(pred, sig.ts, s=4, c="tab:green", label="signal")
    if len(noi):
        d2 = np.hypot(noi.x - est.x0, noi.y - est.y0)
        ax2.scatter(d2 / est.v + est.t0, noi.ts, s=14, c="0.35",
                    label="noise")
    lim = [0, 100]
    ax2.plot(lim, lim, "k--", lw=1)
    rr = sig.ts.to_numpy() - pred.to_numpy()
    ax2.set_xlabel("fitted arrival time"); ax2.set_ylabel("observed arrival time")
    ax2.set_title(f"observed vs fitted\nsignal residual sd = {rr.std():.2e}")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(_OUT / "circ_fig6_wavefront.png", dpi=130)
    plt.close(fig)
    print(f"  → {_OUT/'circ_fig6_wavefront.png'}")

    return pd.DataFrame([est.as_dict()])


# ══════════════════════════════════════════════════════════════════════════
#  ② 精度 vs 噪声率 λ_n（论文图 7）
# ══════════════════════════════════════════════════════════════════════════

def plot_accuracy(df: pd.DataFrame, xcol: str, xlabel: str, logx: bool,
                  title: str, out_name: str) -> None:
    """
    画「四个估计量 vs 扫描变量」。

    ★ R 的 ``relerror.plot`` 三个活跃调用都传 ``plot.value = TRUE``（R:891/911/931），
      所以它画的是**绝对估计值**，不是相对误差 —— 函数名是名不副实的。
      这里照它的实际行为画绝对值，并把真值画成水平虚线。
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (disp, col, truth) in zip(axes, QUANTITIES):
        grp = df.groupby(xcol)[col]
        xs, mean, sd = grp.mean().index.to_numpy(), grp.mean().to_numpy(), \
            grp.std(ddof=1).to_numpy()
        ax.errorbar(xs, mean, yerr=sd, marker="o", ms=4, capsize=3,
                    lw=1.2, color="tab:blue")
        ax.axhline(truth, color="crimson", ls="--", lw=1.4,
                   label=f"truth = {truth:g}")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"estimated {disp}")
        ax.set_title(f"{disp}     mean ± sd over replicates", fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(_OUT / out_name, dpi=130)
    plt.close(fig)
    print(f"  → {_OUT/out_name}")


def part_sweep(plot_mode: int, n_rep: int | None) -> pd.DataFrame:
    hr(f"② 精度扫描：plot_mode={plot_mode}（论文图 "
       f"{ {1: '7', 2: '8', 3: 'σ 曲线'}[plot_mode] }）")
    t0 = time.time()
    df = accuracy_sweep(plot_mode, n_replicates=n_rep)
    dt = time.time() - t0
    print(f"  {len(df)} 次「仿真 + 拟合」，耗时 {dt:.1f} s（{1000*dt/len(df):.0f} ms/次）")

    var = df["variable"].iloc[0]
    summ = df.groupby("scan_value").agg(
        snr=("snr", "mean"), n_noise=("n_noise", "mean"),
        x0=("x0", "mean"), y0=("y0", "mean"), v=("v", "mean"), t0=("t0", "mean"),
        rel_v=("rel_v", "mean"), rel_t0=("rel_t0", "mean"),
        rel_x0=("rel_x0", "mean"), rel_y0=("rel_y0", "mean"))
    print()
    print(f"  {'λ/参数':>10s} {'snr':>9s} {'噪声数':>8s} {'x₀误差':>10s} "
          f"{'y₀误差':>10s} {'v 误差':>10s} {'t₀误差':>10s}")
    print("  " + "-" * 74)
    for val, r in summ.iterrows():
        print(f"  {val:10.4g} {r['snr']:9.2f} {r['n_noise']:8.0f} "
              f"{r['rel_x0']:10.2e} {r['rel_y0']:10.2e} "
              f"{r['rel_v']:10.2e} {r['rel_t0']:10.2e}")

    print()
    print("  ★ 可读性：源点位置 (x₀, y₀) 在所有噪声水平下都很稳（相对误差 ~1e-3），")
    print("     而 v 与 t₀ 会一起漂 —— 噪声点多时，拟合靠「放慢速度 + 推迟激发」")
    print("     去迁就那些离群点。这正是图 7 想展示的退化方式。")

    df.to_csv(_OUT / f"circ_sweep_mode{plot_mode}.csv", index=False)
    print(f"  → {_OUT/f'circ_sweep_mode{plot_mode}.csv'}")

    if plot_mode == 1:
        plot_accuracy(df, "snr", "signal-to-noise ratio (n_signal / n_noise)",
                      True, "Estimation accuracy vs noise rate (paper Fig. 7 style)",
                      "circ_fig7_accuracy_vs_snr.png")
    elif plot_mode == 2:
        plot_accuracy(df, "scan_value", "missing-observation probability  p",
                      False, "Estimation accuracy vs missing probability "
                      "(paper Fig. 8 style)", "circ_fig8_accuracy_vs_p.png")
    else:
        plot_accuracy(df, "scan_value", "measurement error  σ", False,
                      "Estimation accuracy vs measurement error",
                      "circ_sigma_accuracy.png")
    return df


def main(n_rep: int | None) -> int:
    T0 = time.time()

    part_one()
    part_sweep(1, n_rep)
    part_sweep(2, n_rep)
    part_sweep(3, n_rep)

    hr()
    print(f" ✅ 全部完成，总耗时 {time.time()-T0:.1f} s")
    print(f"    图: {_OUT}/circ_fig6_wavefront.png")
    print(f"        {_OUT}/circ_fig7_accuracy_vs_snr.png")
    print(f"        {_OUT}/circ_fig8_accuracy_vs_p.png")
    print(f"        {_OUT}/circ_sigma_accuracy.png")
    print()
    print("    ★ 重复数是【独立】的：每个重复用了不同的 seed_shift。")
    print("      R 的 stomach.sim2d 没有 seed.shift，而且 p=0 时种子恒为 0，")
    print("      所以 R 的 mode 1/3「重复」共享同一条随机数流，曲线被人为抹平。")
    print("      想复现那个效应：accuracy_sweep(mode, independent_replicates=False)")
    print()
    return 0


def _cli() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=None,
                    help="每个网格点的重复次数；默认用 R 提交时的值"
                         "（mode 1→23、mode 2→9、mode 3→10）")
    a = ap.parse_args()
    return main(a.replicates)


if __name__ == "__main__":
    raise SystemExit(_cli())
