"""
仿真示例：论文 §3.1 的检测性能（图 2–5 风格）+ 合成记录上的完整管线

这个脚本回答一个问题：**没有实验数据，这份代码还能证明什么？**

  ① 在已知真值的仿真上跑 RHT → 算出 FPR / FNR（论文图 5）
  ② 把找到的平面画出来，与真值对照（论文图 2/3/4 风格）
  ③ 生成一份格式与真数据相同的合成 MEA 记录 → 跑通阶段 1→2→3 整条管线，
     证明拿不到实验记录时管线同样可以端到端运行

运行：
    python examples/demo_simulation.py                 # 100 个随机种子
    python examples/demo_simulation.py --seeds 20      # 快一点

输出：
    examples/out/sim_fig2_ground_truth.png
    examples/out/sim_fig3_classification.png
    examples/out/sim_fig4_planes.png
    examples/out/sim_fig5_fpr_fnr.png
    examples/out/sim_detection_rates.csv
    examples/out/sim_synthetic_pipeline.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_OUT_EARLY = _HERE / "out"
_OUT_EARLY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_OUT_EARLY / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_OUT_EARLY / ".cache"))

# 允许不安装直接运行：把仓库的 src/ 加入搜索路径。
# 若已 `pip install -e .`，这一句无副作用。
_SRC = _HERE.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                # noqa: E402,F401

from wave_hough_detect import (                        # noqa: E402
    PAPER_2D_PARAMS, build_result_array, detect_all_channels, evaluate_detection,
    fit_circular, fit_linear, hough_plane, load_recording, plane_points,
    simulate_recording, spikes_to_table,
)

OUT = _OUT_EARLY

CLASS_COLOR = {"green": "tab:green", "yellow": "gold",
               "red": "tab:red", "black": "0.25"}
CLASS_LABEL = {"green": "TP  signal→signal", "yellow": "FP  noise→signal",
               "red": "FN  signal→noise", "black": "TN  noise→noise"}
CLASS_ORDER = ["green", "yellow", "red", "black"]


def hr(t=""):
    print()
    print("=" * 78)
    if t:
        print(f" {t}")
        print("=" * 78)


def md_table(df: pd.DataFrame, fmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(float(v), fmt)
                         if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def draw_planes(ax, result, n_grid: int = 8) -> None:
    """
    在 3D 图上画出 RHT 找到的平面（对应 R 的 draw.plane()，fig5_...R:268-277）。

    平面由 ``n1·x + n2·y + n3·t = ρ`` 给定，在网格四角解出 t 即得四个顶点。
    """
    lo, hi = 1.0, float(n_grid)
    u = np.array([lo, hi])
    for k in range(len(result.rhos)):
        n1, n2, n3 = result.normals[k]
        if n3 == 0:
            continue
        rho = result.rhos[k]
        t11 = (rho - n1 * lo - n2 * lo) / n3
        t12 = (rho - n1 * lo - n2 * hi) / n3
        t21 = (rho - n1 * hi - n2 * lo) / n3
        t22 = (rho - n1 * hi - n2 * hi) / n3
        ax.plot_surface(u[None, :].repeat(2, 0), u[:, None].repeat(2, 1),
                        np.array([[t11, t12], [t21, t22]]),
                        alpha=0.30, color=f"C{k % 10}", linewidth=0)


# ══════════════════════════════════════════════════════════════════════════
#  ① 一次仿真 + 检测：真值、四色分类、找到的平面
# ══════════════════════════════════════════════════════════════════════════

def part_detection(seed_shift: int) -> None:
    hr("① 一次仿真 + RHT（论文 §3.1.1 / 图 2–4 风格）")
    t0 = time.time()
    res = evaluate_detection(seed_shift)
    cls = res.classified()
    print(f"  仿真参数: {PAPER_2D_PARAMS}")
    print(f"  点数 {res.rates.n_points}（信号 {res.rates.n_signal}，"
          f"噪声 {res.rates.n_noise}）")
    print(f"  RHT 找到 {res.rates.n_planes} 个平面，耗时 {time.time()-t0:.2f} s")
    print(f"  TP={res.rates.n_true_pos}  FP={res.rates.n_false_pos}  "
          f"FN={res.rates.n_false_neg}  TN={res.rates.n_true_neg}")
    print(f"  FPR={res.rates.false_positive_rate:.4f}  "
          f"FNR={res.rates.false_negative_rate:.4f}"
          f"   ← 论文口径（分母 = 全部点）")
    print(f"  FPR={res.rates.false_positive_rate_among_noise:.4f}  "
          f"FNR={res.rates.false_negative_rate_among_signal:.4f}"
          f"   ← 常规口径（分母 = 各类点数）")

    sig = cls[cls["z"] == 1]
    noi = cls[cls["z"] == 0]

    # ── 图 2：真值（论文图 2 风格）─────────────────────────────────────
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(sig["x"], sig["y"], sig["ts"], s=16, c="tab:green",
               label=f"true signal (z=1), n={len(sig)}", depthshade=False)
    ax.scatter(noi["x"], noi["y"], noi["ts"], s=16, c="0.55",
               label=f"noise (z=0), n={len(noi)}", depthshade=False)
    ax.set_xlabel("x (row)"); ax.set_ylabel("y (col)"); ax.set_zlabel("t")
    ax.set_title("Ground truth of one simulated dataset\n(paper Fig. 2 style)")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=18, azim=-58)

    ax2 = fig.add_subplot(122)
    ax2.scatter(sig["ts"], sig["y"] + 0.12 * (sig["x"] - 4.5), s=16,
                c="tab:green", label="true signal")
    ax2.scatter(noi["ts"], noi["y"] + 0.12 * (noi["x"] - 4.5), s=16,
                c="0.55", label="noise")
    ax2.set_xlabel("t"); ax2.set_ylabel("y (jittered by x)")
    ax2.set_title("Side view — each wavefront is a plane")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "sim_fig2_ground_truth.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'sim_fig2_ground_truth.png'}")

    # ── 图 3：四色分类（论文图 3 风格，对应 R 的 stomach.plot2d）───────
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(121, projection="3d")
    for key in CLASS_ORDER:                    # 与 R 的绘制顺序一致
        sub = cls[cls["class"] == key]
        if len(sub):
            ax.scatter(sub["x"], sub["y"], sub["ts"], s=16,
                       c=CLASS_COLOR[key],
                       label=f"{CLASS_LABEL[key]}  n={len(sub)}",
                       depthshade=False)
    ax.set_xlabel("x (row)"); ax.set_ylabel("y (col)"); ax.set_zlabel("t")
    ax.set_title("RHT classification\n(paper Fig. 3 style)")
    ax.legend(fontsize=7, loc="upper left"); ax.view_init(elev=18, azim=-58)

    ax2 = fig.add_subplot(122)
    counts = cls["class"].value_counts()
    keys = CLASS_ORDER
    bars = ax2.bar([CLASS_LABEL[k] for k in keys],
                   [int(counts.get(k, 0)) for k in keys],
                   color=[CLASS_COLOR[k] for k in keys])
    ax2.bar_label(bars)
    ax2.set_ylabel("number of spikes")
    ax2.set_title("Confusion counts")
    ax2.tick_params(axis="x", labelrotation=12, labelsize=8)
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "sim_fig3_classification.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'sim_fig3_classification.png'}")

    # ── 图 4：找到的平面（论文图 4 风格，对应 R 的 draw.plane）──────────
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(noi["x"], noi["y"], noi["ts"], s=12, c="0.78",
               depthshade=False, label="noise")
    for k in range(1, res.rates.n_planes + 1):
        sub = cls[cls["plane"] == k]
        ax.scatter(sub["x"], sub["y"], sub["ts"], s=20,
                   color=f"C{(k - 1) % 10}",
                   label=f"plane {k} ({len(sub)} pts)", depthshade=False)
    draw_planes(ax, res)
    ax.set_xlabel("x (row)"); ax.set_ylabel("y (col)"); ax.set_zlabel("t")
    ax.set_title("Planes found by RHT\n(paper Fig. 4 style)")
    ax.legend(fontsize=7, loc="upper left"); ax.view_init(elev=18, azim=-58)

    ax2 = fig.add_subplot(122)
    ax2.scatter(cls["ts"], cls["y"] + 0.12 * (cls["x"] - 4.5), s=12, c="0.88",
                label="all spikes")
    for k in range(1, res.rates.n_planes + 1):
        sub = cls[cls["plane"] == k]
        ax2.scatter(sub["ts"], sub["y"] + 0.12 * (sub["x"] - 4.5), s=18,
                    color=f"C{(k - 1) % 10}", label=f"plane {k}")
    ax2.set_xlabel("t"); ax2.set_ylabel("y (jittered by x)")
    ax2.set_title("Side view — planes separate cleanly")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "sim_fig4_planes.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'sim_fig4_planes.png'}")


# ══════════════════════════════════════════════════════════════════════════
#  ② FPR / FNR 分布（论文图 5）
# ══════════════════════════════════════════════════════════════════════════

def part_sweep(n_seeds: int) -> pd.DataFrame:
    hr(f"② FPR / FNR 随随机种子的分布（论文图 5，{n_seeds} 个种子）")
    t0 = time.time()
    rows = []
    for s in range(n_seeds):
        r = evaluate_detection(s)
        rows.append({"seed_shift": s, **r.rates.as_dict()})
        if (s + 1) % 25 == 0:
            print(f"    ... {s + 1}/{n_seeds}  ({time.time() - t0:.0f} s)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sim_detection_rates.csv", index=False)

    dt = time.time() - t0
    print(f"  完成 {n_seeds} 次，总耗时 {dt:.0f} s（平均 {1000 * dt / n_seeds:.0f} ms/次）")
    print()
    print(f"  {'量':24s} {'均值':>9s} {'标准差':>9s} {'中位数':>9s} {'最大':>9s}")
    print("  " + "-" * 64)
    for col, name in [("FPR", "FPR  (论文口径)"),
                      ("FNR", "FNR  (论文口径)"),
                      ("FPR_among_noise", "FPR  (常规口径)"),
                      ("FNR_among_signal", "FNR  (常规口径)"),
                      ("n_planes", "找到的平面数")]:
        v = df[col].to_numpy(dtype=float)
        print(f"  {name:24s} {v.mean():9.4f} {v.std(ddof=1):9.4f} "
              f"{np.median(v):9.4f} {v.max():9.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, [("FPR", "FPR (paper definition)"),
                                       ("FNR", "FNR (paper definition)"),
                                       ("n_planes", "planes found by RHT")]):
        v = df[col].to_numpy(dtype=float)
        bp = ax.boxplot(v, widths=0.5, patch_artist=True,
                        medianprops=dict(color="crimson", lw=2))
        bp["boxes"][0].set_facecolor("lightsteelblue")
        jit = rng.normal(0, 0.045, v.size)
        ax.scatter(np.full(v.size, 1) + jit, v, s=7, c="0.35", alpha=0.45,
                   zorder=3)
        ax.set_title(f"{title}\nmean={v.mean():.4f}  sd={v.std(ddof=1):.4f}  "
                     f"n={v.size}", fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xticks([])
    fig.suptitle(f"Detection performance over {n_seeds} simulated datasets "
                 f"(paper Fig. 5 style)")
    fig.tight_layout(); fig.savefig(OUT / "sim_fig5_fpr_fnr.png", dpi=130)
    plt.close(fig)
    print(f"\n  → {OUT/'sim_fig5_fpr_fnr.png'}")
    print(f"  → {OUT/'sim_detection_rates.csv'}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  ③ 合成 MEA 记录上跑完整管线
# ══════════════════════════════════════════════════════════════════════════

def part_synthetic_pipeline(report) -> list[str]:
    hr("③ 合成 MEA 记录 → 阶段 1→2→3 全管线（已知真值，可判定对错）")
    rec_path = OUT / "synthetic_recording.csv"
    t0 = time.time()
    _, info = simulate_recording(rec_path)
    truth_waves = np.asarray(info["wave_times_ms"])
    print(f"  生成 {rec_path.name}: {info['n_samples']} 行 x "
          f"{info['n_channels'] + 1} 列，{info['duration_ms'] / 1000:.1f} s @ "
          f"{info['sample_rate_hz']:.0f} Hz")
    print(f"  真值：源点 (x0, y0) = ({info['source_x0']}, {info['source_y0']})，"
          f"v = {info['speed']} 格/ms，{len(truth_waves)} 条波前")
    print(f"        源点激发时刻 (ms): {info['wave_times_ms']}")
    print(f"        最大传播时延 {info['max_delay_ms']:.0f} ms")
    print("  ★ time 列按【毫秒】写，与实验记录一致。这一步不能改成 1/200 ms ——")
    print("    管线的 ts/200 是把时间压到与 x,y 同量级的【尺度调理】，不是单位换算。")

    rec = load_recording(rec_path)
    spike_times, _ = detect_all_channels(rec)
    counts = np.array([s.size for s in spike_times])
    print(f"\n  阶段 1  检出 {counts.sum()} 个尖峰"
          f"（每通道 {counts.min()}–{counts.max()}，"
          f"波前 {len(truth_waves)} 条）")

    sp = spikes_to_table(spike_times, rec.channels)
    pts = sp[["x", "y", "t"]].to_numpy(dtype=float)
    res = hough_plane(pts, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)
    print("  阶段 2  " + res.summary().replace("\n", "\n          "))

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

    print("  阶段 3  逐平面拟合（按 t0 升序重排）：")
    print(f"    {'平面':>4s} {'圆 x0':>8s} {'圆 y0':>9s} {'圆 v':>7s} "
          f"{'圆 t0':>9s} {'真值 t0':>9s} {'Δ t0':>7s} {'R²_circ':>9s}")
    for i, (_, c, l) in enumerate(fits, 1):
        tw = truth_waves[i - 1] if i <= truth_waves.size else float("nan")
        print(f"    {i:>4d} {c.x0:8.2f} {c.y0:9.2f} {c.v:7.3f} {c.t0:9.2f} "
              f"{tw:9.2f} {c.t0 - tw:7.2f} {c.r2:9.6f}")

    # ── 逐项判定 ──────────────────────────────────────────────────────
    failures: list[str] = []
    x0s = np.array([c.x0 for _, c, _ in fits])
    y0s = np.array([c.y0 for _, c, _ in fits])
    vs = np.array([c.v for _, c, _ in fits])
    t0s = np.array([c.t0 for _, c, _ in fits])

    checks = [
        ("平面数 = 波前数",
         res.n_planes == len(truth_waves),
         f"{res.n_planes} vs {len(truth_waves)}"),
        ("每个平面恰好覆盖 64 个电极",
         all(int((res.plane_indices == k).sum()) == 64
             for k in range(1, res.n_planes + 1)),
         f"各平面点数 {[int((res.plane_indices == k).sum()) for k in range(1, res.n_planes + 1)]}"),
        ("全部尖峰被归类（无噪声）",
         int(res.prediction.sum()) == counts.sum(),
         f"{int(res.prediction.sum())} / {counts.sum()}"),
        (f"速度 v ≈ {info['speed']}",
         bool(np.abs(vs - info["speed"]).max() < 0.01),
         f"最大偏差 {np.abs(vs - info['speed']).max():.4f}"),
        (f"源点 x0 ≈ {info['source_x0']}",
         bool(np.abs(x0s - info["source_x0"]).max() < 0.5),
         f"最大偏差 {np.abs(x0s - info['source_x0']).max():.3f}"),
        (f"源点 y0 ≈ {info['source_y0']}",
         bool(np.abs(y0s - info["source_y0"]).max() < 0.5),
         f"最大偏差 {np.abs(y0s - info['source_y0']).max():.3f}"),
        ("激发时刻 t0 与真值差 < 2 ms",
         bool(t0s.size == truth_waves.size
              and np.abs(t0s - truth_waves).max() < 2.0),
         f"最大偏差 {np.abs(t0s - truth_waves).max():.2f} ms"
         if t0s.size == truth_waves.size else "平面数与波前数不符"),
    ]
    print("\n  逐项判定：")
    for name, ok, detail in checks:
        print(f"    {'✅' if ok else '❌'} {name:28s} {detail}")
        if not ok:
            failures.append(f"{name}（{detail}）")

    print(f"\n  总耗时 {time.time() - t0:.1f} s")
    if failures:
        print("  ❌ 未通过 —— 合成记录这条路径有问题：")
        for f in failures:
            print(f"      - {f}")
    else:
        print("  ✅ 全项通过：在【已知真值】的合成记录上，整条管线反解出了")
        print("     自己设定的源点位置、速度与各条波前的激发时刻。")
        print("     这比'能跑通'强得多 —— 它同时验证了阶段 1/2/3 的正确性。")
    print("  ★ 这些数字【不是】论文表 2/3/4 的值 —— 那三个表依赖实验记录本身。")

    report.write("③ 合成 MEA 记录（格式与真数据相同）上的完整管线\n\n")
    report.write("这一段是**已知真值的端到端验证**：记录由代码自己生成，")
    report.write("源点位置、速度与各条波前的激发时刻都是设定的，")
    report.write("所以「拟合结果是否落回真值」可以逐项判定。\n\n")
    report.write(f"- 文件：`{rec_path.name}`（time 列单位为毫秒）\n")
    report.write(f"- {info['n_samples']} 行 x {info['n_channels'] + 1} 列，"
                 f"{info['duration_ms'] / 1000:.1f} s @ "
                 f"{info['sample_rate_hz']:.0f} Hz\n")
    report.write(f"- 真值：源点 ({info['source_x0']}, {info['source_y0']})，"
                 f"v = {info['speed']} 格/ms，{len(truth_waves)} 条波前\n")
    report.write(f"- 真值激发时刻 (ms)：{info['wave_times_ms']}\n")
    report.write(f"- 检出 {counts.sum()} 个尖峰，每通道 "
                 f"{counts.min()}–{counts.max()}\n")
    report.write(f"- RHT 找到 {res.n_planes} 个平面，迭代 {res.n_iter_used} 次\n\n")
    report.write("| 判定项 | 结果 | 明细 |\n|---|---|---|\n")
    for name, ok, detail in checks:
        report.write(f"| {name} | {'✅' if ok else '❌'} | {detail} |\n")
    report.write("\n")
    tb = pd.DataFrame([{"plane": i, "x0": c.x0, "y0": c.y0, "v": c.v,
                        "t0": c.t0, "t0_true": truth_waves[i - 1],
                        "R2_circular": c.r2, "a": l.a, "b": l.b,
                        "R2_linear": l.r2}
                       for i, (_, c, l) in enumerate(fits, 1)])
    report.write(md_table(tb, ".4f"))
    report.write("\n")
    return failures


def main(n_seeds: int) -> int:
    T0 = time.time()
    part_detection(seed_shift=1)
    part_sweep(n_seeds)
    with open(OUT / "sim_synthetic_pipeline.txt", "w") as f:
        failures = part_synthetic_pipeline(f)
    print(f"  → {OUT/'sim_synthetic_pipeline.txt'}")

    hr()
    print(f" ✅ 全部完成，总耗时 {time.time() - T0:.1f} s")
    print(f"    图: {OUT}/sim_fig2_ground_truth.png")
    print(f"        {OUT}/sim_fig3_classification.png")
    print(f"        {OUT}/sim_fig4_planes.png")
    print(f"        {OUT}/sim_fig5_fpr_fnr.png")
    print()
    print("    ★ 本脚本覆盖论文 §3.1.1（图 2–5）。")
    print("      §3.1.2 的 96×96 圆波前仿真（图 6–8）是另一套参数、开销大得多，")
    print("      见 docs/reproducibility-notes.md。")
    print()
    if failures:
        print(f" ❌ 合成记录验证有 {len(failures)} 项未通过")
        return 1
    return 0


def _cli() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100,
                    help="FPR/FNR 分布用的随机种子个数（默认 100，对应论文图 5）")
    a = ap.parse_args()
    return main(a.seeds)


if __name__ == "__main__":
    raise SystemExit(_cli())
