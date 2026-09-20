"""
End-to-end example: from a raw MEA recording to a wave-source estimate

This is the "human-readable" script — it runs the whole pipeline once and
produces
  · raw / filtered trace plots in the style of paper Fig. 9
  · a (x,y,t) view with the spikes separated into planes, in the style of
    paper Fig. 10
  · propagation-direction arrow plots in the style of paper Figs. 11/12
    (an extra view provided by this script)
  · paper Tables 2 / 3 / 4

Plane numbering is re-sorted by ascending t₀, matching the paper, so the output
can be compared with it directly.

Run:
    python examples/demo_pipeline.py

Output:
    examples/out/*.png, *.csv, tables.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The matplotlib / fontconfig cache directories must be set before the imports
# below, otherwise a non-writable $HOME produces a flood of warnings
_HERE = Path(__file__).resolve().parent
_OUT_EARLY = _HERE / "out"
_OUT_EARLY.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_OUT_EARLY / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_OUT_EARLY / ".cache"))

# Allow running without installing: add the repository src/ to the search path.
# If the package was already installed with `pip install -e .`, this is a no-op.
_SRC = _HERE.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
from mpl_toolkits.mplot3d import Axes3D               # noqa: E402,F401

from wave_hough_detect import (                       # noqa: E402
    build_result_array, detect_all_channels, find_recording, fit_circular,
    fit_linear, hough_plane, load_recording, plane_points, spikes_to_table,
)

OUT = _OUT_EARLY
GRID_CENTROID = np.array([4.5, 4.5])


def hr(t=""):
    print()
    print("=" * 78)
    if t:
        print(f" {t}")
        print("=" * 78)


def md_table(df: pd.DataFrame, fmt: str = ".4f") -> str:
    """Minimal markdown table builder, to avoid a tabulate dependency."""
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


def main(data_path: Path) -> int:
    T0 = time.time()

    # ==================================================================
    hr("Stage 1/3  spike detection")
    rec = load_recording(data_path)
    print(f"  Data file {data_path}")
    print(f"  {rec.n_samples} samples x {rec.n_channels} channels, "
          f"{rec.sampling_rate_hz:.0f} Hz, {rec.time[-1]:.0f} ms")

    spike_times, filtered = detect_all_channels(rec)
    counts = np.array([s.size for s in spike_times])
    print(f"  Detected {counts.sum()} spikes "
          f"({counts.min()}–{counts.max()} per channel)")
    print(f"  Elapsed {time.time()-T0:.1f} s")

    sp_hough = spikes_to_table(spike_times, rec.channels)     # t/200

    # -- Fig. 1: traces (paper Fig. 9 style) ---------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for ax, ch in zip(axes, ["Ch01", "Ch34"]):
        i = rec.channels.index(ch)
        ax.plot(rec.time, rec.voltage[:, i], color="0.65", lw=0.7, label="raw")
        ax.plot(rec.time, filtered[:, i], color="crimson", lw=1.0,
                label="Butterworth high-pass")
        thr = np.quantile(filtered[:, i], 0.0005)
        ax.axhline(thr, color="darkgreen", ls="--", lw=1,
                   label=f"threshold = 0.05 percentile ({thr:.3f} mV)")
        for t in spike_times[i]:
            ax.axvline(t, color="royalblue", lw=1.1, alpha=0.85)
        ax.set_title(f"{ch}   detected spikes: {spike_times[i].size}")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Voltage (mV)")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Stage 1 — spike detection (paper Fig. 9 style)", y=0.995)
    fig.tight_layout(); fig.savefig(OUT / "fig1_traces.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'fig1_traces.png'}")

    # ==================================================================
    hr("Stage 2/3  randomized Hough transform")
    pts = np.column_stack([sp_hough["x"], sp_hough["y"],
                           sp_hough["t"]]).astype(float)
    t1 = time.time()
    res = hough_plane(pts, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)
    print("  " + res.summary().replace("\n", "\n  "))
    print(f"  Elapsed {time.time()-t1:.1f} s")

    order = np.argsort(sp_hough["t"].to_numpy(), kind="stable")
    x_obs = sp_hough["x"].to_numpy()[order]
    y_obs = sp_hough["y"].to_numpy()[order]
    t_obs = sp_hough["t"].to_numpy()[order]
    pid_raw = res.plane_indices[order]

    # ==================================================================
    hr("Stage 3/3  wavefront model fitting")
    arr = build_result_array(x_obs, y_obs, t_obs, pid_raw, res.n_planes)
    t2 = time.time()

    fits = []
    for k in range(1, res.n_planes + 1):
        p = plane_points(arr, k)
        fits.append(dict(old=k, circ=fit_circular(p), lin=fit_linear(p)))

    # -- Re-sort by ascending t₀ so that the numbering matches paper Table 2 --
    fits.sort(key=lambda d: d["circ"].t0)
    remap = {d["old"]: i + 1 for i, d in enumerate(fits)}
    pid = np.array([remap.get(int(v), 0) if v > 0 else 0 for v in pid_raw])
    print("  Plane indices re-sorted by ascending t₀ (same convention as the paper)")
    for d in fits:
        c, l = d["circ"], d["lin"]
        print(f"    plane {remap[d['old']]}: circ x0={c.x0:7.3f} y0={c.y0:8.3f} "
              f"v={c.v:.4f} t0={c.t0:8.2f} R²={c.r2:.6f} ({c.n_iters} iters)  |  "
              f"lin ã={l.a:+.4f} b̃={l.b:.4f} v={l.v:.4f} R²={l.r2:.6f}")

    C = pd.DataFrame([dict(plane=i + 1, x0=d["circ"].x0, y0=d["circ"].y0,
                           v=d["circ"].v, t0=d["circ"].t0, r2=d["circ"].r2,
                           loss=d["circ"].loss, n_points=d["circ"].n_points,
                           n_iters=d["circ"].n_iters)
                      for i, d in enumerate(fits)])
    L = pd.DataFrame([dict(plane=i + 1, a=d["lin"].a, b=d["lin"].b,
                           v=d["lin"].v, r2=d["lin"].r2, loss=d["lin"].loss,
                           n_points=d["lin"].n_points)
                      for i, d in enumerate(fits)])
    print(f"  Elapsed {time.time()-t2:.1f} s")

    cmap = plt.get_cmap("tab10")

    # -- Fig. 2: spikes coloured by plane (paper Fig. 10 style) -------------
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")
    for k in range(1, len(fits) + 1):
        m = pid == k
        ax.scatter(x_obs[m], y_obs[m], t_obs[m], s=14, color=cmap((k - 1) % 10),
                   label=f"plane {k}", depthshade=False)
    if (pid == 0).any():
        ax.scatter(x_obs[pid == 0], y_obs[pid == 0], t_obs[pid == 0],
                   s=14, c="0.6", label="noise")
    ax.set_xlabel("x (row)"); ax.set_ylabel("y (col)"); ax.set_zlabel("t (/200)")
    ax.set_title("RHT: spikes separated into wavefront planes")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=18, azim=-58)

    ax2 = fig.add_subplot(122)
    for k in range(1, len(fits) + 1):
        m = pid == k
        ax2.scatter(t_obs[m], y_obs[m] + 0.12 * (x_obs[m] - 4.5), s=16,
                    color=cmap((k - 1) % 10), label=f"plane {k}")
    ax2.set_xlabel("t (/200)"); ax2.set_ylabel("y (jittered by x)")
    ax2.set_title("Same data, side view — each plane is a line")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=8)
    fig.suptitle("Stage 2 — randomized Hough transform (paper Fig. 10 style)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_planes.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'fig2_planes.png'}")

    # -- Fig. 3: propagation directions (paper Figs. 11/12 style) -- added
    #    by this script on top of the paper's figure set ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (tab, lab, mode) in zip(
            axes, [(C, "circular wavefront  (paper Fig. 11 style)", "circ"),
                   (L, "linear wavefront  (paper Fig. 12 style)", "lin")]):
        ax.scatter(x_obs, y_obs, s=42, c="0.85", edgecolors="0.6", zorder=1,
                   label="electrodes")
        for _, r in tab.iterrows():
            k = int(r["plane"]); col = cmap((k - 1) % 10)
            if mode == "circ":
                src = np.array([r["x0"], r["y0"]])
            else:
                dv = np.array([r["a"], r["b"]]); dv = dv / np.linalg.norm(dv) * 1.8
                src = GRID_CENTROID - dv
            ax.annotate("", xy=GRID_CENTROID, xytext=src,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                        shrinkA=0, shrinkB=0, alpha=0.92),
                        zorder=3)
            ax.annotate(f"{k}", xy=(src + GRID_CENTROID) / 2, color=col,
                        fontsize=11, fontweight="bold", zorder=4)
        ax.set_xlim(-56, 12); ax.set_ylim(-56, 12); ax.set_aspect("equal")
        ax.axhline(-50, color="0.7", lw=0.9, ls=":")
        ax.text(-55, -48.0, "search-window boundary  y0 = -50",
                fontsize=7, color="0.45")
        ax.set_xlabel("x (row)"); ax.set_ylabel("y (col)")
        ax.set_title(lab); ax.grid(alpha=0.25)
    fig.suptitle("Propagation directions recovered by the two models")
    fig.tight_layout(); fig.savefig(OUT / "fig3_directions.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'fig3_directions.png'}")

    # -- Fig. 4: fit quality ------------------------------------------------
    fig, axes = plt.subplots(1, len(fits), figsize=(3.4 * len(fits), 3.6))
    for k, ax in zip(range(1, len(fits) + 1), np.atleast_1d(axes)):
        p = plane_points(arr, fits[k - 1]["old"])
        r = C[C["plane"] == k].iloc[0]; l = L[L["plane"] == k].iloc[0]
        d = np.sqrt((p[:, 0] - r["x0"]) ** 2 + (p[:, 1] - r["y0"]) ** 2)
        tc = d / r["v"] + r["t0"]
        tl = l["a"] * p[:, 0] + l["b"] * p[:, 1] + (
            p[:, 2].mean() - l["a"] * p[:, 0].mean() - l["b"] * p[:, 1].mean())
        ax.scatter(tc, p[:, 2], s=20, color="tab:blue",
                   label=f"circ  R2={r['r2']:.4f}")
        ax.scatter(tl, p[:, 2], s=22, marker="^", facecolors="none",
                   edgecolors="tab:red", label=f"lin   R2={l['r2']:.4f}")
        lim = [p[:, 2].min() - 1, p[:, 2].max() + 1]
        ax.plot(lim, lim, color="0.5", lw=0.8, ls="--")
        ax.set_title(f"plane {k}"); ax.set_xlabel("fitted t (ms)")
        if k == 1:
            ax.set_ylabel("observed t (ms)")
        ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle("Model fit quality — observed vs fitted activation time")
    fig.tight_layout(); fig.savefig(OUT / "fig4_fit_quality.png", dpi=130)
    plt.close(fig)
    print(f"  → {OUT/'fig4_fit_quality.png'}")

    # ==================================================================
    hr("Paper Tables 2 / 3 / 4")
    print("\nTable 2  circular wavefront model")
    print(f"  {'plane':>5s} {'x0':>9s} {'y0':>9s} {'v':>8s} {'t0':>10s}")
    for _, r in C.iterrows():
        print(f"  {int(r['plane']):>5d} {r['x0']:>9.2f} {r['y0']:>9.2f} "
              f"{r['v']:>8.2f} {r['t0']:>10.2f}")
    on_bound = bool((np.abs(C["y0"] + 50) < 0.01).all())
    print("\n  Do all y0 estimates sit on the search boundary -50? "
          + ("✅ yes — the source is outside the search window, so the linear "
             "model should be used instead" if on_bound else "❌ no"))

    print("\nTable 3  linear wavefront model")
    print(f"  {'plane':>5s} {'a(tilde)':>9s} {'b(tilde)':>9s} {'v':>8s}")
    for _, r in L.iterrows():
        print(f"  {int(r['plane']):>5d} {r['a']:>9.2f} {r['b']:>9.2f} {r['v']:>8.2f}")

    print("\nTable 4  coefficient of determination R^2")
    print(f"  {'plane':>5s} {'R2_circular':>14s} {'R2_linear':>13s}  "
          f"{'circular better':<15s}")
    n_better = 0
    for k in range(1, len(fits) + 1):
        rc = float(C[C['plane'] == k]['r2'].iloc[0])
        rl = float(L[L['plane'] == k]['r2'].iloc[0])
        n_better += rc > rl
        print(f"  {k:>5d} {rc:>14.7f} {rl:>13.7f}  "
              f"{'✅' if rc > rl else '':<15s}")
    print(f"\n  The circular model wins on {n_better}/{len(fits)} planes "
          f"(the paper's conclusion: it is slightly better on every plane)")

    # -- Write the results to disk -----------------------------------------
    C.to_csv(OUT / "table2_circular.csv", index=False)
    L.to_csv(OUT / "table3_linear.csv", index=False)
    T4 = pd.DataFrame({"plane": C["plane"], "R2_circular": C["r2"],
                       "R2_linear": L["r2"]})
    T4.to_csv(OUT / "table4_r2.csv", index=False)

    with open(OUT / "tables.md", "w") as f:
        f.write("# Paper Tables 2 / 3 / 4 (computed by this example)\n\n")
        f.write(f"- Data: `{data_path.name}`\n")
        f.write(f"- {rec.n_channels} channels, {rec.time[-1]:.0f} ms, "
                f"{rec.sampling_rate_hz:.0f} Hz\n")
        f.write(f"- Detected {counts.sum()} spikes "
                f"({counts.min()}–{counts.max()} per channel)\n")
        f.write(f"- RHT found {res.n_planes} planes in "
                f"{res.n_iter_used} iterations\n")
        f.write("- Plane numbering follows ascending t0, as in the paper\n\n")
        f.write("## Table 2  Circular wavefront model\n\n")
        f.write(md_table(C[["plane", "x0", "y0", "v", "t0"]], ".2f"))
        f.write("\n\n## Table 3  Linear wavefront model\n\n")
        f.write(md_table(L[["plane", "a", "b", "v"]], ".2f"))
        f.write("\n\n## Table 4  Coefficient of determination\n\n")
        f.write(md_table(T4, ".7f"))
        f.write("\n")

    hr()
    print(f" ✅ All done, total elapsed {time.time()-T0:.1f} s")
    print(f"    Figures: {OUT}/fig1_traces.png  fig2_planes.png  "
          f"fig3_directions.png  fig4_fit_quality.png")
    print(f"    Tables:  {OUT}/table2_circular.csv  table3_linear.csv  "
          f"table4_r2.csv  tables.md")
    print()
    return 0


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None,
                    help="path to the MEA recording CSV. If omitted, the "
                         "WHD_DATA environment variable, the current directory "
                         "and the repository data/ directory are searched in "
                         "that order")
    args = ap.parse_args()
    try:
        return main(find_recording(args.data))
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
