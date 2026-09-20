"""
Simulation example: the detection performance of paper §3.1 (Figs. 2–5 style)
plus the full pipeline on a synthetic recording

This script answers one question: **without experimental data, what can this
code still prove?**

  ① Run the RHT on a simulation with known ground truth → compute FPR / FNR
     (paper Fig. 5)
  ② Draw the planes that were found and compare them with the ground truth
     (paper Figs. 2/3/4 style)
  ③ Generate a synthetic MEA recording in the same format as the real data →
     run the whole stage 1→2→3 pipeline, showing that the pipeline also runs
     end to end when no experimental recording is available

Run:
    python examples/demo_simulation.py                 # 100 random seeds
    python examples/demo_simulation.py --seeds 20      # faster

Output:
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

# Allow running without installing: add the repository src/ to the search path.
# If the package was already installed with `pip install -e .`, this is a no-op.
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
    Draw the planes found by the RHT on a 3D axes.

    Each plane is given by ``n1·x + n2·y + n3·t = ρ``; solving for t at the four
    corners of the grid gives the four vertices.
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


# ==========================================================================
#  ① One simulation + detection: ground truth, four-colour classification,
#     the planes that were found
# ==========================================================================

def part_detection(seed_shift: int) -> None:
    hr("① One simulation + RHT (paper §3.1.1 / Figs. 2–4 style)")
    t0 = time.time()
    res = evaluate_detection(seed_shift)
    cls = res.classified()
    print(f"  Simulation parameters: {PAPER_2D_PARAMS}")
    print(f"  Points {res.rates.n_points} (signal {res.rates.n_signal}, "
          f"noise {res.rates.n_noise})")
    print(f"  RHT found {res.rates.n_planes} planes in {time.time()-t0:.2f} s")
    print(f"  TP={res.rates.n_true_pos}  FP={res.rates.n_false_pos}  "
          f"FN={res.rates.n_false_neg}  TN={res.rates.n_true_neg}")
    print(f"  FPR={res.rates.false_positive_rate:.4f}  "
          f"FNR={res.rates.false_negative_rate:.4f}"
          f"   ← paper definition (denominator = all points)")
    print(f"  FPR={res.rates.false_positive_rate_among_noise:.4f}  "
          f"FNR={res.rates.false_negative_rate_among_signal:.4f}"
          f"   ← conventional definition (denominator = points of each class)")

    sig = cls[cls["z"] == 1]
    noi = cls[cls["z"] == 0]

    # -- Fig. 2: ground truth (paper Fig. 2 style) ------------------------
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

    # -- Fig. 3: four-colour classification (paper Fig. 3 style) ----------
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(121, projection="3d")
    for key in CLASS_ORDER:                    # fixed drawing order: TP, FP, FN, TN
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

    # -- Fig. 4: the planes that were found (paper Fig. 4 style) ----------
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


# ==========================================================================
#  ② FPR / FNR distribution (paper Fig. 5)
# ==========================================================================

def part_sweep(n_seeds: int) -> pd.DataFrame:
    hr(f"② FPR / FNR across random seeds (paper Fig. 5, {n_seeds} seeds)")
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
    print(f"  {n_seeds} runs completed, total {dt:.0f} s "
          f"(mean {1000 * dt / n_seeds:.0f} ms/run)")
    print()
    print(f"  {'quantity':30s} {'mean':>9s} {'sd':>9s} {'median':>9s} "
          f"{'max':>9s}")
    print("  " + "-" * 70)
    for col, name in [("FPR", "FPR  (paper definition)"),
                      ("FNR", "FNR  (paper definition)"),
                      ("FPR_among_noise", "FPR  (conventional definition)"),
                      ("FNR_among_signal", "FNR  (conventional definition)"),
                      ("n_planes", "planes found by RHT")]:
        v = df[col].to_numpy(dtype=float)
        print(f"  {name:30s} {v.mean():9.4f} {v.std(ddof=1):9.4f} "
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


# ==========================================================================
#  ③ Run the whole pipeline on a synthetic MEA recording
# ==========================================================================

def part_synthetic_pipeline(report) -> list[str]:
    hr("③ Synthetic MEA recording → full stage 1→2→3 pipeline "
       "(known ground truth, so the result can be judged)")
    rec_path = OUT / "synthetic_recording.csv"
    t0 = time.time()
    _, info = simulate_recording(rec_path)
    truth_waves = np.asarray(info["wave_times_ms"])
    print(f"  Wrote {rec_path.name}: {info['n_samples']} rows x "
          f"{info['n_channels'] + 1} columns, {info['duration_ms'] / 1000:.1f} s @ "
          f"{info['sample_rate_hz']:.0f} Hz")
    print(f"  Ground truth: source (x0, y0) = ({info['source_x0']}, "
          f"{info['source_y0']}), v = {info['speed']} cells/ms, "
          f"{len(truth_waves)} wavefronts")
    print(f"        source firing times (ms): {info['wave_times_ms']}")
    print(f"        maximum propagation delay {info['max_delay_ms']:.0f} ms")
    print("  ★ The time column is written in MILLISECONDS, matching the "
          "experimental records.")
    print("    This step must not be changed to 1/200 ms — the pipeline's ts/200")
    print("    is a SCALING step that brings time onto the same order as x,y,")
    print("    not a unit conversion.")

    rec = load_recording(rec_path)
    spike_times, _ = detect_all_channels(rec)
    counts = np.array([s.size for s in spike_times])
    print(f"\n  Stage 1  detected {counts.sum()} spikes "
          f"({counts.min()}–{counts.max()} per channel, "
          f"{len(truth_waves)} wavefronts)")

    sp = spikes_to_table(spike_times, rec.channels)
    pts = sp[["x", "y", "t"]].to_numpy(dtype=float)
    res = hough_plane(pts, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)
    print("  Stage 2  " + res.summary().replace("\n", "\n           "))

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

    print("  Stage 3  per-plane fits (re-sorted by ascending t0):")
    print(f"    {'plane':>5s} {'circ x0':>8s} {'circ y0':>9s} {'circ v':>7s} "
          f"{'circ t0':>9s} {'true t0':>9s} {'Δ t0':>7s} {'R²_circ':>9s}")
    for i, (_, c, l) in enumerate(fits, 1):
        tw = truth_waves[i - 1] if i <= truth_waves.size else float("nan")
        print(f"    {i:>5d} {c.x0:8.2f} {c.y0:9.2f} {c.v:7.3f} {c.t0:9.2f} "
              f"{tw:9.2f} {c.t0 - tw:7.2f} {c.r2:9.6f}")

    # -- Per-item checks ---------------------------------------------------
    failures: list[str] = []
    x0s = np.array([c.x0 for _, c, _ in fits])
    y0s = np.array([c.y0 for _, c, _ in fits])
    vs = np.array([c.v for _, c, _ in fits])
    t0s = np.array([c.t0 for _, c, _ in fits])

    checks = [
        ("number of planes = number of wavefronts",
         res.n_planes == len(truth_waves),
         f"{res.n_planes} vs {len(truth_waves)}"),
        ("every plane covers exactly 64 electrodes",
         all(int((res.plane_indices == k).sum()) == 64
             for k in range(1, res.n_planes + 1)),
         f"points per plane {[int((res.plane_indices == k).sum()) for k in range(1, res.n_planes + 1)]}"),
        ("all spikes classified (no noise left over)",
         int(res.prediction.sum()) == counts.sum(),
         f"{int(res.prediction.sum())} / {counts.sum()}"),
        (f"speed v ≈ {info['speed']}",
         bool(np.abs(vs - info["speed"]).max() < 0.01),
         f"max deviation {np.abs(vs - info['speed']).max():.4f}"),
        (f"source x0 ≈ {info['source_x0']}",
         bool(np.abs(x0s - info["source_x0"]).max() < 0.5),
         f"max deviation {np.abs(x0s - info['source_x0']).max():.3f}"),
        (f"source y0 ≈ {info['source_y0']}",
         bool(np.abs(y0s - info["source_y0"]).max() < 0.5),
         f"max deviation {np.abs(y0s - info['source_y0']).max():.3f}"),
        ("firing time t0 within 2 ms of the ground truth",
         bool(t0s.size == truth_waves.size
              and np.abs(t0s - truth_waves).max() < 2.0),
         f"max deviation {np.abs(t0s - truth_waves).max():.2f} ms"
         if t0s.size == truth_waves.size else "plane count != wavefront count"),
    ]
    print("\n  Per-item checks:")
    for name, ok, detail in checks:
        print(f"    {'✅' if ok else '❌'} {name:43s} {detail}")
        if not ok:
            failures.append(f"{name} ({detail})")

    print(f"\n  Total elapsed {time.time() - t0:.1f} s")
    if failures:
        print("  ❌ FAILED — something is wrong on the synthetic-recording path:")
        for f in failures:
            print(f"      - {f}")
    else:
        print("  ✅ All checks passed: on a synthetic recording with KNOWN")
        print("     ground truth, the whole pipeline recovered the source")
        print("     position, the speed and the firing time of every wavefront")
        print("     that it had set itself.")
        print("     This is far stronger than 'it runs' — it validates the")
        print("     correctness of stages 1/2/3 at the same time.")
    print("  ★ These numbers are NOT the values of paper Tables 2/3/4 — those")
    print("    three tables depend on the experimental recording itself.")

    report.write("③ Full pipeline on a synthetic MEA recording (same format as "
                 "the real data)\n\n")
    report.write("This section is an **end-to-end validation with known ground "
                 "truth**: the recording is generated by the code itself, ")
    report.write("the source position, the speed and the firing time of every "
                 "wavefront are set there, ")
    report.write("so whether the fits fall back onto the ground truth can be "
                 "checked item by item.\n\n")
    report.write(f"- file: `{rec_path.name}` (the time column is in "
                 f"milliseconds)\n")
    report.write(f"- {info['n_samples']} rows x {info['n_channels'] + 1} columns, "
                 f"{info['duration_ms'] / 1000:.1f} s @ "
                 f"{info['sample_rate_hz']:.0f} Hz\n")
    report.write(f"- ground truth: source ({info['source_x0']}, "
                 f"{info['source_y0']}), v = {info['speed']} cells/ms, "
                 f"{len(truth_waves)} wavefronts\n")
    report.write(f"- ground-truth firing times (ms): {info['wave_times_ms']}\n")
    report.write(f"- detected {counts.sum()} spikes, "
                 f"{counts.min()}–{counts.max()} per channel\n")
    report.write(f"- RHT found {res.n_planes} planes in {res.n_iter_used} "
                 f"iterations\n\n")
    report.write("| check | result | detail |\n|---|---|---|\n")
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
    print(f" ✅ All done, total elapsed {time.time() - T0:.1f} s")
    print(f"    Figures: {OUT}/sim_fig2_ground_truth.png")
    print(f"             {OUT}/sim_fig3_classification.png")
    print(f"             {OUT}/sim_fig4_planes.png")
    print(f"             {OUT}/sim_fig5_fpr_fnr.png")
    print()
    print("    ★ This script covers paper §3.1.1 (Figs. 2–5).")
    print("      The 96×96 circular-wavefront simulation of §3.1.2 (Figs. 6–8) is")
    print("      a different study with a much larger cost; see")
    print("      docs/reproducibility-notes.md.")
    print()
    if failures:
        print(f" ❌ {len(failures)} synthetic-recording checks failed")
        return 1
    return 0


def _cli() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100,
                    help="number of random seeds for the FPR/FNR distribution "
                         "(default 100, matching paper Fig. 5)")
    a = ap.parse_args()
    return main(a.seeds)


if __name__ == "__main__":
    raise SystemExit(_cli())
