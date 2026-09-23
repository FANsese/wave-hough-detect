"""
Simulation example (§3.1.2): 96×96 circular wavefront — how the estimation
accuracy varies with noise / missed detections / measurement error

This corresponds to paper Figs. 6–8. It is a DIFFERENT study from
``demo_simulation.py`` (§3.1.1, 8×8 linear wavefront, Figs. 2–5); the two must
not be mixed up.

The ground truth is set by the code itself: source (48, 48), speed 1, firing
time 2, i.e.

    t = √((x−48)² + (y−48)²) / 1 + 2 + ε

so every estimate can be compared with the ground truth directly.

Run:
    python examples/demo_simulation_circular.py                # default replicate counts (~4 min)
    python examples/demo_simulation_circular.py --replicates 2 # quick run (~30 s)

Output:
    examples/out/circ_fig6_wavefront.png       simulated data + fitted circular wavefront
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

# Allow running without installing: add the repository src/ to the search path.
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

#: the four quantities that are tracked: display name, ground truth, result column
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


# ==========================================================================
#  ① One simulation + fit (paper Fig. 6 style)
# ==========================================================================

def part_one() -> pd.DataFrame:
    hr("① One simulation + circular-wavefront fit (paper Fig. 6 style)")
    df = simulate_circular_wavefronts(0, **PAPER_CIRCULAR_PARAMS)
    est = evaluate_circular(0)
    sig = df[df.z == 1]

    print(f"  Simulation parameters λ_n={PAPER_CIRCULAR_PARAMS['noise_freq']}, "
          f"p={PAPER_CIRCULAR_PARAMS['p']}, σ={PAPER_CIRCULAR_PARAMS['sigma']}")
    print(f"  {len(df)} points (signal {est.n_signal} = all 96×96 electrodes, "
          f"noise {est.n_noise})")
    print(f"  truth  x₀={CIRCULAR_TRUTH['x0']:.4f}  y₀={CIRCULAR_TRUTH['y0']:.4f}  "
          f"v={CIRCULAR_TRUTH['v']:.4f}  t₀={CIRCULAR_TRUTH['t0']:.4f}")
    print(f"  fit    x₀={est.x0:.4f}  y₀={est.y0:.4f}  "
          f"v={est.v:.4f}  t₀={est.t0:.4f}   (converged in {est.n_iters} "
          f"iterations)")
    print(f"  relative error " + "  ".join(
        f"{k}={v:.2e}" for k, v in est.rel_error.items()))
    print(f"  R²={est.r2:.6f}   ← note: this is computed over **all signal + "
          f"noise** points; noise makes up {100*est.n_noise/est.n_points:.1f}%")
    print(f"         with the true parameters R²(signal only) = 1.0000000000, so "
          f"0.95 does not mean the model is inaccurate")

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(sig.x, sig.y, sig.ts, s=3, c="tab:green",
               label=f"signal  n={est.n_signal}", depthshade=False)
    noi = df[df.z == 0]
    if len(noi):
        ax.scatter(noi.x, noi.y, noi.ts, s=8, c="0.4",
                   label=f"noise  n={est.n_noise}", depthshade=False)
    # the fitted circular wavefront: t = √((x−x₀)²+(y−y₀)²)/v + t₀
    g = np.arange(1, 97)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    surf = np.hypot(gx - est.x0, gy - est.y0) / est.v + est.t0
    ax.plot_surface(gx, gy, np.where(surf <= 96, surf, np.nan),
                    alpha=0.25, color="tab:red", linewidth=0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("t")
    ax.set_title(f"96×96 circular wavefront\nfitted (x₀,y₀)=({est.x0:.2f},{est.y0:.2f}), "
                 f"v={est.v:.3f}")
    ax.legend(fontsize=8, loc="upper left"); ax.view_init(elev=20, azim=-60)

    # residuals after reordering by distance: with a correct model only the
    # noise should be left
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


# ==========================================================================
#  ② Accuracy vs noise rate λ_n (paper Fig. 7)
# ==========================================================================

def plot_accuracy(df: pd.DataFrame, xcol: str, xlabel: str, logx: bool,
                  title: str, out_name: str) -> None:
    """
    Plot "the four estimates vs the sweep variable".

    ★ This plots the **absolute estimates**, not the relative error (the
      relative errors are reported in the console table instead), and draws the
      ground truth as a horizontal dashed line.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (disp, col, truth) in zip(axes, QUANTITIES):
        # Group by the swept parameter, not by the realised x quantity. For the
        # noise sweep each replicate has its own realised snr, so grouping by
        # snr would give one sample per group and every error bar would be NaN
        # (measured: 46 groups of 1 for 2 replicates x 23 grid points).
        # x is then the mean of the x quantity within each parameter value.
        by_param = df.groupby("scan_value")
        xs = by_param[xcol].mean().to_numpy()
        mean = by_param[col].mean().to_numpy()
        sd = by_param[col].std(ddof=1).to_numpy()
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
    hr(f"② Accuracy sweep: plot_mode={plot_mode} (paper Fig. "
       f"{ {1: '7', 2: '8', 3: 'sigma curve'}[plot_mode] })")
    t0 = time.time()
    df = accuracy_sweep(plot_mode, n_replicates=n_rep)
    dt = time.time() - t0
    print(f"  {len(df)} 'simulate + fit' runs in {dt:.1f} s "
          f"({1000*dt/len(df):.0f} ms/run)")

    var = df["variable"].iloc[0]
    summ = df.groupby("scan_value").agg(
        snr=("snr", "mean"), n_noise=("n_noise", "mean"),
        x0=("x0", "mean"), y0=("y0", "mean"), v=("v", "mean"), t0=("t0", "mean"),
        rel_v=("rel_v", "mean"), rel_t0=("rel_t0", "mean"),
        rel_x0=("rel_x0", "mean"), rel_y0=("rel_y0", "mean"))
    print()
    print(f"  {'λ/param':>10s} {'snr':>9s} {'n_noise':>8s} {'x₀ error':>10s} "
          f"{'y₀ error':>10s} {'v error':>10s} {'t₀ error':>10s}")
    print("  " + "-" * 73)
    for val, r in summ.iterrows():
        print(f"  {val:10.4g} {r['snr']:9.2f} {r['n_noise']:8.0f} "
              f"{r['rel_x0']:10.2e} {r['rel_y0']:10.2e} "
              f"{r['rel_v']:10.2e} {r['rel_t0']:10.2e}")

    print()
    print("  ★ Readability: the source position (x₀, y₀) is very stable at every")
    print("     noise level (relative error ~1e-3), while v and t₀ drift together")
    print("     — with many noise points the fit slows the wave down and delays")
    print("     its onset in order to accommodate those outliers. That is exactly")
    print("     the degradation mode Fig. 7 is meant to show.")

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
    print(f" ✅ All done, total elapsed {time.time()-T0:.1f} s")
    print(f"    Figures: {_OUT}/circ_fig6_wavefront.png")
    print(f"             {_OUT}/circ_fig7_accuracy_vs_snr.png")
    print(f"             {_OUT}/circ_fig8_accuracy_vs_p.png")
    print(f"             {_OUT}/circ_sigma_accuracy.png")
    print()
    print("    ★ The replicates are INDEPENDENT: every replicate uses a different")
    print("      seed_shift. Passing independent_replicates=False instead gives")
    print("      all replicates of a grid point the same seed_shift, so they are")
    print("      no longer independent and the curves come out artificially flat.")
    print("      To see that effect:")
    print("        accuracy_sweep(mode, independent_replicates=False)")
    print()
    return 0


def _cli() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicates", type=int, default=None,
                    help="replicates per grid point; the defaults are the "
                         "values used for the reported results "
                         "(mode 1→23, mode 2→9, mode 3→10)")
    a = ap.parse_args()
    return main(a.replicates)


if __name__ == "__main__":
    raise SystemExit(_cli())
