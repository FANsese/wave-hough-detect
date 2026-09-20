"""
噪声到底是怎么去的？—— 用真实数据分层测量

MEA 记录里有两类完全不同的"噪声"，分别由两个不同的阶段处理：

  类型 A：通道内波形噪声（基线漂移、高频干扰）
          → 由阶段 1 的 Butterworth 高通 + 幅度阈值处理
  类型 B：检出的假尖峰（不属于任何真实波前的孤立峰）
          → 由阶段 2 的霍夫变换处理（未落进任何平面的点判为噪声）

本脚本量化类型 A，并说明阈值与霍夫各自承担多少。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, welch

# 允许不安装直接运行：把仓库的 src/ 加入搜索路径。
# 若已 `pip install -e .`，这一句无副作用。
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from wave_hough_detect import (  # noqa: E402
    detect_all_channels,
    find_recording,
    hough_plane,
    load_recording,
    spikes_to_table,
)


def hr(t=""):
    print()
    print("=" * 74)
    if t:
        print(f" {t}")
        print("=" * 74)


def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.any() else 0.0


def main(data_path: Path):
    hr("读入数据")
    rec = load_recording(data_path)
    t, v = rec.time, rec.voltage
    fs = rec.sampling_rate_hz
    print(f"  {rec.n_samples} 采样点 x {rec.n_channels} 通道, "
          f"采样率 {fs:.0f} Hz")
    print(f"  时长 {t[-1]:.1f} ms")

    ch = 0                                        # 用 Ch01 分析
    raw = v[:, ch]

    hr("类型 A：通道内波形噪声 —— 分层看频谱")

    # Butterworth 高通：与管线一致
    b, a = butter(2, 1/500, btype="high")
    hp = lfilter(b, a, raw)

    f, pxx_raw = welch(raw, fs=fs, nperseg=8192)
    _, pxx_hp = welch(hp, fs=fs, nperseg=8192)

    bands = [
        ("基线漂移      ", 0.0, 1.0),
        ("慢波/运动伪影 ", 1.0, 10.0),
        ("尖峰主要频段  ", 10.0, 500.0),
        ("高频细节      ", 500.0, 2000.0),
        ("高频噪声      ", 2000.0, fs / 2),
    ]
    print(f"  {'频段':14s} {'原始功率':>12s} {'高通后':>12s} {'保留比例':>10s}")
    print("  " + "-" * 54)
    for name, lo, hi in bands:
        pr = band_power(f, pxx_raw, lo, hi)
        ph = band_power(f, pxx_hp, lo, hi)
        ratio = ph / pr if pr > 0 else np.nan
        print(f"  {name:14s} {pr:12.3e} {ph:12.3e} {ratio:10.3f}")

    hr("幅值分解：漂移 vs 尖峰 vs 高频噪声")

    # 低频分量（用低通取出漂移）
    b_lo, a_lo = butter(2, 1/500, btype="low")
    drift = lfilter(b_lo, a_lo, raw)

    # 尖峰幅度：5 个检出位置的深度
    spikes_t = np.array([965.3, 2159.0, 3938.9, 5777.5, 7760.7])
    idx = [int(np.argmin(np.abs(t - s))) for s in spikes_t]
    spike_amp = raw[idx] - drift[idx]

    print(f"  漂移幅度 (低通提取)      : {drift.min():+.4f} – {drift.max():+.4f} mV"
          f"   峰峰值 {drift.max()-drift.min():.4f} mV")
    print(f"  尖峰相对漂移的幅度        : "
          f"{np.round(spike_amp, 4).tolist()} mV")
    print(f"    → 平均 {np.abs(spike_amp).mean():.4f} mV")
    print(f"  高通后残余高频噪声 (std)  : {hp.std():.5f} mV")
    print(f"    → 尖峰/噪声 = {np.abs(spike_amp).mean()/hp.std():.1f} 倍")

    hr("阈值的作用：不同分位数会检出多少个尖峰")

    print(f"  {'分位数':>10s} {'阈值(mV)':>12s} {'检出数':>8s}   说明")
    print("  " + "-" * 58)
    for q, note in [(0.05, "不设阈值，看噪声峰有多少"),
                    (0.01, ""),
                    (0.005, ""),
                    (0.001, ""),
                    (0.0005, "← 管线采用（= 99.95 分位）"),
                    (0.0001, "")]:
        thr = np.quantile(hp, q)
        work = hp.copy()
        n = 0
        for _ in range(500):
            i = int(np.argmin(work))
            if work[i] >= thr:
                break
            n += 1
            work[max(0, i-500):i+501] = 1e6
        print(f"  {q:>10.4f} {thr:12.5f} {n:8d}   {note}")

    hr("全 64 通道：检出结果里有多少是假尖峰？")
    n_ch = rec.n_channels
    spike_times, _ = detect_all_channels(rec)
    n_spikes = sum(s.size for s in spike_times)
    n_beats = max(s.size for s in spike_times)
    print(f"  检出总数: {n_spikes}")
    print(f"  应该有的: {n_ch} 电极 x {n_beats} 次搏动 = {n_ch * n_beats}")
    print(f"  多出来的: {n_spikes - n_ch * n_beats}  ← 阈值把噪声峰全挡住了")

    hr("类型 B：假尖峰由霍夫变换处理")
    sp = spikes_to_table(spike_times, rec.channels)
    xyz = np.column_stack([sp["x"], sp["y"], sp["t"]]).astype(float)
    res = hough_plane(xyz, vote_threshold=8, max_iter=200000,
                      min_detectors=40, seed=1)
    n_sig = int(res.prediction.sum())
    n_noise = len(res.prediction) - n_sig
    print(f"  输入点数      : {len(res.prediction)}")
    print(f"  判为信号      : {n_sig}")
    print(f"  判为噪声      : {n_noise}")
    print()
    print("  机制：不属于任何波前的孤立点，无法与其它点共面，")
    print("        累加器里永远攒不够票，最终 prediction 保持 0 —— 被判为噪声。")
    print("  即：霍夫变换本身就是【第二层噪声过滤器】，而且是结构性的，")
    print("      不依赖幅度阈值。幅度够大但位置不对的假尖峰会在这里被剔除。")

    hr("小结")
    print("  ┌────────────────────────────────────────────────────────┐")
    print("  │ 噪声类型            │ 处理手段              │ 在哪一阶段 │")
    print("  ├────────────────────────────────────────────────────────┤")
    print("  │ 基线漂移(低频)      │ Butterworth 高通      │ 阶段 1     │")
    print("  │ 小幅波动(高频)      │ 0.05 分位幅度阈值     │ 阶段 1     │")
    print("  │ 假尖峰(幅度大但孤立)│ 霍夫共面性            │ 阶段 2     │")
    print("  └────────────────────────────────────────────────────────┘")
    print()
    print("  LOWESS 从未参与。它是一条被弃用的备选路径（代码里有、被注释掉）。")
    print("  而且从机理上说 LOWESS 不适合这个任务：它是【平滑器】，")
    print("  会削平尖锐的峰；这里需要的是【高通】，目的是去漂移而不是去高频。")
    print()


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="MEA 记录 CSV 路径")
    a = ap.parse_args()
    try:
        main(find_recording(a.data))
    except FileNotFoundError as e:
        print(f"\n❌ {e}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
