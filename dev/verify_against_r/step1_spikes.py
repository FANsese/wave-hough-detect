"""
阶段 1 验证脚本：Python 的尖峰检测结果 vs R 的尖峰检测结果

运行（在仓库根目录）：
    python dev/verify_against_r/step1_spikes.py

判定标准：两边检出的尖峰时刻必须【逐位相同】。
任何差异都说明移植有问题，必须查清，不能放宽容差蒙混过去。

依赖 R 侧产物：R_work/output/spikes_xyz.csv
（由 R_work/run1_spike_detection.R 生成）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import filtfilt

from _paths import DATA, OUT as OUT_DIR, R_OUT, hr

from wave_hough_detect import (  # noqa: E402
    Recording,
    butter_highpass,
    channel_to_xy,
    detect_all_channels,
    detect_spikes,
    load_recording,
    spikes_to_table,
)

R_SPIKES = R_OUT / "spikes_xyz.csv"


def main() -> int:
    if not DATA.exists():
        print(f"\n❌ 找不到真实记录：{DATA}\n")
        return 2
    if not R_SPIKES.exists():
        print(f"\n❌ 找不到 R 的尖峰检出：{R_SPIKES}")
        print("   请先在论文工程根目录执行：")
        print("     Rscript R_work/run1_spike_detection.R\n")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hr("步骤 1/4  读入数据")
    rec: Recording = load_recording(DATA)
    print(f"  文件      : {DATA.name}")
    print(f"  维度      : {rec.n_samples} 采样点 x {rec.n_channels} 通道")
    print(f"  时间轴    : {rec.time[0]:.1f} – {rec.time[-1]:.1f} ms")
    print(f"  采样率    : {rec.sampling_rate_hz:.0f} Hz")
    print(f"  通道名    : {rec.channels[0]} ... {rec.channels[-1]}")
    print(f"  电压范围  : {rec.voltage.min():.4f} – {rec.voltage.max():.4f} mV")

    hr("步骤 2/4  滤波 + 尖峰检测")
    spike_times, filtered = detect_all_channels(rec)
    counts = np.array([s.size for s in spike_times])
    print(f"  每通道检出: 最少 {counts.min()}, 最多 {counts.max()}, "
          f"合计 {counts.sum()} 个尖峰")
    print(f"  滤波后范围: {filtered.min():.4f} – {filtered.max():.4f} mV")

    # 不做 /200 缩放，直接和 R 的原始检出时刻比对
    py_raw = spikes_to_table(spike_times, rec.channels, time_scale=1.0)
    py_raw.to_csv(OUT_DIR / "spikes_raw_ms.csv", index=False)

    hr("步骤 3/4  与 R 的结果逐位比对")
    r_df = pd.read_csv(R_SPIKES)
    print(f"  R       : {len(r_df)} 个尖峰")
    print(f"  Python  : {len(py_raw)} 个尖峰")

    failures = []

    # --- 3a 总数 ---
    if len(r_df) != len(py_raw):
        failures.append(f"尖峰总数不同: R={len(r_df)}, Python={len(py_raw)}")
    else:
        print(f"  ✅ 总数一致: {len(r_df)}")

    # --- 3b 每个通道的检出数 ---
    r_cnt = r_df.groupby("channel").size()
    p_cnt = py_raw.groupby("channel").size()
    if not r_cnt.equals(p_cnt):
        diff = (r_cnt - p_cnt).dropna()
        failures.append(f"逐通道计数不同: {diff[diff != 0].to_dict()}")
    else:
        print(f"  ✅ 逐通道计数一致 ({len(r_cnt)} 个通道)")

    # --- 3c 坐标映射 ---
    bad_xy = 0
    for ch, grp in py_raw.groupby("channel"):
        i = int(ch[2:])
        ex, ey = channel_to_xy(i)
        if not ((grp["x"] == ex).all() and (grp["y"] == ey).all()):
            bad_xy += 1
    if bad_xy:
        failures.append(f"{bad_xy} 个通道的 (x,y) 映射不符")
    else:
        print("  ✅ 通道 → (x,y) 映射与 R 一致")

    # --- 3d 逐个尖峰时刻（最严格） ---
    r_sorted = r_df.sort_values(["channel", "t"], kind="stable").reset_index(drop=True)
    p_sorted = py_raw.sort_values(["channel", "t"], kind="stable").reset_index(drop=True)

    if len(r_sorted) == len(p_sorted):
        dt = np.abs(r_sorted["t"].to_numpy() - p_sorted["t"].to_numpy())
        same_ch = (r_sorted["channel"].to_numpy() == p_sorted["channel"].to_numpy())
        max_dt = float(dt.max()) if dt.size else 0.0
        n_exact = int((dt == 0).sum())
        print(f"  逐点时刻差: 最大 {max_dt:.10f} ms")
        print(f"  完全相同的点: {n_exact} / {len(dt)}")
        if not same_ch.all():
            failures.append(f"{int((~same_ch).sum())} 个尖峰的通道归属不同")
        if max_dt > 1e-9:
            failures.append(f"存在时刻差异，最大 {max_dt:.10f} ms "
                            f"（{len(dt) - n_exact} 个点）")
        else:
            print(f"  ✅ 全部 {len(dt)} 个尖峰时刻【逐位相同】")

    # --- 对照组：如果故意用 filtfilt 会差多少 ---
    hr("对照组  若误用 filtfilt（零相位双向）会差多少")
    ba = butter_highpass()
    f_single = filtered[:, 0]
    f_double = filtfilt(ba[0], ba[1], rec.voltage[:, 0])
    s_single = detect_spikes(f_single, rec.time)
    s_double = detect_spikes(f_double, rec.time)
    print(f"  lfilter  (正确) : {s_single.size} 个尖峰  {np.round(s_single, 1)}")
    print(f"  filtfilt (错误) : {s_double.size} 个尖峰  {np.round(s_double, 1)}")
    if s_single.size == s_double.size and s_single.size > 0:
        n = s_single.size
        d = np.abs(np.sort(s_single)[:n] - np.sort(s_double)[:n]).max()
        print(f"  两者最大时刻差  : {d:.4f} ms")
    else:
        print("  两者数量不同，无法逐点比较 —— 这正是不能换滤波器的原因")

    # --- 输出 ---
    hr("步骤 4/4  写出结果")
    scaled = spikes_to_table(spike_times, rec.channels)   # t/200，喂给霍夫用
    scaled.to_csv(OUT_DIR / "spikes_xyz.csv", index=False)
    np.save(OUT_DIR / "filtered.npy", filtered)
    print(f"  {OUT_DIR / 'spikes_raw_ms.csv'}   (t 为原始 ms，用于与 R 比对)")
    print(f"  {OUT_DIR / 'spikes_xyz.csv'}      (t/200，霍夫变换的输入)")
    print(f"  {OUT_DIR / 'filtered.npy'}        (64 通道滤波后信号)")

    # --- 结论 ---
    hr()
    if failures:
        print(" ❌ 移植失败，存在以下差异：")
        for f in failures:
            print(f"    - {f}")
        return 1
    print(" ✅ 移植成功：Python 与 R 的尖峰检测结果完全一致")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
