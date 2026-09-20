"""
阶段 1：尖峰检测（论文 §2.1 / Algorithm 1）

对应 R 代码：R_Codes/cm_hough_grid_8.R 第 18-117 行
本模块是逐行移植，行为必须与 R 完全一致。

移植时容易出错的三个点（都在下面标了 ★）：
  ★1  R 的 filter() 是单向 IIR，不是 scipy.signal.filtfilt（零相位双向）
  ★2  R 的 quantile(type=7) 与 numpy.quantile(线性插值) 相同
  ★3  R 的 which.min 遇并列取【第一个】，numpy.argmin 也是
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

# ── 论文/原代码里的固定参数 ────────────────────────────────────────────────
GRID_SIDE = 8                 # 8 x 8 MEA
BUTTER_ORDER = 2              # signal::butter(2, ...)  —— n 就是阶数
BUTTER_CUTOFF = 1 / 500       # W = 相对 Nyquist 的比例
SPIKE_QUANTILE = 0.0005       # 0.05 分位（= 上尾 99.95 分位）
EXCLUSION_HALF = 500          # 检出后屏蔽 ±500 个样本
MAX_SPIKES_PER_CHANNEL = 200  # 原代码的 for (k in 1:200)
TIME_SCALE = 200              # 原代码 line 115: ts.observ / 200


def channel_to_xy(ch: int, side: int = GRID_SIDE) -> tuple[int, int]:
    """
    通道号 -> (x, y)，与原代码 cm_hough_grid_8.R:82-88 完全一致。
    行优先：x 为行，y 为列。

    >>> channel_to_xy(1), channel_to_xy(8), channel_to_xy(9), channel_to_xy(64)
    ((1, 1), (1, 8), (2, 1), (8, 8))
    """
    if ch % side == 0:
        return ch // side, side
    return ch // side + 1, ch % side


@dataclass
class Recording:
    """一份 MEA 记录。"""

    time: np.ndarray          # (n_samples,)  时间轴，单位 ms
    voltage: np.ndarray       # (n_samples, 64) 电压，单位 mV
    channels: list[str]       # 列名，形如 ["Ch01", ..., "Ch64"]
    sampling_rate_hz: float

    @property
    def n_samples(self) -> int:
        return self.time.size

    @property
    def n_channels(self) -> int:
        return self.voltage.shape[1]


def load_recording(path: str | Path) -> Recording:
    """
    读入 MEA 的 CSV 导出文件。

    原始文件列名形如 ``T(ms), CH1(mV), ..., CH64(mV)``。
    原 R 代码用 ``as.integer(substr(name, 3, 4))`` 解析通道号，
    但 "CH1(mV)" 的第 3-4 个字符是 "1("，as.integer("1(") 返回 NA —— 会崩。
    这里统一重命名为 ``Ch01..Ch64``，使 substr(3,4) 得到 "01".."64"。
    """
    df = pd.read_csv(path)
    time = df.iloc[:, 0].to_numpy(dtype=float)
    voltage = df.iloc[:, 1:].to_numpy(dtype=float)

    n_ch = voltage.shape[1]
    channels = [f"Ch{i:02d}" for i in range(1, n_ch + 1)]

    # 采样率由时间轴步长推得（原始数据 0.1 ms -> 10 kHz）
    dt = float(np.median(np.diff(time)))
    rate = 1000.0 / dt if dt > 0 else np.nan

    return Recording(time=time, voltage=voltage, channels=channels,
                     sampling_rate_hz=rate)


def butter_highpass(order: int = BUTTER_ORDER,
                    cutoff: float = BUTTER_CUTOFF) -> tuple[np.ndarray, np.ndarray]:
    """
    设计 Butterworth 高通滤波器，等价于 R 的 ``signal::butter(2, 1/500, type="high")``。

    注意 ``cutoff`` 是【相对 Nyquist 的比例】，不是 Hz。
    10 kHz 采样 -> Nyquist 5 kHz -> cutoff = 1/500 即 10 Hz。
    """
    return butter(order, cutoff, btype="high")


def filter_channel(x: np.ndarray,
                   ba: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    单向 IIR 滤波。

    ★1 必须用 lfilter（单向），不能用 filtfilt（零相位双向）。
       R 的 ``filter()`` 是单向的，初始条件为 0；``lfilter`` 行为相同。
       用 filtfilt 会得到不同的结果，且相位特性完全改变。
    """
    b, a = ba
    return lfilter(b, a, x)


def detect_spikes(filtered: np.ndarray,
                  time: np.ndarray,
                  exclusion_half: int = EXCLUSION_HALF,
                  quantile: float = SPIKE_QUANTILE,
                  max_spikes: int = MAX_SPIKES_PER_CHANNEL) -> np.ndarray:
    """
    在单通道的滤波后信号上检测尖峰，对应原代码 line 59, 66-99。

    算法：
      阈值 = quantile(信号, 0.0005)
      循环：
        找全局最小点；若它 >= 阈值则停止
        记录该点时刻
        把该点 ±exclusion_half 个样本置为 +1e6（屏蔽，避免重复检出）

    ★2 ``np.quantile(..., method="linear")`` 就是 R 默认的 type=7 分位数，
       两者数值完全相同。
    ★3 ``np.argmin`` 遇并列取第一个，与 R 的 ``which.min`` 一致。

     extracellular 场电位本身是负向的，所以用 argmin 而非 argmax。
    """
    threshold = np.quantile(filtered, quantile, method="linear")

    work = filtered.copy()
    times: list[float] = []
    n = work.size

    for _ in range(max_spikes):
        idx = int(np.argmin(work))
        if work[idx] >= threshold:
            break
        times.append(float(time[idx]))
        lo = max(0, idx - exclusion_half)
        hi = min(n, idx + exclusion_half + 1)
        work[lo:hi] = 1e6

    return np.asarray(times, dtype=float)


def detect_all_channels(rec: Recording) -> tuple[list[np.ndarray], np.ndarray]:
    """
    对 64 个通道逐个跑检测。

    返回 (每个通道的尖峰时刻列表, 滤波后的信号矩阵)。
    """
    ba = butter_highpass()
    filtered = np.empty_like(rec.voltage)
    spike_times: list[np.ndarray] = []

    for i in range(rec.n_channels):
        f = filter_channel(rec.voltage[:, i], ba)
        filtered[:, i] = f
        spike_times.append(detect_spikes(f, rec.time))

    return spike_times, filtered


def spikes_to_table(spike_times: list[np.ndarray],
                    channels: list[str],
                    time_scale: float = TIME_SCALE) -> pd.DataFrame:
    """
    把逐通道的尖峰时刻组装成霍夫变换需要的 (x, y, t) 三元组表。

    ``time_scale`` 对应原代码 line 115 的 ``ts.observ / 200``。
    这个缩放【不是可选的】：不缩放的话平面内标准差约 5.35，
    而霍夫的内点容忍度只有 0.1，一个平面都找不到（已实测验证）。
    """
    rows = []
    for i, times in enumerate(spike_times, start=1):
        if times.size == 0:
            continue
        x, y = channel_to_xy(i)
        for t in times:
            rows.append((x, y, t / time_scale, channels[i - 1]))

    df = pd.DataFrame(rows, columns=["x", "y", "t", "channel"])

    # ★ 必须用稳定排序。R 的 order() 是稳定的，而 pandas 的 sort_values()
    #   默认是快排（不稳定）。这份数据里有 47 个 t 值存在并列（涉及 98 个点），
    #   不稳定排序会让并列点的顺序与 R 不同 —— 后续任何按下标索引的操作
    #   （例如 R 导出的抽样轨迹）都会指向错误的点，而且不会报错，只会静默出错。
    return df.sort_values("t", kind="stable").reset_index(drop=True)
