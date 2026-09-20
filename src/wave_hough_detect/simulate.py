"""
阶段 4：仿真数据生成与检测性能评估（论文 §3.1，图 2–8）

对应 R 代码：
  · R_Codes/generate_stomach_data.r                  （8×8 线性波前仿真）
  · R_Codes/stomach_hough17_newton_square20.r        （96×96 圆波前仿真）
  · R/02_simulation/fig5_detection_performance.R     （FPR / FNR 评估）
  · make_synthetic_recording.R                       （合成 MEA 记录）

【为什么需要这一层】
真数据（论文 §3.2）来自合作实验室，不可分发。所以仓库必须提供两条
"不依赖真数据也能跑通"的路：

  1. ``simulate_linear_wavefronts``  —— 论文 §3.1.1 的 8×8 线性波前仿真。
     生成 (x, y, ts, z) 点集，z 标明每个点是真信号还是噪声。
     把 RHT 跑在这上面就能算 FPR / FNR（论文图 5）。

  2. ``simulate_recording``        —— 生成一份【格式与真数据完全相同】的
     64 通道 MEA 记录 CSV，于是阶段 1→2→3 整条管线都能跑通。

【与 R 的关系：统计等价，不是逐位等价】
R 的随机数发生器（Mersenne-Twister + Rejection 采样）numpy 无法复现，
所以本模块【不可能】逐位复现 R 的仿真数据集。这里做的是：
  · 完全照搬 R 的抽样结构与顺序（哪一步抽几次、抽什么分布）；
  · 完全照搬 R 的种子推导公式（参数的乘积 + 偏移），使"同一组参数 →
    同一个种子 → 同一份数据"这个性质在 Python 内部成立、可复现；
  · 因此得到的 FPR/FNR 与 R 的【分布】一致，但具体数值不同。

★ 若要做到逐位比对，只能像阶段 2 那样让 R 导出抽样轨迹再重放 ——
  本模块的随机数调用点比 R 多且分散，没有做这件事。这是已知的、
  明确的边界，不要把它说成"与 R 一致"。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rht import SIM_PRESET, hough_plane
from .spikes import GRID_SIDE, channel_to_xy

# ── 种子基数（照抄 R，见注释里的行号）────────────────────────────────────
SEED_BASE_2D = 101        # generate_stomach_data.r:10、fig5_...R:119
SEED_BASE_1D = 1000000    # fig5_...R:53（一维仿真的那份拷贝）

# ── 论文 §3.1.1 用的参数（fig5_...R:345-349、generate_stomach_data.r:95-107）
PAPER_2D_PARAMS = dict(
    limit=8,          # 8×8 MEA
    time_max=80.0,    # 图 5 的 time.max = limit*10
    ts=1.0,           # 第一条波前的激发时刻
    x=1, y=1,         # 波前起点（格点坐标）
    u_x=1, u_y=1,     # 每前进一步空间走一格
    p=0.1,            # 每个电极漏检的概率
    noise_freq=1.0,   # 噪声的时间率 λ_n
    miu=1.0,          # 电极间到达时差均值 μ
    sigma=0.1,        # 电极间到达时差标准差 σ
    gap=30.0,         # 两条波前的间隔（卡方分布自由度）
    ratio=2.0,        # 换行时的时差放大倍数
)


def seed_from_params(base: float, shift: int, **factors) -> int:
    """
    照抄 R 的种子推导：``set.seed(SEED_BASE * 各参数连乘 + seed.shift)``。

    R 的 set.seed 会把浮点结果截断成整数，这里用 int() 做同样的事，
    再对 2**32 取模以适配 numpy 的种子范围。
    """
    raw = float(base)
    for v in factors.values():
        raw *= float(v)
    raw += float(shift)
    return int(raw) % (2 ** 32)


# ══════════════════════════════════════════════════════════════════════════
#  §3.1.1  8×8 线性波前仿真（对应 R 的 stomach.sim2d）
# ══════════════════════════════════════════════════════════════════════════

def _simulate_one_line(rng: np.random.Generator, limit: int, time_max: float,
                       ts: float, x: float, u_x: float, p: float,
                       miu: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    对应 R 的 stomach.sim.one()：沿 x 方向走一条直线，走到边界或超时为止。

    R 的写法是"先抽 dt，再判断能不能走"，而且判断失败的那次抽样【已经消耗掉了】。
    这里保持完全相同的抽样顺序：dt 的抽取次数 = 落点数 + 1。

    然后每个落点以概率 p 被丢弃（漏检）。
    """
    xs = [x]
    ts_list = [ts]
    dt = float(rng.normal(miu, sigma))
    while (xs[-1] + u_x >= 1) and (xs[-1] + u_x <= limit) and \
            (ts_list[-1] + dt <= time_max):
        xs.append(xs[-1] + u_x)
        ts_list.append(ts_list[-1] + dt)
        dt = float(rng.normal(miu, sigma))

    x_arr = np.asarray(xs, dtype=float)
    t_arr = np.asarray(ts_list, dtype=float)

    # 对应 R：for (i in 1:length(x)) if (runif(1,0,1) > p) 保留
    keep = rng.random(x_arr.size) > p
    return x_arr[keep], t_arr[keep]


def _simulate_one_line_2d(rng: np.random.Generator, limit: int, time_max: float,
                          ts: float, x: float, y: float, u_x: float, u_y: float,
                          p: float, miu: float, sigma: float,
                          ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对应 R 的 stomach.sim.one2d()：沿 y 方向铺一排平行直线（= 一个平面波前）。

    R 的对照函数只判 y ≥ 1、y ≤ limit、ts ≤ time_max —— 注意【没有】判 ts 下界。
    """
    x_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    t_out: list[np.ndarray] = []

    while (y >= 1) and (y <= limit) and (ts <= time_max):
        lx, lt = _simulate_one_line(rng, limit, time_max, ts, x, u_x, p,
                                    miu, sigma)
        x_out.append(lx)
        y_out.append(np.full(lx.size, y, dtype=float))
        t_out.append(lt)
        y += u_y
        ts += float(rng.normal(ratio * miu, ratio * sigma))

    if not x_out:
        return (np.empty(0), np.empty(0), np.empty(0))
    return (np.concatenate(x_out), np.concatenate(y_out), np.concatenate(t_out))


def simulate_linear_wavefronts(seed_shift: int = 1, *,
                               limit: int = 8, time_max: float = 80.0,
                               ts: float = 1.0, x: float = 1.0, y: float = 1.0,
                               u_x: float = 1.0, u_y: float = 1.0, p: float = 0.1,
                               noise_freq: float = 1.0, miu: float = 1.0,
                               sigma: float = 0.1, gap: float = 30.0,
                               ratio: float = 2.0, seed: int | None = None
                               ) -> pd.DataFrame:
    """
    论文 §3.1.1 的仿真：8×8 网格上若干条平行传播的线性波前 + 泊松噪声。

    对应 R 的 ``stomach.sim2d()``（generate_stomach_data.r:8、fig5_...R:115）。

    参数与论文符号的对照见 ``PAPER_2D_PARAMS``。

    返回
    ----
    DataFrame，列为 ``x, y, ts, z``：
        x, y  —— 电极的格点坐标（整数，1..limit）
        ts    —— 该电极上的激活时刻
        z     —— 1 = 真信号，0 = 噪声。**这是本模块存在的意义**：
                 有了 z 才能算 FPR / FNR。
    行按 ts 稳定排序（R 的 ``order()`` 是稳定排序，这点必须一致）。
    """
    if seed is None:
        seed = seed_from_params(
            SEED_BASE_2D, seed_shift, limit=limit, time_max=time_max, ts=ts,
            x=x, y=y, u_x=u_x, u_y=u_y, p=p, noise_freq=noise_freq,
            miu=miu, sigma=sigma)
    rng = np.random.default_rng(seed)

    # ── 真信号：一波接一波，波前起点按卡方间隔推进（R 131-140）────────
    sig_x: list[np.ndarray] = []
    sig_y: list[np.ndarray] = []
    sig_t: list[np.ndarray] = []
    signal_start_ts = ts
    while signal_start_ts < time_max:
        lx, ly, lt = _simulate_one_line_2d(
            rng, limit, time_max, signal_start_ts, x, y, u_x, u_y, p,
            miu, sigma, ratio)
        sig_x.append(lx); sig_y.append(ly); sig_t.append(lt)
        signal_start_ts += float(rng.chisquare(gap))

    x_sig = np.concatenate(sig_x) if sig_x else np.empty(0)
    y_sig = np.concatenate(sig_y) if sig_y else np.empty(0)
    t_sig = np.concatenate(sig_t) if sig_t else np.empty(0)

    # ── 噪声：时间上服从指数间隔，位置上在网格内均匀（R 142-154）──────
    noise_ts = [0.0]
    while noise_ts[-1] < time_max:
        noise_ts.append(noise_ts[-1] + float(rng.exponential(1.0 / noise_freq)))
    # R 丢掉首尾两点（首点是人为塞的 0，末点已越过 time_max）
    noise_ts = np.asarray(noise_ts[1:-1], dtype=float) if len(noise_ts) > 2 \
        else np.empty(0)
    n_noise = noise_ts.size
    noise_x = rng.integers(1, limit + 1, size=n_noise).astype(float)
    noise_y = rng.integers(1, limit + 1, size=n_noise).astype(float)

    df = pd.DataFrame({
        "x": np.concatenate([x_sig, noise_x]),
        "y": np.concatenate([y_sig, noise_y]),
        "ts": np.concatenate([t_sig, noise_ts]),
        "z": np.concatenate([np.ones(t_sig.size, dtype=int),
                             np.zeros(n_noise, dtype=int)]),
    })
    # ★ R 用 order()（稳定）；pandas 的 sort_values 默认不稳定。
    #   ts 是连续量，理论上不并列，但仿真里有大量重复的噪声时刻，
    #   这里必须显式指定 stable，否则与 R 的行序不同。
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
#  检测性能评估（对应 R 的 visualize2d / optimality.table，论文图 5）
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionRates:
    """一次仿真的检测性能。四类计数的定义与 R 完全一致（fig5_...R:362-365）。"""

    n_points: int
    n_true_pos: int          # z=1 且 prediction=1
    n_false_pos: int         # z=0 且 prediction=1 —— 噪声被当成信号
    n_false_neg: int         # z=1 且 prediction=0 —— 信号被当成噪声
    n_true_neg: int          # z=0 且 prediction=0
    n_planes: int            # RHT 找到的平面数

    @property
    def false_positive_rate(self) -> float:
        """FPR，论文图 5 的纵轴之一。

        ★ R 的定义是 ``num.false.pos / num.all``（fig5_...R:371），
          分母是【全部点】，不是噪声点数。这里如实复现，不改。
          按常规定义算的那个在 ``false_positive_rate_among_noise``。
        """
        return self.n_false_pos / self.n_points if self.n_points else float("nan")

    @property
    def false_negative_rate(self) -> float:
        """FNR，同上：分母是全部点（fig5_...R:372）。"""
        return self.n_false_neg / self.n_points if self.n_points else float("nan")

    @property
    def n_noise(self) -> int:
        return self.n_false_pos + self.n_true_neg

    @property
    def n_signal(self) -> int:
        return self.n_true_pos + self.n_false_neg

    @property
    def false_positive_rate_among_noise(self) -> float:
        """常规口径的 FPR = FP / (FP + TN)。仅作参考，不是论文的口径。"""
        return self.n_false_pos / self.n_noise if self.n_noise else float("nan")

    @property
    def false_negative_rate_among_signal(self) -> float:
        """常规口径的 FNR = FN / (FN + TP)。仅作参考，不是论文的口径。"""
        return self.n_false_neg / self.n_signal if self.n_signal else float("nan")

    def as_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "n_signal": self.n_signal, "n_noise": self.n_noise,
            "TP": self.n_true_pos, "FP": self.n_false_pos,
            "FN": self.n_false_neg, "TN": self.n_true_neg,
            "n_planes": self.n_planes,
            "FPR": self.false_positive_rate,
            "FNR": self.false_negative_rate,
            "FPR_among_noise": self.false_positive_rate_among_noise,
            "FNR_among_signal": self.false_negative_rate_among_signal,
        }


def detection_rates(z: np.ndarray, prediction: np.ndarray,
                    n_planes: int = 0) -> DetectionRates:
    """
    由真值标签 z 与 RHT 的 prediction 算四类计数（对应 R fig5_...R:362-374）。

    ★ R 原代码有一处笔误：``num.true.neg = length(red.indices)``
      （fig5_...R:369，把 red 写成了 TN）。该变量从未被使用，所以不影响论文结果。
      这里用正确的定义，并在文档里记录这处差异。
    """
    z = np.asarray(z).astype(int)
    prediction = np.asarray(prediction).astype(int)
    return DetectionRates(
        n_points=int(z.size),
        n_true_pos=int(np.sum((z == 1) & (prediction == 1))),
        n_false_pos=int(np.sum((z == 0) & (prediction == 1))),
        n_false_neg=int(np.sum((z == 1) & (prediction == 0))),
        n_true_neg=int(np.sum((z == 0) & (prediction == 0))),
        n_planes=int(n_planes),
    )


@dataclass
class DetectionResult:
    """一次仿真 + 检测的完整记录，便于出图（论文图 2/3/4）。"""

    data: pd.DataFrame                     # x, y, ts, z
    prediction: np.ndarray                 # RHT 的 0/1 判据
    plane_indices: np.ndarray
    normals: np.ndarray                    # (n_planes, 3)
    rhos: np.ndarray
    rates: DetectionRates
    seed: int

    def classified(self) -> pd.DataFrame:
        """
        按 R 的 stomach.plot2d（fig5_...R:320-342）给出四类着色标签：

            green  = TP（真信号，判为信号）
            yellow = FP（噪声，判为信号）
            red    = FN（真信号，判为噪声）
            black  = TN（噪声，判为噪声）
        """
        z = self.data["z"].to_numpy()
        p = self.prediction
        label = np.empty(z.size, dtype=object)
        label[(z == 1) & (p == 1)] = "green"     # TP
        label[(z == 0) & (p == 1)] = "yellow"    # FP
        label[(z == 1) & (p == 0)] = "red"       # FN
        label[(z == 0) & (p == 0)] = "black"     # TN
        out = self.data.copy()
        out["prediction"] = p
        out["plane"] = self.plane_indices
        out["class"] = label
        return out


def evaluate_detection(seed_shift: int = 1, *, hough_kwargs: dict | None = None,
                       **params) -> DetectionResult:
    """
    跑一次 §3.1.1 仿真 + RHT，给出检测性能（对应 R 的 ``visualize2d()``，
    fig5_...R:344-375）。

    ``params`` 透传给 :func:`simulate_linear_wavefronts`。

    ★ R 的仿真版 HT 用 ρ 步长 0.5、内点容忍度 1.0（是 ts 尺度决定的），
      与真数据管线的 0.05 / 0.1 不同。这里默认套用 ``SIM_PRESET``，
      所以**不要**用真数据那组参数跑这个仿真。
    """
    kwargs = dict(SIM_PRESET)
    kwargs.update(dict(max_iter=30000, vote_threshold=8, min_detectors=40))
    if hough_kwargs:
        kwargs.update(hough_kwargs)

    df = simulate_linear_wavefronts(seed_shift, **params)
    pts = df[["x", "y", "ts"]].to_numpy(dtype=float)
    res = hough_plane(pts, **kwargs)

    seed = seed_from_params(
        SEED_BASE_2D, seed_shift,
        limit=params.get("limit", PAPER_2D_PARAMS["limit"]),
        time_max=params.get("time_max", PAPER_2D_PARAMS["time_max"]),
        ts=params.get("ts", PAPER_2D_PARAMS["ts"]),
        x=params.get("x", PAPER_2D_PARAMS["x"]),
        y=params.get("y", PAPER_2D_PARAMS["y"]),
        u_x=params.get("u_x", PAPER_2D_PARAMS["u_x"]),
        u_y=params.get("u_y", PAPER_2D_PARAMS["u_y"]),
        p=params.get("p", PAPER_2D_PARAMS["p"]),
        noise_freq=params.get("noise_freq", PAPER_2D_PARAMS["noise_freq"]),
        miu=params.get("miu", PAPER_2D_PARAMS["miu"]),
        sigma=params.get("sigma", PAPER_2D_PARAMS["sigma"]))

    return DetectionResult(
        data=df,
        prediction=res.prediction,
        plane_indices=res.plane_indices,
        normals=np.column_stack([res.n1, res.n2, res.n3]),
        rhos=res.rhos,
        rates=detection_rates(df["z"].to_numpy(), res.prediction,
                              res.n_planes),
        seed=seed,
    )


def detection_sweep(seed_shifts=range(0, 100), *, hough_kwargs: dict | None = None,
                    **params) -> pd.DataFrame:
    """
    复现论文图 5：对一批随机种子各跑一次，得到 FPR / FNR 的分布。

    R 的对应代码是 ``optimality.table()``（fig5_...R:680-705）。注意提交的
    脚本里是 ``shifts = 0``（只跑一次），``shifts = 1:100`` 被注释掉了 ——
    论文图 5 用的是后者。默认值取 ``range(0, 100)``（100 个种子）。

    ★ 注意 seed_shift = 0 与 1 只差 1，但种子推导是"参数连乘 + shift"，
      所以相邻 shift 得到的是完全不同的数据集，不是微扰。
    """
    rows = []
    for s in seed_shifts:
        r = evaluate_detection(int(s), hough_kwargs=hough_kwargs, **params)
        rows.append({"seed_shift": int(s), **r.rates.as_dict()})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
#  合成 MEA 记录（对应 R 的 make_synthetic_recording.R）
# ══════════════════════════════════════════════════════════════════════════

def simulate_recording(out_path: str | Path | None = None, *,
                       sample_rate_hz: float = 10000.0,
                       duration_ms: float = 9000.0,
                       grid_side: int = GRID_SIDE,
                       time_units_per_ms: float = 1.0,
                       first_wave_ms: float = 700.0,
                       beat_interval_ms: float = 1000.0,
                       beat_jitter_ms: float = 60.0,
                       source_x0: float = -3.7, source_y0: float = -50.0,
                       speed: float = 0.45,
                       spike_amp_mv: float = -1.0, spike_decay_ms: float = 0.8,
                       spike_pos_amp: float = 0.30, spike_pos_ms: float = 2.5,
                       spike_pos_decay_ms: float = 1.8,
                       spike_half_ms: float = 10.0,
                       noise_sd_mv: float = 0.02, drift_mv: float = 0.5,
                       drift_period_ms: float = 4000.0,
                       seed: int = 20260915) -> tuple[pd.DataFrame, dict]:
    """
    生成一份【格式与论文 §3.2 的真数据完全相同】的 64 通道 MEA 记录。

    对应 R 的 ``make_synthetic_recording.R``。这样即使拿不到实验记录，
    阶段 1→2→3 整条管线也能端到端跑通。

    ★ 它【不会】复现论文表 2/3/4 的数字 —— 那三个表依赖实验记录本身。
      它的作用有两个，而且第二个更硬：
        1) 格式正确、能跑、能出图；
        2) **在已知真值的情况下端到端验证管线**：源点位置 (source_x0, source_y0)、
           速度 speed、以及每条波前的激发时刻都是自己设的，拟合出来的
           (x0, y0, v, t0) 必须落回这些值。实测 v=0.449–0.451（真值 0.450）、
           x0 −3.7～−3.9（真值 −3.70）、t0 与真值差 <1 ms。

    返回 ``(DataFrame, info)``；给了 ``out_path`` 就同时写出 CSV。
    CSV 列为 ``time, Ch01, ..., Ch64``，与真数据一致。

    ★★ ``time_units_per_ms`` 必须保持 1.0，这是整条管线里最隐蔽的一个坑。
       论文 §3.2 的输入文件里 time 列【就是毫秒】（0..8999.9）。管线内部再做
       ``ts / 200``（cm_hough_grid_8.R:115）**不是单位换算**，而是把时间轴压到与
       网格坐标 (x, y ∈ [1,8]) 同一个量级，好让内点容忍度 0.1 和 ρ 步长 0.05
       同时适用；拟合前又 ``* 200`` 乘回去（:528）。所以：

           time 列必须让 "time/200" 与 x,y 同量级，即 time ∈ [200, ~10000]。

       把 time 列写成 "t_ms × 200"（R 版 make_synthetic_recording.R 里的
       TIME_UNITS_PER_MS <- 200）会让管线内部的 t 变成几千，与 x,y 差三个数量级，
       于是霍夫找到的全是"x ≈ 常数"这种垃圾平面 —— 已实测：
       v ≈ 0、t0 ≈ ±2e7、14 个平面而不是 9 个。R 版本脚本里那句"TIME_UNITS_PER_MS
       = 1 时请注释掉第 115 行"同样是错的：第 115 行任何时候都不能注释掉。

    与 R 版的【有意不同】：
      · R 用 ``seq(0, DURATION_MS, by=...)`` 会多写一个采样点（90001 行）；
        这里用 ``np.arange`` 得到 90000 行，与真数据的行数一致。
      · 随机数发生器不同，因此波形不同（见模块文档）。
      · time 列默认按毫秒写（R 版默认按 1/200 ms 写，那是个 bug，见上）。
    """
    rng = np.random.default_rng(seed)

    t_ms = np.arange(0.0, duration_ms, 1000.0 / sample_rate_hz)
    n_channels = grid_side * grid_side

    # 每个电极到源点的距离 → 到达时延
    dist = np.empty(n_channels)
    for ch in range(1, n_channels + 1):
        cx, cy = channel_to_xy(ch, grid_side)
        dist[ch - 1] = np.hypot(cx - source_x0, cy - source_y0)
    max_delay_ms = float(dist.max() / speed)

    # 源点处的激发时刻序列：间隔均值 beat_interval_ms，抖动 beat_jitter_ms。
    # 录制结束前还没传完的波直接丢掉 —— 真实实验就是这样。
    wave_times = [first_wave_ms]
    while True:
        nxt = wave_times[-1] + float(
            rng.normal(beat_interval_ms, beat_jitter_ms))
        if nxt + max_delay_ms > duration_ms:
            break
        wave_times.append(nxt)
    wave_times = np.asarray(wave_times)

    def spike_shape(u: np.ndarray) -> np.ndarray:
        """胞外场电位：快速负向偏转 + 较小的慢速正向成分。"""
        return (spike_amp_mv * np.exp(-(u / spike_decay_ms) ** 2)
                + spike_pos_amp
                * np.exp(-((u - spike_pos_ms) / spike_pos_decay_ms) ** 2))

    drift = drift_mv * np.sin(2 * np.pi * t_ms / drift_period_ms)

    cols: dict[str, np.ndarray] = {"time": t_ms * time_units_per_ms}
    for ch in range(1, n_channels + 1):
        y = rng.normal(0.0, noise_sd_mv, size=t_ms.size) + drift
        for act in wave_times + dist[ch - 1] / speed:
            idx = np.flatnonzero((t_ms >= act - spike_half_ms)
                                 & (t_ms <= act + spike_half_ms))
            if idx.size:
                y[idx] += spike_shape(t_ms[idx] - act)
        cols[f"Ch{ch:02d}"] = y

    df = pd.DataFrame(cols)
    info = {
        "n_samples": int(df.shape[0]),
        "n_channels": n_channels,
        "sample_rate_hz": sample_rate_hz,
        "duration_ms": duration_ms,
        # ── 真值（用于端到端判定"拟合是否落回设定值"）────────────────
        "source_x0": source_x0,
        "source_y0": source_y0,
        "speed": speed,
        "grid_side": grid_side,
        "time_units_per_ms": time_units_per_ms,
        "wave_times_ms": np.round(wave_times, 1).tolist(),
        "max_delay_ms": max_delay_ms,
        "seed": seed,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        info["path"] = str(out_path)

    return df, info
