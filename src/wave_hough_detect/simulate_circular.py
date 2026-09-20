"""
阶段 4b：96×96 圆波前仿真与拟合精度评估（论文 §3.1.2，图 6–8）

对应 R 代码：R_Codes/stomach_hough17_newton_square20.r
  · ``stomach.sim.one2d``  :100-153   逐电极生成到达时刻
  · ``stomach.sim2d``      :156-198   组装一次仿真数据集
  · ``visualize2d``        :652-730   跑一次「仿真 + 拟合」，返回真值与估计
  · ``relerror.plot``      :782-876   把估计值摊成待画的序列
  · ``optimality.table``   :878-1010  按 plot.mode 做参数扫描

【与 §3.1.1 的区别：这是另一套仿真，不是同一个东西】

§3.1.1（``simulate.py``）是 8×8 网格上的**线性波前**，用来算 FPR/FNR（图 5）；
本节是 96×96 网格上的**圆波前**，用来算源点/速度/激发时刻的估计精度（图 6–8）。
两者连抽样结构都不一样：本节的到达时刻**不是**「沿一条线逐点走」，
而是**每个电极独立算一次** ``t = t₀ + 距离 + ε``（R:121-129）。

【真值】
``visualize2d`` 返回 ``truth = c(x, y, miux, ts) = c(48, 48, 1, 2)``（R:728）。
注意第三项是 ``miux`` 而不是 ``v``：在这套仿真里 miux = miuy = 1，
所以「走一格用 1 个时间单位」，即 v = 1。也就是说圆波前模型的真值是

    t = √((x−48)² + (y−48)²) / 1 + 2 + ε

【为什么 R 的初值在这里恰好是对的】
拟合用的交替最小化从 (u, t₀) = (1, 1) 起步（u = 1/v 是慢度）。
而这套仿真的真值是 v = 1、t₀ = 2 —— 初值几乎正中靶心。
所以 §3.1.1 里我实测到的那个「源点在网格内就收敛到错误极小」的毛病，
在这一节里【不会】发作。这不是巧合，是仿真参数挑出来的。

【与 R 的差异：随机数发生器】
同 ``simulate.py`` —— R 的 MT + Rejection 无法用 numpy 复现，所以是统计等价，
不是逐位等价。这一节没有像阶段 2 那样做「导出抽样轨迹再重放」，
因为这里的随机数调用点又多又分散（每格 1 次 rnorm + 条件 1 次 runif）。
这是明确的边界，不要把它说成「与 R 一致」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .fit import (
    HYBRID_MAX_ITER,
    circular_loss,
    hybrid_optim,
    r_squared,
)

# ── 与 R 一致的固定量 ────────────────────────────────────────────────────
SEED_BASE_CIRCULAR = 100          # R:158 `SEED_BASE = 100`

#: §3.1.2 的真值（R:728 的 `c(x, y, miux, ts)`）。
CIRCULAR_TRUTH = {"x0": 48.0, "y0": 48.0, "v": 1.0, "t0": 2.0}

#: 论文 §3.1.2 用的仿真参数（R:653-662 的调用点）。
PAPER_CIRCULAR_PARAMS = dict(
    limit=96,          # 96×96 网格
    time_max=96.0,     # 观测窗
    ts=2.0,            # 源点激发时刻
    x=48.0, y=48.0,    # 源点位置
    u_x=1.0, u_y=1.0,  # 未使用（R 的形参保留了但函数体里没用到）
    p=0.0,             # 漏检概率（图 8 扫的就是它）
    noise_freq=1.0,    # 噪声率 λ_n
    miu_x=1.0, miu_y=1.0,   # 各向异性传播：真值里的「速度」
    sigma=1e-6,        # 测量误差标准差（图 7 固定它、图 8/σ 曲线扫它）
    gap=1e6,           # 波前间隔的指数分布均值：大到只会产生一条波前
    ratio=10.0,        # 未使用
)

#: §3.1.2 的拟合设置（R:632-636 的 ``hybrid.optim``）。
SEARCH_LOWER_CIRCULAR = np.array([-500.0, -500.0])
SEARCH_UPPER_CIRCULAR = np.array([500.0, 500.0])
CIRCULAR_EPS = 1e-8               # R:635 `epsilon = 1e-8`

#: 论文的三条扫描（R:878-1010 的 ``plot.mode``）。
PLOT_MODES = {
    # variable 是「画在横轴上的量」，param 是它在仿真器里的形参名
    1: {"variable": "snr", "param": "noise_freq",
        "label": "signal-to-noise ratio  λ_n（横轴取倒数）",
        "grid": 2.0 ** (np.arange(-6, 17) / 2.0)},        # R:889-890
    2: {"variable": "p", "param": "p",
        "label": "missing-observation probability  p",
        "grid": np.arange(0, 9) / 10.0},                  # R:908-909
    3: {"variable": "sigma", "param": "sigma",
        "label": "measurement error  σ",
        "grid": np.arange(1, 11) / 10.0},                 # R:924-925
}


def seed_circular(*, limit, time_max, ts, x, y, u_x, u_y, p, noise_freq,
                  miu_x, miu_y, sigma, seed_shift: int = 0) -> int:
    """
    照抄 R 的种子推导（R:160-161）：

        set.seed(100 · limit · time.max · ts · x · y · u.x · u.y
                 · p · noise.freq · miux · miuy · sigma)

    ★ R 这里【没有 seed.shift】，`stomach.sim2d` 的形参表里也没有它 ——
      所以对同一组参数，R 的每一次「重复」都是**同一份数据集**。
      本实现保留这个公式，但多加一个 ``seed_shift``（默认 0，即与 R 相同），
      以便做真正独立的重复。相关的坑见 :func:`accuracy_sweep`。
    """
    raw = (SEED_BASE_CIRCULAR * float(limit) * float(time_max) * float(ts)
           * float(x) * float(y) * float(u_x) * float(u_y) * float(p)
           * float(noise_freq) * float(miu_x) * float(miu_y) * float(sigma))
    return int(raw + seed_shift) % (2 ** 32)


def simulate_circular_wavefronts(seed_shift: int = 0, *,
                                 limit: int = 96, time_max: float = 96.0,
                                 ts: float = 2.0, x: float = 48.0,
                                 y: float = 48.0, u_x: float = 1.0,
                                 u_y: float = 1.0, p: float = 0.0,
                                 noise_freq: float = 1.0, miu_x: float = 1.0,
                                 miu_y: float = 1.0, sigma: float = 1e-6,
                                 gap: float = 1e6, ratio: float = 10.0,
                                 seed: int | None = None) -> pd.DataFrame:
    """
    一次 96×96 圆波前仿真（对应 R 的 ``stomach.sim2d``，R:156-198）。

    返回 DataFrame，列为 ``x, y, ts, z``；z = 1 表示真信号、0 表示噪声。
    行按 ts **稳定排序**（R 的 ``order()`` 是稳定排序，R:195）。

    生成逻辑（严格照抄，包括随机数的调用顺序）：

    1. **每个电极独立算一次**（R:109-139）。对 ``i, j ∈ 1..limit``：
         · 无条件抽一个 ``e ~ N(0, σ)``
         · ``t = ts + √(((i−x)/miux)² + ((j−y)/miuy)²) + e``
           （源点处距离为 0，退化成 ``t = ts + e``）
         · **只有当 t < time_max 时**才抽那个决定漏检的 ``U(0,1)``，
           并以概率 p 丢弃 —— 这个条件很关键，它决定了随机数流的消耗量
    2. 一条波前生成完后抽一次 ``Exp(mean=gap)`` 推进激发时刻（R:171）。
       默认 ``gap = 1e6``，而观测窗只有 96，所以实际上**只会有一条波前**。
    3. 噪声：时刻按 ``Exp(mean=1/λ_n)`` 的泊松流铺满观测窗（首尾各丢一个，
       R:180-184），位置在网格内均匀独立抽取（R:186-187）。
    4. 信号与噪声合并、按 ts 稳定排序。
    """
    if seed is None:
        seed = seed_circular(limit=limit, time_max=time_max, ts=ts, x=x, y=y,
                             u_x=u_x, u_y=u_y, p=p, noise_freq=noise_freq,
                             miu_x=miu_x, miu_y=miu_y, sigma=sigma,
                             seed_shift=seed_shift)
    rng = np.random.default_rng(seed)

    grid = np.arange(1, limit + 1, dtype=float)
    gi, gj = np.meshgrid(grid, grid, indexing="ij")     # i 行, j 列
    travel = np.sqrt(((gi - x) / miu_x) ** 2 + ((gj - y) / miu_y) ** 2)
    at_source = (gi == x) & (gj == y)

    sig_x: list[float] = []
    sig_y: list[float] = []
    sig_t: list[float] = []

    signal_start_ts = float(ts)
    while signal_start_ts < time_max:
        # ── 逐电极（R 的双重 for 循环，行优先：i 外层、j 内层）──────────
        for i in range(limit):
            for j in range(limit):
                e = float(rng.normal(0.0, sigma))
                arrival = signal_start_ts + (
                    e if at_source[i, j] else travel[i, j] + e)
                # ★ runif 只在 arrival < time_max 时才抽 —— 顺序不能改
                if arrival < time_max and rng.random() > p:
                    sig_x.append(gi[i, j])
                    sig_y.append(gj[i, j])
                    sig_t.append(arrival)
        # ── 推进到下一条波前的激发时刻（R:171）──────────────────────
        signal_start_ts += float(rng.exponential(gap))

    # ── 噪声（R:175-187）────────────────────────────────────────────────
    noise_t = [0.0]
    while noise_t[-1] < time_max:
        noise_t.append(noise_t[-1] + float(rng.exponential(1.0 / noise_freq)))
    noise_t = np.asarray(noise_t[1:-1], dtype=float) if len(noise_t) > 2 \
        else np.empty(0)
    n_noise = noise_t.size
    noise_x = rng.integers(1, limit + 1, size=n_noise).astype(float)
    noise_y = rng.integers(1, limit + 1, size=n_noise).astype(float)

    df = pd.DataFrame({
        "x": np.concatenate([np.asarray(sig_x, dtype=float), noise_x]),
        "y": np.concatenate([np.asarray(sig_y, dtype=float), noise_y]),
        "ts": np.concatenate([np.asarray(sig_t, dtype=float), noise_t]),
        "z": np.concatenate([np.ones(len(sig_x), dtype=int),
                             np.zeros(n_noise, dtype=int)]),
    })
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
#  估计与评估
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class WavefrontEstimate:
    """
    一次「仿真 + 拟合」的全部结果，对应 R 的 ``visualize2d`` 返回值（R:728-729）。

    ``n_signal`` / ``n_noise`` / ``mean_travel_time`` 是 R 在 ``hybrid.optim``
    之后追加的三列（R:673-674），用于画图的横轴。
    """

    x0: float
    y0: float
    v: float                 # 速度（已由慢度换算）
    t0: float
    slowness: float          # 1/v —— R 的 pp[[3]]
    n_signal: int
    n_noise: int
    mean_travel_time: float
    n_points: int
    n_iters: int
    loss: float
    r2: float
    truth: dict = field(default_factory=lambda: dict(CIRCULAR_TRUTH))

    # ★ ``r2`` 与 ``loss`` 都是对【信号 + 噪声全部点】算的，因为 R 就是这么做的：
    #   ``visualize2d`` 把整张 sim.matrix（含 z = 0 的噪声行）交给 hybrid.optim
    #   （R:663, 671），损失函数里的 1[t>0] 又把它们全留下。
    #
    #   后果：在 §3.1.2 的典型配置下噪声只占 ~0.9%（9216 信号 + 81 噪声），
    #   但它把 R² 从 1.000000 压到 0.953 —— 实测：
    #       真值参数下 loss(含噪声) = 8.37e4，loss(仅信号) = 9.23e-9
    #       真值参数下 R²(含噪声)   = 0.953330，R²(仅信号) = 1.0000000000
    #   所以这里的 R² ≈ 0.95 表示"模型是精确的、数据里混了噪声"，
    #   **不要**把它和论文 §3.2 表 4 那个 0.94 并列比较 —— 那是另一回事。

    @property
    def snr(self) -> float:
        """
        论文图 7 的横轴（R:788）。

        ★ 名字叫「信噪比」，实际算的是 **真信号点数 / 噪声点数**：

            snr = pp.estimates[i, 6] / pp.estimates[i, 7]

          在 ``accuracy_sweep`` 里，扫描值被 cbind 插到了最前面，所以
          第 6、7 列分别是 ``n_signal`` 与 ``n_noise``。
          这不是幅度意义上的信噪比，读图时要注意。

          在 §3.1.2 的配置下 n_signal = 96×96 = 9216（全部电极都收得到），
          n_noise ≈ λ_n × 96，所以 snr ≈ 96 / λ_n ——
          与 R 实际扫出的范围 [≈0.375, ≈768] 吻合。
        """
        return self.n_signal / self.n_noise if self.n_noise else float("inf")

    @property
    def abs_error(self) -> dict:
        """四个量的绝对误差（图 7/8 纵轴实际画的就是估计值本身，见下）。"""
        t = self.truth
        return {"x0": abs(self.x0 - t["x0"]), "y0": abs(self.y0 - t["y0"]),
                "v": abs(self.v - t["v"]), "t0": abs(self.t0 - t["t0"])}

    @property
    def rel_error(self) -> dict:
        """
        相对误差 —— R 里对应 ``relerror.plot`` 的 ``plot.value = FALSE`` 分支
        （R:796-801）：``|est − truth| / truth``。

        ★ 这个分支在 R 里【不可达】：三个活跃调用都传了 ``plot.value = TRUE``
          （R:891/911/931），所以 R 画的其实是**绝对估计值**，不是相对误差。
          函数名 ``relerror.plot`` 因此是名不副实的。这里两个都给出来。
        """
        t = self.truth
        return {"x0": abs(self.x0 - t["x0"]) / t["x0"],
                "y0": abs(self.y0 - t["y0"]) / t["y0"],
                "v": abs(self.v - t["v"]) / t["v"],
                "t0": abs(self.t0 - t["t0"]) / t["t0"]}

    def as_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "v": self.v, "t0": self.t0,
                "slowness": self.slowness, "n_signal": self.n_signal,
                "n_noise": self.n_noise,
                "mean_travel_time": self.mean_travel_time,
                "n_points": self.n_points, "n_iters": self.n_iters,
                "loss": self.loss, "r2": self.r2, "snr": self.snr}


def estimate_circular_wavefront(df: pd.DataFrame, *, truth: dict | None = None,
                                max_iter: int = HYBRID_MAX_ITER,
                                eps: float = CIRCULAR_EPS,
                                n_starts: int = 1) -> WavefrontEstimate:
    """
    对一次仿真结果跑圆波前拟合（对应 R 的 ``hybrid.optim(sim.matrix)``，R:671）。

    ★ 搜索窗是 **±500**（R:632），不是真数据管线的 ±50；收敛判据是 **1e-8**
      （R:635），比真数据管线的 1e-6 严得多。两者不能混用。
    """
    pts = df[["x", "y", "ts"]].to_numpy(dtype=float)
    trace: list = []            # 只用来数轮数，不参与计算
    p = hybrid_optim(pts, lower=SEARCH_LOWER_CIRCULAR,
                     upper=SEARCH_UPPER_CIRCULAR, max_iter=max_iter, eps=eps,
                     trace=trace)
    # 多起点（本实现的扩展，默认不用）：n_starts >= 2 时额外试一次数据驱动的初值。
    # 在 §3.1.2 里 R 的初值 (u, t₀) = (1, 1) 本来就几乎正中真值 (v = 1, t₀ = 2)，
    # 所以这里默认不需要它；保留这个开关是为了别的量级的数据。
    if n_starts >= 2:
        from .fit import centroid_initial_guess
        u0, t00 = centroid_initial_guess(pts)
        p2 = hybrid_optim(pts, lower=SEARCH_LOWER_CIRCULAR,
                          upper=SEARCH_UPPER_CIRCULAR, max_iter=max_iter,
                          eps=eps, init_u=u0, init_t0=t00)
        if circular_loss(p2, pts) < circular_loss(p, pts):
            p = p2

    sig = df["z"].to_numpy() == 1
    truth = dict(CIRCULAR_TRUTH if truth is None else truth)
    return WavefrontEstimate(
        x0=float(p[0]), y0=float(p[1]), v=float(1.0 / p[2]), t0=float(p[3]),
        slowness=float(p[2]),
        n_signal=int(sig.sum()), n_noise=int((~sig).sum()),
        mean_travel_time=float(df.loc[sig, "ts"].mean()) if sig.any() else np.nan,
        n_points=len(df), n_iters=len(trace), loss=circular_loss(p, pts),
        r2=r_squared(circular_loss, p, pts), truth=truth,
    )


def evaluate_circular(seed_shift: int = 0, *, params: dict | None = None,
                      **overrides) -> WavefrontEstimate:
    """
    跑一次完整的「仿真 + 拟合」—— 对应 R 的 ``visualize2d()``（R:652-730）。

    ``params`` 默认取 :data:`PAPER_CIRCULAR_PARAMS`，``overrides`` 覆盖其中若干项
    （扫描的时候就是这么用的）。
    """
    prm = dict(PAPER_CIRCULAR_PARAMS)
    if params:
        prm.update(params)
    prm.update(overrides)

    df = simulate_circular_wavefronts(seed_shift, **prm)
    return estimate_circular_wavefront(df)


def accuracy_sweep(plot_mode: int = 1, *, n_replicates: int | None = None,
                   independent_replicates: bool = True,
                   params: dict | None = None) -> pd.DataFrame:
    """
    按 ``plot.mode`` 做参数扫描 —— 对应 R 的 ``optimality.table()``（R:878-1010）。

    ``plot_mode``
        1 —— 扫噪声率 λ_n（图 7），固定 p = 0、σ = 1e-6（R:889-890）
        2 —— 扫漏检概率 p（图 8），固定 λ_n = 1、σ = 1e-6（R:908-909）
        3 —— 扫测量误差 σ，固定 λ_n = 1、p = 0（R:924-925）

    ``n_replicates``
        每个网格点重复几次。默认取 R 提交时的重复数（mode 1 → 23、
        mode 2 → 9、mode 3 → 10）。

    ``independent_replicates``
        ★★ 这是本实现与 R 的**一处有意差异**，必须理解清楚。

        R 的 ``stomach.sim2d`` 里种子是 ``100·Π(参数)``，**没有 seed.shift**，
        而 ``optimality.table`` 又是对同一组参数反复调用 —— 所以 R 的「重复」
        根本不是重复，而是**同一份数据集**。更糟的是当 **p = 0** 时种子恒为 0
        （乘积里有一项是 p），于是 mode 1 与 mode 3 的整批点共享同一条随机数流：
        每个电极的测量误差 ε 都是同一个序列，只是整体乘了 σ。
        后果是这两个模式的曲线**异常平滑**，误差棒不代表重复间的波动。

        默认 ``True`` —— 给每个重复加一个不同的 ``seed_shift``，得到真正独立的
        重复。想要复现 R 的行为（看那个平滑效应）就置 ``False``。
    """
    if plot_mode not in PLOT_MODES:
        raise ValueError(f"plot_mode 必须是 {sorted(PLOT_MODES)} 之一")
    spec = PLOT_MODES[plot_mode]
    grid = spec["grid"]
    n_rep = n_replicates if n_replicates is not None else {
        1: 23, 2: 9, 3: 10}[plot_mode]

    # 每个模式扫一个量、固定另外两个（R:889-890 / 908-909 / 924-925）
    var = spec["variable"]
    param = spec["param"]
    fixed = {"snr": {"p": 0.0, "sigma": 1e-6},
             "p": {"noise_freq": 1.0, "sigma": 1e-6},
             "sigma": {"noise_freq": 1.0, "p": 0.0}}[var]

    rows = []
    for value in grid:
        for rep in range(n_rep):
            over = dict(fixed)
            over[param] = float(value)
            shift = (rep + 1) if independent_replicates else 0
            est = evaluate_circular(shift, params=params, **over)
            rows.append({
                "plot_mode": plot_mode,
                "variable": var,
                "scan_value": float(value),
                "replicate": rep,
                **est.as_dict(),
                **{f"abs_{k}": v for k, v in est.abs_error.items()},
                **{f"rel_{k}": v for k, v in est.rel_error.items()},
            })
    return pd.DataFrame(rows)
