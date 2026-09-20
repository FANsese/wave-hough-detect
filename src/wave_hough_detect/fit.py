"""
阶段 3：波前模型拟合（论文 §2.3 / Algorithm 3 + 线性模型）

对应 R 代码：R_Codes/cm_hough_grid_8.R 第 540-779 行

【两个模型】
  圆波前（近场）  t = √((x-x₀)²+(y-y₀)²)/v + t₀      —— 交替最小化
  线波前（远场）  t = ãx + b̃y + c̃                    —— 闭式最小二乘

【移植时最容易踩的坑】
  ★ u = 1/v 槽位：R 的 vt.optim 把【斜率】(=1/v) 存回名为 v 的位置（p[[3]]），
    并且目标函数用 cone.model.inverted（乘 v 而不是除）来配平。
    本模块用 u 作变量名，但数值行为与 R 完全对齐。
  ★ 尺度：拟合用的 ts 必须乘回 200（R 的 line 528），即用原始 ms 尺度。
    霍夫阶段用的 /200 尺度只服务于「内点容忍度 0.1」。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ── 与 R 一致的参数 ────────────────────────────────────────────────────────
GRID_SIDE = 8
SEARCH_LOWER = np.array([-50.0, -50.0])   # hybrid.optim 的搜索窗（x₀, y₀）
SEARCH_UPPER = np.array([50.0, 50.0])
GRID_SPLITS = 4                            # grid.optim 的 num.splits
GRID_EPS = 1e-8                            # grid.optim 的收敛判据
GRID_MAX_ITER = 1000
HYBRID_EPS = 1e-6                          # hybrid.optim 的收敛判据
HYBRID_MAX_ITER = 10000
TIME_SCALE_BACK = 200.0                    # R line 528 的 *200


# ══════════════════════════════════════════════════════════════════════════
#  坐标网格（对应 R 第 536-538 行）
# ══════════════════════════════════════════════════════════════════════════

def electrode_grid(side: int = GRID_SIDE):
    """
    R:  x.grid = array(1:LIMIT, c(LIMIT, LIMIT));  y.grid = t(x.grid)

    R 的 array() 按列填充，所以 x.grid[i,j] = i+1（行号），
    y.grid = t(x.grid) 转置后 y.grid[i,j] = j+1（列号）。
    即 (x, y) = (行, 列)，与通道映射一致。
    """
    xs = np.arange(1, side + 1)
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")   # x_grid[i,j]=i+1
    return x_grid.astype(float), y_grid.astype(float)


def build_result_array(spike_x, spike_y, spike_ts_ms, plane_indices,
                       n_planes: int | None = None):
    """
    对应 R 第 521-533 行：把每个平面在每个电极上的激活时刻装进 (8,8,K) 数组。

    要点：
      · R line 528 把 ts 乘回 200（回到原始 ms 尺度）—— 拟合必须用这个尺度
      · unclassified 的点（plane_indices == 0）跳过
      · 同一 (x, y) 在同一平面出现多次时，【后写入的覆盖先写入的】（R 的同名赋值）
      · 空槽保持 -1（R 用 array(-1, ...) 初始化）
    """
    if n_planes is None:
        n_planes = int(np.max(plane_indices)) if len(plane_indices) else 0

    arr = np.full((GRID_SIDE, GRID_SIDE, n_planes), -1.0)
    for x, y, ts, pid in zip(spike_x, spike_y, spike_ts_ms, plane_indices):
        if pid == 0:
            continue
        arr[int(x) - 1, int(y) - 1, int(pid) - 1] = ts * TIME_SCALE_BACK
    return arr


def plane_points(arr, plane_index: int) -> np.ndarray:
    """
    对应 R 第 727-731 行：取出某个平面的 (x, y, ts) 点集，只保留 ts > 0 的。

    R 的取法是 as.vector(x.grid) / as.vector(y.grid) / as.vector(result.array[,,i])，
    即按列优先顺序展开整个 8x8，再按 ts > 0 过滤。
    """
    x_grid, y_grid = electrode_grid()
    ts = arr[:, :, plane_index - 1]
    # R 的 as.vector 按列优先；numpy 默认 C 序（行优先）
    xs = x_grid.flatten(order="F")
    ys = y_grid.flatten(order="F")
    ts_flat = ts.flatten(order="F")
    keep = ts_flat > 0
    return np.column_stack([xs[keep], ys[keep], ts_flat[keep]])


# ══════════════════════════════════════════════════════════════════════════
#  损失函数（对应 R 第 540-575, 744-749 行）
# ══════════════════════════════════════════════════════════════════════════

def circular_loss(p, points: np.ndarray) -> float:
    """
    圆波前损失 ℓ_circular —— 对应 R 的 cone.model.inverted.all()。

        ℓ = Σ [ √((xᵢ-x₀)²+(yᵢ-y₀)²)·u − (tᵢ−t₀) ]² · 1[tᵢ>0]

    ★ 注意 R 里 p[[3]] 存的是 u = 1/v，所以写成【乘 u】。
      论文式 (3) 写的是【除 v】，两者等价，但代码里是乘。
    """
    x0, y0, u, t0 = p
    x = points[:, 0]; y = points[:, 1]; t = points[:, 2]
    resid = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) * u - (t - t0)
    return float(np.sum(resid ** 2 * (t > 0)))


def circular_loss_xy(xy, points: np.ndarray, u: float, t0: float) -> float:
    """
    固定 (u, t0)，只对 (x₀, y₀) 求值的包装 —— 对应 R 的 cone.model.inverted.all.xy。
    这是 grid.optim 每轮调用的回调。
    """
    return circular_loss([xy[0], xy[1], u, t0], points)


def linear_loss(p, points: np.ndarray) -> float:
    """
    线波前损失 ℓ_linear —— 对应 R 的 plane.model()。

        ℓ = Σ [ (ãxᵢ + b̃yᵢ + c̃) − tᵢ ]² · 1[tᵢ>0]
    """
    a, b, c = p
    x = points[:, 0]; y = points[:, 1]; t = points[:, 2]
    resid = a * x + b * y + c - t
    return float(np.sum(resid ** 2 * (t > 0)))


def r_squared(loss_fn: Callable, params, points: np.ndarray) -> float:
    """
    决定系数 —— 对应 R 的 compute.loss()。

        R² = 1 − SS_res / SS_tot,   SS_tot = Σ(tᵢ − t̄)²

    ★ SS_tot 在 R 里是对【全部】传入的点算的，不额外乘 1[t>0]。
      因为调用前已经过滤过 ts > 0，两者等价。
    """
    ss_res = loss_fn(params, points)
    t = points[:, 2]
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


# ══════════════════════════════════════════════════════════════════════════
#  4x4 网格制表法（对应 R 第 578-642 行）
# ══════════════════════════════════════════════════════════════════════════

def _find_best_point_1d(dim: int, current: np.ndarray, n_splits: int,
                        lower: np.ndarray, upper: np.ndarray,
                        callback: Callable):
    """
    对应 R 的 grid.optim.find.best.point.1d()：沿第 dim 维递归枚举格点。

    R 的循环是 for (i in 0:num.spaces)，num.spaces = num_splits-1 = 3，
    即每维取 4 个点（含两端）：lower, lower+span/3, lower+2span/3, upper。
    """
    n_spaces = n_splits - 1
    best_val = 1e100
    best_pt = current.copy()

    for i in range(n_spaces + 1):                      # R: 0:num.spaces
        current[dim] = lower[dim] + (upper[dim] - lower[dim]) / n_spaces * i
        if dim < len(current) - 1:
            val, pt = _find_best_point_1d(dim + 1, current, n_splits,
                                          lower, upper, callback)
        else:
            val, pt = callback(current), current.copy()
        if val < best_val:
            best_val = val
            best_pt = pt.copy()
    return best_val, best_pt


def grid_optim(lower: np.ndarray, upper: np.ndarray,
               callback: Callable, max_iter: int = GRID_MAX_ITER,
               eps: float = GRID_EPS):
    """
    4x4 网格制表法 —— 对应 R 的 grid.optim()。

    每轮：
      1. 在 [lower, upper] 上布 4^n 个格点，取最小的
      2. 若最优点落在边界上，向内挪一格（★ R 的钳制规则）
      3. 把搜索窗收缩到最优点 ± 一格间距
      4. 窗口宽度 < eps 时停止，返回窗口中点
    """
    n_splits = GRID_SPLITS
    n_spaces = n_splits - 1
    lower = np.asarray(lower, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()

    for _ in range(max_iter):
        val, best = _find_best_point_1d(0, lower.copy(), n_splits,
                                        lower, upper, callback)
        new_space = (upper - lower) / n_spaces

        # ★ 边界钳制（R 第 625-632 行）：
        #   如果最优格点正好落在窗口边界上，说明真极值可能在窗外，
        #   把它向内挪一格，避免窗口持续向外漂移。
        for k in range(len(best)):
            if abs(best[k] - lower[k]) < 1e-12:
                best[k] = lower[k] + new_space[k]
            if abs(best[k] - upper[k]) < 1e-12:
                best[k] = upper[k] - new_space[k]

        lower = best - new_space
        upper = best + new_space

        # R 用 normv(new.space) < epsilon，注意判的是【格距】不是【位移】
        if float(np.linalg.norm(new_space)) < eps:
            break

    return (lower + upper) / 2.0


# ══════════════════════════════════════════════════════════════════════════
#  交替最小化的另一半步：u = 1/v 的线性回归（对应 R 第 656-666 行）
# ══════════════════════════════════════════════════════════════════════════

def vt_optim(p, points: np.ndarray) -> np.ndarray:
    """
    固定 (x₀, y₀)，用最小二乘求 (u, t₀) —— 对应 R 的 vt.optim.all()。

    把 t 看作距离 d 的线性函数：  t = u·d + t₀
      → 斜率 u = 1/v
      → 截距 t₀

    ★ 返回值把【斜率 u】放进 p[2]，与 R 的 p[[3]] 语义一致。
      调用方需要 v 时用 1/p[2]。
    """
    x0, y0 = p[0], p[1]
    t = points[:, 2]
    d = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2)
    keep = t > 0
    d, t = d[keep], t[keep]

    # 对应 R 的 lm(y ~ x)：设计矩阵 [d, 1]
    design = np.column_stack([d, np.ones(len(d))])
    coef, *_ = np.linalg.lstsq(design, t, rcond=None)

    out = np.array(p, dtype=float)
    out[3] = coef[1]     # 截距 -> t₀
    out[2] = coef[0]     # 斜率 -> u = 1/v
    return out


def hybrid_optim(points: np.ndarray,
                 lower: np.ndarray = SEARCH_LOWER,
                 upper: np.ndarray = SEARCH_UPPER,
                 max_iter: int = HYBRID_MAX_ITER,
                 eps: float = HYBRID_EPS,
                 trace: list | None = None,
                 init_u: float = 1.0, init_t0: float = 1.0) -> np.ndarray:
    """
    交替最小化主循环 —— 对应 R 的 hybrid.optim()（论文 Algorithm 3）。

    每轮：
      ① 固定 (u, t₀)，用 4x4 网格制表法求 (x₀, y₀)
      ② 固定 (x₀, y₀)，用最小二乘求 (u, t₀)
      ③ 相对变化 < eps 时停止

    ★ R 的初值是 u = 1、t₀ = 1（代码里写作 ``v = 1; t = 1``，槽位里其实是慢度 u），
      pp.old = c(1,1,1,1) —— 一个与数据完全无关的向量，所以【第一轮】的相对变化是
      相对 (1,1,1,1) 算的。默认值如实复现。
      ``init_u`` / ``init_t0`` 是本实现新增的开关，默认值与 R 相同；
      改动它们就【不再】是与 R 逐位可比的设置，只用于源点落在网格内等
      R 初值会卡住的情形（见 fit_circular 的 n_starts）。

    ★★ 已知的收敛脆弱性（继承自 R，不是移植引入的）：
      交替最小化从 (1,1) 出发时，若源点在网格【内部】，第一轮网格搜索会被推向
      搜索窗边界并且再也爬不回来。实测：
          (x0,y0,v,t0) = (2.5, 3.5, 0.45, 100)   → 收敛到 (−44.7, 14.6)，跑满 10000 轮
          (x0,y0,v,t0) = (−3.0, −50, 0.45, 800)  → 329 轮收敛到真值 ✅
      论文的情形（源点远在网格之外 50 格）属于后者，所以论文的数字是对的。
    """
    u, t0 = float(init_u), float(init_t0)
    p_old = np.array([1.0, 1.0, 1.0, 1.0])  # R: pp.old = c(1,1,1,1)

    p = np.array([0.0, 0.0, u, t0])
    for it in range(1, max_iter + 1):
        # ① (x₀, y₀)
        xy = grid_optim(lower, upper,
                        lambda xy_: circular_loss_xy(xy_, points, u, t0))
        p = np.array([xy[0], xy[1], u, t0])

        # ② (u, t₀)
        p = vt_optim(p, points)
        u, t0 = p[2], p[3]

        if trace is not None:
            trace.append(p.copy())

        reldiff = float(np.linalg.norm(p - p_old) / np.linalg.norm(p_old))
        if reldiff < eps:
            break
        p_old = p.copy()

    return p


# ══════════════════════════════════════════════════════════════════════════
#  便利封装
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CircularFit:
    x0: float
    y0: float
    v: float          # 已换算成速度（不是 u）
    t0: float
    r2: float
    loss: float
    n_points: int
    n_iters: int


@dataclass
class LinearFit:
    a: float          # ã
    b: float          # b̃
    v: float          # 1/√(ã²+b̃²)
    r2: float
    loss: float
    n_points: int


def centroid_initial_guess(points: np.ndarray) -> tuple[float, float]:
    """
    数据驱动的初值：以点集的 (x, y) 质心作临时中心，对 (r, t) 做最小二乘，
    得到慢度 u 与激发时刻 t₀ 的初值。

    这是本实现的【新增】内容，R 里没有。它解决的问题是：
    R 的固定初值 (u, t₀) = (1, 1) 在源点位于网格内部时会收敛到错误的局部极小。
    反过来，这个质心初值在源点远在网格之外时很差（质心离真实源点太远），
    所以两者谁也不该单独当默认值 —— 见 ``fit_circular`` 的 ``n_starts``。
    """
    x, y, t = points[:, 0], points[:, 1], points[:, 2]
    r = np.hypot(x - x.mean(), y - y.mean())
    design = np.column_stack([r, np.ones_like(r)])
    u0, t0 = np.linalg.lstsq(design, t, rcond=None)[0]
    return float(u0), float(t0)


def fit_circular(points: np.ndarray, trace: list | None = None,
                 n_starts: int = 1, **kw) -> CircularFit:
    """
    对一个平面的 (x,y,t) 点集拟合圆波前模型。

    ``n_starts``
        1（默认）—— 只用 R 的初值 (u, t₀) = (1, 1)。
                    **复现论文表 2 / 表 4 必须用这个值。**
        2        —— 再试一次 :func:`centroid_initial_guess` 给出的初值，
                    取最终损失更小的那个解。
                    用于源点落在网格内部等 R 初值会卡住的数据
                    （实测把 (2.5, 3.5, 0.45, 100) 从完全错误救回精确解）。
    无论取值如何，成功收敛的那条路径本身仍是 R 的交替最小化，
    只是起点不同。
    """
    tr = trace if trace is not None else []
    inits = [(1.0, 1.0)]
    if n_starts >= 2:
        inits.append(centroid_initial_guess(points))

    best_p, best_loss, best_iters = None, np.inf, 0
    for k, (u0, t0_0) in enumerate(inits):
        sub = tr if (k == 0 and n_starts == 1) else []
        p = hybrid_optim(points, trace=sub, init_u=u0, init_t0=t0_0, **kw)
        loss = circular_loss(p, points)
        if loss < best_loss:
            best_p, best_loss, best_iters = p, loss, len(sub)
    if n_starts > 1:
        tr.clear()                       # 多起点时逐条 trace 没有单一含义
    p = best_p
    return CircularFit(
        x0=float(p[0]), y0=float(p[1]), v=float(1.0 / p[2]), t0=float(p[3]),
        r2=r_squared(circular_loss, p, points), loss=best_loss,
        n_points=len(points), n_iters=best_iters,
    )


def fit_linear(points: np.ndarray) -> LinearFit:
    """
    线波前模型闭式解 —— 对应 R 第 769-776 行。

    R:  lm(ts.observ ~ x.observ + y.observ)
        a = coef[[2]] (x 的系数), b = coef[[3]] (y 的系数)
        v = 1/sqrt(a^2 + b^2)
    """
    x = points[:, 0]; y = points[:, 1]; t = points[:, 2]
    design = np.column_stack([x, y, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(design, t, rcond=None)
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    v = 1.0 / np.sqrt(a * a + b * b)
    params = np.array([a, b, c])
    return LinearFit(a=a, b=b, v=float(v),
                     r2=r_squared(linear_loss, params, points),
                     loss=linear_loss(params, points),
                     n_points=len(points))
