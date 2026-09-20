"""
阶段 2：随机霍夫变换（论文 §2.2 / Algorithm 2 & 4）

对应 R 代码：R_Codes/cm_hough_grid_8.R 第 197-453 行

【移植原则】
本模块严格保持与 R 代码相同的逻辑与数据处理顺序。所有与 R 的对应关系
都标了行号。三处已知的 R 缺陷（符号规范化失效、θ 病态、空平面崩溃）
**保持原样**，只在注释里说明；需要修正时用开关显式打开。

【与 R 的数值差异说明】
唯一的差异来源是随机数发生器。R 的 sample() 用 R 自己的 Mersenne-Twister
与 Rejection 采样算法，numpy 无法复现其序列。为此本模块支持传入
``trace``：一串预先指定的三元组下标，用于与 R 逐位比对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# ── 与 R 代码一致的固定参数 ────────────────────────────────────────────────
# 下面这组默认值对应【真数据管线】cm_hough_grid_8.R（论文 §3.2，表 2/3/4）。
RHO_STEP = 0.05          # get.key 里 rho 的量化步长
PHI_STEP = 2.0           # phi 的量化步长（度）
THETA_STEP = 2.0         # theta 的量化步长（度）
INLIER_TOL = 0.1         # is.point.on.plane 的容忍度
ANGLE_TOL_DEG = 5.0      # Algorithm 4 的角度容差（度）
SMALL_PLANE_THRESHOLD = 30   # R line 407：尾部扩展平面时的最少点数

# ── 仿真研究（§3.1）用的另一组参数 ─────────────────────────────────────────
# ★ 论文的实验代码其实有两份 hough.plane 拷贝，参数【不一样】：
#     · 真数据管线  cm_hough_grid_8.R            ρ 步长 0.05、容忍度 0.1
#     · 仿真研究    fig5_detection_performance.R ρ 步长 0.50、容忍度 1.0
#   差 10 倍的原因是 ts 尺度：真数据的 ts 已经除以了 200。
#   把两份参数混用不会报任何错，只会静默地一个平面都找不到（或把平面糊在一起）。
SIM_PRESET = {
    "rho_step": 0.5,
    "inlier_tol": 1.0,
    "strict_degenerate_check": True,
}


# ══════════════════════════════════════════════════════════════════════════
#  工具函数（对应 R 第 197-244 行）
# ══════════════════════════════════════════════════════════════════════════

def cross_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """对应 R 的 cross.product()。"""
    return np.array([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])


def _step_key(x: float, step: float) -> float:
    """
    对应 R 的 get.step.key()：``as.integer(x / step) * step``。

    ★ R 的 as.integer() 是【向零取整】，不是 floor。
      对负数：as.integer(-4.81374 / 0.05) = as.integer(-96.2748) = -96
      而 np.floor(-96.2748) = -97。两者差 1 个步长！
      这里必须用 trunc 而非 floor，否则 rho 为负的平面会落到错误的桶。

    ★ 再一个细节：R 的 as.integer() 返回【整数】，所以 -0.0 会变成 0L，
      拼进键里是 "0"；而 Python 的 -0.0 会用 :g 格式化成 "-0"。
      虽然 "-0" 与 "0" 各自内部一致、分组结果相同，但键字符串会与 R 不同。
      这里把 -0.0 归一成 0.0，使键与 R 逐字符可比。
    """
    q = float(np.trunc(x / step) * step)
    return 0.0 if q == 0 else q


def get_key(rho: float, phi: float, theta: float,
            rho_step: float = RHO_STEP) -> str:
    """
    对应 R 的 get.key()：把连续的 (ρ, φ, θ) 量化成字符串键。

    R 里 phi/theta 先由弧度换成度再量化。
    键的格式是 "rho,phi,theta"，例如 "9.9,84,14"。

    ★ rho_step 是【必须可调】的：真数据管线用 0.05，而仿真的那份 R 拷贝
      （fig5_detection_performance.R:401）用的是 0.5 —— 差 10 倍。
      原因是两者的 ts 尺度差 10 倍：真数据的 ts 先除以了 200。
      用错步长不会报错，只会让累加器散架、一个平面都找不到。
    """
    rho_q = _step_key(rho, rho_step)
    phi_q = _step_key(np.degrees(phi), PHI_STEP)
    theta_q = _step_key(np.degrees(theta), THETA_STEP)
    return f"{rho_q:g},{phi_q:g},{theta_q:g}"


def spherical_key(normal: np.ndarray, rho_signed: float,
                  rho_step: float = RHO_STEP) -> str:
    """
    由已接受的平面法向量反算 (ρ, φ, θ) 并给出累加器键。

    对应 R 里重复出现三次的同一段代码（cm_hough_grid_8.R:289-292、:377-378、
    :415-418）：``rho = abs(rho.signed); phi = acos(n3); theta = asin(n2/sin(phi))``。

    ★ φ = 0（法向量恰好平行于 t 轴）时 R 的 ``asin(n2/sin(0))`` 得到 NaN，
      键里会带 "NaN"，但**仍然会继续做内点判定与赋值** —— 键只用于标记，
      不参与任何几何判断。这里取 θ = 0 生成一个确定的键，使 R 的可观察行为
      （点照常被收编）保持一致，同时避免 NaN 进入字符串。
      真数据上 φ 最小也接近 0.8°，这条分支从未触发。
    """
    phi = float(np.arccos(np.clip(normal[2], -1.0, 1.0)))
    if phi == 0.0:
        theta = 0.0
    else:
        theta = float(np.arcsin(np.clip(normal[1] / np.sin(phi), -1.0, 1.0)))
    return get_key(abs(rho_signed), phi, theta, rho_step)


def is_point_on_plane(point: np.ndarray, normal: np.ndarray, rho: float,
                      tol: float = INLIER_TOL) -> bool:
    """
    对应 R 的 is.point.on.plane()：``|n·p − ρ| < tol``。

    ★ tol 必须可调：真数据用 0.1（cm_hough_grid_8.R:212），
      仿真的那份拷贝用 1（fig5_detection_performance.R:410）。
      同样是 ts 尺度差 10 倍导致的。
    """
    return abs(float(np.dot(point, normal)) - rho) < tol


def find_points_on_plane(points: np.ndarray,
                         normal: np.ndarray,
                         rho: float,
                         indices: np.ndarray,
                         tol: float = INLIER_TOL) -> np.ndarray:
    """
    对应 R 的 find.points.on.plane()。

    R 里按 unclassify.indices 的顺序遍历并把命中的下标 append 到结果，
    所以返回顺序与原下标顺序一致。
    """
    hits = [int(i) for i in indices
            if is_point_on_plane(points[i], normal, rho, tol)]
    return np.asarray(hits, dtype=int)


def num_unique_detectors(points: np.ndarray, indices: np.ndarray) -> int:
    """
    对应 R 的 num.unique.detectors()：统计覆盖了多少个【不同电极】。
    电极由 (x, y) 唯一确定，与时间无关。
    """
    if indices.size == 0:
        return 0
    xy = points[indices][:, :2]
    return int(np.unique(xy, axis=0).shape[0])


def r_mode(values: np.ndarray) -> int:
    """
    对应 R 的 mode()：
        as.integer(names(sort(-table(v))[1]))

    ★ 必须复现 R 的并列处理：table() 的表名按【字符串】排序，
      sort() 是稳定排序，所以并列时【字符串序最小的】胜出。
      对平面下标 1..9 而言字符串序 = 数值序，但 10 以上会不同
      （"10" < "2"），这里如实复现。
    """
    vals, counts = np.unique(values, return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: str(vals[i]))
    vals = vals[order]
    counts = counts[order]
    return int(vals[int(np.argmax(counts))])


# ══════════════════════════════════════════════════════════════════════════
#  结果容器
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class HoughResult:
    """对应 R 返回的 data.frame(prediction, keys, n1, n2, n3, plane.indices, rhos)。"""

    prediction: np.ndarray        # 0=噪声, 1=信号
    keys: np.ndarray              # 每个点所属的累加器键
    n1: np.ndarray                # 所属平面的法向量分量
    n2: np.ndarray
    n3: np.ndarray
    plane_indices: np.ndarray     # 所属平面编号，0 表示未归类
    rhos: np.ndarray
    accept_log: list = field(default_factory=list)   # 诊断用：每次接受平面的记录
    n_iter_used: int = 0

    @property
    def n_planes(self) -> int:
        return int(len(np.unique(self.plane_indices[self.plane_indices > 0])))

    def summary(self) -> str:
        lines = [f"找到平面数: {self.n_planes}",
                 f"判为信号的点: {int(self.prediction.sum())} / {len(self.prediction)}",
                 f"实际迭代次数: {self.n_iter_used}"]
        if self.n_planes:
            ids, cnt = np.unique(self.plane_indices[self.plane_indices > 0],
                                 return_counts=True)
            lines.append("各平面点数: " + "  ".join(
                f"P{i}={c}" for i, c in zip(ids, cnt)))
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  主函数（对应 R 第 248-453 行）
# ══════════════════════════════════════════════════════════════════════════

def hough_plane(points: np.ndarray,
                vote_threshold: int = 8,
                max_iter: int = 30000,
                min_detectors: int = 40,
                trace: Sequence[Sequence[int]] | None = None,
                seed: int | None = None,
                apply_angle_rule: bool = True,
                regrow_master_plane: bool = True,
                small_plane_threshold: int = SMALL_PLANE_THRESHOLD,
                rho_step: float = RHO_STEP,
                inlier_tol: float = INLIER_TOL,
                strict_degenerate_check: bool = False) -> HoughResult:
    """
    对应 R 的 hough.plane()。

    参数
    ----
    points : (N, 3) 数组，列为 (x, y, ts)，与 R 的 data.points 一致
    vote_threshold : R 的 threshold，默认 8。注意 R 的判断是
        ``length(accumulator[[key]]) > threshold * 3``，每次投票追加 3 个下标，
        所以实际触发需要【9 票】（threshold+1）。这里如实复现。
    max_iter : R 的 iter.max。R 默认 30000，但实测需要 ≥200000 才能找齐 5 个平面。
    min_detectors : R 硬编码的 40（约为 64 个电极的 2/3）
    trace : 可选。预先指定的抽样下标序列，用于与 R 逐位比对。
            给了 trace 就完全不用随机数发生器。
    seed : trace 为 None 时使用的 numpy 随机种子。
    apply_angle_rule : 是否执行 Algorithm 4 的 5° 单源容差
    regrow_master_plane : 是否执行 R 第 406-450 行的【尾部扩展】——
        主循环与 Algorithm 4 之后，再用主法向量去过一遍剩下的未分类点：
        若某个未分类点所在的同法向量平面能覆盖 > small_plane_threshold 个点，
        就把这些点收成一个新平面。
        ★ 论文正文没写这一步，但它确实在发表的代码里（cm_hough_grid_8.R:406-448）。
          真数据上它必然执行，但循环体一次都不进（320 点全部分完，未分类集为空），
          所以不加时真数据结果看着"完全正确"。只有在存在未分类点时两者才分道扬镳，
          仿真研究（§3.1）正好是这种情况 —— 那里 FNR≠0。
    small_plane_threshold : R 的 small.plane.threshold，默认 30
    rho_step, inlier_tol : 量化步长与内点容忍度。默认值对应【真数据管线】
        （0.05 / 0.1）。仿真研究用的那份 R 拷贝是 0.5 / 1.0，见 SIM_PRESET。
    strict_degenerate_check : R 的两份拷贝在"退化法向量"的判据上不一致 ——
        真数据版只查 (n2,n3) 与 (n1,n3) 两对（cm_hough_grid_8.R:284-285），
        仿真版查全部三对（fig5_detection_performance.R:473-475）。
        默认 False = 跟真数据版一致；仿真研究请置 True。

    返回
    ----
    HoughResult
    """
    n = points.shape[0]
    rng = np.random.default_rng(seed)

    # ── 初始化（R 249-265 行）──────────────────────────────────────────
    prediction = np.zeros(n, dtype=int)
    keys = np.array([""] * n, dtype=object)
    n1 = np.zeros(n); n2 = np.zeros(n); n3 = np.zeros(n)
    rhos = np.zeros(n)
    plane_indices = np.zeros(n, dtype=int)

    n1_by_plane: list[float] = []
    n2_by_plane: list[float] = []
    n3_by_plane: list[float] = []
    rho_by_plane: list[float] = []
    plane_index = 1

    # 对应 R 的 hash()：键 -> 被抽中过的下标列表
    accumulator: dict[str, list[int]] = {}
    accept_log: list[dict] = []

    trace_pos = 0
    iter_used = 0

    # ── 主循环（R 266-349 行）──────────────────────────────────────────
    for it in range(1, max_iter + 1):
        iter_used = it
        unclassified = np.flatnonzero(prediction == 0)
        if unclassified.size < 3:            # R 268
            break

        # ── 抽样 3 个点（R 269）────────────────────────────────────────
        if trace is not None:
            if trace_pos >= len(trace):
                break
            sample_idx = list(trace[trace_pos])
            trace_pos += 1
        else:
            # ★ R 的 sample(x, 3) 是【无放回】抽 3 个不同下标
            sample_idx = rng.choice(unclassified, size=3, replace=False).tolist()

        v1 = points[sample_idx[0]]
        v2 = points[sample_idx[1]]
        v3 = points[sample_idx[2]]

        # ── 叉积求法向量（R 275-277）───────────────────────────────────
        nrm_vec = cross_product(v2 - v1, v3 - v1)
        norm_n = float(np.linalg.norm(nrm_vec))
        if norm_n == 0:
            continue                          # R 276：三点共线
        nrm_vec = nrm_vec / norm_n

        # ── 符号规范化（R 279-281）─────────────────────────────────────
        # 强制 n1 >= 0，消除 (n,rho) 与 (-n,-rho) 的二义性。
        # ★ 已知缺陷：当 n1 ≈ 0（法向量接近 t 轴）时，n1 的正负由噪声决定，
        #   同一个平面会被表示成 phi≈0 和 phi≈180 两个极端。
        #   这份数据恰好命中该情况，是迭代次数偏高的主因之一。
        if nrm_vec[0] != 0:
            nrm_vec = nrm_vec * np.sign(nrm_vec[0])

        # ── 退化情形剔除（R 284-288）──────────────────────────────────
        degenerate = (nrm_vec[1] == 0 and nrm_vec[2] == 0) or \
                     (nrm_vec[0] == 0 and nrm_vec[2] == 0)
        if strict_degenerate_check:                   # 仿真版多查一对
            degenerate = degenerate or (nrm_vec[0] == 0 and nrm_vec[1] == 0)
        if degenerate:
            continue
        if float(np.dot(nrm_vec, np.array([0.0, 0.0, 1.0]))) == 0:
            continue

        # ── 球坐标（R 289-292）─────────────────────────────────────────
        rho_signed = float(np.dot(nrm_vec, v1))
        rho = abs(rho_signed)
        phi = float(np.arccos(np.clip(nrm_vec[2], -1.0, 1.0)))
        if phi == 0:
            continue                          # R 291
        theta = float(np.arcsin(np.clip(nrm_vec[1] / np.sin(phi), -1.0, 1.0)))

        # ── 投票（R 294-300）──────────────────────────────────────────
        key = get_key(rho, phi, theta, rho_step)
        accumulator.setdefault(key, []).extend(sample_idx)

        # ── 票数达标？R 是 > threshold*3，即 ≥9 票（R 301）─────────────
        if len(accumulator[key]) > vote_threshold * 3:

            # 最小二乘精修：用桶里全部点重新拟合平面（R 305-311）
            # ★ 注意 R 的列序：data.points = (x, y, ts)
            #   fit.input = cbind(pts[idx, 2:3], 1) = [y, ts, 1]
            #   响应     = pts[idx, 1]              = x
            #   即拟合 x ~ y + ts，然后 n.fit = c(1, -a, -b)
            bucket = np.asarray(accumulator[key], dtype=int)
            design = np.column_stack([points[bucket, 1],
                                      points[bucket, 2],
                                      np.ones(bucket.size)])
            # 对应 lm.fit(design, points[bucket, 0])
            coef, *_ = np.linalg.lstsq(design, points[bucket, 0], rcond=None)
            n_fit = np.array([1.0, -coef[0], -coef[1]])
            rho_fit = float(coef[2])
            norm_fit = float(np.linalg.norm(n_fit))
            n_fit = n_fit / norm_fit
            rho_fit = rho_fit / norm_fit

            # 找内点（R 317-318）
            inliers = find_points_on_plane(points, n_fit, rho_fit, unclassified,
                                           inlier_tol)

            # 探测器覆盖判据（R 319-327）
            n_det = num_unique_detectors(points, inliers)
            if n_det < min_detectors:
                accumulator[key] = []          # R 325：桶清零后 next
                continue

            # ── 接受这个平面（R 328-348）───────────────────────────────
            keys[inliers] = key
            prediction[inliers] = 1
            n1[inliers] = n_fit[0]
            n2[inliers] = n_fit[1]
            n3[inliers] = n_fit[2]
            rhos[inliers] = rho_fit
            plane_indices[inliers] = plane_index

            n1_by_plane.append(float(n_fit[0]))
            n2_by_plane.append(float(n_fit[1]))
            n3_by_plane.append(float(n_fit[2]))
            rho_by_plane.append(rho_fit)

            accept_log.append({
                "iter": it, "plane": plane_index,
                "votes": len(bucket) // 3, "detectors": n_det,
                "normal": n_fit.copy(), "rho": rho_fit,
            })

            plane_index += 1
            accumulator = {}                   # R 348：清空累加器

    # ── 防空守卫 ────────────────────────────────────────────────────────
    # ★ R 代码在这里没有防空检查：若一个平面都没找到，
    #   n1.by.plane.index 长度为 0，`for (i in 1:length(...))` 退化成
    #   1:0 = c(1,0)，索引越界产生 NA，if() 直接报错。
    #   这里直接返回空结果。
    if not n1_by_plane:
        return HoughResult(prediction, keys, n1, n2, n3, plane_indices, rhos,
                           accept_log, iter_used)

    n1b = np.asarray(n1_by_plane)
    n2b = np.asarray(n2_by_plane)
    n3b = np.asarray(n3_by_plane)
    rb = np.asarray(rho_by_plane)

    # ── 主法向量 + 5° 容差（Algorithm 4，R 353-383）────────────────────
    if apply_angle_rule:
        positive = plane_indices[plane_indices > 0]
        max_index = r_mode(positive)
        max_nv = np.array([n1b[max_index - 1], n2b[max_index - 1], n3b[max_index - 1]])

        keep, drop = [], []
        for i in range(len(n1b)):
            pnv = np.array([n1b[i], n2b[i], n3b[i]])
            cosang = float(np.dot(max_nv, pnv)) / (
                np.linalg.norm(max_nv) * np.linalg.norm(pnv))
            ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            # R 330：取 180° 的补角，因为法向量方向可能整体取反
            ang = min(ang, 180.0 - ang)
            (drop if ang > ANGLE_TOL_DEG else keep).append(i + 1)

        # R 381：被剔除平面的点重新标为未分类
        noise_idx = np.flatnonzero(np.isin(plane_indices, drop))
        prediction[noise_idx] = 0
    else:
        keep = list(range(1, len(n1b) + 1))

    # ── 未分类点回收（R 386-404）──────────────────────────────────────
    for i in keep:
        unclassified = np.flatnonzero(prediction == 0)
        if unclassified.size == 0:
            continue
        normal = np.array([n1b[i - 1], n2b[i - 1], n3b[i - 1]])
        rho_s = rb[i - 1]
        key = spherical_key(normal, rho_s, rho_step)

        inliers = find_points_on_plane(points, normal, rho_s, unclassified,
                                       inlier_tol)
        prediction[inliers] = 1
        keys[inliers] = key
        n1[inliers] = n1b[i - 1]
        n2[inliers] = n2b[i - 1]
        n3[inliers] = n3b[i - 1]
        rhos[inliers] = rho_s
        plane_indices[inliers] = i

    # ── 尾部扩展：拿主法向量去收剩下的未分类点（R 406-450）──────────────
    # 只在 apply_angle_rule 为真时才有 max_nv 可用（R 里没有这个开关，
    # Algorithm 4 与这一段是绑在一起的）。
    if regrow_master_plane and apply_angle_rule:
        plane_index = int(plane_indices.max()) if plane_indices.size else 0
        while True:
            unclassified = np.flatnonzero(prediction == 0)
            if unclassified.size < 3:                     # R 411
                break

            # R 413-414：以第一个未分类点为锚，沿主法向量定出平面
            v = points[unclassified[0]]
            rho_s = float(np.dot(max_nv, v))
            key = spherical_key(max_nv, rho_s, rho_step)

            inliers = find_points_on_plane(points, max_nv, rho_s, unclassified,
                                           inlier_tol)
            if inliers.size > small_plane_threshold:      # R 435
                plane_index += 1                          # R 436
                prediction[inliers] = 1
                keys[inliers] = key
                n1[inliers] = max_nv[0]
                n2[inliers] = max_nv[1]
                n3[inliers] = max_nv[2]
                rhos[inliers] = rho_s
                plane_indices[inliers] = plane_index      # R 443
            else:
                # R 445：先临时标 2，循环结束后统一还原成 0。
                # 这一支保证每轮至少消化掉一个未分类点，否则会死循环。
                prediction[unclassified[0]] = 2

        # R 449-450：临时的 2 全部还原成 0（= 噪声）
        prediction[prediction == 2] = 0

    return HoughResult(prediction, keys, n1, n2, n3, plane_indices, rhos,
                       accept_log, iter_used)
