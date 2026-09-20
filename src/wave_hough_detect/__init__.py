"""
wave_hough_detect — 随机霍夫变换波源检测

论文《Robust Wave Origin Detection from Sensor Array Data via Randomized
Hough Transform and Model Fitting》(arXiv:2609.13248) 方法的 Python 实现。

三阶段管线
----------
    spikes  →  从每个电极的原始电压波形提取尖峰激活时刻
    rht     →  随机霍夫变换，把 (x, y, t) 点云分离成波前平面
    fit     →  对每个平面拟合圆波前 / 线波前模型，估计源点、速度、激发时刻

最小用法
--------
    from wave_hough_detect import (
        load_recording, detect_all_channels, spikes_to_table,
        hough_plane, build_result_array, plane_points,
        fit_circular, fit_linear,
    )

    rec = load_recording("recording.csv")
    spike_times, filtered = detect_all_channels(rec)
    pts = spikes_to_table(spike_times, rec.channels)
    xyz = pts[["x", "y", "t"]].to_numpy()
    res = hough_plane(xyz, max_iter=200000, seed=1)
    arr = build_result_array(pts["x"], pts["y"], pts["t"], res.plane_indices)

    for k in range(1, res.n_planes + 1):
        p = plane_points(arr, k)
        print(k, fit_circular(p), fit_linear(p))

完整示例见 ``examples/demo_pipeline.py``。
"""

from .fit import (
    CircularFit,
    LinearFit,
    SEARCH_LOWER,
    SEARCH_UPPER,
    TIME_SCALE_BACK,
    build_result_array,
    centroid_initial_guess,
    circular_loss,
    fit_circular,
    fit_linear,
    grid_optim,
    hybrid_optim,
    linear_loss,
    plane_points,
    r_squared,
    vt_optim,
)
from .paths import find_recording
from .rht import (
    ANGLE_TOL_DEG,
    INLIER_TOL,
    PHI_STEP,
    RHO_STEP,
    SMALL_PLANE_THRESHOLD,
    THETA_STEP,
    HoughResult,
    cross_product,
    find_points_on_plane,
    get_key,
    hough_plane,
    is_point_on_plane,
    num_unique_detectors,
    r_mode,
    spherical_key,
)
from .simulate import (
    PAPER_2D_PARAMS,
    DetectionRates,
    DetectionResult,
    detection_rates,
    detection_sweep,
    evaluate_detection,
    seed_from_params,
    simulate_linear_wavefronts,
    simulate_recording,
)
from .simulate_circular import (
    CIRCULAR_EPS,
    CIRCULAR_TRUTH,
    PAPER_CIRCULAR_PARAMS,
    PLOT_MODES,
    SEARCH_LOWER_CIRCULAR,
    SEARCH_UPPER_CIRCULAR,
    WavefrontEstimate,
    accuracy_sweep,
    estimate_circular_wavefront,
    evaluate_circular,
    seed_circular,
    simulate_circular_wavefronts,
)
from .spikes import (
    BUTTER_CUTOFF,
    BUTTER_ORDER,
    EXCLUSION_HALF,
    GRID_SIDE,
    SPIKE_QUANTILE,
    TIME_SCALE,
    Recording,
    butter_highpass,
    channel_to_xy,
    detect_all_channels,
    detect_spikes,
    filter_channel,
    load_recording,
    spikes_to_table,
)

__version__ = "0.1.0"

__all__ = [
    # 阶段 1：尖峰检测
    "Recording", "load_recording", "butter_highpass", "filter_channel",
    "detect_spikes", "detect_all_channels", "spikes_to_table",
    "channel_to_xy", "GRID_SIDE", "BUTTER_ORDER", "BUTTER_CUTOFF",
    "SPIKE_QUANTILE", "EXCLUSION_HALF", "TIME_SCALE",
    # 阶段 2：随机霍夫变换
    "HoughResult", "hough_plane", "RHO_STEP", "PHI_STEP", "THETA_STEP",
    "INLIER_TOL", "ANGLE_TOL_DEG", "SMALL_PLANE_THRESHOLD",
    # 阶段 2 的低层构件（对照标准 HT 时要用到）
    "cross_product", "get_key", "spherical_key", "is_point_on_plane",
    "find_points_on_plane", "num_unique_detectors", "r_mode",
    # 阶段 3：波前模型拟合
    "CircularFit", "LinearFit", "circular_loss", "linear_loss",
    "r_squared", "grid_optim", "vt_optim", "hybrid_optim",
    "fit_circular", "fit_linear", "build_result_array", "plane_points",
    "SEARCH_LOWER", "SEARCH_UPPER", "TIME_SCALE_BACK",
    "centroid_initial_guess",
    # 阶段 4：仿真与评估
    "PAPER_2D_PARAMS", "DetectionRates", "DetectionResult",
    "simulate_linear_wavefronts", "simulate_recording",
    "detection_rates", "evaluate_detection", "detection_sweep",
    "seed_from_params",
    # 阶段 4b：§3.1.2 圆波前仿真与精度评估（图 6–8）
    "CIRCULAR_TRUTH", "PAPER_CIRCULAR_PARAMS", "PLOT_MODES", "CIRCULAR_EPS",
    "SEARCH_LOWER_CIRCULAR", "SEARCH_UPPER_CIRCULAR", "WavefrontEstimate",
    "simulate_circular_wavefronts", "estimate_circular_wavefront",
    "evaluate_circular", "accuracy_sweep", "seed_circular",
    # 工具
    "find_recording",
    "__version__",
]
