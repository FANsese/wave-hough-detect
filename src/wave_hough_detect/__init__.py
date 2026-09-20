"""
wave_hough_detect - wave source detection with the randomized Hough transform

Python implementation of the method described in "Robust Wave Origin Detection
from Sensor Array Data via Randomized Hough Transform and Model Fitting"
(arXiv:2609.13248).

Three-stage pipeline
--------------------
    spikes  ->  extract the spike activation times from each electrode's raw voltage trace
    rht     ->  randomized Hough transform; split the (x, y, t) point cloud into wavefront planes
    fit     ->  fit a circular / linear wavefront model per plane to estimate the source point, the speed and the activation time

Minimal usage
-------------
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

A complete example is in ``examples/demo_pipeline.py``.
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
    # Stage 1: spike detection
    "Recording", "load_recording", "butter_highpass", "filter_channel",
    "detect_spikes", "detect_all_channels", "spikes_to_table",
    "channel_to_xy", "GRID_SIDE", "BUTTER_ORDER", "BUTTER_CUTOFF",
    "SPIKE_QUANTILE", "EXCLUSION_HALF", "TIME_SCALE",
    # Stage 2: randomized Hough transform
    "HoughResult", "hough_plane", "RHO_STEP", "PHI_STEP", "THETA_STEP",
    "INLIER_TOL", "ANGLE_TOL_DEG", "SMALL_PLANE_THRESHOLD",
    # Stage 2 low-level building blocks (needed when comparing against the
    # standard, non-randomized Hough transform)
    "cross_product", "get_key", "spherical_key", "is_point_on_plane",
    "find_points_on_plane", "num_unique_detectors", "r_mode",
    # Stage 3: wavefront model fitting
    "CircularFit", "LinearFit", "circular_loss", "linear_loss",
    "r_squared", "grid_optim", "vt_optim", "hybrid_optim",
    "fit_circular", "fit_linear", "build_result_array", "plane_points",
    "SEARCH_LOWER", "SEARCH_UPPER", "TIME_SCALE_BACK",
    "centroid_initial_guess",
    # Stage 4: simulation and evaluation
    "PAPER_2D_PARAMS", "DetectionRates", "DetectionResult",
    "simulate_linear_wavefronts", "simulate_recording",
    "detection_rates", "evaluate_detection", "detection_sweep",
    "seed_from_params",
    # Stage 4b: section 3.1.2 circular-wavefront simulation and accuracy
    # evaluation (Figs. 6-8)
    "CIRCULAR_TRUTH", "PAPER_CIRCULAR_PARAMS", "PLOT_MODES", "CIRCULAR_EPS",
    "SEARCH_LOWER_CIRCULAR", "SEARCH_UPPER_CIRCULAR", "WavefrontEstimate",
    "simulate_circular_wavefronts", "estimate_circular_wavefront",
    "evaluate_circular", "accuracy_sweep", "seed_circular",
    # Utilities
    "find_recording",
    "__version__",
]
