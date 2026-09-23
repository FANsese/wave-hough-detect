"""
Stage 2: randomized Hough transform (section 2.2 / Algorithm 2 & 4)

This module keeps the logic and the order of the data-processing steps exactly
as reported. Two known defects - sign normalisation failing when the plane
normal is nearly parallel to the t axis, and an ill-conditioned theta - are
**kept as they are** and are only described in the comments; where a fix is
available it is guarded by an explicit switch. An empty result is returned when
no plane is accepted instead of failing.

Note on reproducibility: the only randomness in this module comes from numpy's
generator, so a fixed ``seed`` reproduces a run exactly. To replay one specific
sequence of sampled index triples, pass ``trace``: a pre-specified list of index
triples, which bypasses the generator completely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

# -- Fixed parameters of the real-data pipeline (section 3.2, Tables 2/3/4) --
RHO_STEP = 0.05          # quantisation step of rho inside get_key
PHI_STEP = 2.0           # quantisation step of phi, in degrees
THETA_STEP = 2.0         # quantisation step of theta, in degrees
INLIER_TOL = 0.1         # inlier tolerance of is_point_on_plane
ANGLE_TOL_DEG = 5.0      # angular tolerance of Algorithm 4, in degrees
SMALL_PLANE_THRESHOLD = 30   # minimum number of points of a tail-grown plane

# -- The other parameter set, used by the simulation study (section 3.1) -----
# ★ The paper's experiments use two parameter sets for the same routine, and
#   they are *not* the same:
#     - real-data pipeline   rho step 0.05, tolerance 0.1
#     - simulation study     rho step 0.50, tolerance 1.0
#   The factor of 10 comes from the time scale: real-data spike times have
#   already been divided by 200. Mixing the two parameter sets raises no error at
#   all; it just silently finds no plane (or smears the planes together).
SIM_PRESET = {
    "rho_step": 0.5,
    "inlier_tol": 1.0,
    "strict_degenerate_check": True,
}


# ==========================================================================
#  Utility functions
# ==========================================================================

def cross_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Component-wise cross product of two 3-vectors."""
    return np.array([
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ])


def _step_key(x: float, step: float) -> float:
    """
    Quantise a value onto a multiple of ``step``: ``trunc(x / step) * step``.

    ★ Truncation is toward zero, not toward minus infinity, and for negative
      values the two differ by a whole step:
          trunc(-4.81374 / 0.05) = trunc(-96.2748) = -96
          floor(-96.2748)                         = -97
      Truncation must be used here, otherwise planes with a negative rho land in
      the wrong bucket.

    ★ A second detail: truncation produces an integer, so -0.0 becomes 0 and is
      formatted as "0" in the key, whereas Python's -0.0 would be formatted as
      "-0" by the :g format. Although "-0" and "0" are each internally consistent
      and group the same points together, the key string itself would differ.
      -0.0 is therefore normalised to 0.0, so that keys are reproducible
      character by character.
    """
    q = float(np.trunc(x / step) * step)
    return 0.0 if q == 0 else q


def get_key(rho: float, phi: float, theta: float,
            rho_step: float = RHO_STEP) -> str:
    """
    Quantise (rho, phi, theta) into a string key.

    phi and theta are converted from radians to degrees before quantisation.
    The key format is "rho,phi,theta", for example "9.9,84,14".

    ★ ``rho_step`` has to be adjustable: the real-data pipeline uses 0.05 while
      the simulation study uses 0.5 - a factor of 10. The reason is that the two
      time scales differ by the same factor of 10, the real-data spike times
      having been divided by 200 first. Using the wrong step raises no error; it
      makes the accumulator fall apart, so that not a single plane is found.
    """
    rho_q = _step_key(rho, rho_step)
    phi_q = _step_key(np.degrees(phi), PHI_STEP)
    theta_q = _step_key(np.degrees(theta), THETA_STEP)
    return f"{rho_q:g},{phi_q:g},{theta_q:g}"


def spherical_key(normal: np.ndarray, rho_signed: float,
                  rho_step: float = RHO_STEP) -> str:
    """
    Recover (rho, phi, theta) from an accepted plane normal and return the
    accumulator key.

    The same conversion is applied at three places in the main routine:
    ``rho = abs(rho_signed); phi = acos(n3); theta = asin(n2 / sin(phi))``.

    ★ For phi = 0 (normal exactly parallel to the t axis) ``asin(n2 / sin(0))``
      is a division by zero, which would put "NaN" into the key; the inlier test
      and the assignment would still be carried out, because the key is only a
      label and never takes part in any geometric decision. theta is therefore
      set to 0 here, which yields a deterministic key while keeping the
      observable behaviour (the points are absorbed as usual) and never lets a
      NaN reach the string. On real data phi never drops below about 0.8 degrees,
      so this branch has never been triggered.
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
    Plane inlier test: ``|n . p - rho| < tol``.

    ★ ``tol`` has to be adjustable: the real-data pipeline uses 0.1 and the
      simulation study uses 1. Again the two time scales differ by a factor of
      10.
    """
    return abs(float(np.dot(point, normal)) - rho) < tol


def find_points_on_plane(points: np.ndarray,
                         normal: np.ndarray,
                         rho: float,
                         indices: np.ndarray,
                         tol: float = INLIER_TOL) -> np.ndarray:
    """
    Return the indices that lie on the plane.

    The candidate indices are traversed in the order given and every hit is
    appended to the result, so the returned order is the same as the input
    order.
    """
    hits = [int(i) for i in indices
            if is_point_on_plane(points[i], normal, rho, tol)]
    return np.asarray(hits, dtype=int)


def num_unique_detectors(points: np.ndarray, indices: np.ndarray) -> int:
    """
    Count how many *distinct electrodes* the point set covers. An electrode is
    identified by (x, y); time plays no role.
    """
    if indices.size == 0:
        return 0
    xy = points[indices][:, :2]
    return int(np.unique(xy, axis=0).shape[0])


def r_mode(values: np.ndarray) -> int:
    """
    Mode of a set of plane indices, with a specific tie-breaking rule: the
    candidate whose *string* order is smallest wins.

    ★ The tie rule matters. The counts are sorted by their key read as a string,
      and the sort is stable, so among tied counts the smallest string wins. For
      plane indices 1..9 string order equals numeric order, but from 10 upwards
      the two differ ("10" < "2"), and that string rule is reproduced here
      faithfully.
    """
    vals, counts = np.unique(values, return_counts=True)
    order = sorted(range(len(vals)), key=lambda i: str(vals[i]))
    vals = vals[order]
    counts = counts[order]
    return int(vals[int(np.argmax(counts))])


# ==========================================================================
#  Result container
# ==========================================================================

@dataclass
class HoughResult:
    """
    Everything one Hough run produces: ``prediction`` (0 = noise, 1 = signal),
    ``keys`` (the accumulator key each point belongs to), ``n1`` / ``n2`` /
    ``n3`` (the normal-vector components of the plane each point belongs to),
    ``plane_indices`` (the plane number, 0 meaning unclassified) and ``rhos``.
    """

    prediction: np.ndarray        # 0=noise, 1=signal
    keys: np.ndarray              # accumulator key each point belongs to
    n1: np.ndarray                # normal-vector components of the plane
    n2: np.ndarray                # the point was assigned to
    n3: np.ndarray
    plane_indices: np.ndarray     # plane number, 0 means unclassified
    rhos: np.ndarray
    accept_log: list = field(default_factory=list)   # diagnostics: one record per accepted plane
    n_iter_used: int = 0

    @property
    def n_planes(self) -> int:
        return int(len(np.unique(self.plane_indices[self.plane_indices > 0])))

    def summary(self) -> str:
        lines = [f"planes found     : {self.n_planes}",
                 f"points as signal : {int(self.prediction.sum())} / {len(self.prediction)}",
                 f"iterations used  : {self.n_iter_used}"]
        if self.n_planes:
            ids, cnt = np.unique(self.plane_indices[self.plane_indices > 0],
                                 return_counts=True)
            lines.append("points per plane : " + "  ".join(
                f"P{i}={c}" for i, c in zip(ids, cnt)))
        return "\n".join(lines)


# ==========================================================================
#  Main routine
# ==========================================================================

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
    Randomized Hough transform over a (x, y, ts) point cloud.

    Parameters
    ----------
    points : (N, 3) array whose columns are (x, y, ts)
    vote_threshold : default 8. The threshold is effectively 9 votes, not the 8
        stated in the paper: the test is len(bucket) > vote_threshold * 3 and
        each vote appends three indices. This is reproduced faithfully.
    max_iter : upper bound on the number of iterations. The default is 30000,
        but in practice at least 200000 are needed to find all 5 planes.
    min_detectors : hard-coded minimum of 40 detectors (about 2/3 of the 64
        electrodes)
    trace : optional. A pre-specified sequence of sampled indices, used for
        bit-exact replay. When it is given the random number generator is not
        used at all.
    seed : numpy random seed, used when ``trace`` is None.
    apply_angle_rule : whether to apply the 5 degree single-source tolerance of
        Algorithm 4
    regrow_master_plane : whether to run the tail extension:
        after the main loop and Algorithm 4, the master normal is used to sweep
        the remaining unclassified points once more: if the points of one such
        plane with the same normal cover more than ``small_plane_threshold``
        points, they are collected into a new plane.
        ★ The paper's main text does not describe this step, but it is part of
          the published algorithm. On real data it always runs, yet its loop body
          is never entered (all 320 points get classified, the unclassified set
          is empty), so without it the real-data result looks "completely
          correct". The two only diverge once unclassified points remain, which
          is exactly the case in the simulation study (section 3.1) - there
          FNR != 0.
    small_plane_threshold : minimum number of points of a tail-grown plane,
        default 30
    rho_step, inlier_tol : quantisation step and inlier tolerance. The defaults
        correspond to the real-data pipeline (0.05 / 0.1). The simulation study
        uses 0.5 / 1.0, see SIM_PRESET.
    strict_degenerate_check : the two parameter sets disagree on the criterion
        for a "degenerate normal" - the real-data pipeline tests only the
        (n2, n3) and (n1, n3) pairs, the simulation study tests all three pairs.
        The default False matches the real-data pipeline; set it to True for the
        simulation study.

    Returns
    -------
    HoughResult
    """
    n = points.shape[0]
    rng = np.random.default_rng(seed)

    # -- Initialisation --------------------------------------------------
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

    # accumulator: key -> list of indices that have been drawn for that key
    accumulator: dict[str, list[int]] = {}
    accept_log: list[dict] = []

    trace_pos = 0
    iter_used = 0

    # -- Main loop -------------------------------------------------------
    for it in range(1, max_iter + 1):
        iter_used = it
        unclassified = np.flatnonzero(prediction == 0)
        if unclassified.size < 3:            # fewer than 3 points left
            break

        # -- Draw 3 points -----------------------------------------------
        if trace is not None:
            if trace_pos >= len(trace):
                break
            sample_idx = list(trace[trace_pos])
            trace_pos += 1
        else:
            # ★ The three indices are drawn *without replacement*, so they are
            #   always distinct.
            sample_idx = rng.choice(unclassified, size=3, replace=False).tolist()

        v1 = points[sample_idx[0]]
        v2 = points[sample_idx[1]]
        v3 = points[sample_idx[2]]

        # -- Normal vector from the cross product ------------------------
        nrm_vec = cross_product(v2 - v1, v3 - v1)
        norm_n = float(np.linalg.norm(nrm_vec))
        if norm_n == 0:
            continue                          # the three points are collinear
        nrm_vec = nrm_vec / norm_n

        # -- Sign normalisation ------------------------------------------
        # Force n1 >= 0, which removes the (n, rho) / (-n, -rho) ambiguity.
        # ★ Known defect: when n1 is close to 0 (normal nearly parallel to the t
        #   axis) the sign of n1 is decided by noise, so one and the same plane
        #   is represented at the two extremes phi ~ 0 and phi ~ 180. This
        #   dataset happens to hit that case, and it is one of the main reasons
        #   why so many iterations are needed.
        if nrm_vec[0] != 0:
            nrm_vec = nrm_vec * np.sign(nrm_vec[0])

        # -- Rejecting degenerate cases ----------------------------------
        degenerate = (nrm_vec[1] == 0 and nrm_vec[2] == 0) or \
                     (nrm_vec[0] == 0 and nrm_vec[2] == 0)
        if strict_degenerate_check:            # the simulation setting tests one more pair
            degenerate = degenerate or (nrm_vec[0] == 0 and nrm_vec[1] == 0)
        if degenerate:
            continue
        if float(np.dot(nrm_vec, np.array([0.0, 0.0, 1.0]))) == 0:
            continue

        # -- Spherical coordinates ---------------------------------------
        rho_signed = float(np.dot(nrm_vec, v1))
        rho = abs(rho_signed)
        phi = float(np.arccos(np.clip(nrm_vec[2], -1.0, 1.0)))
        if phi == 0:
            continue
        theta = float(np.arcsin(np.clip(nrm_vec[1] / np.sin(phi), -1.0, 1.0)))

        # -- Voting ------------------------------------------------------
        key = get_key(rho, phi, theta, rho_step)
        accumulator.setdefault(key, []).extend(sample_idx)

        # -- Enough votes? The test is len(bucket) > vote_threshold * 3, i.e.
        #    at least 9 votes for the default threshold of 8 -------------
        if len(accumulator[key]) > vote_threshold * 3:

            # Least-squares refinement: refit the plane on every point of the
            # bucket.
            # ★ Note the column order of the design matrix: the point cloud is
            #   (x, y, ts), the design matrix is [y, ts, 1] and the response is
            #   x, i.e. x is regressed on y and ts, and the normal follows as
            #   n_fit = (1, -a, -b).
            bucket = np.asarray(accumulator[key], dtype=int)
            design = np.column_stack([points[bucket, 1],
                                      points[bucket, 2],
                                      np.ones(bucket.size)])
            # regress x on [y, ts, 1]
            coef, *_ = np.linalg.lstsq(design, points[bucket, 0], rcond=None)
            n_fit = np.array([1.0, -coef[0], -coef[1]])
            rho_fit = float(coef[2])
            norm_fit = float(np.linalg.norm(n_fit))
            n_fit = n_fit / norm_fit
            rho_fit = rho_fit / norm_fit

            # -- Inliers -------------------------------------------------
            inliers = find_points_on_plane(points, n_fit, rho_fit, unclassified,
                                           inlier_tol)

            # -- Detector coverage criterion -----------------------------
            n_det = num_unique_detectors(points, inliers)
            if n_det < min_detectors:
                accumulator[key] = []          # empty the bucket and move on
                continue

            # -- Accept this plane ---------------------------------------
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
            accumulator = {}                   # clear the accumulator

    # -- Guard against an empty result -----------------------------------
    # ★ An empty result is returned when no plane is accepted: the per-plane
    #   normal list is then empty, so the master-normal selection and the tail
    #   extension below have nothing to work with.
    if not n1_by_plane:
        return HoughResult(prediction, keys, n1, n2, n3, plane_indices, rhos,
                           accept_log, iter_used)

    n1b = np.asarray(n1_by_plane)
    n2b = np.asarray(n2_by_plane)
    n3b = np.asarray(n3_by_plane)
    rb = np.asarray(rho_by_plane)

    # -- Master normal + 5 degree tolerance (Algorithm 4) -----------------
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
            # supplement to 180 degrees, because the normal direction may be
            # globally flipped
            ang = min(ang, 180.0 - ang)
            (drop if ang > ANGLE_TOL_DEG else keep).append(i + 1)
        _keep_ix = np.array(keep, dtype=int) - 1      # 0-based, kept planes only

        # Points of the dropped planes are marked unclassified again. Clearing
        # only `prediction` is not enough: `plane_indices`, `keys` and the
        # per-point plane parameters would keep pointing at a plane that no
        # longer exists, and HoughResult.n_planes counts distinct positive
        # plane_indices -- so a dropped plane would still be reported as a
        # plane and downstream plane_points()/fit_circular() would fit it.
        # Measured on two synthetic planes 40 degrees apart: n_planes = 2 while
        # only one plane had any point marked as signal.
        noise_idx = np.flatnonzero(np.isin(plane_indices, drop))
        prediction[noise_idx] = 0
        plane_indices[noise_idx] = 0
        keys[noise_idx] = ""
        n1[noise_idx] = 0.0
        n2[noise_idx] = 0.0
        n3[noise_idx] = 0.0
        rhos[noise_idx] = 0.0
        # Drop the rejected planes from the per-plane arrays too, and renumber
        # `keep` to the new contiguous numbering, so that the recovery loop
        # below indexes these arrays consistently.
        n1b, n2b, n3b, rb = n1b[_keep_ix], n2b[_keep_ix], n3b[_keep_ix], rb[_keep_ix]
        keep = list(range(1, len(_keep_ix) + 1))
    else:
        keep = list(range(1, len(n1b) + 1))

    # The dropped planes leave gaps in the numbering, and downstream code
    # (HoughResult.n_planes, build_result_array, plane_points) assumes plane
    # indices run 1..K contiguously. Renumber the surviving planes.
    if apply_angle_rule and len(keep) != len(n1b):
        remap = {old: new for new, old in enumerate(keep, start=1)}
        for old, new in remap.items():
            plane_indices[plane_indices == old] = new
        # accept_log keeps the acceptance order, which is a historical record;
        # after this rule it may differ from the final plane numbering.

    # -- Recovering unclassified points -----------------------------------
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

    # -- Tail extension: sweep the remaining unclassified points with the
    #    master normal --------------------------------------------------
    # max_nv only exists when apply_angle_rule is true; in the algorithm the two
    # steps are bound together.
    if regrow_master_plane and apply_angle_rule:
        plane_index = int(plane_indices.max()) if plane_indices.size else 0
        while True:
            unclassified = np.flatnonzero(prediction == 0)
            if unclassified.size < 3:                     # fewer than 3 points left
                break

            # anchor the plane at the first unclassified point, along the master
            # normal
            v = points[unclassified[0]]
            rho_s = float(np.dot(max_nv, v))
            key = spherical_key(max_nv, rho_s, rho_step)

            inliers = find_points_on_plane(points, max_nv, rho_s, unclassified,
                                           inlier_tol)
            if inliers.size > small_plane_threshold:      # strict inequality
                plane_index += 1
                prediction[inliers] = 1
                keys[inliers] = key
                n1[inliers] = max_nv[0]
                n2[inliers] = max_nv[1]
                n3[inliers] = max_nv[2]
                rhos[inliers] = rho_s
                plane_indices[inliers] = plane_index
            else:
                # tentatively mark the point as 2, restored to 0 after the loop.
                # This branch consumes at least one unclassified point per round,
                # otherwise the loop would never terminate.
                prediction[unclassified[0]] = 2

        # every temporary 2 becomes 0 again (= noise)
        prediction[prediction == 2] = 0

    return HoughResult(prediction, keys, n1, n2, n3, plane_indices, rhos,
                       accept_log, iter_used)
