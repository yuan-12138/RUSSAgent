#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust outlier repair for 3D scan polylines (spine).

Flags interior points that spike in step length, deviate laterally from the
local chord (prev -> next), or jump in Z relative to a local median.
Outliers are replaced by linear interpolation between neighbors (not deleted),
so frame count stays trackable for the executor.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _chord_deviation(points: np.ndarray, idx: int) -> float:
    a = points[idx - 1]
    b = points[idx]
    c = points[idx + 1]
    ac = c - a
    denom = float(np.dot(ac, ac))
    if denom < 1e-12:
        return float(np.linalg.norm(b - a))
    t = float(np.clip(np.dot(b - a, ac) / denom, 0.0, 1.0))
    proj = a + t * ac
    return float(np.linalg.norm(b - proj))


def _local_z_median(points: np.ndarray, idx: int, window: int) -> float:
    lo = max(0, idx - window)
    hi = min(len(points), idx + window + 1)
    return float(np.median(points[lo:hi, 2]))


def filter_trajectory_points(
    points: Sequence[Sequence[float]],
    *,
    lateral_threshold_m: float = 0.035,
    step_spike_factor: float = 2.5,
    z_jump_threshold_m: float = 0.04,
    window: int = 5,
    max_passes: int = 3,
    min_points: int = 2,
) -> Tuple[np.ndarray, List[int]]:
    """
    Returns (filtered_points Nx3, sorted unique outlier indices in the original polyline).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < min_points:
        return pts.copy(), []

    outlier_indices: List[int] = []
    filtered = pts.copy()

    for _ in range(max(1, int(max_passes))):
        if filtered.shape[0] < 3:
            break
        seg = np.linalg.norm(np.diff(filtered, axis=0), axis=1)
        med_step = float(np.median(seg)) if seg.size else 0.0
        step_limit = max(0.015, med_step * float(step_spike_factor))
        changed = False

        for i in range(1, filtered.shape[0] - 1):
            chord_dev = _chord_deviation(filtered, i)
            step_in = float(seg[i - 1])
            step_out = float(seg[i])
            z_jump = abs(float(filtered[i, 2]) - _local_z_median(filtered, i, window))

            is_outlier = (
                chord_dev > lateral_threshold_m
                or (step_in > step_limit and step_out > step_limit)
                or z_jump > z_jump_threshold_m
            )
            if not is_outlier:
                continue

            filtered[i] = 0.5 * (filtered[i - 1] + filtered[i + 1])
            outlier_indices.append(i)
            changed = True

        if not changed:
            break

    return filtered, sorted(set(outlier_indices))
