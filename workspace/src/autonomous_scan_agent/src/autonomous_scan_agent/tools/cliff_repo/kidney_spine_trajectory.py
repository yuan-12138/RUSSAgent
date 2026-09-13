#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Left-kidney trajectory v2 helpers (experimental).

IMPORTANT: This module is intentionally separate from the legacy kidney / gallbladder
trajectory builders in kidney_spine_geometry.py. Importing it does NOT change
any existing pipeline unless new code explicitly calls these functions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from cliff_skel_trajectory import project_points_to_uv
except Exception:
    project_points_to_uv = None  # type: ignore

try:
    from kidney_spine_geometry import (
        _bone_center_xy,
        _infer_lr_and_inferior_dirs,
        extract_spine_centerline_curve_xyz,
    )
except Exception:
    _bone_center_xy = None  # type: ignore
    _infer_lr_and_inferior_dirs = None  # type: ignore
    extract_spine_centerline_curve_xyz = None  # type: ignore


def _require_kidney_deps() -> None:
    if any(
        x is None
        for x in (
            _bone_center_xy,
            _infer_lr_and_inferior_dirs,
            extract_spine_centerline_curve_xyz,
            project_points_to_uv,
        )
    ):
        raise ImportError("kidney_spine_trajectory requires cliff_skel_trajectory / kidney_spine_geometry imports.")


def _lr_coord_of_points(
    points_xyz: np.ndarray,
    center_ref: np.ndarray,
    lr_dir: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    xy = pts[:, :2] - np.asarray(center_ref, dtype=np.float64).reshape(1, 2)
    return (xy @ np.asarray(lr_dir, dtype=np.float64).reshape(2, 1))[:, 0]


def _inferior_coord_of_points(
    points_xyz: np.ndarray,
    center_ref: np.ndarray,
    inferior_dir: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    xy = pts[:, :2] - np.asarray(center_ref, dtype=np.float64).reshape(1, 2)
    return (xy @ np.asarray(inferior_dir, dtype=np.float64).reshape(2, 1))[:, 0]


def filter_spine_segment_thorax_to_lumbar(
    fitter: Any,
    skel_verts: np.ndarray,
    spine_curve_xyz: np.ndarray,
    *,
    margin_m: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Keep spine-centerline points between thorax and lumbar_body bone centers
    (thoracic / thoraco-lumbar segment on the centerline).
    """
    _require_kidney_deps()
    v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    if sc.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.float32), {}

    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, _lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    thorax_xy = _bone_center_xy(weights, v, bone_names, "thorax")
    lumbar_xy = _bone_center_xy(weights, v, bone_names, "lumbar_body")
    if thorax_xy is None or lumbar_xy is None:
        return np.zeros((0, 3), dtype=np.float32), {}

    thorax_inf = float(((thorax_xy - center_ref) @ inferior_dir.reshape(2)))
    lumbar_inf = float(((lumbar_xy - center_ref) @ inferior_dir.reshape(2)))
    lo = min(thorax_inf, lumbar_inf) - float(margin_m)
    hi = max(thorax_inf, lumbar_inf) + float(margin_m)

    sc_inf = _inferior_coord_of_points(sc, center_ref, inferior_dir)
    mask = (sc_inf >= lo) & (sc_inf <= hi)
    seg = sc[mask]
    meta = {
        "thorax_inf": thorax_inf,
        "lumbar_inf": lumbar_inf,
        "segment_lo": lo,
        "segment_hi": hi,
        "num_points_total": float(sc.shape[0]),
        "num_points_segment": float(seg.shape[0]),
    }
    seg_inf = _inferior_coord_of_points(seg, center_ref, inferior_dir)
    meta["seg_inf"] = seg_inf
    return seg.astype(np.float32), meta


def _lumbar_mesh_highest_near_spine(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    center_ref: np.ndarray,
    inferior_dir: np.ndarray,
    lr_dir: np.ndarray,
    thorax_bone_inf: float,
    pelvis_bone_inf: float,
    *,
    threshold: float = 0.1,
    lr_tol_m: float = 0.04,
    transition_floor_above_thorax_m: float = 0.15,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """
    Highest (most superior) point on the lumbar_body mesh near the spine column.

    SKEL lumbar / thorax bone centers sit very close; we therefore ignore lumbar
    vertices still in the mid-thorax zone and only consider mesh vertices at or
    below ``thorax_bone_inf + transition_floor_above_thorax_m`` (TL junction band).
    """
    from cliff_skel_trajectory import extract_specific_bone_mesh

    lb_verts, _faces = extract_specific_bone_mesh(
        fitter,
        skel_verts,
        skel_faces,
        "lumbar_body",
        img_path,
        threshold=threshold,
    )
    if lb_verts is None:
        return None, None
    lb_inf = _inferior_coord_of_points(lb_verts, center_ref, inferior_dir)
    lb_lr = _lr_coord_of_points(lb_verts, center_ref, lr_dir)
    lo = float(thorax_bone_inf) + float(transition_floor_above_thorax_m)
    hi = float(pelvis_bone_inf)
    mask = (np.abs(lb_lr) <= float(lr_tol_m)) & (lb_inf >= lo) & (lb_inf <= hi)
    if int(np.count_nonzero(mask)) < 20:
        mask = (np.abs(lb_lr) <= float(lr_tol_m) * 1.5) & (lb_inf >= lo) & (lb_inf <= hi)
    if not np.any(mask):
        return None, None
    vals = lb_inf[mask]
    verts = np.asarray(lb_verts, dtype=np.float64).reshape(-1, 3)[mask]
    j = int(np.argmin(vals))
    return float(vals[j]), verts[j].astype(np.float32)


def _lumbar_spine_column_superior_extent_m(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    center_ref: np.ndarray,
    inferior_dir: np.ndarray,
    lr_dir: np.ndarray,
    thorax_bone_inf: float,
    pelvis_bone_inf: float,
    *,
    threshold: float = 0.1,
    lr_tol_m: float = 0.04,
    transition_floor_above_thorax_m: float = 0.15,
) -> Optional[float]:
    """Inferior-coordinate of the highest lumbar_body mesh point near the spine."""
    sup_inf, _xyz = _lumbar_mesh_highest_near_spine(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        center_ref,
        inferior_dir,
        lr_dir,
        thorax_bone_inf,
        pelvis_bone_inf,
        threshold=threshold,
        lr_tol_m=lr_tol_m,
        transition_floor_above_thorax_m=transition_floor_above_thorax_m,
    )
    return sup_inf


def _thorax_spine_lowest_from_lumbar_superior(
    spine_curve_xyz: np.ndarray,
    center_ref: np.ndarray,
    inferior_dir: np.ndarray,
    lumbar_superior_inf: float,
    *,
    t12_above_lumbar_m: float = 0.005,
) -> Optional[np.ndarray]:
    """
    T12 on the spine centerline: the most inferior spine point still just above
    the highest lumbar_body mesh level (L1 is at lumbar_superior_inf).
    """
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    if sc.shape[0] < 1:
        return None
    sc_inf = _inferior_coord_of_points(sc, center_ref, inferior_dir)
    target_inf = float(lumbar_superior_inf) - float(t12_above_lumbar_m)
    mask = sc_inf <= target_inf
    if np.any(mask):
        j = int(np.argmax(sc_inf[mask]))
        return sc[mask][j].astype(np.float32)
    # lumbar top is superior to all spine samples: use the most superior spine point
    j = int(np.argmin(sc_inf))
    return sc[j].astype(np.float32)


def _lowest_spine_point_in_inferior_band(
    spine_curve_xyz: np.ndarray,
    center_ref: np.ndarray,
    inferior_dir: np.ndarray,
    lo_m: float,
    hi_m: float,
) -> Optional[np.ndarray]:
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    if sc.shape[0] < 1:
        return None
    sc_inf = _inferior_coord_of_points(sc, center_ref, inferior_dir)
    mask = (sc_inf >= float(lo_m)) & (sc_inf <= float(hi_m))
    if not np.any(mask):
        return None
    j = int(np.argmax(sc_inf[mask]))
    return sc[mask][j].astype(np.float32)


def extract_thorax_spine_lowest_point_xyz(
    fitter: Any,
    skel_verts: np.ndarray,
    spine_curve_xyz: np.ndarray,
    *,
    skel_faces: Optional[np.ndarray] = None,
    img_path: Optional[str] = None,
    margin_m: float = 0.01,
    transition_floor_above_thorax_m: float = 0.15,
    t12_above_lumbar_m: float = 0.005,
) -> Optional[np.ndarray]:
    """
    Lowest point of the thorax spine on the SKEL spine centerline (approx. T12).

    Definition (camera coordinates):
    - On lumbar_body mesh vertices near the spine, find the highest (most superior)
      point that lies below the mid-thorax (>= thorax + transition_floor).
    - Return the spine-centerline point just above that level (T12 above L1).

    Returns:
        (3,) float32 xyz in camera frame, or None if unavailable.
    """
    _require_kidney_deps()
    v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    thorax_xy = _bone_center_xy(weights, v, bone_names, "thorax")
    lumbar_xy = _bone_center_xy(weights, v, bone_names, "lumbar_body")
    pelvis_xy = _bone_center_xy(weights, v, bone_names, "pelvis")
    if thorax_xy is None or lumbar_xy is None or pelvis_xy is None:
        return None

    thorax_inf = float(((thorax_xy - center_ref) @ inferior_dir.reshape(2)))
    pelvis_inf = float(((pelvis_xy - center_ref) @ inferior_dir.reshape(2)))

    if skel_faces is not None and img_path:
        lumbar_superior_inf, _lumbar_top_xyz = _lumbar_mesh_highest_near_spine(
            fitter,
            v,
            np.asarray(skel_faces),
            str(img_path),
            center_ref,
            inferior_dir,
            lr_dir,
            thorax_inf,
            pelvis_inf,
            transition_floor_above_thorax_m=transition_floor_above_thorax_m,
        )
        if lumbar_superior_inf is not None:
            p = _thorax_spine_lowest_from_lumbar_superior(
                sc,
                center_ref,
                inferior_dir,
                lumbar_superior_inf,
                t12_above_lumbar_m=t12_above_lumbar_m,
            )
            if p is not None:
                return p

    seg, meta = filter_spine_segment_thorax_to_lumbar(
        fitter,
        skel_verts,
        spine_curve_xyz,
        margin_m=margin_m,
    )
    return _lowest_point_on_segment(seg, meta)


def extract_lumbar_spine_highest_point_xyz(
    fitter: Any,
    skel_verts: np.ndarray,
    *,
    skel_faces: Optional[np.ndarray] = None,
    img_path: Optional[str] = None,
    transition_floor_above_thorax_m: float = 0.15,
) -> Optional[np.ndarray]:
    """
    Highest (most superior) point on the lumbar_body mesh near the spine column.

    Returns the 3D mesh vertex (camera coordinates), approximating the superior L1 level.
    """
    _require_kidney_deps()
    if skel_faces is None or not img_path:
        return None
    v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)
    thorax_xy = _bone_center_xy(weights, v, bone_names, "thorax")
    pelvis_xy = _bone_center_xy(weights, v, bone_names, "pelvis")
    if thorax_xy is None or pelvis_xy is None:
        return None
    thorax_inf = float(((thorax_xy - center_ref) @ inferior_dir.reshape(2)))
    pelvis_inf = float(((pelvis_xy - center_ref) @ inferior_dir.reshape(2)))
    _sup_inf, xyz = _lumbar_mesh_highest_near_spine(
        fitter,
        v,
        np.asarray(skel_faces),
        str(img_path),
        center_ref,
        inferior_dir,
        lr_dir,
        thorax_inf,
        pelvis_inf,
        transition_floor_above_thorax_m=transition_floor_above_thorax_m,
    )
    return xyz


def extract_thorax_mesh_left_lowest_point_xyz(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    *,
    threshold: float = 0.1,
    lr_band_percentile: float = 15.0,
    thorax_verts: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Left-lowest point on the thorax SKEL mesh (camera coordinates).

    Uses only thorax bone mesh vertices (no rib-margin polyline, no right-side mirror):
      1. Restrict to the leftmost ``lr_band_percentile`` lateral band (body LR axis).
      2. Within that band, take the most inferior vertex (thorax -> pelvis direction).

    This is an auxiliary landmark for comparison with kidney endpoint / spine markers.
    """
    _require_kidney_deps()
    from cliff_skel_trajectory import extract_specific_bone_mesh

    if thorax_verts is None:
        tx_verts, _tx_faces = extract_specific_bone_mesh(
            fitter,
            skel_verts,
            skel_faces,
            "thorax",
            img_path,
            threshold=float(threshold),
        )
    else:
        tx_verts = np.asarray(thorax_verts, dtype=np.float64).reshape(-1, 3)
    if tx_verts is None or len(tx_verts) < 3:
        return None

    v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)

    pts = np.asarray(tx_verts, dtype=np.float64).reshape(-1, 3)
    xy = pts[:, :2] - np.asarray(center_ref, dtype=np.float64).reshape(1, 2)
    coord_lr = (xy @ np.asarray(lr_dir, dtype=np.float64).reshape(2, 1))[:, 0]
    coord_inf = (xy @ np.asarray(inferior_dir, dtype=np.float64).reshape(2, 1))[:, 0]

    p = float(lr_band_percentile)
    p = max(0.0, min(50.0, p))
    thr = float(np.percentile(coord_lr, p))
    band = coord_lr <= thr
    if int(np.count_nonzero(band)) < 3:
        band = coord_lr <= float(np.median(coord_lr))
    if int(np.count_nonzero(band)) < 1:
        band = np.ones((pts.shape[0],), dtype=bool)

    idx = int(np.argmax(np.where(band, coord_inf, -1e18)))
    return pts[idx].astype(np.float32)


def extract_thorax_mesh_left_lowest_point_uv(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    *,
    threshold: float = 0.1,
    lr_band_percentile: float = 15.0,
) -> Optional[np.ndarray]:
    """Project ``extract_thorax_mesh_left_lowest_point_xyz`` to pixel UV."""
    _require_kidney_deps()
    p = extract_thorax_mesh_left_lowest_point_xyz(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        threshold=threshold,
        lr_band_percentile=lr_band_percentile,
    )
    if p is None:
        return None
    uv = project_points_to_uv(np.asarray(p, dtype=np.float32).reshape(1, 3), intrinsics)
    return uv.reshape(2).astype(np.float32)


def _lowest_point_on_segment(seg: np.ndarray, meta: Dict[str, float]) -> Optional[np.ndarray]:
    if seg is None or np.asarray(seg).reshape(-1, 3).shape[0] < 1:
        return None
    seg = np.asarray(seg, dtype=np.float64).reshape(-1, 3)
    thorax_inf = float(meta.get("thorax_inf", 0.0))
    lumbar_inf = float(meta.get("lumbar_inf", 0.0))
    direction = lumbar_inf - thorax_inf
    if abs(direction) < 1e-9:
        return seg[-1].astype(np.float32)
    if "seg_inf" in meta and isinstance(meta["seg_inf"], np.ndarray):
        seg_inf = np.asarray(meta["seg_inf"], dtype=np.float64).reshape(-1)
    else:
        seg_inf = seg[:, 2]
    if direction >= 0:
        j = int(np.argmax(seg_inf))
    else:
        j = int(np.argmin(seg_inf))
    return seg[j].astype(np.float32)


def extract_thorax_spine_lowest_from_cached_frame(
    spine_curve_xyz: np.ndarray,
    center_ref: np.ndarray,
    inferior_dir: np.ndarray,
    thorax_inf: float,
    lumbar_inf: float,
    *,
    margin_m: float = 0.01,
) -> Tuple[Optional[np.ndarray], np.ndarray, Dict[str, float]]:
    """Frame-only variant (no fitter) for offline validation caches."""
    sc = np.asarray(spine_curve_xyz, dtype=np.float64).reshape(-1, 3)
    lo = min(float(thorax_inf), float(lumbar_inf)) - float(margin_m)
    hi = max(float(thorax_inf), float(lumbar_inf)) + float(margin_m)
    sc_inf = _inferior_coord_of_points(sc, center_ref, inferior_dir)
    mask = (sc_inf >= lo) & (sc_inf <= hi)
    seg = sc[mask]
    meta = {
        "thorax_inf": float(thorax_inf),
        "lumbar_inf": float(lumbar_inf),
        "segment_lo": lo,
        "segment_hi": hi,
        "num_points_total": float(sc.shape[0]),
        "num_points_segment": float(seg.shape[0]),
        "seg_inf": sc_inf[mask],
    }
    return _lowest_point_on_segment(seg, meta), seg.astype(np.float32), meta


def compute_spine_segment_frame(
    fitter: Any,
    skel_verts: np.ndarray,
) -> Dict[str, Any]:
    """Serialize thorax/lumbar bounds for offline reuse."""
    _require_kidney_deps()
    v = np.asarray(skel_verts, dtype=np.float64).reshape(-1, 3)
    bone_names = list(getattr(fitter.skel, "bone_names", []))
    weights = fitter.skel.skel_weights.to_dense().detach().cpu().numpy()
    center_ref, lr_dir, inferior_dir = _infer_lr_and_inferior_dirs(weights, v, bone_names)
    thorax_xy = _bone_center_xy(weights, v, bone_names, "thorax")
    lumbar_xy = _bone_center_xy(weights, v, bone_names, "lumbar_body")
    if thorax_xy is None or lumbar_xy is None:
        return {}
    thorax_inf = float(((thorax_xy - center_ref) @ inferior_dir.reshape(2)))
    lumbar_inf = float(((lumbar_xy - center_ref) @ inferior_dir.reshape(2)))
    return {
        "center_ref": center_ref.astype(np.float32),
        "lr_dir": lr_dir.astype(np.float32),
        "inferior_dir": inferior_dir.astype(np.float32),
        "thorax_inf": np.float32(thorax_inf),
        "lumbar_inf": np.float32(lumbar_inf),
    }


def extract_thorax_spine_lowest_point_uv(
    fitter: Any,
    skel_verts: np.ndarray,
    spine_curve_xyz: np.ndarray,
    intrinsics: dict,
    *,
    skel_faces: Optional[np.ndarray] = None,
    img_path: Optional[str] = None,
    margin_m: float = 0.01,
) -> Optional[np.ndarray]:
    """Project `extract_thorax_spine_lowest_point_xyz` to pixel UV."""
    _require_kidney_deps()
    p = extract_thorax_spine_lowest_point_xyz(
        fitter,
        skel_verts,
        spine_curve_xyz,
        skel_faces=skel_faces,
        img_path=img_path,
        margin_m=margin_m,
    )
    if p is None:
        return None
    uv = project_points_to_uv(np.asarray(p, dtype=np.float32).reshape(1, 3), intrinsics)
    return uv.reshape(2).astype(np.float32)


def build_spine_curve_for_v2(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    intrinsics: dict,
    *,
    spine_region: str = "thorax_to_pelvis",
    spine_num_pts: int = 30,
) -> Optional[np.ndarray]:
    """Thin wrapper around the existing spine centerline extractor (legacy logic)."""
    _require_kidney_deps()
    return extract_spine_centerline_curve_xyz(
        fitter,
        skel_verts,
        skel_faces,
        intrinsics,
        num_pts=int(spine_num_pts),
        region=str(spine_region),
    )


def _overlay_out_path(img_path: str, suffix: str, ext: str = ".png") -> str:
    base = os.path.splitext(img_path)[0]
    return base + suffix + ext


def save_thorax_spine_lowest_overlay(
    *,
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    orig_img_bgr: np.ndarray,
    focal_length_val: float,
    intrinsics: Optional[dict] = None,
    spine_curve_xyz: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    save_xyz_npy: bool = True,
    margin_m: float = 0.01,
) -> Optional[str]:
    """
    Render thorax SKEL mesh on the stitched RGB and mark the thorax-spine lowest point.

    Intended as an auxiliary export during trajectory generation (cliff_skel_trajectory).
    Returns the saved PNG path, or None on failure.
    """
    import cv2

    from common.renderer_pyrd import Renderer
    from cliff_skel_trajectory import extract_specific_bone_mesh

    _require_kidney_deps()

    img_bgr = orig_img_bgr
    if img_bgr is None:
        img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    focal = float(focal_length_val)
    intrinsics_use = intrinsics or {
        "fx": focal,
        "fy": focal,
        "cx": w * 0.5,
        "cy": h * 0.5,
        "width": int(w),
        "height": int(h),
    }

    spine = spine_curve_xyz
    if spine is None:
        spine = build_spine_curve_for_v2(
            fitter,
            skel_verts,
            skel_faces,
            intrinsics_use,
        )
    if spine is None or len(spine) < 3:
        return None

    lowest_xyz = extract_thorax_spine_lowest_point_xyz(
        fitter,
        skel_verts,
        spine,
        skel_faces=skel_faces,
        img_path=img_path,
        margin_m=margin_m,
    )
    if lowest_xyz is None:
        return None

    tx_verts, tx_faces = extract_specific_bone_mesh(
        fitter,
        skel_verts,
        skel_faces,
        "thorax",
        img_path,
        threshold=0.1,
    )
    if tx_verts is None:
        return None

    renderer = Renderer(
        focal_length=focal,
        img_w=w,
        img_h=h,
        faces=tx_faces,
        same_mesh_color=True,
    )
    overlay_rgb = renderer.render_front_view(
        [tx_verts],
        bg_img_rgb=img_bgr[:, :, ::-1].copy(),
    )
    renderer.delete()
    vis = overlay_rgb[:, :, ::-1].copy()

    # Match Renderer principal point (image center).
    intrinsics_overlay = {
        "fx": focal,
        "fy": focal,
        "cx": w * 0.5,
        "cy": h * 0.5,
    }
    uv = project_points_to_uv(
        np.asarray(lowest_xyz, dtype=np.float32).reshape(1, 3),
        intrinsics_overlay,
    )
    u = int(np.clip(round(float(uv[0, 0])), 0, w - 1))
    v = int(np.clip(round(float(uv[0, 1])), 0, h - 1))

    cv2.circle(vis, (u, v), 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 16, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        "thorax spine lowest",
        (u + 18, max(20, v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    out_png = out_path or _overlay_out_path(img_path, "_demo_thorax_spine_lowest", ".png")
    out_dir = os.path.dirname(os.path.abspath(img_path))
    if save_xyz_npy:
        np.save(
            os.path.join(out_dir, "thorax_spine_lowest_xyz_cam.npy"),
            np.asarray(lowest_xyz, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "thorax_spine_lowest_uv.npy"),
            uv.reshape(2).astype(np.float32),
        )

    cv2.imwrite(out_png, vis)
    return out_png


def save_lumbar_spine_highest_overlay(
    *,
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    orig_img_bgr: np.ndarray,
    focal_length_val: float,
    intrinsics: Optional[dict] = None,
    out_path: Optional[str] = None,
    save_xyz_npy: bool = True,
    transition_floor_above_thorax_m: float = 0.15,
    label: str = "lumbar spine highest",
) -> Optional[str]:
    """
    Render lumbar_body SKEL mesh on the stitched RGB and mark the highest lumbar
    spine point (superior L1 on the lumbar mesh near the spine column).
    """
    import cv2

    from common.renderer_pyrd import Renderer
    from cliff_skel_trajectory import extract_specific_bone_mesh

    _require_kidney_deps()

    img_bgr = orig_img_bgr
    if img_bgr is None:
        img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    focal = float(focal_length_val)

    highest_xyz = extract_lumbar_spine_highest_point_xyz(
        fitter,
        skel_verts,
        skel_faces=skel_faces,
        img_path=img_path,
        transition_floor_above_thorax_m=transition_floor_above_thorax_m,
    )
    if highest_xyz is None:
        return None

    lb_verts, lb_faces = extract_specific_bone_mesh(
        fitter,
        skel_verts,
        skel_faces,
        "lumbar_body",
        img_path,
        threshold=0.1,
    )
    if lb_verts is None:
        return None

    renderer = Renderer(
        focal_length=focal,
        img_w=w,
        img_h=h,
        faces=lb_faces,
        same_mesh_color=True,
    )
    overlay_rgb = renderer.render_front_view(
        [lb_verts],
        bg_img_rgb=img_bgr[:, :, ::-1].copy(),
    )
    renderer.delete()
    vis = overlay_rgb[:, :, ::-1].copy()

    intrinsics_overlay = {
        "fx": focal,
        "fy": focal,
        "cx": w * 0.5,
        "cy": h * 0.5,
    }
    uv = project_points_to_uv(
        np.asarray(highest_xyz, dtype=np.float32).reshape(1, 3),
        intrinsics_overlay,
    )
    u = int(np.clip(round(float(uv[0, 0])), 0, w - 1))
    v = int(np.clip(round(float(uv[0, 1])), 0, h - 1))

    cv2.circle(vis, (u, v), 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 16, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        label,
        (u + 18, max(20, v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    out_png = out_path or _overlay_out_path(img_path, "_demo_lumbar_spine_highest", ".png")
    out_dir = os.path.dirname(os.path.abspath(img_path))
    if save_xyz_npy:
        np.save(
            os.path.join(out_dir, "lumbar_spine_highest_xyz_cam.npy"),
            np.asarray(highest_xyz, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "lumbar_spine_highest_uv.npy"),
            uv.reshape(2).astype(np.float32),
        )

    cv2.imwrite(out_png, vis)
    return out_png


def save_thorax_mesh_left_lowest_overlay(
    *,
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    orig_img_bgr: np.ndarray,
    focal_length_val: float,
    intrinsics: Optional[dict] = None,
    out_path: Optional[str] = None,
    save_xyz_npy: bool = True,
    threshold: float = 0.1,
    lr_band_percentile: float = 15.0,
    label: str = "thorax mesh left lowest",
) -> Optional[str]:
    """
    Render thorax SKEL mesh on the stitched RGB and mark the left-lowest thorax mesh point.

    Auxiliary export during trajectory generation (does not affect trajectory outputs).
    """
    import cv2

    from common.renderer_pyrd import Renderer
    from cliff_skel_trajectory import extract_specific_bone_mesh

    _require_kidney_deps()

    img_bgr = orig_img_bgr
    if img_bgr is None:
        img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    focal = float(focal_length_val)
    intrinsics_use = intrinsics or {
        "fx": focal,
        "fy": focal,
        "cx": w * 0.5,
        "cy": h * 0.5,
        "width": int(w),
        "height": int(h),
    }

    tx_verts, tx_faces = extract_specific_bone_mesh(
        fitter,
        skel_verts,
        skel_faces,
        "thorax",
        img_path,
        threshold=float(threshold),
    )
    if tx_verts is None:
        return None

    lowest_xyz = extract_thorax_mesh_left_lowest_point_xyz(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        threshold=float(threshold),
        lr_band_percentile=float(lr_band_percentile),
        thorax_verts=tx_verts,
    )
    if lowest_xyz is None:
        return None

    renderer = Renderer(
        focal_length=focal,
        img_w=w,
        img_h=h,
        faces=tx_faces,
        same_mesh_color=True,
    )
    overlay_rgb = renderer.render_front_view(
        [tx_verts],
        bg_img_rgb=img_bgr[:, :, ::-1].copy(),
    )
    renderer.delete()
    vis = overlay_rgb[:, :, ::-1].copy()

    intrinsics_overlay = {
        "fx": focal,
        "fy": focal,
        "cx": w * 0.5,
        "cy": h * 0.5,
    }
    uv = project_points_to_uv(
        np.asarray(lowest_xyz, dtype=np.float32).reshape(1, 3),
        intrinsics_overlay,
    )
    u = int(np.clip(round(float(uv[0, 0])), 0, w - 1))
    v = int(np.clip(round(float(uv[0, 1])), 0, h - 1))

    cv2.circle(vis, (u, v), 12, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, (u, v), 16, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        label,
        (u + 18, max(20, v - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    out_png = out_path or _overlay_out_path(img_path, "_demo_thorax_mesh_left_lowest", ".png")
    out_dir = os.path.dirname(os.path.abspath(img_path))
    if save_xyz_npy:
        np.save(
            os.path.join(out_dir, "thorax_mesh_left_lowest_xyz_cam.npy"),
            np.asarray(lowest_xyz, dtype=np.float32),
        )
        np.save(
            os.path.join(out_dir, "thorax_mesh_left_lowest_uv.npy"),
            uv.reshape(2).astype(np.float32),
        )

    cv2.imwrite(out_png, vis)
    return out_png


# ---------------------------------------------------------------------------
# Left-kidney hybrid trajectory (experimental; does not replace legacy exports)
# ---------------------------------------------------------------------------

def _import_kidney_post():
    import kidney_spine_geometry as kidney_post

    return kidney_post


def compute_kidney_end_xyz_legacy(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    *,
    spine_curve_xyz: Optional[np.ndarray] = None,
    rib_trim_head_ratio: float = 0.40,
    kidney_side: str = "left",
    end_up_along_rib_m: float = 0.0,
    end_lr_band_percentile: float = 15.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    End point for the kidney offset line (same logic as legacy try_build_kidney).

    Returns:
        (end_xyz, rib_line_3d_trimmed) in camera coordinates.
    """
    kp = _import_kidney_post()
    side_lc = str(kidney_side or "left").strip().lower()
    rib_side = "left" if side_lc == "left" else "right"
    trim_r = float(rib_trim_head_ratio or 0.0)
    if side_lc == "left" and trim_r <= 0.0:
        trim_r = 0.40

    rib_line_3d = None
    try:
        _cross, rib_line_3d = kp.find_spine_costal_margin_cross_point_on_spine_curve(
            fitter=fitter,
            skel_verts=skel_verts,
            skel_faces=skel_faces,
            img_path=img_path,
            intrinsics=intrinsics,
            spine_curve_xyz=spine_curve_xyz,
            rib_trim_head_ratio=trim_r,
            side=rib_side,
        )
    except Exception:
        rib_line_3d = None

    if rib_line_3d is None or len(rib_line_3d) < 2:
        rib_line_3d = kp.extract_right_costal_margin(
            fitter,
            skel_verts,
            skel_faces,
            img_path,
            intrinsics,
            side=rib_side,
        )
    if rib_line_3d is None or len(rib_line_3d) < 2:
        return None, None

    rib_xyz = np.asarray(rib_line_3d, dtype=np.float32).reshape(-1, 3)
    rib_trimmed = kp._trim_head_points(rib_xyz, trim_r) if trim_r > 0 else rib_xyz
    if len(rib_trimmed) < 2:
        rib_trimmed = rib_xyz

    if rib_side == "left":
        end_xyz_raw = kp.find_costal_margin_end_point_leftest_lowest(
            fitter,
            skel_verts,
            rib_trimmed,
            lr_band_percentile=float(end_lr_band_percentile),
        )
    else:
        end_xyz_raw = kp.find_costal_margin_end_point_rightest_lowest(
            fitter,
            skel_verts,
            rib_trimmed,
            lr_band_percentile=float(end_lr_band_percentile),
        )
    if end_xyz_raw is None:
        end_xyz_raw = rib_trimmed[-1].astype(np.float32)

    if float(end_up_along_rib_m) > 1e-6:
        d_end = np.linalg.norm(rib_trimmed.reshape(-1, 3) - end_xyz_raw.reshape(1, 3), axis=1)
        end_idx = int(np.argmin(d_end)) if d_end.size > 0 else (len(rib_trimmed) - 1)
        end_xyz = kp._advance_point_along_polyline_by_arclen(
            rib_trimmed,
            start_idx=end_idx,
            delta_m=float(end_up_along_rib_m),
            direction="backward",
        )
    else:
        end_xyz = end_xyz_raw

    if end_xyz is None:
        return None, rib_trimmed
    return np.asarray(end_xyz, dtype=np.float32).reshape(3), rib_trimmed


def compute_kidney_end_xyz_v2_thorax_mesh(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    *,
    kidney_side: str = "left",
    end_lr_band_percentile: float = 15.0,
    thorax_mesh_threshold: float = 0.1,
) -> Optional[np.ndarray]:
    """
    V2 kidney endpoint (left kidney only): left-lowest vertex on the thorax mesh.

    Does not use rib-margin polyline or right-side mirror. ``end_up_along_rib_m`` is
    not applied here (no rib polyline to walk along).
    """
    side_lc = str(kidney_side or "left").strip().lower()
    if side_lc != "left":
        return None
    return extract_thorax_mesh_left_lowest_point_xyz(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        threshold=float(thorax_mesh_threshold),
        lr_band_percentile=float(end_lr_band_percentile),
    )


def build_kidney_offset_uv_from_start_end(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    intrinsics: dict,
    *,
    spine_curve_uv: Optional[np.ndarray] = None,
    spine_curve_xyz: Optional[np.ndarray] = None,
    kidney_line_shift_px: float = 8.0,
) -> Optional[np.ndarray]:
    """UV offset kidney line from 3D start/end (same post-process as legacy builder)."""
    kp = _import_kidney_post()
    uv_s = kp._project_xyz_to_uv_single(start_xyz, intrinsics)
    uv_e = kp._project_xyz_to_uv_single(end_xyz, intrinsics)
    if uv_s is None or uv_e is None:
        return None

    uv_s = np.asarray(uv_s, dtype=np.float64).reshape(2)
    uv_e = np.asarray(uv_e, dtype=np.float64).reshape(2)
    d = uv_e - uv_s
    dn = float(np.linalg.norm(d))
    if dn < 1e-6:
        return None
    d = d / dn
    n_down = kp._normal_down_from_dir_uv(d)

    uv_line = np.stack([uv_s, uv_e], axis=0).astype(np.float32)
    uv_line_shift = (uv_line + n_down.reshape(1, 2) * float(kidney_line_shift_px)).astype(np.float32)

    if spine_curve_xyz is not None and len(spine_curve_xyz) >= 5 and spine_curve_uv is not None:
        d_uv = (uv_line[1] - uv_line[0]).astype(np.float64)
        d_uvn = float(np.linalg.norm(d_uv))
        if d_uvn > 1e-6:
            d_uv = d_uv / d_uvn
            hit = kp._intersect_infinite_line_with_polyline_uv(
                line_p=uv_line_shift[0],
                line_dir=d_uv,
                poly_uv=spine_curve_uv,
                prefer_forward=True,
            )
            if hit is not None:
                hit_uv, _t = hit
                length = float(np.linalg.norm(uv_line_shift[1] - uv_line_shift[0]))
                uv_line_shift = np.stack(
                    [hit_uv, (hit_uv.astype(np.float64) + d_uv * length).astype(np.float32)],
                    axis=0,
                ).astype(np.float32)

    return uv_line_shift


# V2-only: extra UV shift along image +v (down / toward feet), after kidney_line_shift_px + spine snap.
V2_EXTRA_IMAGE_DOWN_SHIFT_PX = 25.0


def _shift_uv_line_image_down_px(uv_line: np.ndarray, down_px: float) -> np.ndarray:
    """Translate a (2,2) UV polyline by +v (image y toward bottom)."""
    pts = np.asarray(uv_line, dtype=np.float32).reshape(-1, 2)
    delta = np.array([0.0, float(down_px)], dtype=np.float32)
    return (pts + delta.reshape(1, 2)).astype(np.float32)


def try_build_left_kidney_offset_line_uv_legacy(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_uv: Optional[np.ndarray],
    spine_curve_xyz: Optional[np.ndarray],
    **kwargs: Any,
) -> Optional[np.ndarray]:
    """Thin wrapper around the existing legacy kidney trajectory builder."""
    kp = _import_kidney_post()
    return kp.try_build_kidney_offset_line_uv_shifted(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        intrinsics,
        spine_curve_uv,
        spine_curve_xyz,
        **kwargs,
    )


def try_build_left_kidney_offset_line_uv_v2(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_uv: Optional[np.ndarray],
    spine_curve_xyz: Optional[np.ndarray],
    *,
    rib_trim_head_ratio: float = 0.40,
    kidney_side: str = "left",
    end_up_along_rib_m: float = 0.0,
    kidney_line_shift_px: float = 8.0,
    end_lr_band_percentile: float = 15.0,
    transition_floor_above_thorax_m: float = 0.15,
    t12_above_lumbar_m: float = 0.005,
    extra_image_down_shift_px: float = V2_EXTRA_IMAGE_DOWN_SHIFT_PX,
) -> Optional[np.ndarray]:
    """
    Left-kidney v2: start at thorax-spine lowest (T12);
    end at left-lowest point on the thorax mesh (not rib-margin legacy).

    After the standard UV offset + spine snap, applies an additional +v image shift
    (``extra_image_down_shift_px``, default 25 px toward image bottom).
    """
    _require_kidney_deps()
    if spine_curve_xyz is None or len(spine_curve_xyz) < 3:
        return None

    start_xyz = extract_thorax_spine_lowest_point_xyz(
        fitter,
        skel_verts,
        spine_curve_xyz,
        skel_faces=skel_faces,
        img_path=img_path,
        transition_floor_above_thorax_m=transition_floor_above_thorax_m,
        t12_above_lumbar_m=t12_above_lumbar_m,
    )
    if start_xyz is None:
        return None

    end_xyz = compute_kidney_end_xyz_v2_thorax_mesh(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        kidney_side=kidney_side,
        end_lr_band_percentile=end_lr_band_percentile,
    )
    if end_xyz is None:
        return None

    uv_line_shift = build_kidney_offset_uv_from_start_end(
        start_xyz,
        end_xyz,
        intrinsics,
        spine_curve_uv=spine_curve_uv,
        spine_curve_xyz=spine_curve_xyz,
        kidney_line_shift_px=kidney_line_shift_px,
    )
    if uv_line_shift is None:
        return None
    if float(extra_image_down_shift_px) != 0.0:
        uv_line_shift = _shift_uv_line_image_down_px(uv_line_shift, extra_image_down_shift_px)
    return uv_line_shift


KIDNEY_START_MODE_LEGACY = "legacy"
KIDNEY_START_MODE_THORAX_SPINE_LOWEST = "thorax_spine_lowest"
KIDNEY_START_MODE_DEFAULT = KIDNEY_START_MODE_THORAX_SPINE_LOWEST
KIDNEY_START_MODES = (KIDNEY_START_MODE_LEGACY, KIDNEY_START_MODE_THORAX_SPINE_LOWEST)


def normalize_kidney_start_mode(mode: Optional[str]) -> str:
    m = str(mode or KIDNEY_START_MODE_DEFAULT).strip().lower()
    if m in ("v2", "t12", "thorax_lowest"):
        m = KIDNEY_START_MODE_THORAX_SPINE_LOWEST
    if m not in KIDNEY_START_MODES:
        m = KIDNEY_START_MODE_DEFAULT
    return m


def build_kidney_offset_line_uv_shifted_by_mode(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_uv: Optional[np.ndarray],
    spine_curve_xyz: Optional[np.ndarray],
    *,
    kidney_start_mode: str = KIDNEY_START_MODE_DEFAULT,
    cross_to_start_len_m: float = 0.03,
    cross_to_start_dir: str = "inferior",
    end_up_along_rib_m: float = 0.0,
    kidney_line_shift_px: float = 8.0,
    rib_trim_head_ratio: float = 0.40,
    kidney_side: str = "left",
    end_lr_band_percentile: float = 15.0,
    transition_floor_above_thorax_m: float = 0.15,
    t12_above_lumbar_m: float = 0.005,
    extra_image_down_shift_px: float = V2_EXTRA_IMAGE_DOWN_SHIFT_PX,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Build kidney offset UV line using the selected start-point strategy.

    Returns:
        (uv_line_shifted, mode_used)
    """
    mode = normalize_kidney_start_mode(kidney_start_mode)
    side_lc = str(kidney_side or "left").strip().lower()

    if mode == KIDNEY_START_MODE_THORAX_SPINE_LOWEST:
        if side_lc != "left":
            print(
                "[kidney] thorax_spine_lowest start applies to left kidney only; using legacy"
            )
            mode = KIDNEY_START_MODE_LEGACY
        else:
            uv = try_build_left_kidney_offset_line_uv_v2(
                fitter,
                skel_verts,
                skel_faces,
                img_path,
                intrinsics,
                spine_curve_uv,
                spine_curve_xyz,
                rib_trim_head_ratio=rib_trim_head_ratio,
                kidney_side=kidney_side,
                end_up_along_rib_m=end_up_along_rib_m,
                kidney_line_shift_px=kidney_line_shift_px,
                end_lr_band_percentile=end_lr_band_percentile,
                transition_floor_above_thorax_m=transition_floor_above_thorax_m,
                t12_above_lumbar_m=t12_above_lumbar_m,
                extra_image_down_shift_px=extra_image_down_shift_px,
            )
            return uv, mode

    uv = try_build_left_kidney_offset_line_uv_legacy(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        intrinsics,
        spine_curve_uv,
        spine_curve_xyz,
        cross_to_start_len_m=cross_to_start_len_m,
        cross_to_start_dir=cross_to_start_dir,
        end_up_along_rib_m=end_up_along_rib_m,
        kidney_line_shift_px=kidney_line_shift_px,
        rib_trim_head_ratio=rib_trim_head_ratio,
        kidney_side=kidney_side,
        end_lr_band_percentile=end_lr_band_percentile,
    )
    return uv, mode


def build_left_kidney_hybrid_uv_trajectories(
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_uv: Optional[np.ndarray],
    spine_curve_xyz: Optional[np.ndarray],
    *,
    cross_to_start_len_m: float = 0.03,
    cross_to_start_dir: str = "inferior",
    end_up_along_rib_m: float = 0.0,
    kidney_line_shift_px: float = 8.0,
    rib_trim_head_ratio: float = 0.40,
    kidney_side: str = "left",
    end_lr_band_percentile: float = 15.0,
    transition_floor_above_thorax_m: float = 0.15,
    t12_above_lumbar_m: float = 0.005,
) -> Dict[str, Any]:
    """
    Build legacy and v2 left-kidney UV trajectories side-by-side.

    Returns dict with keys:
      - legacy_uv: rib-cross start (existing pipeline)
      - v2_uv: thorax-spine-lowest start, thorax-mesh left-lowest end
    """
    legacy_uv = try_build_left_kidney_offset_line_uv_legacy(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        intrinsics,
        spine_curve_uv,
        spine_curve_xyz,
        cross_to_start_len_m=cross_to_start_len_m,
        cross_to_start_dir=cross_to_start_dir,
        end_up_along_rib_m=end_up_along_rib_m,
        kidney_line_shift_px=kidney_line_shift_px,
        rib_trim_head_ratio=rib_trim_head_ratio,
        kidney_side=kidney_side,
        end_lr_band_percentile=end_lr_band_percentile,
    )
    v2_uv = try_build_left_kidney_offset_line_uv_v2(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        intrinsics,
        spine_curve_uv,
        spine_curve_xyz,
        rib_trim_head_ratio=rib_trim_head_ratio,
        kidney_side=kidney_side,
        end_up_along_rib_m=end_up_along_rib_m,
        kidney_line_shift_px=kidney_line_shift_px,
        end_lr_band_percentile=end_lr_band_percentile,
        transition_floor_above_thorax_m=transition_floor_above_thorax_m,
        t12_above_lumbar_m=t12_above_lumbar_m,
    )
    return {
        "legacy_uv": legacy_uv,
        "v2_uv": v2_uv,
        "legacy_start_mode": "rib_costal_margin_cross",
        "v2_start_mode": "thorax_spine_lowest",
    }


def export_left_kidney_v2_trajectory(
    *,
    fitter: Any,
    skel_verts: np.ndarray,
    skel_faces: np.ndarray,
    img_path: str,
    intrinsics: dict,
    spine_curve_uv: Optional[np.ndarray],
    spine_curve_xyz: Optional[np.ndarray],
    skel_overlay_bgr: Optional[np.ndarray] = None,
    cross_to_start_len_m: float = 0.03,
    cross_to_start_dir: str = "inferior",
    end_up_along_rib_m: float = 0.0,
    kidney_line_shift_px: float = 8.0,
    rib_trim_head_ratio: float = 0.40,
    kidney_side: str = "left",
    end_lr_band_percentile: float = 15.0,
    transition_floor_above_thorax_m: float = 0.15,
    t12_above_lumbar_m: float = 0.005,
    save_hybrid_npy: bool = True,
    save_overlay: bool = True,
) -> Dict[str, Any]:
    """
    Export experimental v2 left-kidney outputs without overwriting legacy files.

    Writes (when available):
      - kidney_line_uv_shifted_v2.npy
      - kidney_line_uv_shifted_hybrid_legacy.npy  (copy of legacy for side-by-side)
      - rgb_stitched_*_demo_skel_overlay_kidney_offset_only_v2.png
    """
    out_dir = os.path.dirname(os.path.abspath(img_path))
    base = os.path.splitext(os.path.basename(img_path))[0]

    hybrid = build_left_kidney_hybrid_uv_trajectories(
        fitter,
        skel_verts,
        skel_faces,
        img_path,
        intrinsics,
        spine_curve_uv,
        spine_curve_xyz,
        cross_to_start_len_m=cross_to_start_len_m,
        cross_to_start_dir=cross_to_start_dir,
        end_up_along_rib_m=end_up_along_rib_m,
        kidney_line_shift_px=kidney_line_shift_px,
        rib_trim_head_ratio=rib_trim_head_ratio,
        kidney_side=kidney_side,
        end_lr_band_percentile=end_lr_band_percentile,
        transition_floor_above_thorax_m=transition_floor_above_thorax_m,
        t12_above_lumbar_m=t12_above_lumbar_m,
    )

    result: Dict[str, Any] = dict(hybrid)
    if save_hybrid_npy:
        if hybrid.get("v2_uv") is not None:
            p_v2 = os.path.join(out_dir, "kidney_line_uv_shifted_v2.npy")
            np.save(p_v2, np.asarray(hybrid["v2_uv"], dtype=np.float32))
            result["v2_npy"] = p_v2
        if hybrid.get("legacy_uv") is not None:
            p_leg = os.path.join(out_dir, "kidney_line_uv_shifted_hybrid_legacy.npy")
            np.save(p_leg, np.asarray(hybrid["legacy_uv"], dtype=np.float32))
            result["hybrid_legacy_npy"] = p_leg

    if save_overlay and skel_overlay_bgr is not None and hybrid.get("v2_uv") is not None:
        import cv2

        try:
            from cliff_skel_trajectory import _draw_uv_polyline
        except Exception:
            _draw_uv_polyline = None  # type: ignore

        if _draw_uv_polyline is not None:
            vis = skel_overlay_bgr.copy()
            vis = _draw_uv_polyline(vis, hybrid["v2_uv"], color=(0, 255, 0), thickness=4)
            if hybrid.get("legacy_uv") is not None:
                vis = _draw_uv_polyline(vis, hybrid["legacy_uv"], color=(0, 0, 255), thickness=2)
            out_png = os.path.join(out_dir, f"{base}_demo_skel_overlay_kidney_offset_only_v2.png")
            cv2.imwrite(out_png, vis)
            result["overlay_v2_png"] = out_png

    return result
